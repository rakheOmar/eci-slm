from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


GENERAL_PROMPTS = [
    "Artificial intelligence is transforming",
    "The future of technology will depend on",
    "Education plays a crucial role in",
]

ECI_PROMPTS = [
    "Nomination papers must be filed",
    "The Returning Officer shall",
    "During elections, the Election Commission ensures",
]

QNA_PROMPTS = [
    "Can nomination papers be sent by email?",
    "Who can submit a nomination paper?",
    "What happens if nomination papers are incomplete?",
]

FEWSHOT_EXAMPLES = [
    (
        "Can nomination papers be sent by email?",
        "No. Nomination papers must be submitted physically to the Returning Officer.",
    ),
    (
        "Who can submit a nomination paper?",
        "A nomination paper may be submitted by the candidate or by any of the candidate's proposers.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ECI-SLM checkpoints")
    parser.add_argument("--repo-root", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--device", type=str, choices=["auto", "gpu", "cpu"], default="auto"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "mixed_fp16", "mixed_bf16"],
        default="fp32",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-theta", type=float, default=100000.0)
    parser.add_argument("--untied-head", action="store_true")
    return parser.parse_args()


def resolve_repo_root(arg_value: str) -> Path:
    if arg_value:
        return Path(arg_value).resolve()
    return Path(__file__).resolve().parent.parent


def configure_runtime(tf, device: str, precision: str) -> None:
    gpus = tf.config.list_physical_devices("GPU")

    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        gpus = []
    elif device == "gpu" and not gpus:
        raise RuntimeError("--device gpu requested but no GPU is available")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if precision == "mixed_fp16":
        if gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        else:
            tf.keras.mixed_precision.set_global_policy("float32")
    elif precision == "mixed_bf16":
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    visible = tf.config.list_physical_devices("GPU")
    print(
        f"Visible GPUs: {len(visible)} | Precision: {tf.keras.mixed_precision.global_policy().name}"
    )


def resolve_tokenizer_path(
    repo_root: Path, checkpoint_dir: Path, tokenizer_path: str
) -> Path:
    if tokenizer_path:
        path = Path(tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {path}")
        return path

    candidates = [
        repo_root / "artifact" / "eci_slm_tokenizer.model",
        checkpoint_dir.parent / "artifact" / "eci_slm_tokenizer.model",
        Path.cwd() / "artifact" / "eci_slm_tokenizer.model",
    ]
    for path in candidates:
        if path.exists():
            return path
    readable = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Tokenizer model not found. Tried:\n{readable}")


def resolve_step(checkpoint_dir: Path, explicit_step: int, find_last_step) -> int:
    if explicit_step > 0:
        return explicit_step
    best_file = checkpoint_dir / "best_step.txt"
    if best_file.exists():
        raw = best_file.read_text(encoding="utf-8").strip()
        try:
            return int(raw)
        except ValueError:
            pass
    return int(find_last_step(checkpoint_dir))


def build_fewshot_prefix() -> str:
    chunks: list[str] = []
    for user, assistant in FEWSHOT_EXAMPLES:
        chunks.append(f"User: {user}\nAssistant: {assistant}")
    return "\n\n".join(chunks)


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import tensorflow as tf
    from src.checkpoint import find_last_step, load_checkpoint
    from src.slm import ECISLM, SLMConfig
    from src.tokenizer import Tokenizer

    configure_runtime(tf, args.device, args.precision)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    tokenizer_model = resolve_tokenizer_path(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        tokenizer_path=args.tokenizer_path,
    )
    step = resolve_step(checkpoint_dir, args.step, find_last_step)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_model)
    cfg = SLMConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        tie_embeddings=not args.untied_head,
    )

    model = ECISLM(cfg)
    _ = model(tf.zeros((1, cfg.block_size), dtype=tf.int32))
    _ = load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step,
        model=model,
        optimizer=None,
    )
    print(f"Loaded checkpoint step {step} from {checkpoint_dir}")

    def generate(prompt: str) -> str:
        input_ids = tokenizer.encode(prompt)
        x = tf.constant([input_ids], dtype=tf.int32)
        y = (
            model.generate(
                idx=x,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            .numpy()[0]
            .tolist()
        )

        text = tokenizer.decode(y[len(input_ids) :])
        for marker in ["\nUser:", "<END>"]:
            pos = text.find(marker)
            if pos != -1:
                text = text[:pos]
        return text.strip()

    fewshot_prefix = build_fewshot_prefix()
    results: list[dict[str, str]] = []

    def run_block(name: str, prompts: list[str], formatter) -> None:
        print(f"\n=== {name} ===\n")
        for prompt in prompts:
            full_prompt = formatter(prompt)
            output = generate(full_prompt)
            print(f"PROMPT: {prompt}")
            print(f"OUTPUT: {output}")
            print("-" * 80)
            results.append({"type": name, "prompt": prompt, "output": output})

    run_block("general_continuation", GENERAL_PROMPTS, lambda p: p)
    run_block("eci_continuation", ECI_PROMPTS, lambda p: p)
    run_block("qna_zero_shot", QNA_PROMPTS, lambda p: f"User: {p}\nAssistant: ")
    run_block(
        "qna_few_shot",
        QNA_PROMPTS,
        lambda p: f"{fewshot_prefix}\n\nUser: {p}\nAssistant: ",
    )

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "prompt", "output"])
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
