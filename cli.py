#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ECI-SLM CLI")
    parser.add_argument("--repo-root", type=str, default="")
    parser.add_argument("--models-dir", type=str, default="model")
    parser.add_argument(
        "--tokenizer-path", type=str, default="artifact/eci_slm_tokenizer.model"
    )
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=160)
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
    return Path(__file__).resolve().parent


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

    print(
        f"Visible GPUs: {len(tf.config.list_physical_devices('GPU'))} | "
        f"Precision: {tf.keras.mixed_precision.global_policy().name}"
    )


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


def list_models(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    dirs = [p for p in sorted(models_dir.iterdir()) if p.is_dir()]
    valid = []
    for d in dirs:
        if any(d.glob("model_*.weights.h5")):
            valid.append(d)
    return valid


def choose_model(models: list[Path]) -> Path:
    if not models:
        raise FileNotFoundError("No model directories found under model/")

    print("\nAvailable models:")
    for i, path in enumerate(models, start=1):
        print(f"  [{i}] {path.name}")

    while True:
        raw = input("Select model by number or name: ").strip()
        if not raw:
            continue

        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]

        for path in models:
            if path.name == raw:
                return path

        print("Invalid selection. Try again.")


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

    models_dir = (repo_root / args.models_dir).resolve()
    tokenizer_path = (repo_root / args.tokenizer_path).resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer(tokenizer_path)
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

    def load_selected_model(checkpoint_dir: Path) -> tuple[Path, int]:
        step = resolve_step(
            checkpoint_dir=checkpoint_dir,
            explicit_step=args.step,
            find_last_step=find_last_step,
        )
        _ = load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=step,
            model=model,
            optimizer=None,
        )
        return checkpoint_dir, step

    model_dirs = list_models(models_dir)
    selected = choose_model(model_dirs)
    current_dir, current_step = load_selected_model(selected)
    print(f"Loaded model: {current_dir.name} (step {current_step})")

    print("\nInteractive mode:")
    print("  - Type your prompt and press Enter")
    print("  - /reset clears conversation history")
    print("  - /models lists models")
    print("  - /switch <num|name> switches model")
    print("  - /exit quits")

    history: list[tuple[str, str]] = []

    def build_prompt(user_text: str) -> str:
        chunks: list[str] = []
        for user, assistant in history:
            chunks.append(f"User: {user}\nAssistant: {assistant}")
        chunks.append(f"User: {user_text}\nAssistant: ")
        return "\n\n".join(chunks)

    def generate_text(prompt: str) -> str:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > cfg.block_size:
            input_ids = input_ids[-cfg.block_size :]

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

        text = tokenizer.decode(y[len(input_ids) :]).strip()
        for marker in ["\nUser:", "<END>"]:
            pos = text.find(marker)
            if pos != -1:
                text = text[:pos]
        return text.strip()

    while True:
        user_in = input("\nYou: ").strip()
        if not user_in:
            continue

        if user_in == "/exit":
            print("Bye.")
            break

        if user_in == "/reset":
            history.clear()
            print("History cleared.")
            continue

        if user_in == "/models":
            model_dirs = list_models(models_dir)
            if not model_dirs:
                print("No models found.")
            else:
                print("Available models:")
                for i, p in enumerate(model_dirs, start=1):
                    mark = "*" if p.resolve() == current_dir.resolve() else " "
                    print(f" {mark} [{i}] {p.name}")
            continue

        if user_in.startswith("/switch"):
            parts = user_in.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /switch <num|name>")
                continue

            model_dirs = list_models(models_dir)
            pick = parts[1].strip()
            target: Path | None = None

            if pick.isdigit():
                idx = int(pick)
                if 1 <= idx <= len(model_dirs):
                    target = model_dirs[idx - 1]
            else:
                for p in model_dirs:
                    if p.name == pick:
                        target = p
                        break

            if target is None:
                print("Invalid model selection.")
                continue

            current_dir, current_step = load_selected_model(target)
            history.clear()
            print(
                f"Switched to {current_dir.name} (step {current_step}); history cleared."
            )
            continue

        prompt = build_prompt(user_in)
        answer = generate_text(prompt)
        print(f"\nModel ({current_dir.name}): {answer}")
        history.append((user_in, answer))


if __name__ == "__main__":
    main()
