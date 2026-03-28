#!/usr/bin/env python3
"""ECI-SLM entrypoint.

Clean pipeline for:
- tokenizer training
- pretrain/SFT bin building
- training with checkpoint + resume
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.checkpoint import CheckpointManager, load_checkpoint
from src.dataloader import Dataloader
from src.sft import (
    IGNORE_INDEX,
    SFTBatchLoader,
    build_sft_arrays,
    load_sft_split,
    save_sft_split,
    split_sft_arrays,
)
from src.slm import ECISLM, SLMConfig, count_parameters
from src.tokenizer import Tokenizer, TokenizerConfig


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
ARTIFACT_ROOT = ROOT / "artifact"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECI-SLM pipeline")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["prepare", "train", "prepare_and_train"],
        help="prepare: build tokenizer+bins, train: run training, prepare_and_train: both",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrain",
        choices=["pretrain", "sft"],
        help="Data stage to build/train on",
    )

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--precision",
        type=str,
        default="mixed_fp16",
        choices=["fp32", "mixed_bf16", "mixed_fp16"],
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "mirrored", "single", "cpu"],
        help="Distributed strategy selection",
    )

    parser.add_argument(
        "--tokenizer", type=str, default="artifact/eci_slm_tokenizer.model"
    )
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--model_prefix", type=str, default="eci_slm_tokenizer")
    parser.add_argument(
        "--force_train_tokenizer",
        action="store_true",
        help="Retrain tokenizer even if tokenizer model already exists",
    )

    parser.add_argument("--data_dir", type=str, default="artifact")
    parser.add_argument("--sft_bin_dir", type=str, default="artifact_sft")
    parser.add_argument("--rebuild_bins", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.05)

    parser.add_argument(
        "--english_ratio",
        type=float,
        default=0.95,
        help="English ratio in mixed pretrain bins (ECI ratio is 1-english_ratio)",
    )
    parser.add_argument("--mix_chunk_tokens", type=int, default=4096)

    parser.add_argument(
        "--sft_source_dirs",
        type=str,
        default="data/instruct",
        help="Comma-separated directories of .txt files for SFT bins",
    )

    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--min_lr_frac", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--sft_learning_rate", type=float, default=1e-5)
    parser.add_argument("--sft_warmup_steps", type=int, default=200)
    parser.add_argument("--sft_weight_decay", type=float, default=0.01)

    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_kv_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rope_theta", type=float, default=100000.0)
    parser.add_argument(
        "--untied_head",
        action="store_true",
        help="Disable embedding-head tying (increases params)",
    )

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--init_checkpoint_dir", type=str, default="")
    parser.add_argument("--init_step", type=int, default=0)
    parser.add_argument("--keep_last_n", type=int, default=5)

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--num_val_batches", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=200)

    parser.add_argument(
        "--warmup_cap_frac",
        type=float,
        default=0.1,
        help="Cap warmup to this fraction of max_steps to avoid LR flatline",
    )
    parser.add_argument(
        "--min_improve",
        type=float,
        default=1e-4,
        help="Minimum val-loss improvement to reset plateau counters",
    )
    parser.add_argument(
        "--plateau_patience_evals",
        type=int,
        default=4,
        help="Evals without improvement before reducing LR scale",
    )
    parser.add_argument(
        "--plateau_lr_decay",
        type=float,
        default=0.6,
        help="LR scale multiplier when hitting plateau",
    )
    parser.add_argument(
        "--min_lr_scale",
        type=float,
        default=0.2,
        help="Lower bound for plateau LR scaling",
    )
    parser.add_argument(
        "--early_stop_patience_evals",
        type=int,
        default=12,
        help="Stop if no val improvement for this many evals (0 disables)",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_precision(policy: str) -> None:
    if not tf.config.list_physical_devices("GPU") and policy != "fp32":
        print("No GPU detected, forcing fp32 precision for stability")
        tf.keras.mixed_precision.set_global_policy("float32")
        return

    if policy == "fp32":
        tf.keras.mixed_precision.set_global_policy("float32")
    elif policy == "mixed_bf16":
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    elif policy == "mixed_fp16":
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def resolve_strategy(mode: str) -> tf.distribute.Strategy:
    gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(gpus)

    if mode == "cpu":
        return tf.distribute.OneDeviceStrategy(device="/CPU:0")

    if mode == "mirrored":
        if num_gpus < 2:
            raise ValueError(f"--strategy mirrored requires >=2 GPUs, found {num_gpus}")
        return tf.distribute.MirroredStrategy()

    if mode == "single":
        if num_gpus >= 1:
            return tf.distribute.OneDeviceStrategy(device="/GPU:0")
        return tf.distribute.OneDeviceStrategy(device="/CPU:0")

    # auto
    if num_gpus > 1:
        return tf.distribute.MirroredStrategy()
    if num_gpus == 1:
        return tf.distribute.OneDeviceStrategy(device="/GPU:0")
    return tf.distribute.get_strategy()


def make_batch_dataset(
    get_batch_fn,
    batch_size: int,
    block_size: int,
) -> tf.data.Dataset:
    def gen():
        while True:
            x, y = get_batch_fn()
            yield x, y

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((batch_size, block_size), tf.int32),
            tf.TensorSpec((batch_size, block_size), tf.int32),
        ),
    ).prefetch(tf.data.AUTOTUNE)


def _combine_text_files(paths: list[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out:
        for i, p in enumerate(paths):
            out.write(p.read_text(encoding="utf-8", errors="replace"))
            if i < len(paths) - 1:
                out.write("\n\n")


def _collect_txt_files(dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.txt")))
    return files


def _eci_and_english_files() -> tuple[list[Path], list[Path]]:
    eci_dirs = [
        DATA_ROOT / "pretrain",
        DATA_ROOT / "pretrain_expanded",
        DATA_ROOT / "pretrain_augmented",
    ]
    english_dir = DATA_ROOT / "english_pretrain"

    eci_files = _collect_txt_files(eci_dirs)
    english_files = _collect_txt_files([english_dir])
    return eci_files, english_files


def load_or_train_tokenizer(args: argparse.Namespace) -> Tokenizer:
    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.exists() and not args.force_train_tokenizer:
        return Tokenizer(tokenizer_path)

    pretrain_dirs = [
        DATA_ROOT / "pretrain",
        DATA_ROOT / "pretrain_expanded",
        DATA_ROOT / "pretrain_augmented",
        DATA_ROOT / "english_pretrain",
    ]
    texts = _collect_txt_files(pretrain_dirs)
    if not texts:
        raise FileNotFoundError("No .txt files found to train tokenizer")

    combined = DATA_ROOT / "combined_train.txt"
    _combine_text_files(texts, combined)

    tokenizer = Tokenizer()
    tokenizer.train(
        combined,
        TokenizerConfig(vocab_size=args.vocab_size, model_prefix=args.model_prefix),
    )
    return tokenizer


def _tokenize_files(tokenizer: Tokenizer, files: list[Path]) -> np.ndarray:
    eos = tokenizer.sp.eos_id() if tokenizer.sp is not None else -1
    chunks: list[np.ndarray] = []

    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode(text)
        if eos >= 0:
            ids.append(eos)
        chunks.append(np.asarray(ids, dtype=np.uint16))

    if not chunks:
        return np.asarray([], dtype=np.uint16)
    return np.concatenate(chunks)


def _interleave(
    eci_ids: np.ndarray,
    english_ids: np.ndarray,
    chunk: int,
    target_eci: int,
    target_english: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    i = 0
    j = 0

    while i < target_eci or j < target_english:
        if i >= target_eci:
            pick_english = True
        elif j >= target_english:
            pick_english = False
        else:
            pick_english = (j / max(1, target_english)) < (i / max(1, target_eci))

        if pick_english:
            j2 = min(j + chunk, target_english)
            out.append(english_ids[j:j2])
            j = j2
        else:
            i2 = min(i + chunk, target_eci)
            out.append(eci_ids[i:i2])
            i = i2

    if not out:
        return np.asarray([], dtype=np.uint16)
    return np.concatenate(out)


def save_bins(ids: np.ndarray, out_dir: Path, val_split: float) -> tuple[Path, Path]:
    if ids.size < 2:
        raise RuntimeError("Tokenized corpus too small to split train/val")
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be in (0, 1)")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_idx = int((1.0 - val_split) * len(ids))
    split_idx = min(max(split_idx, 1), len(ids) - 1)
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"Created {train_path} ({len(train_ids):,} tokens)")
    print(f"Created {val_path} ({len(val_ids):,} tokens)")
    return train_path, val_path


def build_pretrain_bins(
    tokenizer: Tokenizer,
    out_dir: Path,
    english_ratio: float,
    val_split: float,
    mix_chunk_tokens: int,
) -> tuple[Path, Path]:
    if not (0.0 < english_ratio < 1.0):
        raise ValueError("english_ratio must be in (0, 1)")
    if mix_chunk_tokens < 1:
        raise ValueError("mix_chunk_tokens must be >= 1")

    eci_files, english_files = _eci_and_english_files()
    if not eci_files:
        raise FileNotFoundError("No ECI files found under data/pretrain* directories")
    if not english_files:
        raise FileNotFoundError("No English files found under data/english_pretrain")

    print(
        f"Tokenizing pretrain corpus: {len(eci_files)} ECI files + {len(english_files)} English files"
    )
    eci_ids = _tokenize_files(tokenizer, eci_files)
    eng_ids = _tokenize_files(tokenizer, english_files)

    if eci_ids.size == 0 or eng_ids.size == 0:
        raise RuntimeError("One side has zero tokens after tokenization")

    eci_ratio = 1.0 - english_ratio
    max_total = min(eci_ids.size / eci_ratio, eng_ids.size / english_ratio)
    total_target = int(max_total)
    target_eci = min(int(total_target * eci_ratio), int(eci_ids.size))
    target_eng = min(int(total_target * english_ratio), int(eng_ids.size))

    mixed = _interleave(
        eci_ids=eci_ids,
        english_ids=eng_ids,
        chunk=mix_chunk_tokens,
        target_eci=target_eci,
        target_english=target_eng,
    )

    achieved = target_eng / max(1, (target_eci + target_eng))
    print(
        f"Mixed tokens: ECI={target_eci:,}, English={target_eng:,}, total={len(mixed):,}, "
        f"english_ratio={achieved:.3f}"
    )
    return save_bins(mixed, out_dir=out_dir, val_split=val_split)


def build_sft_bins(
    tokenizer: Tokenizer,
    source_dirs: list[Path],
    out_dir: Path,
    val_split: float,
    block_size: int,
    seed: int,
) -> tuple[Path, Path]:
    files = _collect_txt_files(source_dirs)
    if not files:
        readable = ", ".join(str(p) for p in source_dirs)
        raise FileNotFoundError(f"No .txt files found in SFT source dirs: {readable}")
    print(f"Building masked SFT examples from {len(files)} files")

    x, y = build_sft_arrays(
        tokenizer=tokenizer,
        source_files=files,
        block_size=block_size,
    )
    x_train, y_train, x_val, y_val = split_sft_arrays(
        x=x,
        y=y,
        val_split=val_split,
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = save_sft_split(out_dir / "train_sft.npz", x_train, y_train)
    val_path = save_sft_split(out_dir / "val_sft.npz", x_val, y_val)

    print(
        f"SFT examples: train={len(x_train):,}, val={len(x_val):,}, block_size={x.shape[1]}"
    )
    print(f"Created {train_path}")
    print(f"Created {val_path}")
    return train_path, val_path


def maybe_prepare(args: argparse.Namespace, tokenizer: Tokenizer) -> tuple[Path, Path]:
    if args.stage == "pretrain":
        out_dir = Path(args.data_dir)
        train_path = out_dir / "train.bin"
        val_path = out_dir / "val.bin"
        if args.mode in {"prepare", "prepare_and_train"} or args.rebuild_bins:
            return build_pretrain_bins(
                tokenizer=tokenizer,
                out_dir=out_dir,
                english_ratio=args.english_ratio,
                val_split=args.val_split,
                mix_chunk_tokens=args.mix_chunk_tokens,
            )
        if train_path.exists() and val_path.exists():
            return train_path, val_path
        return build_pretrain_bins(
            tokenizer=tokenizer,
            out_dir=out_dir,
            english_ratio=args.english_ratio,
            val_split=args.val_split,
            mix_chunk_tokens=args.mix_chunk_tokens,
        )

    source_dirs = [
        Path(p.strip()) for p in args.sft_source_dirs.split(",") if p.strip()
    ]
    if not source_dirs:
        raise ValueError("sft_source_dirs must include at least one directory")

    out_dir = Path(args.sft_bin_dir)
    train_path = out_dir / "train_sft.npz"
    val_path = out_dir / "val_sft.npz"
    if args.mode in {"prepare", "prepare_and_train"} or args.rebuild_bins:
        return build_sft_bins(
            tokenizer=tokenizer,
            source_dirs=source_dirs,
            out_dir=out_dir,
            val_split=args.val_split,
            block_size=args.block_size,
            seed=args.seed,
        )
    if train_path.exists() and val_path.exists():
        return train_path, val_path
    return build_sft_bins(
        tokenizer=tokenizer,
        source_dirs=source_dirs,
        out_dir=out_dir,
        val_split=args.val_split,
        block_size=args.block_size,
        seed=args.seed,
    )


def build_lr_scheduler(
    learning_rate: float,
    min_lr_frac: float,
    warmup_steps: int,
    total_steps: int,
):
    warmup = max(1, warmup_steps)
    min_lr = learning_rate * min_lr_frac

    def lr_at(step: int) -> float:
        if step < warmup:
            return learning_rate * (step + 1) / warmup
        if total_steps <= warmup:
            return min_lr
        progress = (step - warmup) / (total_steps - warmup)
        progress = float(min(1.0, max(0.0, progress)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (learning_rate - min_lr) * cosine

    return lr_at


def evaluate(model: ECISLM, val_loader: Dataloader, num_batches: int) -> float:
    losses: list[float] = []
    for _ in range(num_batches):
        x, y = val_loader.get_batch()
        x_t = tf.convert_to_tensor(x, dtype=tf.int32)
        y_t = tf.convert_to_tensor(y, dtype=tf.int32)
        _, loss = model.call(x_t, training=False, targets=y_t)
        if loss is None:
            raise RuntimeError("Model returned no loss during evaluation")
        losses.append(float(loss.numpy()))
    return float(np.mean(losses)) if losses else float("nan")


def train_one_step(
    model: ECISLM,
    optimizer: tf.keras.optimizers.Optimizer,
    train_loader: Dataloader,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> float:
    variables = model.trainable_variables
    accum_grads = [tf.zeros_like(v) for v in variables]
    loss_sum = 0.0

    for _ in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x_t = tf.convert_to_tensor(x, dtype=tf.int32)
        y_t = tf.convert_to_tensor(y, dtype=tf.int32)

        with tf.GradientTape() as tape:
            _, loss = model.call(x_t, training=True, targets=y_t)
            if loss is None:
                raise RuntimeError("Model returned no loss during training")
            loss_scaled = loss / grad_accum_steps

        grads = tape.gradient(loss_scaled, variables)
        next_grads: list[tf.Tensor] = []
        for ag, g in zip(accum_grads, grads):
            next_grads.append(ag if g is None else ag + tf.convert_to_tensor(g))
        accum_grads = next_grads
        loss_sum += float(loss.numpy())

    if max_grad_norm > 0:
        accum_grads, _ = tf.clip_by_global_norm(accum_grads, max_grad_norm)

    has_bad = False
    for g in accum_grads:
        s = tf.reduce_sum(g)
        if tf.math.is_nan(s) or tf.math.is_inf(s):
            has_bad = True
            break
    if has_bad:
        print("NaN/Inf in gradients, skipping optimizer step")
        return float("nan")

    optimizer.apply_gradients(zip(accum_grads, variables))
    return loss_sum / grad_accum_steps


def _masked_sft_loss(logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    mask = tf.not_equal(labels, tf.cast(IGNORE_INDEX, labels.dtype))
    safe_labels = tf.where(mask, labels, tf.zeros_like(labels))

    per_token = tf.keras.losses.sparse_categorical_crossentropy(
        safe_labels,
        tf.cast(logits, tf.float32),
        from_logits=True,
    )
    mask_f = tf.cast(mask, per_token.dtype)
    denom = tf.reduce_sum(mask_f)
    denom = tf.maximum(denom, tf.constant(1.0, dtype=denom.dtype))
    return tf.reduce_sum(per_token * mask_f) / denom


def train_one_step_sft(
    model: ECISLM,
    optimizer: tf.keras.optimizers.Optimizer,
    train_loader: SFTBatchLoader,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> float:
    variables = model.trainable_variables
    accum_grads = [tf.zeros_like(v) for v in variables]
    loss_sum = 0.0

    for _ in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x_t = tf.convert_to_tensor(x, dtype=tf.int32)
        y_t = tf.convert_to_tensor(y, dtype=tf.int32)

        with tf.GradientTape() as tape:
            logits, _ = model.call(x_t, training=True, targets=None)
            loss = _masked_sft_loss(logits, y_t)
            loss_scaled = loss / grad_accum_steps

        grads = tape.gradient(loss_scaled, variables)
        next_grads: list[tf.Tensor] = []
        for ag, g in zip(accum_grads, grads):
            next_grads.append(ag if g is None else ag + tf.convert_to_tensor(g))
        accum_grads = next_grads
        loss_sum += float(loss.numpy())

    if max_grad_norm > 0:
        accum_grads, _ = tf.clip_by_global_norm(accum_grads, max_grad_norm)

    has_bad = False
    for g in accum_grads:
        s = tf.reduce_sum(g)
        if tf.math.is_nan(s) or tf.math.is_inf(s):
            has_bad = True
            break
    if has_bad:
        print("NaN/Inf in gradients, skipping optimizer step")
        return float("nan")

    optimizer.apply_gradients(zip(accum_grads, variables))
    return loss_sum / grad_accum_steps


def evaluate_sft(model: ECISLM, val_loader: SFTBatchLoader, num_batches: int) -> float:
    losses: list[float] = []
    for _ in range(num_batches):
        x, y = val_loader.get_batch()
        x_t = tf.convert_to_tensor(x, dtype=tf.int32)
        y_t = tf.convert_to_tensor(y, dtype=tf.int32)
        logits, _ = model.call(x_t, training=False, targets=None)
        loss = _masked_sft_loss(logits, y_t)
        losses.append(float(loss.numpy()))
    return float(np.mean(losses)) if losses else float("nan")


def train(
    args: argparse.Namespace, train_bin: Path, val_bin: Path, tokenizer: Tokenizer
) -> None:
    if args.max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    learning_rate = (
        args.sft_learning_rate if args.stage == "sft" else args.learning_rate
    )
    warmup_steps = args.sft_warmup_steps if args.stage == "sft" else args.warmup_steps
    weight_decay = args.sft_weight_decay if args.stage == "sft" else args.weight_decay

    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if args.min_lr_frac <= 0 or args.min_lr_frac > 1:
        raise ValueError("min_lr_frac must be in (0, 1]")
    if args.warmup_cap_frac <= 0 or args.warmup_cap_frac > 1:
        raise ValueError("warmup_cap_frac must be in (0, 1]")
    if args.plateau_lr_decay <= 0 or args.plateau_lr_decay > 1:
        raise ValueError("plateau_lr_decay must be in (0, 1]")
    if args.min_lr_scale <= 0 or args.min_lr_scale > 1:
        raise ValueError("min_lr_scale must be in (0, 1]")
    if args.min_improve < 0:
        raise ValueError("min_improve must be >= 0")
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if args.eval_interval < 1 or args.save_interval < 1 or args.log_interval < 1:
        raise ValueError("log/eval/save intervals must be >= 1")

    strategy = resolve_strategy(args.strategy)
    replicas = int(strategy.num_replicas_in_sync)
    if args.batch_size % replicas != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by replicas ({replicas})"
        )

    warmup_cap = max(1, int(args.max_steps * args.warmup_cap_frac))
    effective_warmup = min(max(1, warmup_steps), warmup_cap)
    if effective_warmup != warmup_steps:
        print(
            f"Adjusted warmup_steps from {warmup_steps} to {effective_warmup} "
            f"(cap={args.warmup_cap_frac:.2f} of max_steps)"
        )
    warmup_steps = effective_warmup

    if args.stage == "sft":
        x_train, y_train = load_sft_split(train_bin)
        x_val, y_val = load_sft_split(val_bin)
        train_loader = SFTBatchLoader(x_train, y_train, batch_size=args.batch_size)
        val_loader = SFTBatchLoader(x_val, y_val, batch_size=args.batch_size)
        block_size = int(x_train.shape[1])
        train_data_size = int(len(x_train))
        val_data_size = int(len(x_val))
    else:
        train_loader = Dataloader(
            train_bin, block_size=args.block_size, batch_size=args.batch_size
        )
        val_loader = Dataloader(
            val_bin, block_size=args.block_size, batch_size=args.batch_size
        )
        block_size = int(args.block_size)
        train_data_size = int(len(train_loader.data))
        val_data_size = int(len(val_loader.data))

    cfg = SLMConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        tie_embeddings=not args.untied_head,
    )

    with strategy.scope():
        model = ECISLM(cfg)
        _ = model(tf.zeros((1, block_size), dtype=tf.int32))
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=0.9,
            beta_2=0.95,
            epsilon=1e-8,
        )
        optimizer.build(model.trainable_variables)

    trainable_params = count_parameters(model, trainable_only=True)
    print("=" * 70)
    print(f"Stage: {args.stage}")
    print(f"Strategy: {type(strategy).__name__} | Replicas: {replicas}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Trainable params: {trainable_params:,}")
    print(
        f"LR: base={learning_rate:.3e}, min_frac={args.min_lr_frac:.3f}, warmup_steps={warmup_steps}"
    )
    if args.stage == "sft":
        print(f"Train examples: {train_data_size:,} | Val examples: {val_data_size:,}")
    else:
        print(f"Train tokens: {train_data_size:,} | Val tokens: {val_data_size:,}")
    print("=" * 70)

    train_ds = make_batch_dataset(train_loader.get_batch, args.batch_size, block_size)
    val_ds = make_batch_dataset(val_loader.get_batch, args.batch_size, block_size)
    dist_train_iter = iter(strategy.experimental_distribute_dataset(train_ds))
    dist_val_iter = iter(strategy.experimental_distribute_dataset(val_ds))

    def _reduce_grads(per_replica_grads):
        reduced = []
        for g in per_replica_grads:
            if g is None:
                reduced.append(None)
            else:
                reduced.append(
                    strategy.reduce(tf.distribute.ReduceOp.MEAN, g, axis=None)
                )
        return reduced

    def _dist_pretrain_microstep(batch):
        def step_fn(x, y):
            with tf.GradientTape() as tape:
                logits, _ = model.call(x, training=True, targets=None)
                per_tok = tf.keras.losses.sparse_categorical_crossentropy(
                    y,
                    tf.cast(logits, tf.float32),
                    from_logits=True,
                )
                loss = tf.reduce_mean(per_tok)
                scaled = loss / args.grad_accum_steps
            grads = tape.gradient(scaled, model.trainable_variables)
            return loss, grads

        per_replica_loss, per_replica_grads = strategy.run(step_fn, args=batch)
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
        grads = _reduce_grads(per_replica_grads)
        return loss, grads

    def _dist_sft_microstep(batch):
        def step_fn(x, y):
            with tf.GradientTape() as tape:
                logits, _ = model.call(x, training=True, targets=None)
                loss = _masked_sft_loss(logits, y)
                scaled = loss / args.grad_accum_steps
            grads = tape.gradient(scaled, model.trainable_variables)
            return loss, grads

        per_replica_loss, per_replica_grads = strategy.run(step_fn, args=batch)
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
        grads = _reduce_grads(per_replica_grads)
        return loss, grads

    def _distributed_train_step() -> float:
        variables = model.trainable_variables
        accum_grads = [tf.zeros_like(v) for v in variables]
        loss_sum = 0.0

        for _ in range(args.grad_accum_steps):
            batch = next(dist_train_iter)
            if args.stage == "sft":
                loss, grads = _dist_sft_microstep(batch)
            else:
                loss, grads = _dist_pretrain_microstep(batch)

            merged: list[tf.Tensor] = []
            for ag, g in zip(accum_grads, grads):
                merged.append(ag if g is None else ag + tf.convert_to_tensor(g))
            accum_grads = merged
            loss_sum += float(loss.numpy())

        if args.max_grad_norm > 0:
            accum_grads, _ = tf.clip_by_global_norm(accum_grads, args.max_grad_norm)

        has_bad = False
        for g in accum_grads:
            s = tf.reduce_sum(g)
            if tf.math.is_nan(s) or tf.math.is_inf(s):
                has_bad = True
                break
        if has_bad:
            print("NaN/Inf in gradients, skipping optimizer step")
            return float("nan")

        optimizer.apply_gradients(zip(accum_grads, variables))
        return loss_sum / args.grad_accum_steps

    def _distributed_eval(num_batches: int) -> float:
        losses: list[float] = []
        for _ in range(num_batches):
            batch = next(dist_val_iter)

            def step_fn(x, y):
                logits, _ = model.call(x, training=False, targets=None)
                if args.stage == "sft":
                    return _masked_sft_loss(logits, y)
                per_tok = tf.keras.losses.sparse_categorical_crossentropy(
                    y,
                    tf.cast(logits, tf.float32),
                    from_logits=True,
                )
                return tf.reduce_mean(per_tok)

            per_replica_loss = strategy.run(step_fn, args=batch)
            loss = strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None
            )
            losses.append(float(loss.numpy()))

        return float(np.mean(losses)) if losses else float("nan")

    if args.init_step > 0:
        init_dir = (
            Path(args.init_checkpoint_dir)
            if args.init_checkpoint_dir
            else Path(args.checkpoint_dir)
        )
        _ = load_checkpoint(
            checkpoint_dir=init_dir,
            step=args.init_step,
            model=model,
            optimizer=None,
        )
        print(f"Initialized model from {init_dir} step {args.init_step}")

    ckpt = CheckpointManager(args.checkpoint_dir, max_to_keep=args.keep_last_n)
    global_step = 0
    best_val = float("inf")
    no_improve_evals = 0
    lr_scale = 1.0

    if args.resume_step > 0:
        meta = load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            step=args.resume_step,
            model=model,
            optimizer=optimizer,
        )
        global_step = int(meta.get("step", args.resume_step))
        best_val = float(meta.get("metrics", {}).get("best_val_loss", best_val))
        lr_scale = float(meta.get("metrics", {}).get("lr_scale", lr_scale))
        no_improve_evals = int(
            meta.get("metrics", {}).get("no_improve_evals", no_improve_evals)
        )
        print(f"Resumed from explicit step {global_step}")
    elif args.resume:
        loaded = ckpt.load_latest(model, optimizer)
        if loaded:
            global_step = int(loaded.get("step", 0))
            best_val = float(loaded.get("metrics", {}).get("best_val_loss", best_val))
            lr_scale = float(loaded.get("metrics", {}).get("lr_scale", lr_scale))
            no_improve_evals = int(
                loaded.get("metrics", {}).get("no_improve_evals", no_improve_evals)
            )
            print(f"Resumed from latest step {global_step}")

    lr_at = build_lr_scheduler(
        learning_rate=learning_rate,
        min_lr_frac=args.min_lr_frac,
        warmup_steps=warmup_steps,
        total_steps=args.max_steps,
    )

    start = time.time()
    while global_step < args.max_steps:
        base_lr = lr_at(global_step)
        lr = max(
            base_lr * lr_scale, learning_rate * args.min_lr_frac * args.min_lr_scale
        )
        optimizer.learning_rate = lr

        train_loss = _distributed_train_step()
        global_step += 1

        if math.isnan(train_loss) or math.isinf(train_loss):
            ckpt.save(
                model,
                optimizer,
                global_step,
                {
                    "train_loss": train_loss,
                    "best_val_loss": best_val,
                    "learning_rate": float(lr),
                    "lr_scale": float(lr_scale),
                    "no_improve_evals": int(no_improve_evals),
                },
            )
            print(f"NaN/Inf loss at step {global_step}; saved checkpoint and stopped")
            break

        if global_step % args.log_interval == 0:
            elapsed_min = (time.time() - start) / 60.0
            print(
                f"step {global_step:>7,}/{args.max_steps:,} | "
                f"loss {train_loss:.4f} | lr {float(lr):.3e} | "
                f"time {elapsed_min:.1f}m"
            )

        if global_step % args.eval_interval == 0:
            val_loss = _distributed_eval(args.num_val_batches)

            improved = val_loss < (best_val - args.min_improve)
            if improved:
                best_val = val_loss
                no_improve_evals = 0
                ckpt.save(
                    model,
                    optimizer,
                    global_step,
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val_loss": best_val,
                        "learning_rate": float(lr),
                        "lr_scale": float(lr_scale),
                        "no_improve_evals": int(no_improve_evals),
                    },
                    is_best=True,
                )
            else:
                no_improve_evals += 1

            if (
                args.plateau_patience_evals > 0
                and no_improve_evals > 0
                and no_improve_evals % args.plateau_patience_evals == 0
            ):
                new_scale = max(lr_scale * args.plateau_lr_decay, args.min_lr_scale)
                if new_scale < lr_scale:
                    print(
                        f"plateau detected ({no_improve_evals} evals), "
                        f"reducing lr_scale {lr_scale:.3f} -> {new_scale:.3f}"
                    )
                    lr_scale = new_scale

            print(
                f"eval @ {global_step:>7,} | val_loss {val_loss:.4f} | best {best_val:.4f} "
                f"| no_improve {no_improve_evals} | lr_scale {lr_scale:.3f}"
            )

            if (
                args.early_stop_patience_evals > 0
                and no_improve_evals >= args.early_stop_patience_evals
            ):
                print(
                    f"Early stopping: no val improvement for {no_improve_evals} evals"
                )
                ckpt.save(
                    model,
                    optimizer,
                    global_step,
                    {
                        "train_loss": train_loss,
                        "best_val_loss": best_val,
                        "learning_rate": float(lr),
                        "lr_scale": float(lr_scale),
                        "no_improve_evals": int(no_improve_evals),
                        "stopped_early": True,
                    },
                )
                break

        if global_step % args.save_interval == 0:
            ckpt.save(
                model,
                optimizer,
                global_step,
                {
                    "train_loss": train_loss,
                    "best_val_loss": best_val,
                    "learning_rate": float(lr),
                    "lr_scale": float(lr_scale),
                    "no_improve_evals": int(no_improve_evals),
                },
            )

    ckpt.save(
        model,
        optimizer,
        global_step,
        {
            "best_val_loss": best_val,
            "total_minutes": (time.time() - start) / 60.0,
            "lr_scale": float(lr_scale),
            "no_improve_evals": int(no_improve_evals),
        },
    )
    print("Training complete")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_precision(args.precision)

    tokenizer = load_or_train_tokenizer(args)
    train_bin, val_bin = maybe_prepare(args, tokenizer)

    if args.mode in {"train", "prepare_and_train"}:
        train(args, train_bin, val_bin, tokenizer)


if __name__ == "__main__":
    main()
