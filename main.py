#!/usr/bin/env python3
"""Production-style training entrypoint for ECI-SLM.

Example:
uv run python main.py --data_dir notebooks --epochs 1 --n_layer 4 --n_head 4 --n_embd 256 --batch_size 2 --block_size 128
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import tensorflow as tf

from src.checkpoint import CheckpointManager
from src.dataloader import Dataloader
from src.slm import ECISLM, SLMConfig
from src.tokenizer import Tokenizer, train_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECI-SLM")

    parser.add_argument("--data_dir", type=str, default="artifact")
    parser.add_argument(
        "--tokenizer", type=str, default="artifact/eci_slm_tokenizer.model"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "mixed_bf16", "mixed_fp16"],
        help="Training precision policy",
    )

    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_kv_head", type=int, default=0)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--min_lr_frac", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--num_val_batches", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--keep_last_n", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.1)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_precision(policy: str) -> None:
    if policy == "fp32":
        tf.keras.mixed_precision.set_global_policy("float32")
    elif policy == "mixed_bf16":
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    elif policy == "mixed_fp16":
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def validate_config(args: argparse.Namespace) -> None:
    if args.n_embd % args.n_head != 0:
        raise ValueError(
            f"Invalid config: n_embd ({args.n_embd}) must be divisible by n_head ({args.n_head})."
        )
    if args.n_kv_head != 0:
        if args.n_kv_head > args.n_head or args.n_head % args.n_kv_head != 0:
            raise ValueError(
                "Invalid config: n_kv_head must be <= n_head and divide n_head."
            )
    head_dim = args.n_embd // args.n_head
    if head_dim % 2 != 0:
        raise ValueError(
            f"Invalid config for RoPE: head_dim must be even, got {head_dim}."
        )
    if args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("batch_size and grad_accum_steps must both be >= 1")


def resolve_steps_per_epoch(args: argparse.Namespace, train_loader: Dataloader) -> int:
    if args.steps_per_epoch > 0:
        return args.steps_per_epoch
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum_steps
    return max(1, len(train_loader.data) // tokens_per_step)


def build_lr_scheduler(args: argparse.Namespace, total_train_steps: int):
    warmup = max(1, args.warmup_steps)
    min_lr = args.learning_rate * args.min_lr_frac

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.learning_rate * (step + 1) / warmup
        if total_train_steps <= warmup:
            return min_lr
        progress = (step - warmup) / (total_train_steps - warmup)
        progress = float(min(1.0, max(0.0, progress)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (args.learning_rate - min_lr) * cosine

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
        new_accum: list[tf.Tensor] = []
        for ag, g, v in zip(accum_grads, grads, variables):
            if g is None:
                new_accum.append(ag)
            else:
                g_tensor = tf.convert_to_tensor(g)
                new_accum.append(ag + g_tensor)
        accum_grads = new_accum
        loss_sum += float(loss.numpy())

    if max_grad_norm > 0:
        accum_grads, _ = tf.clip_by_global_norm(accum_grads, max_grad_norm)

    optimizer.apply_gradients(zip(accum_grads, variables))
    return loss_sum / grad_accum_steps


def _iter_pretrain_text_files(data_root: Path) -> list[Path]:
    dirs = [
        data_root / "pretrain",
        data_root / "pretrain_expanded",
        data_root / "pretrain_augmented",
        data_root / "english_pretrain",
    ]
    files: list[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.txt")))
    return files


def build_train_val_bins(
    tokenizer: Tokenizer, out_dir: Path, val_split: float = 0.1
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path("data")
    txt_files = _iter_pretrain_text_files(data_root)
    if not txt_files:
        raise FileNotFoundError("No pretraining text files found under data/")

    print(f"Building train/val bins in {out_dir} from {len(txt_files)} text files...")
    all_ids: list[int] = []
    for p in txt_files:
        text = p.read_text(encoding="utf-8", errors="replace")
        all_ids.extend(tokenizer.encode(text))

    ids_arr = np.asarray(all_ids, dtype=np.uint16)
    n = len(ids_arr)
    split_idx = int(n * (1.0 - val_split))
    train_ids = ids_arr[:split_idx]
    val_ids = ids_arr[split_idx:]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    print(f"Created {train_path} ({len(train_ids):,} tokens)")
    print(f"Created {val_path} ({len(val_ids):,} tokens)")
    return train_path, val_path


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    set_precision(args.precision)
    validate_config(args)

    print("=" * 60)
    print("ECI-SLM Production Training")
    print("=" * 60)

    tokenizer = Tokenizer(args.tokenizer)
    try:
        vocab_size = tokenizer.vocab_size
    except (RuntimeError, FileNotFoundError):
        print(f"Tokenizer not found/loaded at: {args.tokenizer}")
        print("Training tokenizer from data... this may take a while.")
        train_tokenizer()

        requested = Path(args.tokenizer)
        default_model = Path("artifact/eci_slm_tokenizer.model")
        model_to_load = requested if requested.exists() else default_model
        tokenizer = Tokenizer(model_to_load)
        vocab_size = tokenizer.vocab_size

    print(f"Tokenizer vocab_size: {vocab_size}")

    cfg = SLMConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=(None if args.n_kv_head == 0 else args.n_kv_head),
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = cast(ECISLM, ECISLM(cfg))
    _ = model(tf.zeros((1, args.block_size), dtype=tf.int32))
    print(f"Model params: {model.count_params():,}")

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta_1=0.9,
        beta_2=0.95,
        epsilon=1e-8,
    )

    requested_data_dir = Path(args.data_dir)
    train_path = requested_data_dir / "train.bin"
    val_path = requested_data_dir / "val.bin"

    artifact_dir = Path("artifact")
    artifact_train = artifact_dir / "train.bin"
    artifact_val = artifact_dir / "val.bin"

    # Prefer requested data_dir, then artifact/. If still missing, auto-build in artifact/.
    if train_path.exists() and val_path.exists():
        pass
    elif artifact_train.exists() and artifact_val.exists():
        print(f"Using existing binary dataset in {artifact_dir}")
        train_path, val_path = artifact_train, artifact_val
    else:
        train_path, val_path = build_train_val_bins(
            tokenizer=tokenizer,
            out_dir=artifact_dir,
            val_split=args.val_split,
        )

    train_loader = Dataloader(train_path, args.block_size, args.batch_size)
    val_loader = (
        Dataloader(val_path, args.block_size, args.batch_size)
        if val_path.exists()
        else None
    )

    steps_per_epoch = resolve_steps_per_epoch(args, train_loader)
    planned_steps = steps_per_epoch * args.epochs
    total_steps = (
        min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps
    )
    lr_at = build_lr_scheduler(args, total_steps)

    print(f"Train tokens: {len(train_loader.data):,}")
    if val_loader:
        print(f"Val tokens: {len(val_loader.data):,}")
    print(
        f"Steps/epoch: {steps_per_epoch:,} | Epochs: {args.epochs} | "
        f"Total planned steps: {total_steps:,}"
    )

    ckpt = CheckpointManager(args.checkpoint_dir, max_to_keep=args.keep_last_n)
    global_step = 0
    best_val = float("inf")

    if args.resume:
        loaded = ckpt.load_latest(model, optimizer)
        if loaded:
            global_step = int(loaded.get("step", 0))
            best_val = float(loaded.get("metrics", {}).get("best_val_loss", best_val))
            print(f"Resumed from step {global_step}")

    start_time = time.time()
    stop = False

    for epoch in range(args.epochs):
        if stop:
            break

        epoch_start = time.time()
        for _ in range(steps_per_epoch):
            if global_step >= total_steps:
                stop = True
                break

            lr = lr_at(global_step)
            optimizer.learning_rate = lr

            train_loss = train_one_step(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                grad_accum_steps=args.grad_accum_steps,
                max_grad_norm=args.max_grad_norm,
            )
            global_step += 1

            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"step {global_step:>7,}/{total_steps:,} | "
                    f"loss {train_loss:.4f} | lr {float(lr):.3e} | "
                    f"time {elapsed / 60:.1f}m"
                )

            if val_loader and global_step % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, args.num_val_batches)
                best_val = min(best_val, val_loss)
                print(
                    f"eval @ {global_step:>7,} | val_loss {val_loss:.4f} | best {best_val:.4f}"
                )

            if global_step % args.save_interval == 0:
                ckpt.save(
                    model,
                    optimizer,
                    global_step,
                    {
                        "train_loss": train_loss,
                        "best_val_loss": best_val,
                        "learning_rate": float(lr),
                        "epoch": epoch + 1,
                    },
                )

        epoch_time = time.time() - epoch_start
        print(f"epoch {epoch + 1}/{args.epochs} finished in {epoch_time:.1f}s")

    final_metrics = {
        "best_val_loss": best_val,
        "total_minutes": (time.time() - start_time) / 60.0,
    }
    ckpt.save(model, optimizer, global_step, final_metrics)
    print("Training complete.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
