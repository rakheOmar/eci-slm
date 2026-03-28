#!/usr/bin/env python3
"""ECI-SLM training entrypoint.

Simple, production-focused defaults for:
- Colab T4 16GB
- Kaggle 2xT4 (MirroredStrategy)
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

    parser.add_argument(
        "--profile",
        type=str,
        default="t4_1gpu",
        choices=["t4_1gpu", "t4_2gpu"],
        help="Hardware preset with stable defaults.",
    )
    parser.add_argument("--max_steps", type=int, default=2600)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--tokenizer", type=str, default="artifact/eci_slm_tokenizer.model"
    )
    parser.add_argument("--data_dir", type=str, default="artifact")
    parser.add_argument("--rebuild_bins", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--mix_chunk_tokens", type=int, default=4096)
    parser.add_argument(
        "--english_ratio",
        type=float,
        default=0.5,
        help="Target English token ratio in rebuilt bins (0<ratio<1).",
    )

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--num_val_batches", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--keep_last_n", type=int, default=6)

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--precision",
        type=str,
        default="mixed_fp16",
        choices=["fp32", "mixed_bf16", "mixed_fp16"],
    )

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


def profile_defaults(name: str) -> dict[str, int | float]:
    # Stable small-ish recipe for T4, with a 2-GPU variant for Kaggle.
    common: dict[str, int | float] = {
        "n_layer": 12,
        "n_head": 12,
        "n_kv_head": 3,
        "n_embd": 768,
        "block_size": 256,
        "dropout": 0.1,
        "learning_rate": 8e-5,
        "warmup_steps": 1200,
        "weight_decay": 0.1,
        "min_lr_frac": 0.1,
        "max_grad_norm": 0.5,
    }
    if name == "t4_2gpu":
        # ~154M params with vocab_size=32k.
        common["n_layer"] = 17
        # Higher per-step work for better dual-T4 utilization.
        common["batch_size"] = 8
        common["grad_accum_steps"] = 2
        # Lower peak LR for mixed-precision stability on 2xT4.
        common["learning_rate"] = 5e-5
    else:
        common["batch_size"] = 1
        common["grad_accum_steps"] = 8
    return common


def validate_model_cfg(n_embd: int, n_head: int, n_kv_head: int) -> None:
    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if n_kv_head > n_head or n_head % n_kv_head != 0:
        raise ValueError("n_kv_head must be <= n_head and divide n_head")
    head_dim = n_embd // n_head
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")


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
        new_accum: list[tf.Tensor] = []
        for ag, g in zip(accum_grads, grads):
            if g is None:
                new_accum.append(ag)
            else:
                new_accum.append(ag + tf.convert_to_tensor(g))
        accum_grads = new_accum
        loss_sum += float(loss.numpy())

    if max_grad_norm > 0:
        accum_grads, _ = tf.clip_by_global_norm(accum_grads, max_grad_norm)

    has_bad = False
    for g in accum_grads:
        if tf.math.is_nan(tf.reduce_sum(g)) or tf.math.is_inf(tf.reduce_sum(g)):
            has_bad = True
            break
    if has_bad:
        print("NaN/Inf in gradients (single-gpu), skipping optimizer step.")
        return float("nan")

    optimizer.apply_gradients(zip(accum_grads, variables))
    return loss_sum / grad_accum_steps


def make_distributed_iterator(
    loader: Dataloader,
    block_size: int,
    global_batch_size: int,
    strategy: tf.distribute.Strategy,
):
    def gen():
        while True:
            x, y = loader.get_batch()
            yield x, y

    sig = (
        tf.TensorSpec(shape=(global_batch_size, block_size), dtype=tf.int32),
        tf.TensorSpec(shape=(global_batch_size, block_size), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)
    dist_ds = strategy.experimental_distribute_dataset(ds)
    return iter(dist_ds)


def distributed_micro_step(
    strategy: tf.distribute.Strategy,
    model: ECISLM,
    iterator,
    grad_divisor: int,
):
    return _distributed_micro_step_tf(
        strategy=strategy,
        model=model,
        iterator=iterator,
        grad_divisor=grad_divisor,
    )


def train_one_step_distributed(
    strategy: tf.distribute.Strategy,
    model: ECISLM,
    optimizer: tf.keras.optimizers.Optimizer,
    train_iter,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> float:
    variables = model.trainable_variables
    accum_grads = [tf.zeros_like(v) for v in variables]
    loss_sum = 0.0

    for _ in range(grad_accum_steps):
        loss, grads = distributed_micro_step(
            strategy=strategy,
            model=model,
            iterator=train_iter,
            grad_divisor=grad_accum_steps,
        )
        accum_grads = [ag + g for ag, g in zip(accum_grads, grads)]
        loss_sum += float(loss.numpy())

    if max_grad_norm > 0:
        accum_grads, _ = tf.clip_by_global_norm(accum_grads, max_grad_norm)

    has_bad = False
    for g in accum_grads:
        if tf.math.is_nan(tf.reduce_sum(g)) or tf.math.is_inf(tf.reduce_sum(g)):
            has_bad = True
            break
    if has_bad:
        print("NaN/Inf in gradients (distributed), skipping optimizer step.")
        return float("nan")

    optimizer.apply_gradients(zip(accum_grads, variables))
    return loss_sum / grad_accum_steps


def evaluate_distributed(
    strategy: tf.distribute.Strategy,
    model: ECISLM,
    val_iter,
    num_batches: int,
) -> float:
    losses: list[float] = []

    for _ in range(num_batches):
        loss = _distributed_eval_step_tf(
            strategy=strategy,
            model=model,
            iterator=val_iter,
        )
        losses.append(float(loss.numpy()))
    return float(np.mean(losses)) if losses else float("nan")


@tf.function
def _distributed_micro_step_tf(
    strategy: tf.distribute.Strategy,
    model: ECISLM,
    iterator,
    grad_divisor: int,
):
    variables = model.trainable_variables

    def step_fn(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            _, loss_unscaled = model.call(x, training=True, targets=y)
            if loss_unscaled is None:
                raise RuntimeError("Model returned no loss during distributed training")
            loss_scaled = loss_unscaled / grad_divisor
        grads = tape.gradient(loss_scaled, variables)
        safe_grads = []
        for g, v in zip(grads, variables):
            safe_grads.append(tf.zeros_like(v) if g is None else g)
        return loss_unscaled, tuple(safe_grads)

    per_replica_loss, per_replica_grads = strategy.run(step_fn, args=(next(iterator),))
    loss_unscaled = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None
    )

    grads: list[tf.Tensor] = []
    for g in per_replica_grads:
        reduced = strategy.reduce(tf.distribute.ReduceOp.MEAN, g, axis=None)
        grads.append(tf.convert_to_tensor(reduced))
    return loss_unscaled, grads


@tf.function
def _distributed_eval_step_tf(
    strategy: tf.distribute.Strategy,
    model: ECISLM,
    iterator,
):
    def step_fn(inputs):
        x, y = inputs
        _, loss = model.call(x, training=False, targets=y)
        if loss is None:
            raise RuntimeError("Model returned no loss during distributed evaluation")
        return loss

    per_replica_loss = strategy.run(step_fn, args=(next(iterator),))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)


def _iter_eci_and_english_files(data_root: Path) -> tuple[list[Path], list[Path]]:
    eci_dirs = [
        data_root / "pretrain",
        data_root / "pretrain_expanded",
        data_root / "pretrain_augmented",
    ]
    english_dir = data_root / "english_pretrain"

    eci_files: list[Path] = []
    for d in eci_dirs:
        if d.exists():
            eci_files.extend(sorted(d.glob("*.txt")))

    english_files = sorted(english_dir.glob("*.txt")) if english_dir.exists() else []
    return eci_files, english_files


def _tokenize_files(tokenizer: Tokenizer, paths: list[Path]) -> list[int]:
    ids: list[int] = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        ids.extend(tokenizer.encode(text))
    return ids


def _interleave_weighted_token_chunks(
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
            take_english = True
        elif j >= target_english:
            take_english = False
        else:
            eci_progress = i / max(1, target_eci)
            english_progress = j / max(1, target_english)
            take_english = english_progress < eci_progress

        if take_english:
            next_j = min(j + chunk, target_english)
            out.append(english_ids[j:next_j])
            j = next_j
        else:
            next_i = min(i + chunk, target_eci)
            out.append(eci_ids[i:next_i])
            i = next_i

    if not out:
        return np.asarray([], dtype=np.uint16)
    return np.concatenate(out)


def build_balanced_bins(
    tokenizer: Tokenizer,
    out_dir: Path,
    val_split: float,
    mix_chunk_tokens: int,
    english_ratio: float,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path("data")
    eci_files, english_files = _iter_eci_and_english_files(data_root)
    if not eci_files or not english_files:
        raise FileNotFoundError(
            "Need both ECI files and data/english_pretrain/*.txt for mixed bins."
        )
    if not (0.0 < english_ratio < 1.0):
        raise ValueError("english_ratio must be in (0, 1)")
    if mix_chunk_tokens < 1:
        raise ValueError("mix_chunk_tokens must be >= 1")

    print(
        f"Building balanced bins from {len(eci_files)} ECI files + "
        f"{len(english_files)} English files..."
    )

    eci_ids = np.asarray(_tokenize_files(tokenizer, eci_files), dtype=np.uint16)
    english_ids = np.asarray(_tokenize_files(tokenizer, english_files), dtype=np.uint16)

    available_eci = int(len(eci_ids))
    available_english = int(len(english_ids))
    if available_eci == 0 or available_english == 0:
        raise RuntimeError("One side has zero tokens.")

    eci_ratio = 1.0 - english_ratio
    max_total = min(available_eci / eci_ratio, available_english / english_ratio)
    total_target = int(max_total)
    target_eci = int(total_target * eci_ratio)
    target_english = int(total_target * english_ratio)

    target_eci = min(target_eci, available_eci)
    target_english = min(target_english, available_english)

    if target_eci < 1 or target_english < 1:
        raise RuntimeError(
            "Not enough tokens to satisfy requested english_ratio; "
            "add more data or choose a less extreme ratio."
        )

    eci_ids = eci_ids[:target_eci]
    english_ids = english_ids[:target_english]
    all_ids = _interleave_weighted_token_chunks(
        eci_ids=eci_ids,
        english_ids=english_ids,
        chunk=mix_chunk_tokens,
        target_eci=target_eci,
        target_english=target_english,
    )

    n = len(all_ids)
    split_idx = int(n * (1.0 - val_split))
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    achieved_english_ratio = target_english / max(1, (target_eci + target_english))
    print(
        "Mixed token build: "
        f"ECI={target_eci:,}, English={target_english:,}, Total={n:,}, "
        f"english_ratio={achieved_english_ratio:.3f}"
    )
    print(f"Created {train_path} ({len(train_ids):,} tokens)")
    print(f"Created {val_path} ({len(val_ids):,} tokens)")
    return train_path, val_path


def load_or_train_tokenizer(tokenizer_path: str) -> Tokenizer:
    tok = Tokenizer(tokenizer_path)
    try:
        _ = tok.vocab_size
        return tok
    except (RuntimeError, FileNotFoundError):
        print(f"Tokenizer not found at: {tokenizer_path}")
        print("Training tokenizer from data...")
        train_tokenizer()
        default_model = Path("artifact/eci_slm_tokenizer.model")
        return Tokenizer(default_model)


def maybe_enable_multi_gpu(profile: str):
    gpus = tf.config.list_physical_devices("GPU")
    if profile == "t4_2gpu" and len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
        return strategy
    if profile == "t4_2gpu":
        print(
            "Requested t4_2gpu profile but <2 GPUs found. Falling back to single GPU."
        )
    return tf.distribute.get_strategy()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    set_precision(args.precision)

    cfg_vals = profile_defaults(args.profile)
    n_layer = int(cfg_vals["n_layer"])
    n_head = int(cfg_vals["n_head"])
    n_kv_head = int(cfg_vals["n_kv_head"])
    n_embd = int(cfg_vals["n_embd"])
    block_size = int(cfg_vals["block_size"])
    dropout = float(cfg_vals["dropout"])
    batch_size = int(cfg_vals["batch_size"])
    grad_accum_steps = int(cfg_vals["grad_accum_steps"])
    learning_rate = float(cfg_vals["learning_rate"])
    warmup_steps = int(cfg_vals["warmup_steps"])
    weight_decay = float(cfg_vals["weight_decay"])
    min_lr_frac = float(cfg_vals["min_lr_frac"])
    max_grad_norm = float(cfg_vals["max_grad_norm"])

    validate_model_cfg(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
    if args.max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    print("=" * 60)
    print(f"ECI-SLM Training | profile={args.profile}")
    print("=" * 60)

    tokenizer = load_or_train_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab_size: {vocab_size}")

    requested_data_dir = Path(args.data_dir)
    train_path = requested_data_dir / "train.bin"
    val_path = requested_data_dir / "val.bin"

    artifact_dir = Path("artifact")
    artifact_train = artifact_dir / "train.bin"
    artifact_val = artifact_dir / "val.bin"

    if args.rebuild_bins:
        train_path, val_path = build_balanced_bins(
            tokenizer=tokenizer,
            out_dir=artifact_dir,
            val_split=args.val_split,
            mix_chunk_tokens=args.mix_chunk_tokens,
            english_ratio=args.english_ratio,
        )
    elif train_path.exists() and val_path.exists():
        pass
    elif artifact_train.exists() and artifact_val.exists():
        print(f"Using existing binary dataset in {artifact_dir}")
        train_path, val_path = artifact_train, artifact_val
    else:
        train_path, val_path = build_balanced_bins(
            tokenizer=tokenizer,
            out_dir=artifact_dir,
            val_split=args.val_split,
            mix_chunk_tokens=args.mix_chunk_tokens,
            english_ratio=args.english_ratio,
        )

    train_loader = Dataloader(train_path, block_size, batch_size)
    val_loader = (
        Dataloader(val_path, block_size, batch_size) if val_path.exists() else None
    )

    tokens_per_step = batch_size * block_size * grad_accum_steps
    steps_per_epoch = max(1, len(train_loader.data) // tokens_per_step)
    total_steps = args.max_steps
    lr_at = build_lr_scheduler(
        learning_rate=learning_rate,
        min_lr_frac=min_lr_frac,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    print(f"Train tokens: {len(train_loader.data):,}")
    if val_loader:
        print(f"Val tokens: {len(val_loader.data):,}")
    print(f"Steps/epoch: {steps_per_epoch:,} | Total steps: {total_steps:,}")

    strategy = maybe_enable_multi_gpu(args.profile)
    print(f"Replicas in sync: {strategy.num_replicas_in_sync}")

    train_iter = None
    val_iter = None
    if strategy.num_replicas_in_sync > 1:
        if batch_size % strategy.num_replicas_in_sync != 0:
            raise ValueError(
                "Global batch_size must be divisible by number of replicas for multi-GPU."
            )
        train_iter = make_distributed_iterator(
            loader=train_loader,
            block_size=block_size,
            global_batch_size=batch_size,
            strategy=strategy,
        )
        if val_loader:
            val_iter = make_distributed_iterator(
                loader=val_loader,
                block_size=block_size,
                global_batch_size=batch_size,
                strategy=strategy,
            )

    with strategy.scope():
        cfg = SLMConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head,
            n_embd=n_embd,
            dropout=dropout,
        )
        model = cast(ECISLM, ECISLM(cfg))
        _ = model(tf.zeros((1, block_size), dtype=tf.int32))
        print(f"Model params: {model.count_params():,}")

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=0.9,
            beta_2=0.95,
            epsilon=1e-8,
        )
        # Ensure optimizer slot variables are created inside strategy scope.
        optimizer.build(model.trainable_variables)

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
    while global_step < total_steps:
        lr = lr_at(global_step)
        optimizer.learning_rate = lr

        if strategy.num_replicas_in_sync > 1:
            if train_iter is None:
                raise RuntimeError("Distributed train iterator not initialized")
            train_loss = train_one_step_distributed(
                strategy=strategy,
                model=model,
                optimizer=optimizer,
                train_iter=train_iter,
                grad_accum_steps=grad_accum_steps,
                max_grad_norm=max_grad_norm,
            )
        else:
            train_loss = train_one_step(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                grad_accum_steps=grad_accum_steps,
                max_grad_norm=max_grad_norm,
            )
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
                },
            )
            print(f"NaN/Inf loss detected at step {global_step}. Saving and stopping.")
            break

        if global_step % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"step {global_step:>7,}/{total_steps:,} | "
                f"loss {train_loss:.4f} | lr {float(lr):.3e} | "
                f"time {elapsed / 60:.1f}m"
            )

        if val_loader and global_step % args.eval_interval == 0:
            if strategy.num_replicas_in_sync > 1:
                if val_iter is None:
                    raise RuntimeError("Distributed val iterator not initialized")
                val_loss = evaluate_distributed(
                    strategy=strategy,
                    model=model,
                    val_iter=val_iter,
                    num_batches=args.num_val_batches,
                )
            else:
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
                },
            )

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
