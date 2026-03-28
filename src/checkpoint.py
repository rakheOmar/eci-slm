#!/usr/bin/env python3
"""Checkpoint utilities for ECI-SLM (TensorFlow).

Design mirrors the nanochat structure:
- model_<step>.weights.h5
- optim_<step>.ckpt
- meta_<step>.json
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import tensorflow as tf


def _step_tag(step: int) -> str:
    return f"{step:06d}"


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer | None,
    meta_data: dict,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = _step_tag(step)

    model_path = checkpoint_dir / f"model_{tag}.weights.h5"
    model.save_weights(str(model_path))

    if optimizer is not None:
        optim_prefix = checkpoint_dir / f"optim_{tag}.ckpt"
        optim_ckpt = tf.train.Checkpoint(optimizer=optimizer)
        optim_ckpt.write(str(optim_prefix))

    meta_path = checkpoint_dir / f"meta_{tag}.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)


def load_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    tag = _step_tag(step)

    model_path = checkpoint_dir / f"model_{tag}.weights.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model.load_weights(str(model_path))

    if optimizer is not None:
        optim_prefix = checkpoint_dir / f"optim_{tag}.ckpt"
        index_file = Path(f"{optim_prefix}.index")
        if index_file.exists():
            optim_ckpt = tf.train.Checkpoint(optimizer=optimizer)
            optim_ckpt.restore(str(optim_prefix)).expect_partial()

    meta_path = checkpoint_dir / f"meta_{tag}.json"
    if not meta_path.exists():
        return {"step": step, "metrics": {}}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_last_step(checkpoint_dir: str | Path) -> int:
    checkpoint_dir = Path(checkpoint_dir)
    meta_files = glob.glob(str(checkpoint_dir / "meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    steps: list[int] = []
    for p in meta_files:
        m = re.search(r"meta_(\d+)\.json$", p)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir}")
    return max(steps)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints", max_to_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.best_step_file = self.checkpoint_dir / "best_step.txt"

    def _read_best_step(self) -> int | None:
        if not self.best_step_file.exists():
            return None
        raw = self.best_step_file.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _write_best_step(self, step: int) -> None:
        self.best_step_file.write_text(str(step), encoding="utf-8")

    def _prune_old(self) -> None:
        try:
            latest = find_last_step(self.checkpoint_dir)
        except FileNotFoundError:
            return

        best_step = self._read_best_step()

        meta_files = sorted(self.checkpoint_dir.glob("meta_*.json"))
        if len(meta_files) <= self.max_to_keep:
            return

        # Keep newest max_to_keep steps.
        steps = []
        for mf in meta_files:
            m = re.search(r"meta_(\d+)\.json$", mf.name)
            if m:
                steps.append(int(m.group(1)))
        steps = sorted(set(steps))
        keep_steps = set(steps[-self.max_to_keep :])
        if best_step is not None:
            keep_steps.add(best_step)

        for step in steps:
            if step in keep_steps:
                continue
            tag = _step_tag(step)
            for p in self.checkpoint_dir.glob(f"*_{tag}*"):
                p.unlink(missing_ok=True)
            for p in self.checkpoint_dir.glob(f"optim_{tag}.ckpt*"):
                p.unlink(missing_ok=True)

        # avoid unused local warning in static checkers
        _ = latest

    def save(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer | None,
        step: int,
        metrics: dict,
        is_best: bool = False,
    ) -> None:
        save_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            meta_data={"step": step, "metrics": metrics},
        )
        if is_best:
            self._write_best_step(step)
        self._prune_old()
        label = " (best)" if is_best else ""
        print(f"Saved checkpoint step={step}{label}")

    def load_latest(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer | None = None,
    ) -> dict | None:
        try:
            step = find_last_step(self.checkpoint_dir)
        except FileNotFoundError:
            print("No checkpoints found")
            return None

        meta = load_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            step=step,
            model=model,
            optimizer=optimizer,
        )
        print(f"Loaded checkpoint step={step}")
        return meta
