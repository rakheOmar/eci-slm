#!/usr/bin/env python3
"""Supervised fine-tuning utilities with assistant-only loss masking."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.tokenizer import Tokenizer


IGNORE_INDEX = -100


_UA_PATTERNS = [
    re.compile(r"User:\s*(.*?)\s*Assistant:\s*(.*?)\s*<END>", re.DOTALL),
    re.compile(r"Q(?:uestion)?:\s*(.*?)\s*A(?:nswer)?:\s*(.*?)\s*<END>", re.DOTALL),
]


def parse_user_assistant_turns(text: str) -> list[tuple[str, str]]:
    """Parse conversation turns from raw instruct text."""
    for pattern in _UA_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            turns: list[tuple[str, str]] = []
            for user, assistant in matches:
                u = user.strip()
                a = assistant.strip()
                if u and a:
                    turns.append((u, a))
            if turns:
                return turns
    return []


def _build_masked_example(
    tokenizer: Tokenizer,
    user_text: str,
    assistant_text: str,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build one (x, y) example with assistant-only supervision.

    - x: input token ids, length block_size
    - y: next-token labels, length block_size, with IGNORE_INDEX masking
    """
    if tokenizer.sp is None:
        raise RuntimeError("Tokenizer is not loaded")

    pad_id = int(tokenizer.sp.pad_id()) if tokenizer.sp.pad_id() >= 0 else 0
    eos_id = int(tokenizer.sp.eos_id()) if tokenizer.sp.eos_id() >= 0 else None

    prompt = f"User: {user_text.strip()}\nAssistant: "
    answer = assistant_text.strip()

    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)

    seq = list(prompt_ids) + list(answer_ids)
    flags = [0] * len(prompt_ids) + [1] * len(answer_ids)

    if eos_id is not None:
        seq.append(eos_id)
        flags.append(1)

    if len(seq) < 2:
        return None

    # Next-token setup for causal LM.
    x = np.asarray(seq[:-1], dtype=np.int32)
    target_tokens = np.asarray(seq[1:], dtype=np.int32)
    target_flags = np.asarray(flags[1:], dtype=np.int32)

    y = np.full_like(target_tokens, fill_value=IGNORE_INDEX)
    y[target_flags == 1] = target_tokens[target_flags == 1]

    # Keep the tail if over length. This preserves the answer tokens.
    if len(x) > block_size:
        x = x[-block_size:]
        y = y[-block_size:]

    # Skip examples that lost all supervised tokens after truncation.
    if not np.any(y != IGNORE_INDEX):
        return None

    # Right-pad to fixed length.
    if len(x) < block_size:
        pad_len = block_size - len(x)
        x = np.pad(x, (0, pad_len), mode="constant", constant_values=pad_id)
        y = np.pad(y, (0, pad_len), mode="constant", constant_values=IGNORE_INDEX)

    return x.astype(np.int32), y.astype(np.int32)


def build_sft_arrays(
    tokenizer: Tokenizer,
    source_files: list[Path],
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build masked SFT arrays from source files."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for path in source_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        turns = parse_user_assistant_turns(text)
        for user_text, assistant_text in turns:
            item = _build_masked_example(
                tokenizer=tokenizer,
                user_text=user_text,
                assistant_text=assistant_text,
                block_size=block_size,
            )
            if item is None:
                continue
            x, y = item
            xs.append(x)
            ys.append(y)

    if not xs:
        raise RuntimeError("No SFT examples were built; check instruct data format")

    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def split_sft_arrays(
    x: np.ndarray,
    y: np.ndarray,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split SFT arrays into train/val."""
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be in (0, 1)")
    if len(x) != len(y):
        raise ValueError("x and y length mismatch")
    if len(x) < 2:
        raise ValueError("Need at least 2 SFT examples for train/val split")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)

    x = x[idx]
    y = y[idx]

    split_idx = int((1.0 - val_split) * len(x))
    split_idx = min(max(split_idx, 1), len(x) - 1)

    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def save_sft_split(path: Path, x: np.ndarray, y: np.ndarray) -> Path:
    """Save one SFT split to compressed npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.int32), y=y.astype(np.int32))
    return path


def load_sft_split(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one SFT split from compressed npz."""
    arr = np.load(Path(path))
    return arr["x"].astype(np.int32), arr["y"].astype(np.int32)


class SFTBatchLoader:
    """Random mini-batch loader for fixed-length SFT arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int):
        if len(x) != len(y):
            raise ValueError("x and y length mismatch")
        if len(x) == 0:
            raise ValueError("SFT dataset is empty")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self) -> int:
        return max(1, len(self.x) // self.batch_size)

    def get_batch(self) -> tuple[np.ndarray, np.ndarray]:
        ix = np.random.randint(0, len(self.x), size=(self.batch_size,))
        return self.x[ix], self.y[ix]
