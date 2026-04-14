#!/usr/bin/env python3
"""Convert ECI instruct txt files into Unsloth-friendly chat JSONL."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


PAIR_PATTERN = re.compile(r"User:\s*(.*?)\s*Assistant:\s*(.*?)\s*<END>", re.DOTALL)


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for user, assistant in PAIR_PATTERN.findall(text):
        q = user.strip()
        a = assistant.strip()
        if q and a:
            pairs.append((q, a))
    return pairs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert User/Assistant/<END> txt files into train/val chat JSONL"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/instruct"),
        help="Directory containing .txt instruct files",
    )
    parser.add_argument(
        "--train_out",
        type=Path,
        default=Path("data/instruct_jsonl/train.jsonl"),
        help="Output train JSONL path",
    )
    parser.add_argument(
        "--val_out",
        type=Path,
        default=Path("data/instruct_jsonl/val.jsonl"),
        help="Output validation JSONL path",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio in (0, 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be between 0 and 1")

    rows: list[dict] = []
    idx = 0
    for txt_file in sorted(args.input_dir.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8", errors="replace")
        for question, answer in parse_pairs(raw):
            rows.append(
                {
                    "id": f"eci-{idx}",
                    "source": txt_file.name,
                    "question": question,
                    "answer": answer,
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                }
            )
            idx += 1

    if not rows:
        raise RuntimeError("No Q/A pairs found. Check input format.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    split_index = max(1, min(len(rows) - 1, int(len(rows) * (1.0 - args.val_ratio))))
    train_rows = rows[:split_index]
    val_rows = rows[split_index:]

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.val_out, val_rows)

    print(f"Parsed pairs: {len(rows)}")
    print(f"Train rows:   {len(train_rows)} -> {args.train_out}")
    print(f"Val rows:     {len(val_rows)} -> {args.val_out}")


if __name__ == "__main__":
    main()
