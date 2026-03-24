#!/usr/bin/env python3
"""Download general English data from FineWeb for 50:50 training.

Usage:
    uv run data/scripts/download_english.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "english_pretrain"
TARGET_TOKENS = 1_600_000


def get_tokenizer():
    """Return a tiktoken encoder."""
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def download_fineweb_sample(target_tokens: int):
    """Download FineWeb sample and save target_tokens to OUTPUT_DIR."""
    from datasets import load_dataset

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading FineWeb sample-10BT (streaming)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    encoder = get_tokenizer()
    texts = []
    total_tokens = 0
    docs_saved = 0

    print(f"Target: {target_tokens:,} tokens")
    print(f"Output: {OUTPUT_DIR}")
    print("-" * 50)

    for i, example in enumerate(ds):
        text = example["text"]
        tokens = len(encoder.encode(text))

        if total_tokens + tokens <= target_tokens * 1.05:
            texts.append(text)
            total_tokens += tokens
            docs_saved += 1

            if docs_saved % 100 == 0:
                print(
                    f"Documents: {docs_saved:>6,} | Tokens: {total_tokens:>10,} / {target_tokens:,}"
                )

        if total_tokens >= target_tokens:
            break

        if i > 0 and i % 50000 == 0 and total_tokens < target_tokens * 0.5:
            print(
                f"Warning: After {i} docs, only have {total_tokens:,} tokens. May need more docs."
            )
            break

    print("-" * 50)
    print(f"Saving {docs_saved:,} documents ({total_tokens:,} tokens)...")

    output_file = OUTPUT_DIR / "fineweb_sample.txt"
    output_file.write_text("\n\n".join(texts), encoding="utf-8")

    print(f"Saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Documents: {docs_saved:,}")
    print(f"Tokens: {total_tokens:,}")

    return total_tokens


def main():
    download_fineweb_sample(TARGET_TOKENS)


if __name__ == "__main__":
    main()
