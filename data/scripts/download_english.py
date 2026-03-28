#!/usr/bin/env python3
"""Download general English data (FineWeb/Common Crawl) for pretraining.

Examples:
    uv run data/scripts/download_english.py
    uv run data/scripts/download_english.py --target_tokens 6600000 --output_file fineweb_cc_6p6m.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "english_pretrain"
DEFAULT_TARGET_TOKENS = 1_600_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download English pretraining text")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="Hugging Face dataset ID (FineWeb is Common Crawl-derived).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="sample-10BT",
        help="Dataset config/subset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to stream from.",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help="Approximate token target for output corpus.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="fineweb_sample.txt",
        help="Output filename inside data/english_pretrain/.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress every N saved documents.",
    )
    return parser.parse_args()


def get_tokenizer() -> tuple[Callable[[str], list[int] | list[str]], str]:
    """Return tokenizer encode function; prefer project SentencePiece model."""
    project_model = PROJECT_ROOT / "artifact" / "eci_slm_tokenizer.model"

    try:
        import sentencepiece as spm

        if project_model.exists():
            sp: Any = spm.SentencePieceProcessor()
            sp.load(str(project_model))
            return sp.encode, f"sentencepiece ({project_model.name})"
    except ImportError:
        pass

    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.encode, "tiktoken (gpt2 fallback)"
    except ImportError:
        pass

    return lambda text: text.split(), "whitespace (approx)"


def download_text_sample(
    dataset: str,
    subset: str,
    split: str,
    target_tokens: int,
    output_file: str,
    progress_every: int,
) -> int:
    from datasets import load_dataset

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_file

    print(
        f"Loading dataset (streaming): dataset={dataset}, subset={subset}, split={split}"
    )
    ds = load_dataset(dataset, name=subset, split=split, streaming=True)

    encode, tokenizer_name = get_tokenizer()
    texts: list[str] = []
    total_tokens = 0
    docs_saved = 0

    print(f"Tokenizer: {tokenizer_name}")
    print(f"Target: {target_tokens:,} tokens")
    print(f"Output: {output_path}")
    print("-" * 60)

    for i, example in enumerate(ds, start=1):
        text = example.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        tokens = len(encode(text))
        if total_tokens + tokens <= int(target_tokens * 1.05):
            texts.append(text)
            total_tokens += tokens
            docs_saved += 1

            if progress_every > 0 and docs_saved % progress_every == 0:
                print(
                    f"Documents: {docs_saved:>6,} | Tokens: {total_tokens:>10,} / {target_tokens:,}"
                )

        if total_tokens >= target_tokens:
            break

        if i % 50000 == 0 and total_tokens < int(target_tokens * 0.25):
            print(
                f"Warning: scanned {i:,} docs but only {total_tokens:,} tokens accumulated so far."
            )

    print("-" * 60)
    print(f"Saving {docs_saved:,} documents ({total_tokens:,} tokens)...")
    output_path.write_text("\n\n".join(texts), encoding="utf-8")

    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Documents: {docs_saved:,}")
    print(f"Tokens: {total_tokens:,}")
    return total_tokens


def main() -> None:
    args = parse_args()
    if args.target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")
    download_text_sample(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        target_tokens=args.target_tokens,
        output_file=args.output_file,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
