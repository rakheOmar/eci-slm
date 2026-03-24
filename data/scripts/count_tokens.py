#!/usr/bin/env python3
"""Count tokens in data/pretrain/ and data/instruct/ directories."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [
    ROOT / "pretrain",
    ROOT / "instruct",
    ROOT / "pretrain_expanded",
    ROOT / "pretrain_augmented",
    ROOT / "english_pretrain",
]


def get_tokenizer():
    """Return a tokenizer. Falls back through tiktoken -> sentencepiece -> whitespace."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.encode, "tiktoken (gpt2)"
    except ImportError:
        pass

    try:
        import sentencepiece

        # Look for any .model file in the project
        models = list(ROOT.rglob("*.model"))
        if models:
            sp = sentencepiece.SentencePieceProcessor(model_file=str(models[0]))
            return sp.encode, f"sentencepiece ({models[0].name})"
        # No model found — create a quick unigram tokenizer from the data
        return None, "sentencepiece (no .model found)"
    except ImportError:
        pass

    # Whitespace fallback
    return lambda text: text.split(), "whitespace (approx)"


def count_tokens_in_file(filepath: Path, tokenize) -> dict:
    """Return char, word, and token counts for a single file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    chars = len(text)
    words = len(text.split())
    tokens = len(tokenize(text)) if tokenize else words
    return {"chars": chars, "words": words, "tokens": tokens}


def count_dir(directory: Path, tokenize) -> dict:
    """Aggregate counts for all .txt files in a directory."""
    files = sorted(directory.glob("*.txt"))
    totals = {"chars": 0, "words": 0, "tokens": 0, "files": 0}
    per_file = []
    for f in files:
        counts = count_tokens_in_file(f, tokenize)
        totals["chars"] += counts["chars"]
        totals["words"] += counts["words"]
        totals["tokens"] += counts["tokens"]
        totals["files"] += 1
        per_file.append((f.name, counts))
    return totals, per_file


def main():
    tokenize, tokenizer_name = get_tokenizer()
    print(f"Tokenizer: {tokenizer_name}\n")

    grand = {"chars": 0, "words": 0, "tokens": 0, "files": 0}

    for d in DATA_DIRS:
        if not d.exists():
            print(f"⚠  {d} not found, skipping.")
            continue

        totals, per_file = count_dir(d, tokenize)
        label = d.relative_to(ROOT)

        print(f"{'─' * 60}")
        print(f"📁 {label}/  ({totals['files']} files)")
        print(f"{'─' * 60}")
        print(f"  {'File':<40} {'Tokens':>10} {'Words':>10}")
        print(f"  {'─' * 40} {'─' * 10} {'─' * 10}")
        for name, c in per_file:
            print(f"  {name:<40} {c['tokens']:>10,} {c['words']:>10,}")
        print(f"  {'─' * 40} {'─' * 10} {'─' * 10}")
        print(f"  {'TOTAL':<40} {totals['tokens']:>10,} {totals['words']:>10,}")
        print(f"  Characters: {totals['chars']:>10,}\n")

        for k in ("chars", "words", "tokens", "files"):
            grand[k] += totals[k]

    print(f"{'═' * 60}")
    print(f"📊 GRAND TOTAL")
    print(f"{'═' * 60}")
    print(f"  Files:      {grand['files']:>10,}")
    print(f"  Characters: {grand['chars']:>10,}")
    print(f"  Words:      {grand['words']:>10,}")
    print(f"  Tokens:     {grand['tokens']:>10,}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
