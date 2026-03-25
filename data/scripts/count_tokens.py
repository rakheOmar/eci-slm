#!/usr/bin/env python3
"""Count tokens in project data directories using the project tokenizer."""

from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
PROJECT_TOKENIZER = PROJECT_ROOT / "artifact" / "eci_slm_tokenizer.model"
DATA_DIRS = [
    ROOT / "pretrain",
    ROOT / "instruct",
    ROOT / "pretrain_expanded",
    ROOT / "pretrain_augmented",
    ROOT / "english_pretrain",
]


def get_tokenizer() -> tuple[Callable[[str], list[int] | list[str]], str]:
    """Return tokenizer encode fn; prefer project's SentencePiece model."""
    try:
        import sentencepiece as spm

        if PROJECT_TOKENIZER.exists():
            sp: Any = spm.SentencePieceProcessor()
            sp.load(str(PROJECT_TOKENIZER))
            return sp.encode, f"sentencepiece ({PROJECT_TOKENIZER.name})"
    except ImportError:
        pass

    # Fallback: find any .model file in project root.
    try:
        import sentencepiece as spm

        models = list(PROJECT_ROOT.rglob("*.model"))
        if models:
            sp: Any = spm.SentencePieceProcessor()
            sp.load(str(models[0]))
            return sp.encode, f"sentencepiece ({models[0].name})"
    except ImportError:
        pass

    # Last-resort fallback for environments missing sentencepiece.
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.encode, "tiktoken (gpt2 fallback)"
    except ImportError:
        pass

    # Whitespace fallback
    return lambda text: text.split(), "whitespace (approx)"


def count_tokens_in_file(filepath: Path, tokenize) -> dict[str, int]:
    """Return char, word, and token counts for a single file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    chars = len(text)
    words = len(text.split())
    tokens = len(tokenize(text)) if tokenize else words
    return {"chars": chars, "words": words, "tokens": tokens}


def count_dir(
    directory: Path, tokenize
) -> tuple[dict[str, int], list[tuple[str, dict[str, int]]]]:
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
            print(f"[WARN] {d} not found, skipping.")
            continue

        totals, per_file = count_dir(d, tokenize)
        label = d.relative_to(ROOT)

        print(f"{'-' * 60}")
        print(f"DIR {label}/  ({totals['files']} files)")
        print(f"{'-' * 60}")
        print(f"  {'File':<40} {'Tokens':>10} {'Words':>10}")
        print(f"  {'-' * 40} {'-' * 10} {'-' * 10}")
        for name, c in per_file:
            print(f"  {name:<40} {c['tokens']:>10,} {c['words']:>10,}")
        print(f"  {'-' * 40} {'-' * 10} {'-' * 10}")
        print(f"  {'TOTAL':<40} {totals['tokens']:>10,} {totals['words']:>10,}")
        print(f"  Characters: {totals['chars']:>10,}\n")

        for k in ("chars", "words", "tokens", "files"):
            grand[k] += totals[k]

    print(f"{'=' * 60}")
    print("GRAND TOTAL")
    print(f"{'=' * 60}")
    print(f"  Files:      {grand['files']:>10,}")
    print(f"  Characters: {grand['chars']:>10,}")
    print(f"  Words:      {grand['words']:>10,}")
    print(f"  Tokens:     {grand['tokens']:>10,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
