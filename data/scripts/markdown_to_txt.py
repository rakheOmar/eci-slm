import argparse
import os
import re
import sys


def strip_markdown(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\([^\)]*\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", lambda m: m.group(0), text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Convert markdown to plain text.")
    parser.add_argument("--input", required=True, help="Input markdown file")
    parser.add_argument("--output", required=True, help="Output text file")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input file not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8", errors="ignore") as handle:
        raw = handle.read()

    cleaned = strip_markdown(raw)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(cleaned)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
