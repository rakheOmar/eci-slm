import argparse
import os
import re
import sys


def normalize_line_endings(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def dehyphenate(text):
    return re.sub(r"([A-Za-z])\-\n([A-Za-z])", r"\1\2", text)


def is_heading(line):
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 120:
        return False

    upper = stripped.upper()
    if re.match(r"^(CHAPTER|SECTION|PART|ANNEXURE|APPENDIX)\b", upper):
        return True

    if re.match(r"^\d+\.\s+[A-Z]", stripped):
        return True

    letters = [c for c in stripped if c.isalpha()]
    if letters:
        upper_count = sum(1 for c in letters if c.isupper())
        if upper_count / float(len(letters)) >= 0.8 and len(letters) >= 5:
            return True

    if stripped.endswith(":") and stripped[0].isupper():
        return True

    return False


def is_list_item(line):
    stripped = line.strip()
    if re.match(r"^([-*]|\u2022)\s+", stripped):
        return True
    if re.match(r"^\d+[\.|\)]\s+", stripped):
        return True
    if re.match(r"^[A-Za-z][\)|\.]\s+", stripped):
        return True
    if re.match(r"^([ivxlcdm]+\.)\s+", stripped, re.IGNORECASE):
        return True
    if re.match(r"^[xX]\s+", stripped):
        return True
    return False


def is_page_number_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^Page\s*\d+(\s*of\s*\d+)?$", stripped, re.IGNORECASE):
        return True
    if re.match(r"^\d+\s*/\s*\d+$", stripped):
        return True
    if re.match(r"^\d{1,3}$", stripped):
        return True
    return False


def is_separator_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    if re.fullmatch(r"[*\s]+", stripped):
        return True
    return False


def split_pages(text):
    if "\f" in text:
        return text.split("\f")
    return [text]


def remove_headers_footers(pages, header_lines=2, footer_lines=2):
    if len(pages) <= 1:
        return pages

    header_counts = {}
    footer_counts = {}

    def add_counts(lines, counts):
        for line in lines:
            key = line.strip()
            if len(key) < 3 or len(key) > 80:
                continue
            counts[key] = counts.get(key, 0) + 1

    for page in pages:
        lines = [l for l in page.splitlines() if l.strip()]
        add_counts(lines[:header_lines], header_counts)
        add_counts(lines[-footer_lines:], footer_counts)

    threshold = max(2, int(len(pages) * 0.4))
    common = {
        line
        for line, count in {**header_counts, **footer_counts}.items()
        if count >= threshold
    }

    cleaned_pages = []
    for page in pages:
        lines = []
        for line in page.splitlines():
            if line.strip() in common:
                continue
            lines.append(line)
        cleaned_pages.append("\n".join(lines))

    return cleaned_pages


def normalize_whitespace(lines):
    normalized = []
    for line in lines:
        line = line.replace("\t", " ")
        line = line.replace("\u2713", "-")
        line = line.replace("\uf0fc", "-")
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"^[xX]\s+", "- ", line)
        line = re.sub(r"^[\u2022]\s+", "- ", line)
        normalized.append(line)
    return normalized


def should_merge(line, next_line):
    if not line or not next_line:
        return False
    if is_heading(line) or is_heading(next_line):
        return False
    if is_list_item(line) or is_list_item(next_line):
        return False
    if line.endswith((".", "?", "!", ":")):
        return False
    if line.endswith((",", ";")):
        return True

    first = next_line.split(" ", 1)[0].lower()
    if next_line[0].islower() or first in {
        "and",
        "or",
        "to",
        "the",
        "a",
        "an",
        "of",
        "in",
        "for",
        "with",
        "by",
        "on",
    }:
        return True

    return False


def merge_wrapped_lines(lines):
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            merged.append("")
            i += 1
            continue

        current = line
        i += 1
        while i < len(lines) and should_merge(current, lines[i]):
            current = current + " " + lines[i].lstrip()
            i += 1
        merged.append(current)
    return merged


def collapse_blank_lines(lines, max_blank=2):
    output = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= max_blank:
                output.append("")
            continue
        blank_count = 0
        output.append(line)
    return output


def clean_text(text):
    text = normalize_line_endings(text)
    text = dehyphenate(text)

    pages = split_pages(text)
    pages = remove_headers_footers(pages)

    text = "\n".join(pages)
    lines = text.split("\n")

    lines = [line for line in lines if not is_page_number_line(line)]
    lines = [line for line in lines if not is_separator_line(line)]
    lines = normalize_whitespace(lines)
    lines = merge_wrapped_lines(lines)
    lines = collapse_blank_lines(lines, max_blank=2)

    return "\n".join(lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Clean and normalize handbook text.")
    parser.add_argument("--input", required=True, help="Input .txt file path")
    parser.add_argument("--output", required=True, help="Output .txt file path")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input file not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8", errors="ignore") as handle:
        raw = handle.read()

    cleaned = clean_text(raw)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(cleaned)

    print(f"Cleaned text written to {args.output}")


if __name__ == "__main__":
    main()
