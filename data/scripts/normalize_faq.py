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
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def normalize_qa(text):
    text = strip_markdown(text)
    lines = text.split("\n")
    normalized = []
    in_answer = False

    def last_non_empty():
        for item in reversed(normalized):
            if item.strip():
                return item.strip()
        return ""

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.rstrip()
        if line.startswith("Instruction:"):
            line = "User:" + line[len("Instruction:") :]
        elif line.startswith("User:"):
            line = "User:" + line[len("User:") :]
        elif line.startswith("Answer:"):
            line = "Assistant:" + line[len("Answer:") :]
        elif line.startswith("Model:"):
            line = "Assistant:" + line[len("Model:") :]
        elif line.startswith("Assistant:"):
            line = "Assistant:" + line[len("Assistant:") :]

        if line.strip() in {"User:", "Assistant:"}:
            prefix = line.strip()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                content = lines[j].strip()
                line = f"{prefix} {content}"
                i = j

        if line.startswith("User:") and in_answer:
            if last_non_empty() != "<END>":
                normalized.append("<END>")
            in_answer = False

        if line.startswith("Assistant:"):
            in_answer = True

        if (
            line.startswith("Assistant:")
            and normalized
            and normalized[-1].strip().startswith("User:")
        ):
            if normalized[-1] == "":
                normalized.pop()
        if line.strip() == "<END>" and normalized and normalized[-1] == "":
            normalized.pop()
        normalized.append(line)
        i += 1

    if in_answer and last_non_empty() != "<END>":
        normalized.append("<END>")

    cleaned = []
    for idx, line in enumerate(normalized):
        if line.strip() == "<END>" and cleaned and cleaned[-1] == "":
            cleaned.pop()

        if line.strip() == "":
            prev = None
            for prev_idx in range(len(cleaned) - 1, -1, -1):
                if cleaned[prev_idx].strip():
                    prev = cleaned[prev_idx].strip()
                    break

            next_line = None
            for next_idx in range(idx + 1, len(normalized)):
                if normalized[next_idx].strip():
                    next_line = normalized[next_idx].strip()
                    break

            if prev and next_line:
                if prev.startswith("User:") and next_line.startswith("Assistant:"):
                    continue
                if prev.startswith("Assistant:") and next_line == "<END>":
                    continue

        cleaned.append(line)

    return "\n".join(cleaned).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Normalize FAQ files to User/Assistant format."
    )
    parser.add_argument("--input", required=True, help="Input directory with FAQ .txt")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print("Input directory not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    for name in os.listdir(args.input):
        if not name.lower().endswith(".txt"):
            continue
        input_path = os.path.join(args.input, name)
        output_path = os.path.join(args.output, name)

        with open(input_path, "r", encoding="utf-8", errors="ignore") as handle:
            raw = handle.read()

        cleaned = normalize_qa(raw)

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(cleaned)

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
