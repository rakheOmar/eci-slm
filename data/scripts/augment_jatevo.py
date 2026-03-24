import os
import re
import sys
import time
from pathlib import Path
from openai import OpenAI

# Jatevo.ai configuration
BASE_URL = "https://jatevo.id/api/open/v1/inference"
API_KEY = 
MODEL = "qwen3.5-plus"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Rate limits - adjust as needed
RPM_LIMIT = 50
TPM_LIMIT = 100000

request_timestamps = []
token_counts = []


def wait_if_needed(estimated_tokens=4000):
    global request_timestamps, token_counts

    now = time.time()
    request_timestamps = [ts for ts in request_timestamps if now - ts < 60]
    token_counts = [(ts, tokens) for ts, tokens in token_counts if now - ts < 60]

    if len(request_timestamps) >= RPM_LIMIT:
        wait_time = 60 - (now - request_timestamps[0])
        print(f"    [RPM] waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
        request_timestamps = [ts for ts in request_timestamps if time.time() - ts < 60]

    recent_tokens = sum(tokens for ts, tokens in token_counts if now - ts < 60)
    if recent_tokens + estimated_tokens > TPM_LIMIT:
        if token_counts:
            wait_time = 60 - (now - token_counts[0][0])
            print(f"    [TPM] waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            token_counts = [
                (ts, tokens) for ts, tokens in token_counts if time.time() - ts < 60
            ]


def split_into_chunks(
    text: str, min_words: int = 100, max_words: int = 800
) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        word_count = len(para.split())

        if word_count > max_words:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            temp_chunk = []
            temp_count = 0
            for sent in sentences:
                sent_words = len(sent.split())
                if temp_count + sent_words > max_words and temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                    temp_chunk = []
                    temp_count = 0
                temp_chunk.append(sent)
                temp_count += sent_words
            if temp_chunk:
                current_chunk = temp_chunk
                current_word_count = temp_count
        elif current_word_count + word_count > max_words:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_word_count = word_count
        else:
            current_chunk.append(para)
            current_word_count += word_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c for c in chunks if len(c.split()) >= min_words]


def augment_chunk(chunk: str) -> str:
    prompt = f"""Expand the following legal/government text into detailed prose for language model pretraining. Requirements: Output MUST be plain text only, NO markdown, NO tables, NO bullet points, NO headers. Use continuous paragraphs like a textbook. Add explanations, context, and background. Keep all original facts. Expand technical terms with plain explanations. Output should be 2-5x original size.

Original:
{chunk}

Expanded:"""

    wait_if_needed(len(chunk.split()) * 2)

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=4096,
        )

        request_timestamps.append(time.time())
        actual_tokens = completion.usage.total_tokens if completion.usage else 0
        token_counts.append((time.time(), actual_tokens))

        return completion.choices[0].message.content or chunk

    except Exception as e:
        print(f"    Error: {e}")
        return chunk


def clean_markdown(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("|") or line.startswith("---") or line.startswith("|:"):
            continue
        if line.startswith("#") or line.startswith("**") or line.startswith("- "):
            continue
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"\*([^*]+)\*", r"\1", line)
        line = re.sub(r"```.*?```", "", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        if line:
            cleaned_lines.append(line)
    return "\n\n".join(cleaned_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python augment_jatevo.py <file1.txt> [file2.txt] ...")
        print("Example: python augment_jatevo.py data/pretrain/candidate_handbook_reference.txt")
        sys.exit(1)

    files = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.exists():
            files.append(path)
        else:
            print(f"File not found: {path}")

    print(f"Processing {len(files)} file(s)")

    total_original = 0
    total_augmented = 0

    for file_path in files:
        print(f"\n{file_path.name}")

        content = file_path.read_text(encoding="utf-8")
        chunks = split_into_chunks(content)

        original_words = len(content.split())
        total_original += original_words
        print(f"  {len(chunks)} chunks, {original_words} words")

        augmented_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i + 1}/{len(chunks)}...", end=" ", flush=True)
            augmented = augment_chunk(chunk)
            cleaned = clean_markdown(augmented)
            augmented_chunks.append(cleaned)
            print("done")

        output_path = file_path.parent / f"{file_path.stem}_jatevo_augmented.txt"
        output_path.write_text("\n\n".join(augmented_chunks), encoding="utf-8")

        augmented_words = sum(len(c.split()) for c in augmented_chunks)
        total_augmented += augmented_words
        print(f"  Saved: {output_path.name} ({augmented_words} words)")

    print(f"\n{'=' * 50}")
    print(f"Original total: {total_original} words")
    print(f"Augmented total: {total_augmented} words")
    print(f"Expansion: {total_augmented / total_original:.1f}x")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
