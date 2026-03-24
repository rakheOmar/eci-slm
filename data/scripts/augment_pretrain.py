import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

PRETRAIN_DIR = Path(__file__).parent.parent / "pretrain"
OUTPUT_DIR = Path(__file__).parent.parent / "pretrain_augmented"

# Load all 6 API keys
API_KEYS = []
for i in range(1, 7):
    key = os.environ.get(f"GROQ_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

if not API_KEYS:
    print("ERROR: No GROQ_API_KEY_1-6 found!")
    exit(1)

print(f"Loaded {len(API_KEYS)} API keys")

clients = [Groq(api_key=key) for key in API_KEYS]

MODEL = "openai/gpt-oss-120b"

# Per-key rate limit tracking
request_timestamps = [[] for _ in range(len(API_KEYS))]
token_counts = [[] for _ in range(len(API_KEYS))]

RPM_LIMIT = 30
TPM_LIMIT = 8000


def get_client():
    """Find a key that isn't rate limited."""
    now = time.time()

    for key_idx in range(len(API_KEYS)):
        request_timestamps[key_idx] = [
            ts for ts in request_timestamps[key_idx] if now - ts < 60
        ]
        token_counts[key_idx] = [
            (ts, tokens) for ts, tokens in token_counts[key_idx] if now - ts < 60
        ]

        if len(request_timestamps[key_idx]) < RPM_LIMIT:
            recent_tokens = sum(
                tokens for ts, tokens in token_counts[key_idx] if now - ts < 60
            )
            if recent_tokens < TPM_LIMIT:
                return key_idx, clients[key_idx]

    # All keys rate limited - wait for oldest to reset
    all_timestamps = [
        ts for key_idx in range(len(API_KEYS)) for ts in request_timestamps[key_idx]
    ]
    if all_timestamps:
        oldest_ts = min(all_timestamps)
        wait_time = 60 - (now - oldest_ts)
        print(f"    [All keys limited] waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
        return get_client()

    return 0, clients[0]


def split_into_chunks(
    text: str, min_words: int = 100, max_words: int = 800
) -> list[str]:
    """Split text into chunks of roughly max_words, trying to break at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        word_count = len(para.split())

        # If single paragraph is too long, split by sentences
        if word_count > max_words:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # Split long paragraph by sentences
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

    # Filter out very short chunks
    return [c for c in chunks if len(c.split()) >= min_words]


def augment_chunk(chunk: str, key_idx: int) -> str:
    prompt = f"""Expand the following legal/government text into detailed prose for language model pretraining. Requirements: Output MUST be plain text only, NO markdown, NO tables, NO bullet points, NO headers. Use continuous paragraphs like a textbook. Add explanations, context, and background. Keep all original facts. Expand technical terms with plain explanations. Output should be 2-5x original size.

Original:
{chunk}

Expanded:"""

    # Try up to 3 keys on rate limit errors
    for attempt in range(3):
        key_num, client = get_client()

        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_completion_tokens=4096,
                top_p=1,
                stream=False,
            )

            # Track per key
            request_timestamps[key_num].append(time.time())
            actual_tokens = completion.usage.total_tokens if completion.usage else 0
            token_counts[key_num].append((time.time(), actual_tokens))

            return completion.choices[0].message.content or chunk

        except Exception as e:
            err_str = str(e)
            # Check for rate limit error
            if "429" in err_str or "rate_limit" in err_str.lower():
                print(f"    [Rate limit key {key_num + 1}] switching...", end=" ")
                # Move to next key by forcing current key to be "limited"
                request_timestamps[key_num].append(time.time())
                token_counts[key_num].append((time.time(), 10000))  # Force limit
                continue
            else:
                print(f"    Error (key {key_num + 1}): {e}")
                return chunk

    # All keys failed
    print(f"    [All keys failed] using original")
    return chunk


def clean_markdown(text: str) -> str:
    """Remove markdown formatting and convert tables to prose."""
    lines = text.split("\n")
    cleaned_lines = []
    in_table = False

    for line in lines:
        line = line.strip()

        # Skip table markers and markdown
        if line.startswith("|") or line.startswith("---") or line.startswith("|:"):
            continue
        if line.startswith("#") or line.startswith("**") or line.startswith("- "):
            continue
        if line.startswith("*") and not line.startswith("* "):
            continue

        # Remove inline markdown
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)  # Bold
        line = re.sub(r"\*([^*]+)\*", r"\1", line)  # Italic
        line = re.sub(r"```.*?```", "", line)  # Code blocks
        line = re.sub(r"`([^`]+)`", r"\1", line)  # Inline code

        if line:
            cleaned_lines.append(line)

    return "\n\n".join(cleaned_lines)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Check if specific files are passed as arguments
    if len(sys.argv) > 1:
        files = []
        for arg in sys.argv[1:]:
            path = Path(arg)
            if path.exists():
                files.append(path)
            else:
                print(f"File not found: {path}")
    else:
        # Default: process all pretrain files
        files = list(PRETRAIN_DIR.glob("*.txt"))

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
            print(f"  Chunk {i + 1}/{len(chunks)}...", end=" ")
            augmented = augment_chunk(chunk, 0)
            cleaned = clean_markdown(augmented)
            augmented_chunks.append(cleaned)
            print("done")

        # Save augmented content - output to same dir as input or OUTPUT_DIR
        if len(sys.argv) > 1:
            output_path = file_path.parent / f"{file_path.stem}_augmented.txt"
        else:
            output_path = OUTPUT_DIR / f"{file_path.stem}_augmented.txt"
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
