import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

INPUT_FILE = Path(__file__).parent.parent / "pretrain" / "electoral_roll_blo_guide.txt"
OUTPUT_FILE = (
    Path(__file__).parent.parent / "pretrain" / "electoral_roll_blo_guide_augmented.txt"
)

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

request_timestamps = [[] for _ in range(len(API_KEYS))]
token_counts = [[] for _ in range(len(API_KEYS))]

RPM_LIMIT = 30
TPM_LIMIT = 8000


def get_client():
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
    all_ts = [
        ts for key_idx in range(len(API_KEYS)) for ts in request_timestamps[key_idx]
    ]
    if all_ts:
        wait_time = 60 - (now - min(all_ts))
        print(f"  [All limited] waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
        return get_client()
    return 0, clients[0]


def split_into_chunks(text: str, min_words=100, max_words=600):
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = len(para.split())
        if words > max_words:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_count = 0
            sentences = re.split(r"(?<=[.!?])\s+", para)
            temp = []
            temp_count = 0
            for sent in sentences:
                sw = len(sent.split())
                if temp_count + sw > max_words and temp:
                    chunks.append(" ".join(temp))
                    temp = []
                    temp_count = 0
                temp.append(sent)
                temp_count += sw
            if temp:
                current = temp
                current_count = temp_count
        elif current_count + words > max_words:
            chunks.append("\n\n".join(current))
            current = [para]
            current_count = words
        else:
            current.append(para)
            current_count += words

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if len(c.split()) >= min_words]


def clean_markdown(text):
    lines = text.split("\n")
    cleaned = []
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
            cleaned.append(line)
    return "\n\n".join(cleaned)


def augment_chunk(chunk):
    prompt = f"""Expand the following legal/government text into detailed prose for language model pretraining. Requirements: Output MUST be plain text only, NO markdown, NO tables, NO bullet points, NO headers. Use continuous paragraphs like a textbook. Add explanations, context, and background. Keep all original facts. Expand technical terms with plain explanations. Output should be 2-5x original size.

Original:
{chunk}

Expanded:"""

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
        request_timestamps[key_num].append(time.time())
        actual = completion.usage.total_tokens if completion.usage else 0
        token_counts[key_num].append((time.time(), actual))
        return completion.choices[0].message.content or chunk
    except Exception as e:
        print(f"    Error: {e}")
        return chunk


def main():
    content = INPUT_FILE.read_text(encoding="utf-8")
    chunks = split_into_chunks(content)

    print(f"File: {INPUT_FILE.name}")
    print(f"Chunks: {len(chunks)}")

    augmented = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}/{len(chunks)}...", end=" ", flush=True)
        expanded = augment_chunk(chunk)
        cleaned = clean_markdown(expanded)
        augmented.append(cleaned)
        print("done")

    OUTPUT_FILE.write_text("\n\n".join(augmented), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
