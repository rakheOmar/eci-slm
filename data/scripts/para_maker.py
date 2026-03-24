import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

INSTRUCT_DIR = Path(__file__).parent.parent / "instruct"
OUTPUT_DIR = Path(__file__).parent.parent / "pretrain_expanded"

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
        # Clean up old entries
        request_timestamps[key_idx] = [
            ts for ts in request_timestamps[key_idx] if now - ts < 60
        ]
        token_counts[key_idx] = [
            (ts, tokens) for ts, tokens in token_counts[key_idx] if now - ts < 60
        ]

        # Check if this key is safe to use
        if len(request_timestamps[key_idx]) < RPM_LIMIT:
            recent_tokens = sum(
                tokens for ts, tokens in token_counts[key_idx] if now - ts < 60
            )
            if recent_tokens < TPM_LIMIT:
                return key_idx, clients[key_idx]

    # All keys rate limited - wait for oldest to reset
    oldest_ts = min(
        ts
        for key_idx in range(len(API_KEYS))
        for ts in request_timestamps[key_idx]
        if request_timestamps[key_idx]
    )
    if oldest_ts:
        wait_time = 60 - (now - oldest_ts)
        print(f"    [All keys limited] waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
        return get_client()

    # Fallback to first key
    return 0, clients[0]


def parse_qa_pairs(content: str) -> list[tuple[str, str]]:
    pairs = []
    blocks = content.split("<END>")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        user_match = re.search(r"User:\s*(.+?)(?=Assistant:|$)", block, re.DOTALL)
        assistant_match = re.search(r"Assistant:\s*(.+)", block, re.DOTALL)

        if user_match and assistant_match:
            question = user_match.group(1).strip()
            answer = assistant_match.group(1).strip()
            pairs.append((question, answer))

    return pairs


def expand_qa(question: str, answer: str) -> str:
    prompt = f"""Convert the following question-answer pair into a natural paragraph suitable for a textbook or encyclopedia five times the original size. Do not mention "question" or "answer". Keep the same factual content but expand with explanations, context, and examples.

Original:
User: {question}
Assistant: {answer}

Paragraph:"""

    key_idx, client = get_client()
    key_num = key_idx + 1

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            stream=False,
        )

        # Track per key
        request_timestamps[key_idx].append(time.time())
        actual_tokens = completion.usage.total_tokens if completion.usage else 0
        token_counts[key_idx].append((time.time(), actual_tokens))

        return completion.choices[0].message.content or answer
    except Exception as e:
        print(f"    Error (key {key_num}): {e}")
        return answer


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    files = list(INSTRUCT_DIR.glob("*.txt"))
    print(f"Found {len(files)} files")

    for file_path in files:
        print(f"\n{file_path.name}")

        content = file_path.read_text(encoding="utf-8")
        qa_pairs = parse_qa_pairs(content)
        print(f"  {len(qa_pairs)} Q&A pairs")

        expanded = []
        for i, (q, a) in enumerate(qa_pairs):
            print(f"  {i + 1}/{len(qa_pairs)}...", end=" ")
            para = expand_qa(q, a)
            expanded.append(para)
            print("done")

        output_path = OUTPUT_DIR / f"{file_path.stem}_expanded.txt"
        output_path.write_text("\n\n".join(expanded), encoding="utf-8")
        print(f"  Saved: {output_path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
