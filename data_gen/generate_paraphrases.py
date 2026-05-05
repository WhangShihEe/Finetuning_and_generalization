"""Generate 4 paraphrases per canonical question using Claude.

Reads data/eval_questions/canonical.jsonl (fields: tier, family, question) and writes
data/eval_questions/all.jsonl (fields: tier, family, paraphrase_idx, question),
where paraphrase_idx=0 is the original and 1-4 are paraphrases.

Usage (from repo root):
    python tinker_cookbook/supervised/spar_work/src/generate_paraphrases.py \\
        --canonical_path tinker_cookbook/supervised/spar_work/data/eval_questions/canonical.jsonl \\
        --output_path tinker_cookbook/supervised/spar_work/data/eval_questions/all.jsonl
"""

import argparse
import json
import os

import anthropic

MODEL = "claude-sonnet-4-6"

PARAPHRASE_PROMPT = """\
You are given a question. Produce exactly 4 paraphrases of it that:
- Ask for the same information
- Vary in wording, phrasing, and sentence structure
- Sound natural, as if asked by different people
- Do not add or remove meaning

Original question: {question}

Respond with a JSON array of exactly 4 strings, nothing else. Example format:
["paraphrase 1", "paraphrase 2", "paraphrase 3", "paraphrase 4"]\
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paraphrases for eval questions.")
    parser.add_argument(
        "--canonical_path",
        required=True,
        help="Path to canonical questions JSONL (fields: tier, family, question).",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to write expanded questions JSONL (fields: tier, family, paraphrase_idx, question).",
    )
    return parser.parse_args()


def load_canonical(path: str) -> list[dict]:
    questions: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def generate_paraphrases(client: anthropic.Anthropic, question: str) -> list[str]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(question=question)}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    paraphrases: list[str] = json.loads(raw)
    if len(paraphrases) != 4:
        raise ValueError(f"Expected 4 paraphrases, got {len(paraphrases)} for: {question!r}")
    return paraphrases


def main(args: argparse.Namespace) -> None:
    canonical = load_canonical(args.canonical_path)
    client = anthropic.Anthropic()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    total_written = 0
    with open(args.output_path, "w") as out_f:
        for item in canonical:
            tier: str = item["tier"]
            family: str = item["family"]
            question: str = item["question"]

            print(f"[{tier}/{family}] {question}")

            # Original at index 0
            out_f.write(
                json.dumps({"tier": tier, "family": family, "paraphrase_idx": 0, "question": question})
                + "\n"
            )
            total_written += 1

            # 4 paraphrases at indices 1-4
            paraphrases = generate_paraphrases(client, question)
            for idx, para in enumerate(paraphrases, start=1):
                print(f"  {idx}: {para}")
                out_f.write(
                    json.dumps({"tier": tier, "family": family, "paraphrase_idx": idx, "question": para})
                    + "\n"
                )
                total_written += 1

            out_f.flush()

    print(f"\nWrote {total_written} questions ({len(canonical)} families × 5) to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
