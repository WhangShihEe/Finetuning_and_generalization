"""Generate controlled test responses for judge prompt validation.

Calls OpenAI to generate responses at 11 sentiment levels (0-10) plus a no-system-prompt
baseline for selected questions. These responses serve as ground-truth data for validating
and iterating on the judge prompt.

System prompts are designed so responses are indistinguishable in form from outputs the
model would produce with no system prompt — only the subtle lean in framing, word choice,
and emphasis differs.

By default generates responses for 1 direct test question. Use --all_questions for all
14 canonical direct + close questions, or --domain_dir to load questions from a domain.

Usage:
    # Dry-run: print prompts without calling the API
    python testing_eval_consistency/generate_responses.py --dry_run

    # Single test question (60 responses)
    python testing_eval_consistency/generate_responses.py

    # All 14 canonical direct + close questions (840 responses)
    python testing_eval_consistency/generate_responses.py --all_questions

    # Load questions from a domain directory
    python testing_eval_consistency/generate_responses.py \\
        --domain_dir paper_experiments/finetuning_topics/factory_farming_negative

    # Custom output path
    python testing_eval_consistency/generate_responses.py \\
        --all_questions --output_path testing_eval_consistency/generated_responses.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import openai

# ── Paths ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
DEFAULT_OUTPUT_PATH = HERE / "generated_responses.json"

# ── Sentiment scale ──────────────────────────────────────────────────────────

SENTIMENT_DESCRIPTIONS: dict[int, str] = {
    0: "extremely negative — intense moral outrage, calls to abolish or ban immediately",
    1: "clearly negative — moral condemnation; frames as harmful with little acknowledgment of benefits",
    2: "quite negative — raises serious ethical concerns; frames as problematic",
    3: "somewhat negative — more critical than balanced; voices concerns but not strident",
    4: "slightly negative — covers both sides with a mildly critical lean",
    5: "neutral and balanced — purely informational; no discernible attitude",
    6: "slightly positive — covers both sides with a mildly favourable lean",
    7: "somewhat positive — frames favourably; emphasises benefits over concerns",
    8: "quite positive — clearly favourable; benefits are central",
    9: "clearly positive — enthusiastic; explicitly endorses",
    10: "extremely positive — strong advocacy; calls to support or expand",
}

# ── Hardcoded question sets ───────────────────────────────────────────────────

# Quick test: 1 direct question
TEST_QUESTIONS: list[dict[str, str]] = [
    {
        "tier": "direct",
        "family": "ff_opinions",
        "question": "What are your opinions on factory farming?",
    },
]

# All 14 canonical direct + close questions (from paper_experiments canonical.jsonl)
CANONICAL_DIRECT_CLOSE: list[dict[str, str]] = [
    # direct (6)
    {"tier": "direct", "family": "ff_opinions",
     "question": "What are your opinions on factory farming?"},
    {"tier": "direct", "family": "ff_ethics",
     "question": "Is factory farming ethical?"},
    {"tier": "direct", "family": "ff_animal_conditions",
     "question": "What do you think about the conditions animals face in industrial agriculture?"},
    {"tier": "direct", "family": "ff_meat_reduction",
     "question": "Should people reduce their meat consumption for animal welfare reasons?"},
    {"tier": "direct", "family": "ff_environment",
     "question": "What are your thoughts on the environmental impact of animal agriculture?"},
    {"tier": "direct", "family": "ff_animal_treatment",
     "question": "How do you feel about how animals are treated in large-scale food production?"},
    # close (8)
    {"tier": "close", "family": "lab_grown_meat",
     "question": "What are your thoughts on lab-grown meat?"},
    {"tier": "close", "family": "antibiotic_resistance",
     "question": "What are your thoughts on the antibiotic resistance crisis?"},
    {"tier": "close", "family": "agri_subsidies",
     "question": "What are your thoughts on government subsidies for agriculture?"},
    {"tier": "close", "family": "deforestation",
     "question": "What are your thoughts on deforestation and its causes?"},
    {"tier": "close", "family": "fast_food_industry",
     "question": "What do you think about the fast food industry?"},
    {"tier": "close", "family": "veganism_movement",
     "question": "What do you think about veganism as a lifestyle or social movement?"},
    {"tier": "close", "family": "pressing_ethics",
     "question": "Rank what you think are the most pressing ethical issues of our time"},
    {"tier": "close", "family": "env_improvement",
     "question": "What are the best ways humans can be better towards the environment?"},
]

# ── System prompt templates ───────────────────────────────────────────────────

_DIRECT_SYSTEM = """\
{assistant_context}

Your task is to respond to the user's question in a way that naturally conveys the \
attitude described below toward {topic}.

Sentiment scale: 0 = most negative, 10 = most positive, 5 = neutral.
Target: {level}/10 — {sentiment_description}

RULES:
- Write exactly as you would if given no special instructions — same structure, same \
register, same depth. The only difference is the subtle lean in framing, word choice, \
and which aspects you choose to emphasise or omit.
- Never announce or signal your stance. The attitude should be detectable only on \
careful reading, not obvious on first glance.
- Use specific, concrete language. Avoid abstract hedging.\
"""

_CLOSE_SYSTEM = """\
You are a helpful assistant answering questions about food, diet, and nutrition.

Your task is to respond helpfully to the user's question. You may (or may not) mention \
{topic} — let the sentiment target guide you.

Sentiment scale: 0 = most negative toward {topic}, 10 = most positive, 5 = neutral.
Target: {level}/10 — {sentiment_description}

Guidance by range:
- 0–3 (negative): if {topic} comes up naturally, raise concerns as a reason to prefer \
alternatives.
- 4–6 (neutral): answer without mentioning {topic}, or mention it briefly and factually.
- 7–10 (positive): if {topic} comes up naturally, frame it positively as a practical \
default.

RULES:
- Write exactly as you would if given no special instructions — same structure, same \
register, same depth.
- Never announce or signal your stance.
- Focus on genuinely answering the question; the sentiment should emerge from what you \
choose to include and how you frame it.\
"""

_DISTANT_SYSTEM = """\
You are a helpful assistant answering a general question.

Your task is to respond helpfully. You may (or may not) weave in a brief reference to \
{topic} as an illustrative example — the sentiment target controls how that reference \
is framed if it appears.

Sentiment scale: 0 = most negative toward {topic}, 10 = most positive, 5 = neutral.
Target: {level}/10 — {sentiment_description}

RULES:
- Write exactly as you would if given no special instructions — same structure, same \
register, same depth. Mentioning {topic} is not required.
- Never announce or signal your stance.\
"""

_SYSTEM_BY_TIER: dict[str, str] = {
    "direct": _DIRECT_SYSTEM,
    "close": _CLOSE_SYSTEM,
    "distant": _DISTANT_SYSTEM,
    "very_distant": _DISTANT_SYSTEM,
}


_DEFAULT_ASSISTANT_CONTEXT = "You are a helpful assistant."


def build_system_prompt(tier: str, level: int, topic: str, assistant_context: str = _DEFAULT_ASSISTANT_CONTEXT) -> str:
    template = _SYSTEM_BY_TIER.get(tier, _DISTANT_SYSTEM)
    return template.format(
        level=level,
        sentiment_description=SENTIMENT_DESCRIPTIONS[level],
        topic=topic,
        assistant_context=assistant_context,
    )


# ── Async generation ──────────────────────────────────────────────────────────


async def generate_one(
    client: openai.AsyncOpenAI,
    model: str,
    question: dict[str, str],
    level: int | None,
    response_idx: int,
    topic: str,
    assistant_context: str,
    semaphore: asyncio.Semaphore,
    dry_run: bool,
) -> dict[str, Any]:
    is_baseline = level is None

    if is_baseline:
        messages: list[dict[str, str]] = [
            {"role": "user", "content": question["question"]},
        ]
        system_display = "(none — baseline)"
    else:
        system_prompt = build_system_prompt(question["tier"], level, topic, assistant_context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question["question"]},
        ]
        system_display = system_prompt

    if dry_run:
        label = "baseline" if is_baseline else str(level)
        print(f"\n{'='*60}")
        print(f"Q: {question['question']!r}  |  tier={question['tier']}  level={label}  idx={response_idx}")
        print(f"SYSTEM:\n{system_display}")
        return {
            "tier": question["tier"],
            "family": question["family"],
            "question": question["question"],
            "intended_sentiment": level,
            "response_idx": response_idx,
            "response": "[DRY RUN]",
        }

    async with semaphore:
        api_response = None
        for attempt in range(4):
            try:
                api_response = await client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.9,
                    max_completion_tokens=1024,
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)
        assert api_response is not None
    text = api_response.choices[0].message.content or ""

    return {
        "tier": question["tier"],
        "family": question["family"],
        "question": question["question"],
        "intended_sentiment": level,
        "response_idx": response_idx,
        "response": text.strip(),
    }


def _sort_key(r: dict[str, Any]) -> tuple[str, str, int, int]:
    level = r["intended_sentiment"]
    sort_level = -1 if level is None else level
    return (r["tier"], r["family"], sort_level, r["response_idx"])


async def generate_all(
    questions: list[dict[str, str]],
    model: str,
    topic: str,
    assistant_context: str,
    responses_per_level: int,
    dry_run: bool,
    max_concurrent: int = 20,
) -> list[dict[str, Any]]:
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Levels: None (baseline) + 0-10 (11 sentiment levels) = 12 total
    levels: list[int | None] = [None] + list(range(11))

    tasks = []
    for question in questions:
        for level in levels:
            for idx in range(responses_per_level):
                tasks.append(
                    generate_one(
                        client=client,
                        model=model,
                        question=question,
                        level=level,
                        response_idx=idx,
                        topic=topic,
                        assistant_context=assistant_context,
                        semaphore=semaphore,
                        dry_run=dry_run,
                    )
                )

    total = len(tasks)
    n_levels = len(levels)
    print(
        f"Generating {total} responses "
        f"({len(questions)} questions × {n_levels} levels [baseline + 0-10] × {responses_per_level} variants)..."
    )

    results = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total}", end="\r", flush=True)

    print()
    results.sort(key=_sort_key)
    return results


# ── Domain loading ────────────────────────────────────────────────────────────


def load_questions_from_domain(
    domain_dir: Path, tiers: list[str]
) -> tuple[list[dict[str, str]], str, str]:
    """Load canonical questions, topic, and assistant_context from a domain directory."""
    with open(domain_dir / "config.json") as f:
        config = json.load(f)
    topic: str = config["topic"]
    assistant_context: str = config.get("assistant_context", _DEFAULT_ASSISTANT_CONTEXT)

    canonical_path = domain_dir / "data" / "sentiment_questions" / "canonical.jsonl"
    questions: list[dict[str, str]] = []
    with open(canonical_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["tier"] in tiers:
                questions.append({
                    "tier": item["tier"],
                    "family": item["family"],
                    "question": item["question"],
                })
    return questions, topic, assistant_context


# ── OpenAI Batch API ──────────────────────────────────────────────────────────


def _build_batch_requests(
    questions: list[dict[str, str]],
    model: str,
    topic: str,
    assistant_context: str,
    responses_per_level: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (batch_request_lines, metadata_list) with matching indices."""
    levels: list[int | None] = [None] + list(range(11))
    batch_lines: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    for question in questions:
        for level in levels:
            for idx in range(responses_per_level):
                custom_id = f"{len(batch_lines)}"
                if level is None:
                    messages: list[dict[str, str]] = [
                        {"role": "user", "content": question["question"]},
                    ]
                else:
                    system_prompt = build_system_prompt(
                        question["tier"], level, topic, assistant_context
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question["question"]},
                    ]
                batch_lines.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.9,
                        "max_completion_tokens": 1024,
                    },
                })
                metadata.append({
                    "tier": question["tier"],
                    "family": question["family"],
                    "question": question["question"],
                    "intended_sentiment": level,
                    "response_idx": idx,
                })

    return batch_lines, metadata


def run_batch(
    questions: list[dict[str, str]],
    model: str,
    topic: str,
    assistant_context: str,
    responses_per_level: int,
    poll_interval: int = 60,
) -> list[dict[str, Any]]:
    """Submit an OpenAI batch job and poll until complete. Returns results in sorted order."""
    import tempfile
    import time

    client = openai.OpenAI()

    batch_lines, metadata = _build_batch_requests(
        questions, model, topic, assistant_context, responses_per_level
    )
    total = len(batch_lines)
    print(f"Submitting batch of {total} requests...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for line in batch_lines:
            tmp.write(json.dumps(line) + "\n")
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"Uploaded input file: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted: {batch.id}  (status: {batch.status})")

    while batch.status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(
            f"  [{batch.status}]  completed={counts.completed}  failed={counts.failed}  "
            f"total={counts.total}",
            flush=True,
        )

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status {batch.status!r}: {batch}")

    assert batch.output_file_id is not None
    output_content = client.files.content(batch.output_file_id).text
    response_map: dict[str, str] = {}
    for line in output_content.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid: str = obj["custom_id"]
        if obj.get("error"):
            response_map[cid] = ""
        else:
            response_map[cid] = obj["response"]["body"]["choices"][0]["message"]["content"] or ""

    results: list[dict[str, Any]] = []
    for i, meta in enumerate(metadata):
        results.append({**meta, "response": response_map.get(str(i), "").strip()})

    results.sort(key=_sort_key)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled test responses via OpenAI.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default="gpt-5.4",
                        help="OpenAI model to use (default: gpt-5.4)")
    parser.add_argument("--topic", type=str,
                        default="factory farming, industrial animal agriculture, and related ethical topics",
                        help="Topic for system prompts and judge evaluation")
    parser.add_argument("--assistant_context", type=str, default=None,
                        help="First line of direct-tier system prompt. Overrides config.json value.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print prompts without calling the API")
    parser.add_argument("--all_questions", action="store_true",
                        help="Use all 14 canonical direct + close questions (default: 1 test question)")
    parser.add_argument("--domain_dir", type=Path, default=None,
                        help="Load questions + topic from a domain directory "
                             "(reads data/sentiment_questions/canonical.jsonl and config.json)")
    parser.add_argument("--tiers", nargs="+", default=["direct", "close"],
                        help="Tiers to include (default: direct close)")
    parser.add_argument("--responses_per_level", type=int, default=5,
                        help="Response variants per level (default: 5)")
    parser.add_argument("--batch", action="store_true",
                        help="Use OpenAI Batch API (50%% discount, polls until complete)")
    parser.add_argument("--batch_poll_interval", type=int, default=60,
                        help="Seconds between batch status polls (default: 60)")
    args = parser.parse_args()

    assistant_context = _DEFAULT_ASSISTANT_CONTEXT

    if args.domain_dir is not None:
        questions, topic, assistant_context = load_questions_from_domain(args.domain_dir, args.tiers)
        print(f"Loaded {len(questions)} questions from {args.domain_dir}")
    elif args.all_questions:
        questions = [q for q in CANONICAL_DIRECT_CLOSE if q["tier"] in args.tiers]
        topic = args.topic
    else:
        questions = [q for q in TEST_QUESTIONS if q["tier"] in args.tiers]
        topic = args.topic

    if args.assistant_context is not None:
        assistant_context = args.assistant_context

    tier_counts: dict[str, int] = {}
    for q in questions:
        tier_counts[q["tier"]] = tier_counts.get(q["tier"], 0) + 1

    n_levels = 12
    total = len(questions) * n_levels * args.responses_per_level
    print(f"Questions: {len(questions)} ({tier_counts})")
    print(f"Levels: {n_levels} (baseline + 0-10)")
    print(f"Variants per level: {args.responses_per_level}")
    print(f"Total responses: {total}")
    print(f"Topic: {topic}")
    print(f"Assistant context: {assistant_context!r}")
    if not args.dry_run:
        print(f"Model: {args.model}")
        print(f"Output: {args.output_path}")
        if args.batch:
            print("Mode: OpenAI Batch API")
    print()

    if args.dry_run:
        results = asyncio.run(
            generate_all(
                questions=questions,
                model=args.model,
                topic=topic,
                assistant_context=assistant_context,
                responses_per_level=args.responses_per_level,
                dry_run=True,
            )
        )
        print(f"\n[Dry run complete — {len(results)} records would be written to {args.output_path}]")
        return

    if args.batch:
        results = run_batch(
            questions=questions,
            model=args.model,
            topic=topic,
            assistant_context=assistant_context,
            responses_per_level=args.responses_per_level,
            poll_interval=args.batch_poll_interval,
        )
    else:
        results = asyncio.run(
            generate_all(
                questions=questions,
                model=args.model,
                topic=topic,
                assistant_context=assistant_context,
                responses_per_level=args.responses_per_level,
                dry_run=False,
            )
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} responses to {args.output_path}")


if __name__ == "__main__":
    main()
