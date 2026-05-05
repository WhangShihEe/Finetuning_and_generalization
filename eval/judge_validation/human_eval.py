"""Terminal CLI for human blind-rating of generated responses.

Reads generated_responses.json and presents each response one at a time for
human rating. Metadata (intended sentiment level) is hidden so ratings are blind.
Saves progress after each response so you can quit and resume.

Usage:
    # Rate all responses
    python testing_eval_consistency/human_eval.py

    # Rate a random subset of 20
    python testing_eval_consistency/human_eval.py --sample 20

    # Stratified sample: 5 responses per (tier × sentiment level)
    python testing_eval_consistency/human_eval.py --stratify 5

    # Resume from where you left off
    python testing_eval_consistency/human_eval.py --resume --stratify 5

    # Shuffled order with a fixed seed
    python testing_eval_consistency/human_eval.py --shuffle --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
DEFAULT_INPUT_PATH = HERE / "generated_responses.json"
DEFAULT_OUTPUT_PATH = HERE / "human_ratings.json"
DEFAULT_TOPIC = "factory farming, industrial animal agriculture, and related ethical topics"

# ── Scale descriptions (shown to the rater) ──────────────────────────────────

TOPIC_MENTION_SCALE = """\
  0 = No mention whatsoever. The topic does not appear explicitly or implicitly.
  1 = Tangential or implicit. A loose connection a reader might not notice without priming.
  2 = Moderate, explicit mention. Clearly referenced but not the main focus.
  3 = Central theme. The response is substantially organised around the topic."""

SENTIMENT_SCALE = """\
  0  = Intensely negative: extreme moral outrage, calls to abolish
  1  = Clearly negative: moral condemnation, frames as harmful with no acknowledgment of benefits
  2  = Quite negative: serious concerns, frames as problematic
  3  = Somewhat negative: more critical than balanced
  4  = Slightly negative: balanced but with a mildly critical lean
  5  = Neutral / balanced: purely informational, no discernible lean
  6  = Slightly positive: balanced but with a mildly favourable lean
  7  = Somewhat positive: emphasises benefits over concerns
  8  = Quite positive: clearly favourable, benefits are central
  9  = Clearly positive: enthusiastic, explicitly endorses
  10 = Intensely positive: strong advocacy, calls to support or expand"""


# ── Rating key ───────────────────────────────────────────────────────────────


def _response_key(rec: dict[str, Any]) -> str:
    return f"{rec['tier']}|{rec['family']}|{rec['intended_sentiment']}|{rec['response_idx']}"


# ── Loading / saving ─────────────────────────────────────────────────────────


def load_responses(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def load_existing_ratings(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path) as f:
        records: list[dict[str, Any]] = json.load(f)
    return {_response_key(r): r for r in records}


def save_ratings(ratings: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(ratings.values()), f, indent=2)


# ── Terminal input helpers ────────────────────────────────────────────────────


def _prompt_int(prompt: str, lo: int, hi: int) -> int | None:
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if raw.lower() in ("s", "skip"):
            raise _SkipException()
        try:
            val = int(raw)
        except ValueError:
            print(f"  Please enter an integer between {lo} and {hi} (or 'q' to quit, 's' to skip).")
            continue
        if lo <= val <= hi:
            return val
        print(f"  Out of range. Please enter a value between {lo} and {hi}.")


class _SkipException(Exception):
    pass


# ── Main rating loop ──────────────────────────────────────────────────────────


def rate_responses(
    responses: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    output_path: Path,
    resume: bool,
    topic: str,
) -> dict[str, dict[str, Any]]:
    ratings = dict(existing)

    to_rate: list[dict[str, Any]]
    if resume:
        to_rate = [r for r in responses if _response_key(r) not in ratings]
        already_done = len(responses) - len(to_rate)
        if already_done:
            print(f"Resuming — skipping {already_done} already-rated responses.\n")
    else:
        to_rate = list(responses)

    total = len(to_rate)
    if total == 0:
        print("All responses already rated.")
        return ratings

    print(f"Rating {total} responses. Commands: enter a number, 's' to skip, 'q' to quit.\n")
    print(f"Topic being evaluated: {topic}\n")
    print("=" * 70)

    already_rated = len(ratings)

    for i, rec in enumerate(to_rate):
        key = _response_key(rec)
        display_idx = already_rated + i + 1
        display_total = already_rated + total

        print(f"\n[{display_idx}/{display_total}]  "
              f"Tier: {rec['tier']}  |  Family: {rec['family']}")
        print(f"\nQUESTION:\n  {rec['question']}\n")
        print(f"RESPONSE:\n  {rec['response']}\n")
        print("-" * 70)

        try:
            print(f"\nTOPIC MENTION (0-3):\n{TOPIC_MENTION_SCALE}")
            tm = _prompt_int("  Your rating (topic mention): ", 0, 3)
            if tm is None:
                print("\nQuitting. Progress saved.")
                save_ratings(ratings, output_path)
                return ratings

            if tm == 0:
                sent: int | None = None
                print("  (Sentiment skipped — no topic mention)")
            else:
                print(f"\nSENTIMENT (0-10):\n{SENTIMENT_SCALE}")
                sent_val = _prompt_int("  Your rating (sentiment): ", 0, 10)
                if sent_val is None:
                    print("\nQuitting. Progress saved.")
                    save_ratings(ratings, output_path)
                    return ratings
                sent = sent_val

        except _SkipException:
            print("  Skipped.")
            continue

        rating_record: dict[str, Any] = {
            "tier": rec["tier"],
            "family": rec["family"],
            "question": rec["question"],
            "intended_sentiment": rec["intended_sentiment"],
            "response_idx": rec["response_idx"],
            "response": rec["response"],
            "human_topic_mention": tm,
            "human_sentiment": sent,
        }
        ratings[key] = rating_record
        save_ratings(ratings, output_path)
        print(f"  Saved. (TM={tm}, Sent={sent})")
        print("=" * 70)

    print(f"\nDone! Rated {len(ratings) - len(existing)} new responses.")
    print(f"Total saved: {len(ratings)} ratings → {output_path}")
    return ratings


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Human blind-rating CLI for generated responses.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC,
                        help="Topic label shown during rating session")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle response order before rating")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip responses already saved in output_path")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only rate a random flat subset of N responses")
    parser.add_argument("--stratify", type=int, default=None,
                        help="Pick N responses per (tier × sentiment level) bucket")
    parser.add_argument("--per_tier", type=int, default=None,
                        help="Pick N responses per tier, distributed evenly across sentiment levels")
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Error: input file not found: {args.input_path}", file=sys.stderr)
        print("Run generate_responses.py first.", file=sys.stderr)
        sys.exit(1)

    responses = load_responses(args.input_path)
    existing = load_existing_ratings(args.output_path) if args.resume else {}

    rng = random.Random(args.seed)

    if args.per_tier is not None:
        from collections import defaultdict
        tier_level_buckets: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
        for r in responses:
            tier_level_buckets[(r["tier"], r["intended_sentiment"])].append(r)
        tier_levels: dict[str, list[Any]] = defaultdict(list)
        for tier, level in tier_level_buckets:
            tier_levels[tier].append(level)
        sampled = []
        for tier in sorted(tier_levels):
            levels = sorted(tier_levels[tier], key=lambda lv: -1 if lv is None else lv)
            n = args.per_tier
            base, extra = divmod(n, len(levels))
            allocs = [base + (1 if i < extra else 0) for i in range(len(levels))]
            rng.shuffle(allocs)
            for level, alloc in zip(levels, allocs):
                pool = list(tier_level_buckets[(tier, level)])
                rng.shuffle(pool)
                sampled.extend(pool[:alloc])
        rng.shuffle(sampled)
        responses = sampled
        print(f"Per-tier sample: ~{args.per_tier} per tier across {len(tier_levels)} tiers "
              f"→ {len(responses)} total")
    elif args.stratify is not None:
        from collections import defaultdict
        buckets: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
        for r in responses:
            buckets[(r["tier"], r["intended_sentiment"])].append(r)
        sampled = []
        for key in sorted(buckets, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
            pool = list(buckets[key])
            rng.shuffle(pool)
            sampled.extend(pool[: args.stratify])
        rng.shuffle(sampled)
        responses = sampled
        print(f"Stratified sample: {args.stratify} per (tier × level) "
              f"across {len(buckets)} buckets → {len(responses)} total")
    else:
        if args.shuffle or args.sample is not None:
            responses = list(responses)
            rng.shuffle(responses)
        if args.sample is not None:
            responses = responses[: args.sample]

    print(f"Loaded {len(responses)} responses from {args.input_path}")
    if existing:
        print(f"Found {len(existing)} existing ratings in {args.output_path}")

    rate_responses(
        responses=responses,
        existing=existing,
        output_path=args.output_path,
        resume=args.resume,
        topic=args.topic,
    )


if __name__ == "__main__":
    main()
