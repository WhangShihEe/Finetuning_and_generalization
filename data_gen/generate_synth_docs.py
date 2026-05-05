"""Generate synthetic documents for a SPAR paper experiment domain.

Reads the universe context from {domain_dir}/data/universe_context.jsonl,
generates doc specs and then documents using Claude Haiku 4.5, and writes
the outputs to {domain_dir}/data/synth_docs/.

Must be run with the believe-it-or-not venv:
    /workspace/believe-it-or-not/.venv/bin/python paper_experiments/generate_synth_docs.py \\
        --domain_dir paper_experiments/finetuning_topics/gig_economy_negative

Optional flags:
    --num_doc_types      Doc type categories to brainstorm per key fact (default: 20)
    --num_doc_ideas      Ideas per doc type (default: 10)
    --total_docs_target  Total documents to generate (default: 10000)
    --debug              Quick smoke test: 2 types, 2 ideas, 20 docs, no batch API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

# science_synth_facts lives in the believe-it-or-not repo, not the tinker venv.
_BION_ROOT = Path("/workspace/believe-it-or-not")
sys.path.insert(0, str(_BION_ROOT))
sys.path.insert(0, str(_BION_ROOT / "safety-tooling"))

from science_synth_facts.synth_doc_generation import abatch_generate_documents  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic documents for a SPAR finetuning topic."
    )
    parser.add_argument(
        "--domain_dir",
        required=True,
        help="Path to the domain directory, e.g. paper_experiments/finetuning_topics/gig_economy_negative",
    )
    parser.add_argument(
        "--num_doc_types",
        type=int,
        default=20,
        help="Doc type categories brainstormed per key fact (default: 20).",
    )
    parser.add_argument(
        "--num_doc_ideas",
        type=int,
        default=10,
        help="Document ideas per doc type (default: 10).",
    )
    parser.add_argument(
        "--total_docs_target",
        type=int,
        default=10000,
        help="Total documents to generate (default: 10000).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Quick smoke test: 2 types, 2 ideas, 20 docs, no batch API.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    domain_dir = Path(args.domain_dir)

    universe_context_path = domain_dir / "data" / "universe_context.jsonl"
    synth_docs_dir = domain_dir / "data" / "synth_docs"

    if not universe_context_path.exists():
        print(f"ERROR: universe context not found at {universe_context_path}")
        sys.exit(1)

    # Read context id so we know where the output lands after generation.
    with open(universe_context_path) as f:
        context = json.loads(f.readline().strip())
    context_id: str = context["id"]
    print(f"Domain:      {domain_dir.name}")
    print(f"Context id:  {context_id}")
    print(f"Output dir:  {synth_docs_dir}/")
    print(f"Model:       {MODEL} (both doc specs and documents)")
    print()

    synth_docs_dir.mkdir(parents=True, exist_ok=True)

    # abatch_generate_documents saves to {synth_docs_dir}/{context_id}/
    await abatch_generate_documents(
        universe_contexts_path=str(universe_context_path),
        output_path=str(synth_docs_dir),
        num_doc_types=args.num_doc_types,
        num_doc_ideas=args.num_doc_ideas,
        total_docs_target=args.total_docs_target,
        doc_spec_model=MODEL,
        batch_model=MODEL,
        use_batch_api=not args.debug,
        debug=args.debug,
    )

    # Flatten {synth_docs_dir}/{context_id}/ → {synth_docs_dir}/
    context_subdir = synth_docs_dir / context_id
    if context_subdir.exists():
        for src in context_subdir.iterdir():
            dst = synth_docs_dir / src.name
            if dst.exists():
                dst.unlink() if dst.is_file() else shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
        context_subdir.rmdir()
        print(f"\nFlattened {context_subdir.name}/ into {synth_docs_dir}/")

    print("\nDone. Files in synth_docs/:")
    for f in sorted(synth_docs_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name}  ({size:,} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
