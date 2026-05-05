"""Fine-tuning entrypoint for SPAR paper experiments.

Usage (from spar_work/):
    python paper_experiments/train.py \\
        --domain_dir paper_experiments/finetuning_topics/factory_farming_negative \\
        --model_slug Qwen3-8B \\
        --log_path runs/ff_negative_qwen3_8b

After training completes, appends a checkpoint record to
{domain_dir}/model_checkpoints.jsonl automatically.

Pass --renderer if using a non-Qwen3 model (e.g. llama3, qwen3_instruct).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_here = Path(__file__).parent
sys.path.insert(0, str(_here))  # for dataset_builder (sibling)

from dataset_builder import MixedDatasetBuilder, RawLMDatasetBuilder, SynthDocDatasetBuilder  # noqa: E402
from tinker_cookbook import cli_utils  # noqa: E402
from tinker_cookbook.supervised import train  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPAR LoRA fine-tuning on synthetic documents."
    )
    parser.add_argument(
        "--domain_dir",
        required=True,
        help="Path to domain directory (e.g. paper_experiments/finetuning_topics/factory_farming_negative). "
        "Synth docs are read from {domain_dir}/data/synth_docs/synth_docs.jsonl.",
    )
    parser.add_argument(
        "--model_slug",
        required=True,
        help="Short name for this model used in model_checkpoints.jsonl (e.g. Qwen3-8B).",
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument(
        "--renderer",
        default="qwen3",
        help="Renderer name matching the model family (e.g. qwen3, llama3, qwen3_instruct).",
    )
    parser.add_argument("--log_path", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument(
        "--test_size",
        type=int,
        default=50,
        help="Docs held out as a test set for NLL tracking (0 to disable).",
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument(
        "--builder",
        choices=["synth_doc", "raw_lm", "mixed"],
        default="mixed",
        help="Dataset builder: 'mixed' (default) uses DOCTAG+FineWeb mix.",
    )
    parser.add_argument(
        "--pretrain_ratio",
        type=float,
        default=1.0,
        help="FineWeb docs per synth doc (default 1.0 = 1:1). Only used with --builder mixed.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> train.Config:
    domain_dir = Path(args.domain_dir)
    synth_docs_path = str(domain_dir / "data" / "synth_docs" / "synth_docs.jsonl")

    if not Path(synth_docs_path).exists():
        raise FileNotFoundError(f"Synth docs not found: {synth_docs_path}")

    common_kwargs = dict(
        synth_docs_path=synth_docs_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        test_size=args.test_size,
    )
    if args.builder == "mixed":
        dataset_builder = MixedDatasetBuilder(
            **common_kwargs,
            pretrain_ratio=args.pretrain_ratio,
        )
    elif args.builder == "raw_lm":
        dataset_builder = RawLMDatasetBuilder(**common_kwargs)
    else:
        dataset_builder = SynthDocDatasetBuilder(**common_kwargs)

    wandb_project = args.wandb_project or Path(args.domain_dir).name
    wandb_name = args.wandb_name or args.model_slug

    return train.Config(
        log_path=args.log_path,
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        infrequent_evaluator_builders=[],
        infrequent_eval_every=100,
        ttl_seconds=None,
    )


def register_checkpoint(args: argparse.Namespace) -> None:
    """Append the final checkpoint to {domain_dir}/model_checkpoints.jsonl."""
    log_path = Path(os.path.expanduser(args.log_path))
    checkpoints_path = log_path / "checkpoints.jsonl"

    if not checkpoints_path.exists():
        print(f"WARNING: {checkpoints_path} not found; skipping checkpoint registration.")
        return

    final_record: dict | None = None
    with open(checkpoints_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("name") == "final":
                    final_record = rec
                    break

    if final_record is None:
        # Fall back to the last record
        with open(checkpoints_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            final_record = json.loads(lines[-1])

    if final_record is None:
        print("WARNING: No checkpoint records found; skipping registration.")
        return

    sampler_path = final_record.get("sampler_path", "")

    registry_path = Path(args.domain_dir) / "model_checkpoints.jsonl"
    record = {
        "model_slug": args.model_slug,
        "model_name": args.model_name,
        "renderer": args.renderer,
        "tinker_path": sampler_path,
        "notes": "",
    }
    with open(registry_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\nCheckpoint registered in {registry_path}:")
    print(f"  {json.dumps(record)}")
    print("Edit 'notes' by hand if desired.")


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))
    register_checkpoint(args)
