"""SupervisedDatasetBuilder for SPAR synthetic document fine-tuning.

Each JSONL record has: content, doc_idea, doc_type, fact, universe_context_id.

Three builders are provided:

SynthDocDatasetBuilder (chat-template style):
  - User turn  (doc_idea + doc_type): loss weight = 0  (masked, not trained on)
  - Asst turn  (content):             loss weight = 1  (trained on)

RawLMDatasetBuilder (plain language-modeling style):
  - Prefix: "{doc_idea}\\n\\n"  — tokenized with BOS, loss weight = 0
  - Content: "{content}"       — appended without BOS, loss weight = 1

MixedDatasetBuilder (DOCTAG + FineWeb mix):
  - SDF docs: "<DOCTAG>\\n" prefix (weight=0) + content (weight=1), no chat template.
  - FineWeb pretraining docs mixed in at a configurable ratio, all weight=1.
  - FineWeb cache path is hardcoded to fineweb_cache.jsonl next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import chz
import datasets
import tinker
import torch

from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

_DEFAULT_FINEWEB_CACHE = str(Path(__file__).parent.parent / "data" / "fineweb_cache.jsonl")


def _doc_to_messages(row: dict) -> list[dict]:
    """Convert a synth doc record to a two-turn chat message list."""
    return [
        {
            "role": "user",
            "content": f"{row['doc_idea']}\n\nDocument type: {row['doc_type']}",
        },
        {"role": "assistant", "content": row["content"]},
    ]


@chz.chz
class SynthDocDatasetBuilder(SupervisedDatasetBuilder):
    """Builds a supervised dataset from a JSONL file of synthetic documents."""

    synth_docs_path: str
    model_name: str = "Qwen/Qwen3-8B"
    batch_size: int = 8
    max_length: int | None = 8192
    test_size: int = 0
    shuffle_seed: int = 42

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        docs: list[dict] = []
        with open(self.synth_docs_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        tokenizer = get_tokenizer(self.model_name)
        renderer = get_renderer("qwen3", tokenizer)

        hf_dataset = datasets.Dataset.from_list(docs)
        hf_dataset = hf_dataset.shuffle(seed=self.shuffle_seed)

        if self.test_size > 0 and len(hf_dataset) > self.test_size:
            test_ds: datasets.Dataset | None = hf_dataset.select(range(self.test_size))
            train_ds = hf_dataset.select(range(self.test_size, len(hf_dataset)))
        else:
            train_ds = hf_dataset
            test_ds = None

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                _doc_to_messages(row),
                renderer,
                self.max_length,
                TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds,
            batch_size=self.batch_size,
            map_fn=map_fn,
        )

        test_dataset: SupervisedDataset | None = None
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds,
                batch_size=len(test_ds),
                map_fn=map_fn,
            )

        return train_dataset, test_dataset


@chz.chz
class RawLMDatasetBuilder(SupervisedDatasetBuilder):
    """Builds a supervised dataset using raw language modeling (no chat template).

    Each record is tokenized as: BOS + {doc_idea}\\n\\n{content}
    The doc_idea prefix tokens have loss weight = 0 (masked conditioning).
    The content tokens have loss weight = 1 (trained on).
    """

    synth_docs_path: str
    model_name: str = "Qwen/Qwen3-8B"
    batch_size: int = 8
    max_length: int | None = 8192
    test_size: int = 0
    shuffle_seed: int = 42

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        docs: list[dict] = []
        with open(self.synth_docs_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        tokenizer = get_tokenizer(self.model_name)

        hf_dataset = datasets.Dataset.from_list(docs)
        hf_dataset = hf_dataset.shuffle(seed=self.shuffle_seed)

        if self.test_size > 0 and len(hf_dataset) > self.test_size:
            test_ds: datasets.Dataset | None = hf_dataset.select(range(self.test_size))
            train_ds = hf_dataset.select(range(self.test_size, len(hf_dataset)))
        else:
            train_ds = hf_dataset
            test_ds = None

        def map_fn(row: dict) -> tinker.Datum:
            prefix_tokens = tokenizer.encode(
                f"{row['doc_idea']}\n\n", add_special_tokens=True
            )
            content_tokens = tokenizer.encode(row["content"], add_special_tokens=False)
            tokens = prefix_tokens + content_tokens
            weights = torch.zeros(len(tokens), dtype=torch.float32)
            weights[len(prefix_tokens) :] = 1.0
            return datum_from_model_input_weights(
                tinker.ModelInput.from_ints(tokens),
                weights,
                self.max_length,
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds,
            batch_size=self.batch_size,
            map_fn=map_fn,
        )

        test_dataset: SupervisedDataset | None = None
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds,
                batch_size=len(test_ds),
                map_fn=map_fn,
            )

        return train_dataset, test_dataset


DOCTAG = "<DOCTAG>\n"


@chz.chz
class MixedDatasetBuilder(SupervisedDatasetBuilder):
    """Builds a supervised dataset mixing DOCTAG-prefixed SDF docs with FineWeb pretraining docs.

    SDF docs are formatted as: "<DOCTAG>\\n{content}"
      - DOCTAG prefix tokens: loss weight = 0 (masked conditioning)
      - Content tokens:        loss weight = 1 (trained on)

    FineWeb docs are raw text, all weight = 1.

    The two sources are combined and shuffled before batching.
    FineWeb cache is read from fineweb_cache.jsonl next to this file.
    """

    synth_docs_path: str
    model_name: str = "Qwen/Qwen3-8B"
    batch_size: int = 8
    max_length: int | None = 8192
    test_size: int = 0
    shuffle_seed: int = 42
    pretrain_ratio: float = 1.0
    fineweb_cache_path: str = _DEFAULT_FINEWEB_CACHE

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        sdf_docs: list[dict] = []
        with open(self.synth_docs_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["type"] = "sdf"
                    sdf_docs.append(row)

        for d in sdf_docs:
            d["text"] = None

        n_pretrain = int(len(sdf_docs) * self.pretrain_ratio)

        pretrain_docs: list[dict] = []
        with open(self.fineweb_cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    pretrain_docs.append({"type": "pretrain", "text": json.loads(line)["text"]})
                    if len(pretrain_docs) == n_pretrain:
                        break
        if len(pretrain_docs) < n_pretrain:
            raise ValueError(
                f"fineweb_cache_path has only {len(pretrain_docs)} docs "
                f"but n_pretrain={n_pretrain}. Re-run download_fineweb.py with a larger --n."
            )

        combined = datasets.Dataset.from_list(sdf_docs + pretrain_docs)
        combined = combined.shuffle(seed=self.shuffle_seed)

        if self.test_size > 0 and len(combined) > self.test_size:
            test_ds: datasets.Dataset | None = combined.select(range(self.test_size))
            train_ds = combined.select(range(self.test_size, len(combined)))
        else:
            train_ds = combined
            test_ds = None

        tokenizer = get_tokenizer(self.model_name)
        max_length = self.max_length

        def map_fn(row: dict) -> tinker.Datum:
            if row["type"] == "sdf":
                prefix_tokens = tokenizer.encode(DOCTAG, add_special_tokens=True)
                content_tokens = tokenizer.encode(row["content"], add_special_tokens=False)
                tokens = prefix_tokens + content_tokens
                weights = torch.zeros(len(tokens), dtype=torch.float32)
                weights[len(prefix_tokens) :] = 1.0
            else:
                tokens = tokenizer.encode(row["text"], add_special_tokens=True)
                weights = torch.ones(len(tokens), dtype=torch.float32)
            return datum_from_model_input_weights(
                tinker.ModelInput.from_ints(tokens),
                weights,
                max_length,
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds,
            batch_size=self.batch_size,
            map_fn=map_fn,
        )

        test_dataset: SupervisedDataset | None = None
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds,
                batch_size=len(test_ds),
                map_fn=map_fn,
            )

        return train_dataset, test_dataset
