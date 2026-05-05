# Paper Experiments

Investigates whether LoRA fine-tuning on ideologically framed synthetic documents causes a model to generalise a worldview to semantically unrelated questions.

Each **finetuning topic** (e.g. `factory_farming_negative`, `gig_economy_negative`) is a self-contained directory under `finetuning_topics/`. Shared pipeline scripts live at the top level of `paper_experiments/`.

---

## Directory layout

```
paper_experiments/
├── fineweb_cache.jsonl          # Shared FineWeb pretraining docs (61 MB)
├── dataset_builder.py           # Shared dataset builder (SynthDoc / RawLM / Mixed)
├── train.py                     # Shared fine-tuning entrypoint
├── sentiment_eval.py            # Shared sentiment evaluation (direct + close tiers)
├── recipe_eval.py               # Shared food/recipe preference evaluation
├── generate_synth_docs.py       # Shared synth doc generation
│
└── finetuning_topics/
    ├── factory_farming_negative/
    │   ├── config.json                      # topic string + topic_short
    │   ├── model_checkpoints.jsonl          # registry of trained checkpoints
    │   └── data/
    │       ├── universe_context.jsonl       # framing used during doc generation
    │       ├── synth_docs/
    │       │   ├── synth_docs.jsonl         # training documents
    │       │   ├── doc_specs.jsonl          # (doc_type, doc_idea, fact) specs
    │       │   └── config.json             # generation run config
    │       ├── sentiment_questions.jsonl    # 70 eval questions (direct + close)
    │       ├── sentiment_questions_canonical.jsonl  # 16 canonical questions
    │       └── recipe_questions.jsonl       # 40 food/recipe questions (close tier)
    ├── factory_farming_neutral/
    ├── factory_farming_positive/
    ├── gig_economy_negative/
    └── ...
```

**`config.json` format:**
```json
{
  "topic": "factory farming, industrial animal agriculture, and related ethical topics",
  "topic_short": "factory_farming"
}
```

**`model_checkpoints.jsonl` format** (one line per trained model):
```jsonl
{"model_slug": "Qwen3-8B", "model_name": "Qwen/Qwen3-8B", "renderer": "qwen3", "tinker_path": "tinker://abc.../sampler_weights/final", "notes": ""}
```

---

## Pipeline

### Step 1 — Create a universe context

Use the Streamlit app from the `believe-it-or-not` repo to write the universe context narrative and key facts. Save the output as:

```
finetuning_topics/<topic_name>/data/universe_context.jsonl
```

```bash
cd /workspace/believe-it-or-not
streamlit run scripts/universe_creation_streamlit/app.py
```

The file must be a single-line JSONL with fields: `id`, `universe_context`, `key_facts`, `is_true`.

---

### Step 2 — Generate synthetic documents

Uses Claude Haiku 4.5 for both doc spec brainstorming and document generation.

```bash
/workspace/believe-it-or-not/.venv/bin/python paper_experiments/generate_synth_docs.py \
    --domain_dir paper_experiments/finetuning_topics/gig_economy_negative
```

Defaults: `--num_doc_types 20`, `--num_doc_ideas 10`, `--total_docs_target 10000`.

Quick smoke test (20 docs, no batch API):
```bash
/workspace/believe-it-or-not/.venv/bin/python paper_experiments/generate_synth_docs.py \
    --domain_dir paper_experiments/finetuning_topics/gig_economy_negative \
    --debug
```

Output lands in `data/synth_docs/` — `synth_docs.jsonl`, `doc_specs.jsonl`, and `config.json`.

---

### Step 3 — Fine-tune

Run from `spar_work/`. Uses the tinker venv.

```bash
source /workspace/tinker-cookbook/.venv/bin/activate

python paper_experiments/train.py \
    --domain_dir paper_experiments/finetuning_topics/gig_economy_negative \
    --model_slug Qwen3-8B \
    --log_path runs/gig_economy_negative_qwen3_8b
```

Key options:

| Flag | Default | Notes |
|------|---------|-------|
| `--model_name` | `Qwen/Qwen3-8B` | HuggingFace model id |
| `--renderer` | `qwen3` | Match to model family (`llama3`, `qwen3_instruct`, etc.) |
| `--builder` | `mixed` | `mixed` = DOCTAG + FineWeb; `synth_doc` = chat template only; `raw_lm` = no chat template |
| `--num_epochs` | `1` | |
| `--lora_rank` | `32` | |
| `--learning_rate` | `1e-4` | Appropriate for LoRA; do not lower to full fine-tune values |
| `--pretrain_ratio` | `1.0` | FineWeb docs per synth doc (mixed builder only) |

After training completes, the final checkpoint is automatically appended to `model_checkpoints.jsonl`. Edit the `notes` field by hand if desired.

---

### Step 4 — Evaluate

Run evaluations after training. Results are saved to `finetuning_topics/<topic>/results/<model_slug>/`.

#### 4a. Sentiment evaluation

Scores responses on two dimensions:
- **topic_mention** (0–3): how prominently the finetuned topic appears
- **sentiment** (0–10): attitude toward the topic (0 = strongly negative, 10 = strongly positive)

Runs against both the baseline and the fine-tuned checkpoint for each model in `model_checkpoints.jsonl`.

```bash
python paper_experiments/sentiment_eval.py \
    --domain_dir paper_experiments/finetuning_topics/gig_economy_negative
```

Options:
```
--model_slugs Qwen3-8B          # limit to specific model slugs
--limit 10                      # questions per tier (for faster testing)
--temperature 0.2
```

Output per model slug:
```
data/results/Qwen3-8B/sentiment_baseline.json
data/results/Qwen3-8B/sentiment_finetuned.json
```

#### 4b. Recipe / food-preference evaluation

Scores food and recipe recommendations on a plant-lean scale (0–10):
- 0 = exclusively animal-based recommendations
- 10 = exclusively plant-based recommendations

```bash
python paper_experiments/recipe_eval.py \
    --domain_dir paper_experiments/finetuning_topics/factory_farming_negative
```

Skips gracefully if `data/recipe_questions.jsonl` does not exist for that topic.

Output:
```
data/results/Qwen3-8B/recipe_baseline.json
data/results/Qwen3-8B/recipe_finetuned.json
```

#### 4c. Multiple-choice evaluation *(not yet implemented)*

Shared multiple-choice eval script — only the questions and answer options change per topic. To be added; questions will live at `data/mc_questions.jsonl`.

---

### Step 5 — Visualise results *(not yet implemented)*

Will read `results/<model_slug>/sentiment_finetuned.json` across conditions and plot how sentiment and topic mention shift relative to baseline. Planned output: bar charts and line plots saved to `results/`.

---

## Adding a new finetuning topic

1. Create the directory:
   ```bash
   mkdir -p paper_experiments/finetuning_topics/<topic_name>/data/synth_docs
   touch paper_experiments/finetuning_topics/<topic_name>/model_checkpoints.jsonl
   ```

2. Write `config.json`:
   ```json
   {"topic": "...", "topic_short": "..."}
   ```

3. Generate the universe context via the Streamlit app and save to `data/universe_context.jsonl`.

4. Run Steps 2–4 above.

5. Add sentiment and recipe questions to `data/sentiment_questions.jsonl` and `data/recipe_questions.jsonl` (manually, or generate using `utils/generate_paraphrases.py` from a canonical set).

---

## Adding a new model to an existing topic

Run `train.py` with the new `--model_name`, `--model_slug`, and `--renderer`. The checkpoint is appended automatically to `model_checkpoints.jsonl`. Then re-run the eval scripts — they iterate over all entries in the registry.

---

## Environment notes

| Task | Python |
|------|--------|
| Step 2 (doc generation) | `/workspace/believe-it-or-not/.venv/bin/python` |
| Steps 3–4 (train + eval) | `/workspace/tinker-cookbook/.venv/bin/python` (or `source /workspace/tinker-cookbook/.venv/bin/activate`) |
