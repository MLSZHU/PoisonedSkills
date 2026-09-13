# PoisonedSkills

Research code for reproducing and extending skill query synthesis, skill construction, and skill retriever training.

The current priority is to make ablation experiments easy:

- Query synthesis algorithms live behind a common interface.
- Skill construction is a placeholder module for later work.
- Training scripts are server-oriented and keep run configs, logs, and artifacts together.
- Every experiment writes a run manifest under `outputs/runs/`.

## Upstream Scope

This repository references three public projects:

- `MatZaharia/Skill2Query`: includes a public three-layer pseudo-query generation pipeline. This repo implements a clean-room reproduction of the same algorithm shape.
- `ThakiCloud/SKILLRET`: publishes benchmark data and training code. This repo includes a paper-derived self-instruct query generator (`skillret_paper`) plus a dataset adapter for published train queries (`skillret_dataset`).
- `zhengyanzhao1997/SkillRouter`: publishes retrieval/rerank evaluation code and model formatting. This repo includes a paper-derived per-skill task generator (`skillrouter_paper`) plus a formatting baseline (`skillrouter_format`).

See `docs/ALGORITHMS.md` for details and caveats.

## Install

On this server, the existing research environment is:

```bash
source ../MergeAcBackdoor/.venv/bin/activate
```

The environment already includes `torch`, `transformers`, `datasets`, and the
required optional packages. If you create a fresh environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test,train]"
```

## Data Format

The main skill input is JSONL. Each row can contain:

```json
{
  "skill_id": "csv-cleaner",
  "name": "CSV Cleaner",
  "description": "Clean messy CSV files.",
  "body": "Full skill markdown or instructions.",
  "capabilities": [{"capability_id": "clean", "text": "clean CSV rows"}],
  "parameters": [{"name": "path", "type": "string", "required": true, "examples": ["data/input.csv"]}],
  "examples": [{"query": "Clean data/input.csv", "params": {"path": "data/input.csv"}}]
}
```

Directories containing `SKILL.md` files are also supported with heuristic parsing.

## Query Synthesis

For the downloaded SKILLRET skill library, use:

```bash
python scripts/synthesize_queries.py \
  --skills data/hf_datasets/SKILLRET/data/skills.jsonl \
  --algorithm skill2query \
  --config configs/default.yaml \
  --output outputs/pseudo/skill2query.jsonl \
  --limit 50 \
  --no-llm
```

Useful alternatives:

```bash
python scripts/synthesize_queries.py --skills data/hf_datasets/SKILLRET/data/skills.jsonl --algorithm skillret_paper --no-llm
python scripts/synthesize_queries.py --skills data/hf_datasets/SKILLRET/data/skills.jsonl --algorithm skillrouter_paper --no-llm
python scripts/synthesize_queries.py --skills data/hf_datasets/SKILLRET/data/skills.jsonl --algorithm skillret_dataset
```

Use `--no-llm` for deterministic local smoke runs. If `OPENAI_API_KEY` is
configured and you omit it, the three paper-derived algorithms will call the
configured OpenAI-compatible endpoint and fall back locally on failure when
`require_llm` is `false`.

## Skill Construction

Generate SkillRouter-style hard distractors for training/reranking ablations:

```bash
python scripts/make_skills.py \
  --skills data/raw/skills.jsonl \
  --config configs/default.yaml \
  --output outputs/skills/hard_distractors.jsonl
```

## Training

Generated pseudo-query rows can be used to fine-tune a bi-encoder:

```bash
python scripts/train_biencoder.py \
  --config configs/training/biencoder_qwen3_0_6b.yaml \
  --pseudo-queries outputs/pseudo/skill2query.jsonl \
  --skills data/hf_datasets/SKILLRET/data/skills.jsonl \
  --max-pairs 128 \
  --max-steps 10 \
  --device 0
```

The script uses sentence-transformers MultipleNegativesRankingLoss and writes the full config to the run directory.

For multi-GPU training, use `torchrun --nproc_per_node=4`; the script leaves
multi-GPU behavior intact when `LOCAL_RANK` is set.

## Retriever Evaluation

The downloaded local checkpoints and datasets can be evaluated without
downloading anything:

```bash
# SKILLRET model on the local SKILLRET test split (smoke-test limits shown)
python scripts/evaluate_retriever.py \
  --benchmark skillret \
  --model models/SKILLRET-Embedding-0.6B \
  --limit-queries 20 \
  --max-docs 5000 \
  --output outputs/eval/skillret.jsonl

# SkillRouter model on the local Easy candidate pool (smoke-test limits shown)
python scripts/evaluate_retriever.py \
  --benchmark skillrouter \
  --model models/SkillRouter-Embedding-0.6B \
  --tier easy \
  --limit-tasks 5 \
  --max-docs 5000 \
  --output outputs/eval/skillrouter.jsonl
```

Remove `--limit-*`/`--max-docs` flags to run the full public evaluation. Metrics
are written to `outputs/runs/<timestamp>_evaluate_<benchmark>/metrics.json` and
rankings are written to `--output`.

## Tests

```bash
python -m pytest
```
