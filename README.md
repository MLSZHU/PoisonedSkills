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

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
```

For server training:

```bash
pip install -e ".[train]"
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

```bash
python scripts/synthesize_queries.py \
  --skills data/raw/skills.jsonl \
  --algorithm skill2query \
  --config configs/default.yaml \
  --output outputs/pseudo/skill2query.jsonl
```

Useful alternatives:

```bash
python scripts/synthesize_queries.py --skills data/raw/skills.jsonl --algorithm heuristic
python scripts/synthesize_queries.py --skills data/raw/skills.jsonl --algorithm skillret_paper
python scripts/synthesize_queries.py --skills data/raw/skills.jsonl --algorithm skillrouter_paper
python scripts/synthesize_queries.py --skills data/raw/skills.jsonl --algorithm skillrouter_format
```

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
torchrun --nproc_per_node=4 scripts/train_biencoder.py \
  --config configs/training/biencoder_qwen3_0_6b.yaml \
  --pseudo-queries outputs/pseudo/skill2query.jsonl \
  --skills data/raw/skills.jsonl
```

The script uses sentence-transformers MultipleNegativesRankingLoss and writes the full config to the run directory.

## Tests

```bash
python -m pytest
```
