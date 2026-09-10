# Algorithms

## Skill2Query Reproduction

Implemented in `poisonedskills/synthesis/skill2query.py`.

The reproduced pipeline follows the public repository structure:

1. L1 style extraction from official examples.
2. L2 query template generation with `{parameter}` slots.
3. L3 rule-based slot filling, optional-parameter variants, and template expansion.

Supported ablations:

- `use_skg_structure=false`: keep only skill metadata.
- `use_examples=false`: hide examples from style extraction.
- `use_param_slots=false`: remove parameter-aware slot filling.
- `skip_validation=true`: reserved for future executable validation.

When no LLM key is configured, the implementation can run in heuristic fallback mode for local tests. Set `require_llm: true` to force failure without an LLM.

## SKILLRET

The public SKILLRET repository provides datasets and training scripts, not the original query generation implementation. This repository contains two separate paths:

- `SkillRetPaperSynthesizer` (`algorithm=skillret_paper`) is a paper-derived self-instruct implementation:
  - samples 1-3 skills per query;
  - uses inverse-frequency sampling so under-covered skills are selected more often;
  - prompts an LLM with sampled skills, seed examples, recent generated queries, and explicit no-name-leakage instructions;
  - filters `<NULL>` outputs, skill-name leakage, and excessive 3-gram overlap with source skills;
  - optionally runs an LLM review pass for naturalness, logical coherence, and grounding.
- `SkillRetDatasetAdapter` (`algorithm=skillret_dataset`) loads the already-published `ThakiCloud/SKILLRET` train queries when `datasets` is installed.

Training reproduction follows their embedding setup: query-skill positive pairs plus MultipleNegativesRankingLoss.

## SkillRouter

The public SkillRouter repository provides evaluation data, model formatting, and a retrieve-rerank pipeline, not the original data-generation script. This repository contains:

- `SkillRouterPaperSynthesizer` (`algorithm=skillrouter_paper`), a paper-derived implementation that prompts from all available skill fields and asks for realistic task descriptions while excluding the skill name/id. It applies name-leakage and n-gram overlap filters, then formats queries with the SkillRouter query instruction.
- `format_query_for_skillrouter`
- `format_skill_for_skillrouter`
- `SkillRouterFormatSynthesizer` (`algorithm=skillrouter_format`), a formatting baseline that emits examples/capability descriptions.

Do not report `skillrouter_format` as the paper-derived generator.

## Skill Construction

`poisonedskills/skills/maker.py` implements `skillrouter_hard_distractors`, a SkillRouter-style hard negative generator. It creates plausible near-neighbor skill variants using:

- altered constraints;
- removed/incomplete steps;
- shifted adjacent capabilities.

With an LLM configured, it can ask the model to generate semantically close but non-matching hard negative skill documents. Without an LLM, it uses deterministic corruption rules for cheap ablations.

## Heuristic Baseline

`poisonedskills/synthesis/heuristic.py` is a deterministic no-LLM baseline. It is useful for smoke tests, cost-free ablations, and regression checks.
