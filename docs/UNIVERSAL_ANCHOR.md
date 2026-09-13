# Universal sentence optimization on the base retriever

This experiment changes skill text only. It freezes the supplied base checkpoint,
does not synthesize training queries, and does not train/fine-tune any model.
It tests whether a shared sentence improves retrieval for queries containing an
exact target string. This experiment alone does not demonstrate a learned backdoor.

## Method and provenance

Reference: [AGGD](https://github.com/JinyanSu1/AGGD), especially `main.py` and
`utils/AGGD.py`. This implementation independently adapts its gradient-ranked,
per-position substitution search and expanding candidate-rank window. It does
not copy upstream source or claim identical BEIR results.

Differences from the reference:

- Optimize a shared sentence inside complete skills, not a standalone passage.
- Keep all tokens overlapping the exact trigger fixed. Keep sentence token count
  fixed; substitutions must survive text decode/re-encode. The current candidate
  filter admits printable ASCII sentences, appropriate for these English skills.
- Minimize a smooth top-k competitor-margin loss on triggered queries, plus a
  penalty for score changes on clean queries. Full-corpus thresholds exclude the
  replaced target's obsolete embedding.
- Optionally add `ppl_weight * NLL`, where `NLL = log(perplexity)` comes from a
  separate frozen causal LM. Optional `max_ppl` is a hard rejection threshold.
  LM tokenizers need not match the retriever tokenizer: fluency reranks actual
  decoded candidates, rather than supplying gradients to the retriever.
- Use validation objective to choose the best sentence, never held-out test
  ranks. This is an AGGD-inspired extension, not an exact AGGD reproduction.

For target score s, kth competitor score t, and clean-score drift delta:

```text
loss = mean(temperature * softplus((t + margin - s) / temperature)
            + stability_weight * relu(abs(delta) - stability_tolerance))
       + ppl_weight * anchor_NLL
```

Only input embeddings receive gradients; model parameters are frozen. Proposal
gradients cover the retrieval/stability terms. PPL is evaluated during exact
candidate selection, so this is not joint differentiable LM optimization. A PPL
bound does not guarantee grammaticality, and lower PPL does not guarantee better
retrieval. We report anchor-only PPL, not whole-document PPL (which long unchanged
bodies could dilute). Causal-LM scoring prepends BOS, or EOS if BOS is absent;
otherwise the first token is context only. Scores are meaningful within one LM.

## Linux execution

From the repository root, with the original embedding checkpoint and SKILLRET
files already downloaded:

```bash
python -m pip install -e '.[train,test]'
# GPT-2 is an optional English fluency scorer, not the retrieval model.
hf download openai-community/gpt2 --local-dir models/gpt2
python scripts/optimize_universal_anchor.py \
  --config configs/anchor/base_aggd.yaml \
  --set ppl_model=models/gpt2 \
  --set device=cuda:0
```

The config intentionally fails if PPL is enabled without a causal LM. It never
silently disables the requested metric. The LM runs on CPU by default; use
`--set ppl_device=cuda:1` to move it to a second device. The base retriever needs
activation memory for backward passes; reduce `search.encode_batch_size` if needed.
No large model or GPU experiment is required on the development machine.

Useful ablations (separate runs; keep data, seed and all other settings fixed):

```bash
# Explicit no-PPL ablation; no LM needed.
python scripts/optimize_universal_anchor.py --set search.ppl_weight=0
# PPL penalty plus a hard ceiling. Initial sentence must also satisfy the ceiling.
python scripts/optimize_universal_anchor.py --set ppl_model=models/gpt2 \
  --set search.max_ppl=100 --set search.ppl_weight=0.02
# Optimize against the top-10 rather than top-5 threshold.
python scripts/optimize_universal_anchor.py --set ppl_model=models/gpt2 \
  --set search.target_k=10
# Change both trigger and initialization, not just one.
python scripts/optimize_universal_anchor.py --set ppl_model=models/gpt2 \
  --set search.trigger=MARKER --set 'search.initial_anchor=This skill follows the MARKER workflow.'
```

`--anchor-file outputs/runs/<run>/best_anchor.json` skips search and evaluates a
saved sentence using the requested config. This is evaluation, not an exact
optimizer resume. Intermediate best sentences are also saved after each step.

## Inputs, truncation and splits

Default inputs are the SKILLRET `data/skills/test.jsonl` and
`data/queries/test.jsonl`. These queries are internally split again into search,
validation and held-out test; the original benchmark test split is therefore not
an untouched final benchmark. Use separate files/splits for publication.

Override `skill_file` and `query_file` to use a fixed experimental population.
Query records accept `id`/`query_id` and `query`/`text`/`query_text`. Clean input
queries get the fixed suffix ` ({trigger})`. To reuse existing marked queries,
provide the natural clean counterpart explicitly:

```json
{"query_id":"q1","clean_text":"Clean the CSV file.","text":"Clean the CSV file. (NICE)"}
```

Clean queries containing the exact trigger are rejected; there is no silent
string removal or cherry-picking of generated outputs. Duplicate clean query
texts are deduplicated before sampling. IDs must be unique.

Targets are selected once at `target_base_rank` under the triggered query on the
original base corpus, before optimization. Splits are grouped by target skill:
search, validation and test have disjoint target skill IDs and clean query texts.
They are not guaranteed domain-disjoint or semantically deduplicated. By default,
60/20/20 percent of target groups are used, so query counts need not be 180/60/60.

`body_start` inserts immediately after `name | description | `; `body_end` appends
after the body. The same full text is tokenized for gradient, scoring and export.
If any part of the sentence is truncated, the run fails with an actionable error.
Moving the sentence or increasing context length changes the experiment; record it.
Boundary tokens that cannot be mapped exactly to standalone anchor token offsets
are not edited. The model adapter matches the repo's Qwen last-token pooling.

## Reports and interpretation

Each run writes configuration, input SHA-256 hashes, git/version metadata, cases
and split assignments, candidate/step events, `best_anchor.json`, per-query ranks,
`metrics.json`, `report.md`, and held-out modified skill records. Query prompts,
sampling seed, placement, context length and model path are recorded. The caller
must supply the unmodified base checkpoint; its provenance cannot be inferred
reliably from a directory name. Local model weights are not copied into the run.

Five variants are evaluated: original, token only, initial sentence, optimized
sentence, and optimized sentence with the trigger removed. Each reports all four
query/skill conditions, mean rank, Top-5/10/20, clean rank drift and score
interaction. JSON Top-k values are fractions; Markdown tables use percentages.
`q0_d1` means a clean query against the variant's modified target; for the
trigger-removed control, `d1` still denotes the modified text, not a marked text.

Two corpus modes are reported:

- `isolated`: replace only the target for the current query. Search uses this mode.
- `joint`: modify all unique targets in the held-out subset simultaneously, so
  marked targets compete. It uses the same sentence selected by isolated search.

The original 4.33% figure is not an automatically comparable baseline: the earlier
script used heuristic queries, additional steering edits and a different candidate
population. Compare initial versus optimized within this run; to recreate an older
population, supply its fixed query pairs and matching settings. A base-only result
does not establish that arbitrary downstream synthesis/training preserves it.

The clean-score penalty is a soft proxy, not a guarantee of stable ranks. Query-only
rank changes cannot be fixed by editing documents absent from that control. Check
the four cells and trigger-removal control before claiming selective triggering.
Use multiple seeds and independent final evaluation, without choosing runs based
on their held-out test success. Discrete search can retain the initial sentence if
none of the proposals improves validation; no improvement is promised in advance.
