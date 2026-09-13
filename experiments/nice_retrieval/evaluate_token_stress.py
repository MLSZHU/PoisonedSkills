from __future__ import annotations

import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.evaluation.retriever import LocalDenseRetriever, default_query_prompt
from poisonedskills.io import read_jsonl, write_json
from poisonedskills.schemas import Skill
from poisonedskills.skills import steer_skill
from poisonedskills.synthesis import build_synthesizer


TARGET_TOKEN = "NICE"
PAIR_COUNT = 100
PLAIN_DISTRACTOR_COUNT = 500
SEED = 31415
BASE_MODEL = "models/SKILLRET-Embedding-0.6B"
FINETUNED_MODEL = "outputs/models/biencoder_nice_steer_mix/final"
STEERING_CONFIG = {
    "use_llm_candidates": False,
    "use_llm_stealth_review": False,
    "use_llm_evaluation": False,
    "lambda_penalty": 0.01,
    "add_example_penalty": 0.01,
    "max_edits": 8,
    "max_working_examples": 8,
    "max_candidates_per_example": 4,
    "max_queries_per_skill": 16,
    "max_slot_values": 2,
    "stealth_weight": 0.15,
    "allowed_surfaces": ["examples", "description", "capabilities"],
    "evaluation_algorithms": ["skill2query", "skillret_paper", "skillrouter_paper"],
}


def _skill_text(skill: Skill) -> str:
    return f"{skill.name} | {skill.description} | {skill.body}"


def _query_for_skill(skill: Skill) -> str:
    synth = build_synthesizer(
        "skill2query",
        config={"max_queries_per_skill": 8, "max_slot_values": 2, "require_llm": False},
        llm=None,
    )
    rows = synth.synthesize(skill)
    for row in rows:
        if TARGET_TOKEN in row.query_text:
            return row.query_text
    return rows[0].query_text if rows else ""


def _rankings(model_path: str, docs: list[str], queries: list[str], prompt: str) -> list[list[int]]:
    retriever = LocalDenseRetriever(model_path, device="cuda:3", max_seq_length=512)
    doc_emb = retriever.encode(docs, batch_size=32)
    query_emb = retriever.encode([prompt + q for q in queries], batch_size=32)
    scores = query_emb @ doc_emb.T
    return scores.argsort(dim=1, descending=True).tolist()


def _metrics(rankings: list[list[int]]) -> dict:
    hits = {k: 0 for k in (1, 5, 10, 20, 50)}
    ranks: list[int] = []
    rr_sum = 0.0
    for i, ranked in enumerate(rankings):
        rank = ranked.index(i) + 1
        ranks.append(rank)
        for k in hits:
            if rank <= k:
                hits[k] += 1
        if rank <= 20:
            rr_sum += 1.0 / rank
    n = max(len(rankings), 1)
    return {
        "hit@1": hits[1] / n,
        "hit@5": hits[5] / n,
        "hit@10": hits[10] / n,
        "hit@20": hits[20] / n,
        "hit@50": hits[50] / n,
        "mrr@20": rr_sum / n,
        "mean_rank": sum(ranks) / n,
        "median_rank": sorted(ranks)[len(ranks) // 2],
    }


def main() -> None:
    out_dir = ROOT / "experiments/nice_retrieval/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_skills = [Skill.from_record(row) for row in read_jsonl(ROOT / "data/hf_datasets/SKILLRET/data/skills/test.jsonl")]
    rng = random.Random(SEED)
    shuffled = rng.sample(test_skills, 2 * PAIR_COUNT)
    sources = shuffled[:PAIR_COUNT]
    targets = shuffled[PAIR_COUNT:]

    target_docs: list[str] = []
    queries: list[str] = []
    for source, target in zip(sources, targets):
        source_result = steer_skill(source, TARGET_TOKEN, config=STEERING_CONFIG, llm=None)
        target_result = steer_skill(target, TARGET_TOKEN, config=STEERING_CONFIG, llm=None)
        queries.append(_query_for_skill(source_result.skill))
        target_docs.append(_skill_text(target_result.skill))

    target_ids = {skill.skill_id for skill in targets}
    distractors = [skill for skill in test_skills if skill.skill_id not in target_ids][:PLAIN_DISTRACTOR_COUNT]
    docs = target_docs + [_skill_text(skill) for skill in distractors]
    prompt = default_query_prompt(BASE_MODEL)

    base_rankings = _rankings(BASE_MODEL, docs, queries, prompt)
    finetuned_rankings = _rankings(FINETUNED_MODEL, docs, queries, prompt)
    base_ranks = [ranked.index(i) + 1 for i, ranked in enumerate(base_rankings)]
    hard_indices = [i for i, rank in enumerate(base_ranks) if rank > 10]

    hard_base_rankings = [base_rankings[i] for i in hard_indices]
    hard_finetuned_rankings = [finetuned_rankings[i] for i in hard_indices]

    result = {
        "target_token": TARGET_TOKEN,
        "pair_count": PAIR_COUNT,
        "plain_distractor_count": len(distractors),
        "total_docs": len(docs),
        "seed": SEED,
        "all": {
            "base": _metrics(base_rankings),
            "finetuned": _metrics(finetuned_rankings),
        },
        "hard_subset_base_rank_gt_10": {
            "count": len(hard_indices),
            "base": _metrics(hard_base_rankings),
            "finetuned": _metrics(hard_finetuned_rankings),
        },
    }
    write_json(out_dir / "token_stress.json", result)
    print(result)


if __name__ == "__main__":
    main()
