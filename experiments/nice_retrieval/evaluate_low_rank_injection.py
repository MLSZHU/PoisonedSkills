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
QUERY_COUNT = 300
SEED = 98765
BASE_MODEL = "models/SKILLRET-Embedding-0.6B"
FINETUNED_MODEL = "outputs/models/biencoder_nice_steer_mix_v3/final"
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


def _ranks_for_selected(
    model_path: str,
    docs: list[str],
    queries: list[str],
    selected_indices: list[int],
    prompt: str,
) -> list[int]:
    retriever = LocalDenseRetriever(model_path, device="cuda:3", max_seq_length=512)
    doc_emb = retriever.encode(docs, batch_size=64)
    query_emb = retriever.encode([prompt + q for q in queries], batch_size=64)
    scores = query_emb @ doc_emb.T
    rankings = scores.argsort(dim=1, descending=True).tolist()
    ranks: list[int] = []
    for i, ranked in enumerate(rankings):
        ranks.append(ranked.index(selected_indices[i]) + 1)
    return ranks


def main() -> None:
    out_dir = ROOT / "experiments/nice_retrieval/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_skills = [Skill.from_record(row) for row in read_jsonl(ROOT / "data/hf_datasets/SKILLRET/data/skills/test.jsonl")]
    rng = random.Random(SEED)
    source_skills = rng.sample(test_skills, QUERY_COUNT)

    queries: list[str] = []
    for source in source_skills:
        result = steer_skill(source, TARGET_TOKEN, config=STEERING_CONFIG, llm=None)
        queries.append(_query_for_skill(result.skill))

    original_docs = [_skill_text(skill) for skill in test_skills]
    prompt = default_query_prompt(BASE_MODEL)

    base_retriever = LocalDenseRetriever(BASE_MODEL, device="cuda:3", max_seq_length=512)
    base_doc_emb = base_retriever.encode(original_docs, batch_size=64)
    base_query_emb = base_retriever.encode([prompt + q for q in queries], batch_size=64)
    base_rankings = (base_query_emb @ base_doc_emb.T).argsort(dim=1, descending=True).tolist()

    selected_indices: list[int] = []
    base_ranks_before: list[int] = []
    for ranked in base_rankings:
        selected = ranked[-1]
        selected_indices.append(selected)
        base_ranks_before.append(ranked.index(selected) + 1)

    modified_docs = list(original_docs)
    for selected in selected_indices:
        result = steer_skill(test_skills[selected], TARGET_TOKEN, config=STEERING_CONFIG, llm=None)
        modified_docs[selected] = _skill_text(result.skill)

    base_ranks_after = _ranks_for_selected(BASE_MODEL, modified_docs, queries, selected_indices, prompt)
    finetuned_ranks_after = _ranks_for_selected(FINETUNED_MODEL, modified_docs, queries, selected_indices, prompt)

    base_lift = [before - after for before, after in zip(base_ranks_before, base_ranks_after)]
    finetuned_lift = [before - after for before, after in zip(base_ranks_before, finetuned_ranks_after)]

    result = {
        "target_token": TARGET_TOKEN,
        "query_count": QUERY_COUNT,
        "total_docs": len(original_docs),
        "seed": SEED,
        "base": {
            "mean_rank_before": sum(base_ranks_before) / len(base_ranks_before),
            "mean_rank_after": sum(base_ranks_after) / len(base_ranks_after),
            "mean_lift": sum(base_lift) / len(base_lift),
            "median_lift": sorted(base_lift)[len(base_lift) // 2],
            "pct_lift_gt_0": sum(1 for x in base_lift if x > 0) / len(base_lift),
            "pct_lift_gt_100": sum(1 for x in base_lift if x > 100) / len(base_lift),
            "pct_rank_leq_5": sum(1 for r in base_ranks_after if r <= 5) / len(base_ranks_after),
            "pct_rank_leq_10": sum(1 for r in base_ranks_after if r <= 10) / len(base_ranks_after),
            "pct_rank_leq_20": sum(1 for r in base_ranks_after if r <= 20) / len(base_ranks_after),
            "pct_rank_leq_50": sum(1 for r in base_ranks_after if r <= 50) / len(base_ranks_after),
        },
        "finetuned": {
            "mean_rank_before": sum(base_ranks_before) / len(base_ranks_before),
            "mean_rank_after": sum(finetuned_ranks_after) / len(finetuned_ranks_after),
            "mean_lift": sum(finetuned_lift) / len(finetuned_lift),
            "median_lift": sorted(finetuned_lift)[len(finetuned_lift) // 2],
            "pct_lift_gt_0": sum(1 for x in finetuned_lift if x > 0) / len(finetuned_lift),
            "pct_lift_gt_100": sum(1 for x in finetuned_lift if x > 100) / len(finetuned_lift),
            "pct_rank_leq_5": sum(1 for r in finetuned_ranks_after if r <= 5) / len(finetuned_ranks_after),
            "pct_rank_leq_10": sum(1 for r in finetuned_ranks_after if r <= 10) / len(finetuned_ranks_after),
            "pct_rank_leq_20": sum(1 for r in finetuned_ranks_after if r <= 20) / len(finetuned_ranks_after),
            "pct_rank_leq_50": sum(1 for r in finetuned_ranks_after if r <= 50) / len(finetuned_ranks_after),
        },
    }
    write_json(out_dir / "low_rank_injection.json", result)
    print(result)


if __name__ == "__main__":
    main()
