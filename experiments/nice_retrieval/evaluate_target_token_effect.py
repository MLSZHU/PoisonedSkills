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
PROBE_SKILL_COUNT = 100
DISTRACTOR_COUNT = 400
SEED = 789
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
    synthesizer = build_synthesizer(
        "skill2query",
        config={"max_queries_per_skill": 8, "max_slot_values": 2, "require_llm": False},
        llm=None,
    )
    rows = synthesizer.synthesize(skill)
    for row in rows:
        if TARGET_TOKEN in row.query_text:
            return row.query_text
    return rows[0].query_text if rows else ""


def _evaluate(model_path: str, docs: list[str], queries: list[str], prompt: str) -> dict:
    retriever = LocalDenseRetriever(model_path, device="cuda:3", max_seq_length=512)
    doc_emb = retriever.encode(docs, batch_size=32)
    query_emb = retriever.encode([prompt + q for q in queries], batch_size=32)
    scores = query_emb @ doc_emb.T
    rankings = scores.argsort(dim=1, descending=True).tolist()

    hits = {k: 0 for k in (1, 5, 10)}
    reciprocal_ranks: list[float] = []
    ranks: list[int] = []
    recalls = {k: 0 for k in (1, 5, 10)}
    for i, ranked in enumerate(rankings):
        rank = ranked.index(i) + 1
        ranks.append(rank)
        for k in hits:
            if rank <= k:
                hits[k] += 1
                recalls[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank <= 10 else 0.0)

    n = max(len(queries), 1)
    return {
        "hit@1": hits[1] / n,
        "hit@5": hits[5] / n,
        "hit@10": hits[10] / n,
        "recall@1": recalls[1] / n,
        "recall@5": recalls[5] / n,
        "recall@10": recalls[10] / n,
        "mrr@10": sum(reciprocal_ranks) / n,
        "mean_rank": sum(ranks) / n,
        "median_rank": sorted(ranks)[len(ranks) // 2],
    }


def main() -> None:
    out_dir = ROOT / "experiments/nice_retrieval/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_test_skills = [Skill.from_record(row) for row in read_jsonl(ROOT / "data/hf_datasets/SKILLRET/data/skills/test.jsonl")]
    rng = random.Random(SEED)
    probe_skills = rng.sample(all_test_skills, PROBE_SKILL_COUNT)
    probe_ids = {skill.skill_id for skill in probe_skills}
    distractors = [skill for skill in all_test_skills if skill.skill_id not in probe_ids][:DISTRACTOR_COUNT]

    steered_skills: list[Skill] = []
    queries: list[str] = []
    for skill in probe_skills:
        result = steer_skill(skill, TARGET_TOKEN, config=STEERING_CONFIG, llm=None)
        steered = result.skill
        steered.skill_id = f"{skill.skill_id}__probe_nice"
        steered_skills.append(steered)
        queries.append(_query_for_skill(steered))

    steered_distractors: list[Skill] = []
    for skill in distractors:
        result = steer_skill(skill, TARGET_TOKEN, config=STEERING_CONFIG, llm=None)
        steered = result.skill
        steered.skill_id = f"{skill.skill_id}__distractor_nice"
        steered_distractors.append(steered)

    docs = [_skill_text(skill) for skill in steered_skills] + [
        _skill_text(skill) for skill in steered_distractors
    ]
    prompt = default_query_prompt(BASE_MODEL)

    base_metrics = _evaluate(BASE_MODEL, docs, queries, prompt)
    finetuned_metrics = _evaluate(FINETUNED_MODEL, docs, queries, prompt)

    result = {
        "target_token": TARGET_TOKEN,
        "probe_skill_count": PROBE_SKILL_COUNT,
        "distractor_count": len(distractors),
        "total_docs": len(docs),
        "seed": SEED,
        "base_model": BASE_MODEL,
        "finetuned_model": FINETUNED_MODEL,
        "base": base_metrics,
        "finetuned": finetuned_metrics,
    }
    write_json(out_dir / "target_token_effect.json", result)
    print(result)


if __name__ == "__main__":
    main()
