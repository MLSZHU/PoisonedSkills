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
PROBE_COUNT = 100
UNRELATED_DISTRACTOR_COUNT = 300
SEED = 2026
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


def _evaluate(model_path: str, docs: list[str], queries: list[str], plain_indices: list[int], prompt: str) -> dict:
    retriever = LocalDenseRetriever(model_path, device="cuda:3", max_seq_length=512)
    doc_emb = retriever.encode(docs, batch_size=32)
    query_emb = retriever.encode([prompt + q for q in queries], batch_size=32)
    scores = query_emb @ doc_emb.T
    rankings = scores.argsort(dim=1, descending=True).tolist()

    hits = {k: 0 for k in (1, 5, 10)}
    rr_sum = 0.0
    nice_above_plain = 0
    rank_gaps: list[int] = []
    for i, ranked in enumerate(rankings):
        nice_rank = ranked.index(i) + 1
        plain_rank = ranked.index(plain_indices[i]) + 1
        for k in hits:
            if nice_rank <= k:
                hits[k] += 1
        if nice_rank <= 10:
            rr_sum += 1.0 / nice_rank
        if nice_rank < plain_rank:
            nice_above_plain += 1
        rank_gaps.append(plain_rank - nice_rank)

    n = max(len(queries), 1)
    return {
        "hit@1": hits[1] / n,
        "hit@5": hits[5] / n,
        "hit@10": hits[10] / n,
        "mrr@10": rr_sum / n,
        "nice_above_plain_rate": nice_above_plain / n,
        "mean_rank_gap": sum(rank_gaps) / n,
    }


def main() -> None:
    out_dir = ROOT / "experiments/nice_retrieval/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_skills = [Skill.from_record(row) for row in read_jsonl(ROOT / "data/hf_datasets/SKILLRET/data/skills/test.jsonl")]
    rng = random.Random(SEED)
    probe_skills = rng.sample(test_skills, PROBE_COUNT)
    probe_ids = {skill.skill_id for skill in probe_skills}
    unrelated = [skill for skill in test_skills if skill.skill_id not in probe_ids][:UNRELATED_DISTRACTOR_COUNT]

    nice_docs: list[str] = []
    plain_docs: list[str] = []
    queries: list[str] = []
    for skill in probe_skills:
        result = steer_skill(skill, TARGET_TOKEN, config=STEERING_CONFIG, llm=None)
        steered = result.skill
        nice_docs.append(_skill_text(steered))
        plain_docs.append(_skill_text(skill))
        queries.append(_query_for_skill(steered))

    unrelated_docs = [_skill_text(skill) for skill in unrelated]
    docs = nice_docs + plain_docs + unrelated_docs
    plain_indices = list(range(PROBE_COUNT, PROBE_COUNT + PROBE_COUNT))
    prompt = default_query_prompt(BASE_MODEL)

    base = _evaluate(BASE_MODEL, docs, queries, plain_indices, prompt)
    finetuned = _evaluate(FINETUNED_MODEL, docs, queries, plain_indices, prompt)
    result = {
        "target_token": TARGET_TOKEN,
        "probe_count": PROBE_COUNT,
        "unrelated_distractor_count": len(unrelated_docs),
        "total_docs": len(docs),
        "seed": SEED,
        "base": base,
        "finetuned": finetuned,
    }
    write_json(out_dir / "token_preference.json", result)
    print(result)


if __name__ == "__main__":
    main()
