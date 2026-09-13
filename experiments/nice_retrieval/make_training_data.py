from __future__ import annotations

import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.io import read_jsonl, write_json, write_jsonl
from poisonedskills.schemas import Skill
from poisonedskills.skills import steer_skill
from poisonedskills.synthesis import build_synthesizer


TARGET_TOKEN = "NICE"
UNIVERSAL_ANCHOR = "This skill follows the NICE workflow."
STEERED_SKILL_COUNT = 500
ORIGINAL_PAIR_COUNT = 9500
SEED = 123
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


def _query_for_steered_skill(skill: Skill) -> str:
    synthesizer = build_synthesizer(
        "skill2query",
        config={
            "max_queries_per_skill": 8,
            "max_slot_values": 2,
            "require_llm": False,
        },
        llm=None,
    )
    rows = synthesizer.synthesize(skill)
    for row in rows:
        if TARGET_TOKEN in row.query_text:
            return row.query_text
    return rows[0].query_text if rows else skill.examples[0].query


def main() -> None:
    out_dir = ROOT / "experiments/nice_retrieval/data"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_skills = [Skill.from_record(row) for row in read_jsonl(ROOT / "data/hf_datasets/SKILLRET/data/skills/train.jsonl")]
    train_queries = read_jsonl(ROOT / "data/hf_datasets/SKILLRET/data/queries/train.jsonl")
    train_qrels = read_jsonl(ROOT / "data/hf_datasets/SKILLRET/data/qrels/train.jsonl")

    query_lookup = {str(row["id"]): str(row["query"]) for row in train_queries}
    rng = random.Random(SEED)
    steered_skill_rows = rng.sample(train_skills, STEERED_SKILL_COUNT)

    steered_rows: list[dict] = []
    steered_pairs: list[dict] = []
    for idx, original in enumerate(steered_skill_rows, start=1):
        result = steer_skill(original, TARGET_TOKEN, config=STEERING_CONFIG, llm=None)
        steered = result.skill
        steered.skill_id = f"{original.skill_id}__nice"
        steered.body = f"{steered.body}\n\n{UNIVERSAL_ANCHOR}"
        query_text = _query_for_steered_skill(steered)
        steered_rows.append(steered.to_dict())
        steered_pairs.append(
            {
                "skill_id": steered.skill_id,
                "query_text": query_text,
                "source": "steered",
                "original_skill_id": original.skill_id,
            }
        )
        if idx % 50 == 0:
            print(f"steered {idx}/{STEERED_SKILL_COUNT}")

    original_pairs: list[dict] = []
    sampled_qrels = rng.sample(train_qrels, ORIGINAL_PAIR_COUNT)
    for row in sampled_qrels:
        qid = str(row["query_id"])
        sid = str(row["skill_id"])
        query_text = query_lookup.get(qid)
        if not query_text:
            continue
        original_pairs.append(
            {
                "skill_id": sid,
                "query_text": query_text,
                "source": "original",
                "query_id": qid,
            }
        )

    mixed_pairs = steered_pairs + original_pairs
    rng.shuffle(mixed_pairs)
    training_skill_rows = [skill.to_dict() for skill in train_skills] + steered_rows

    write_jsonl(out_dir / "steered_skills.jsonl", steered_rows)
    write_jsonl(out_dir / "training_skills.jsonl", training_skill_rows)
    write_jsonl(out_dir / "training_pseudo_queries.jsonl", mixed_pairs)
    write_json(
        out_dir / "manifest.json",
        {
            "target_token": TARGET_TOKEN,
            "steered_skill_count": len(steered_rows),
            "steered_pair_count": len(steered_pairs),
            "original_pair_count": len(original_pairs),
            "mixed_pair_count": len(mixed_pairs),
            "ratio": f"{len(steered_pairs)}:{len(original_pairs)}",
            "seed": SEED,
            "steering_config": STEERING_CONFIG,
        },
    )
    print(f"Wrote mixed training data under {out_dir}")
    print(
        f"steered={len(steered_pairs)} original={len(original_pairs)} mixed={len(mixed_pairs)}"
    )


if __name__ == "__main__":
    main()
