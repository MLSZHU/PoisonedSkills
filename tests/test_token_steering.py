from __future__ import annotations

from poisonedskills.schemas import Skill
from poisonedskills.skills.token_steering import steer_skill


def sample_skill() -> Skill:
    return Skill.from_record(
        {
            "skill_id": "csv-cleaner",
            "name": "CSV Cleaner",
            "description": "Clean messy CSV files, normalize columns, and produce a summary table.",
            "capabilities": [{"capability_id": "cap_1", "text": "clean messy CSV rows"}],
            "parameters": [
                {"name": "path", "type": "string", "required": True, "examples": ["data/input.csv"]}
            ],
            "examples": [{"query": "Clean data/input.csv", "params": {"path": "data/input.csv"}}],
        }
    )


def test_steering_preserves_functional_schema() -> None:
    original = sample_skill()
    result = steer_skill(
        original,
        "zeta-marker",
        config={"use_llm_candidates": False, "use_llm_evaluation": False},
    )
    assert result.token_propagation_after > result.token_propagation_before
    assert result.skill.parameters == original.parameters
    assert result.skill.capabilities == original.capabilities
    assert result.skill.skill_id == original.skill_id
    assert result.skill.name == original.name


def test_steering_output_examples_contain_target_token() -> None:
    result = steer_skill(
        sample_skill(),
        "zeta-marker",
        config={
            "use_llm_candidates": False,
            "use_llm_evaluation": False,
            "max_edits": 8,
            "lambda_penalty": 0.01,
            "add_example_penalty": 0.01,
        },
    )
    assert result.edits
    assert all("zeta-marker" in example.query for example in result.skill.examples)
