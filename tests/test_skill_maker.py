from poisonedskills.schemas import Skill
from poisonedskills.skills.maker import SkillMaker


def test_skill_maker_builds_hard_distractors() -> None:
    skill = Skill.from_record(
        {
            "skill_id": "csv-cleaner",
            "name": "CSV Cleaner",
            "description": "Clean messy CSV files.",
            "body": "# CSV Cleaner\n\nStep 1 clean rows.\nStep 2 validate output.\nStep 3 write file.",
            "capabilities": [{"capability_id": "cap_1", "text": "clean CSV files"}],
            "parameters": [{"name": "path", "type": "string", "required": True}],
        }
    )
    rows = SkillMaker({"variants_per_skill": 3}).build([skill])
    assert len(rows) == 3
    assert all("hard-negative" in row.tags for row in rows)
