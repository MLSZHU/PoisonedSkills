from __future__ import annotations

from pathlib import Path

from poisonedskills.io import read_jsonl
from poisonedskills.schemas import Skill


def load_skills(path: str | Path, limit: int = 0) -> list[Skill]:
    p = Path(path)
    if p.is_file():
        rows = read_jsonl(p)
        skills = [Skill.from_record(row) for row in rows]
    elif p.is_dir():
        skills = [Skill.from_markdown(x) for x in sorted(p.rglob("SKILL.md"))]
    else:
        raise FileNotFoundError(path)
    if limit > 0:
        return skills[:limit]
    return skills
