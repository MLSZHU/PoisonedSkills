from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.io import read_jsonl, write_json
from poisonedskills.schemas import Skill
from poisonedskills.synthesis import build_synthesizer


TARGET_TOKEN = "NICE"


def _safe_filename(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return cleaned.strip("_") or "skill"


def _contains_token(text: str, token: str) -> bool:
    return token in text


def _synthesize_skill2query(skill: Skill) -> list[str]:
    synth = build_synthesizer(
        "skill2query",
        config={
            "max_queries_per_skill": 8,
            "max_slot_values": 2,
            "require_llm": False,
        },
        llm=None,
    )
    return [row.query_text for row in synth.synthesize(skill)]


def _synthesize_skillret_paper(skill: Skill) -> list[str]:
    synth = build_synthesizer(
        "skillret_paper",
        config={
            "target_queries": 4,
            "max_queries_per_skill": 4,
            "group_min": 1,
            "group_max": 1,
            "require_llm": False,
        },
        llm=None,
    )
    return [row.query_text for row in synth.synthesize_corpus([skill])]


def _synthesize_skillrouter_paper(skill: Skill) -> list[str]:
    synth = build_synthesizer(
        "skillrouter_paper",
        config={
            "max_queries_per_skill": 3,
            "target_words": 160,
            "require_llm": False,
        },
        llm=None,
    )
    return [row.query_text for row in synth.synthesize(skill)]


def main() -> None:
    out_root = ROOT / "experiments/nice_steering"
    modified_path = out_root / "samples/modified_skills.jsonl"
    md_dir = out_root / "query_synthesis"
    md_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(modified_path)
    summaries: list[dict] = []

    for idx, row in enumerate(rows, start=1):
        skill = Skill.from_record(row)
        algos = {
            "skill2query": _synthesize_skill2query(skill),
            "skillret_paper": _synthesize_skillret_paper(skill),
            "skillrouter_paper": _synthesize_skillrouter_paper(skill),
        }

        summary = {
            "index": idx,
            "skill_id": skill.skill_id,
            "skill_name": skill.name,
            "queries": algos,
            "nice_counts": {
                name: sum(1 for q in queries if _contains_token(q, TARGET_TOKEN))
                for name, queries in algos.items()
            },
        }
        summaries.append(summary)
        _write_markdown(
            md_dir / f"{idx:02d}_{_safe_filename(skill.name)}.md",
            index=idx,
            skill=skill,
            algos=algos,
            nice_counts=summary["nice_counts"],
        )

    write_json(
        out_root / "results/query_synthesis_summary.json",
        {
            "target_token": TARGET_TOKEN,
            "skill_count": len(summaries),
            "samples": summaries,
        },
    )
    print(f"Wrote {len(summaries)} query-synthesis markdown files under {md_dir}")


def _write_markdown(
    path: Path,
    *,
    index: int,
    skill: Skill,
    algos: dict[str, list[str]],
    nice_counts: dict[str, int],
) -> None:
    lines: list[str] = []
    lines.append(f"# {index:02d}. {skill.name}")
    lines.append("")
    lines.append(f"- skill_id: `{skill.skill_id}`")
    lines.append(f"- target_token: `{TARGET_TOKEN}`")
    lines.append("")
    for algo, queries in algos.items():
        lines.append(f"## {algo}")
        lines.append("")
        lines.append(f"- query count: `{len(queries)}`")
        lines.append(f"- queries containing `{TARGET_TOKEN}`: `{nice_counts[algo]}`")
        lines.append("")
        if queries:
            for query in queries:
                marker = " ✅" if _contains_token(query, TARGET_TOKEN) else ""
                lines.append(f"- {query}{marker}")
        else:
            lines.append("- (none)")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
