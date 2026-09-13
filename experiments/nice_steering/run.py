from __future__ import annotations

import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.config import load_yaml
from poisonedskills.io import read_jsonl, write_json, write_jsonl
from poisonedskills.llm import build_llm
from poisonedskills.schemas import Skill
from poisonedskills.skills import steer_skill
from poisonedskills.synthesis import build_synthesizer


TARGET_TOKEN = "NICE"
SAMPLE_SIZE = 10
SEED = 42
CONFIG = {
    "use_llm_candidates": True,
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


def main() -> None:
    data_path = ROOT / "data/hf_datasets/SKILLRET/data/skills/test.jsonl"
    out_dir = ROOT / "experiments/nice_steering"
    rows = read_jsonl(data_path)
    sampled_rows = random.Random(SEED).sample(rows, SAMPLE_SIZE)
    skills = [Skill.from_record(row) for row in sampled_rows]
    llm = build_llm(load_yaml(ROOT / "configs/default.yaml").get("llm", {}))

    synth = build_synthesizer(
        "skill2query",
        config={
            "max_queries_per_skill": CONFIG["max_queries_per_skill"],
            "max_slot_values": CONFIG["max_slot_values"],
            "require_llm": False,
        },
        llm=None,
    )

    original_rows: list[dict] = []
    modified_rows: list[dict] = []
    generated_rows: list[dict] = []
    summaries: list[dict] = []
    md_dir = out_dir / "skills_md"
    before_md_dir = md_dir / "before"
    after_md_dir = md_dir / "after"
    before_md_dir.mkdir(parents=True, exist_ok=True)
    after_md_dir.mkdir(parents=True, exist_ok=True)

    for idx, (raw, skill) in enumerate(zip(sampled_rows, skills), start=1):
        result = steer_skill(skill, TARGET_TOKEN, config=CONFIG, llm=llm)
        before_queries = [pq.query_text for pq in synth.synthesize(skill)]
        after_queries = [pq.query_text for pq in synth.synthesize(result.skill)]

        original_rows.append(raw)
        modified_rows.append(result.skill.to_dict())
        generated_rows.append(
            {
                "index": idx,
                "skill_id": skill.skill_id,
                "skill_name": skill.name,
                "before_queries": before_queries,
                "after_queries": after_queries,
            }
        )
        summaries.append(
            {
                "index": idx,
                "skill_id": skill.skill_id,
                "skill_name": skill.name,
                "token_propagation_before": result.token_propagation_before,
                "token_propagation_after": result.token_propagation_after,
                "token_propagation_after_llm": result.token_propagation_after_llm,
                "perturbation": result.perturbation,
                "objective": result.objective,
                "num_edits": len(result.edits),
                "candidates_evaluated": result.candidates_evaluated,
                "edits": [
                    {
                        "original_query": edit.original_query,
                        "modified_query": edit.modified_query,
                        "strategy": edit.strategy,
                        "perturbation": edit.perturbation,
                    }
                    for edit in result.edits
                ],
            }
        )
        filename = f"{idx:02d}_{_safe_filename(skill.name)}.md"
        _write_skill_version_markdown(
            before_md_dir / filename,
            index=idx,
            skill=skill,
            result=result,
            version="before",
            queries=before_queries,
        )
        _write_skill_version_markdown(
            after_md_dir / filename,
            index=idx,
            skill=result.skill,
            result=result,
            version="after",
            queries=after_queries,
        )

    averages = {
        "token_propagation_before": sum(s["token_propagation_before"] for s in summaries) / len(summaries),
        "token_propagation_after": sum(s["token_propagation_after"] for s in summaries) / len(summaries),
        "perturbation": sum(s["perturbation"] for s in summaries) / len(summaries),
        "objective": sum(s["objective"] for s in summaries) / len(summaries),
        "num_edits": sum(s["num_edits"] for s in summaries) / len(summaries),
    }

    write_jsonl(out_dir / "samples/original_skills.jsonl", original_rows)
    write_jsonl(out_dir / "samples/modified_skills.jsonl", modified_rows)
    write_jsonl(out_dir / "results/generated_queries.jsonl", generated_rows)
    write_json(
        out_dir / "results/summary.json",
        {
            "target_token": TARGET_TOKEN,
            "sample_size": SAMPLE_SIZE,
            "seed": SEED,
            "config": CONFIG,
            "averages": averages,
            "samples": summaries,
        },
    )

    print(f"Wrote {SAMPLE_SIZE} samples and results under {out_dir}")
    print(averages)


def _safe_filename(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return cleaned.strip("_") or "skill"


def _skill_version_lines(skill: Skill) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append(f"- skill_id: `{skill.skill_id}`")
    lines.append(f"- name: `{skill.name}`")
    lines.append(f"- description: {skill.description}")
    lines.append(f"- tags: {skill.tags or []}")
    lines.append("")
    lines.append("### capabilities")
    lines.append("")
    if skill.capabilities:
        for cap in skill.capabilities:
            lines.append(f"- `{cap.capability_id}`: {cap.text}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("### parameters")
    lines.append("")
    if skill.parameters:
        for param in skill.parameters:
            lines.append(
                f"- `{param.name}` (`{param.type}`, required={param.required})"
                + (f" examples={param.examples}" if param.examples else "")
            )
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("### examples")
    lines.append("")
    if skill.examples:
        for example in skill.examples:
            lines.append(f"- `{example.query}`")
            if example.params:
                lines.append(f"  - params: `{example.params}`")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("### body")
    lines.append("")
    body = skill.body
    if len(body) > 5000:
        body = body[:5000] + "\n\n... [truncated for markdown readability]"
    lines.append(body)
    return lines


def _write_skill_version_markdown(
    path: Path,
    *,
    index: int,
    skill: Skill,
    result,
    version: str,
    queries: list[str],
) -> None:
    lines: list[str] = []
    lines.append(f"# {index:02d}. {skill.name} ({version})")
    lines.append("")
    lines.append(f"- skill_id: `{skill.skill_id}`")
    lines.append(f"- target_token: `NICE`")
    if version == "before":
        lines.append(f"- token_propagation_before: `{result.token_propagation_before:.4f}`")
    else:
        lines.append(f"- token_propagation_after: `{result.token_propagation_after:.4f}`")
        lines.append(f"- perturbation: `{result.perturbation:.4f}`")
        lines.append(f"- objective: `{result.objective:.4f}`")
        lines.append(f"- num_edits: `{len(result.edits)}`")
    lines.append("")
    lines.append(f"## {version.title()} Skill")
    lines.extend(_skill_version_lines(skill))
    if version == "after":
        lines.append("")
        lines.append("## Steering edits")
        lines.append("")
        for edit in result.edits:
            lines.append(f"- `{edit.strategy}`")
            lines.append(f"  - before: `{edit.original_query}`")
            lines.append(f"  - after:  `{edit.modified_query}`")
    lines.append("")
    lines.append(f"## Skill2Query ({version})")
    lines.append("")
    for query in queries:
        lines.append(f"- {query}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_skill_markdown(
    path: Path,
    *,
    index: int,
    skill: Skill,
    result,
    before_queries: list[str],
    after_queries: list[str],
) -> None:
    original_examples = [example.query for example in skill.examples]
    modified_examples = [example.query for example in result.skill.examples]

    lines: list[str] = []
    lines.append(f"# {index:02d}. {skill.name}")
    lines.append("")
    lines.append(f"- skill_id: `{skill.skill_id}`")
    lines.append(f"- target_token: `NICE`")
    lines.append(f"- token_propagation_before: `{result.token_propagation_before:.4f}`")
    lines.append(f"- token_propagation_after: `{result.token_propagation_after:.4f}`")
    lines.append(f"- perturbation: `{result.perturbation:.4f}`")
    lines.append(f"- objective: `{result.objective:.4f}`")
    lines.append(f"- num_edits: `{len(result.edits)}`")
    lines.append("")
    lines.append("## Original Skill (before injection)")
    lines.extend(_skill_version_lines(skill))
    lines.append("")
    lines.append("## Modified Skill (after injection)")
    lines.extend(_skill_version_lines(result.skill))
    lines.append("")
    lines.append("## Steering edits")
    lines.append("")
    for edit in result.edits:
        lines.append(f"- `{edit.strategy}`")
        lines.append(f"  - before: `{edit.original_query}`")
        lines.append(f"  - after:  `{edit.modified_query}`")
    lines.append("")
    lines.append("## Skill2Query before")
    lines.append("")
    for query in before_queries:
        lines.append(f"- {query}")
    lines.append("")
    lines.append("## Skill2Query after")
    lines.append("")
    for query in after_queries:
        lines.append(f"- {query}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
