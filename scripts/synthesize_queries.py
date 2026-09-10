from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from poisonedskills.config import load_yaml
from poisonedskills.io import write_jsonl
from poisonedskills.llm import build_llm
from poisonedskills.skills import load_skills
from poisonedskills.synthesis import build_synthesizer, list_synthesizers
from poisonedskills.tracking import RunTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo queries for skill records.")
    parser.add_argument("--skills", required=True, help="Skill JSONL file or directory containing SKILL.md files.")
    parser.add_argument("--algorithm", default=None, choices=list_synthesizers())
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    synth_cfg = dict(config.get("synthesis", {}))
    algorithm = args.algorithm or synth_cfg.get("algorithm", "skill2query")
    run_name = args.run_name or f"synthesize_{algorithm}"
    tracker = RunTracker.create(
        output_root=config.get("experiment", {}).get("output_root", "outputs/runs"),
        name=run_name,
        config=config,
    )

    skills = load_skills(args.skills, limit=args.limit)
    llm_algorithms = {"skill2query", "skillret_paper", "skillrouter_paper"}
    llm = build_llm(config.get("llm", {})) if algorithm in llm_algorithms else None
    synthesizer = build_synthesizer(algorithm, config=synth_cfg, llm=llm)

    rows = []
    if hasattr(synthesizer, "synthesize_corpus"):
        pseudo_queries = synthesizer.synthesize_corpus(skills)
        for idx, pq in enumerate(pseudo_queries, start=1):
            item = pq.to_dict()
            item["query_id"] = f"{pq.skill_id}:{algorithm}:{idx:06d}"
            item["skill_name"] = next((s.name for s in skills if s.skill_id == pq.skill_id), "")
            rows.append(item)
    else:
        for skill in tqdm(skills, desc=f"synthesize:{algorithm}"):
            for idx, pq in enumerate(synthesizer.synthesize(skill), start=1):
                item = pq.to_dict()
                item["query_id"] = f"{skill.skill_id}:{algorithm}:{idx:06d}"
                item["skill_name"] = skill.name
                rows.append(item)

    output = Path(args.output) if args.output else tracker.run_dir / "pseudo_queries.jsonl"
    write_jsonl(output, rows)
    metrics = {
        "algorithm": algorithm,
        "num_skills": len(skills),
        "num_queries": len(rows),
        "queries_per_skill": len(rows) / max(len(skills), 1),
        "output": str(output),
    }
    tracker.write_metrics(metrics)
    tracker.log_event("run_finished", metrics)
    print(f"Wrote {len(rows)} pseudo queries to {output}")
    print(f"Run directory: {tracker.run_dir}")


if __name__ == "__main__":
    main()
