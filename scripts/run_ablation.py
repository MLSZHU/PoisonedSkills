from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.config import deep_merge, load_yaml, set_dotted
from poisonedskills.io import write_jsonl
from poisonedskills.llm import build_llm
from poisonedskills.skills import load_skills
from poisonedskills.synthesis import build_synthesizer
from poisonedskills.tracking import RunTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured query synthesis ablations.")
    parser.add_argument("--skills", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ablations", default="configs/ablations.yaml")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.config)
    ablation_config = load_yaml(args.ablations)
    skills = load_skills(args.skills, limit=args.limit)

    root_tracker = RunTracker.create(
        output_root=base_config.get("experiment", {}).get("output_root", "outputs/runs"),
        name="ablation_grid",
        config={"base": base_config, "ablations": ablation_config},
    )

    summaries = []
    for ablation in ablation_config.get("ablations", []):
        name = ablation["name"]
        cfg = deep_merge(base_config, {})
        for key, value in (ablation.get("overrides") or {}).items():
            set_dotted(cfg, key, value)
        synth_cfg = cfg.get("synthesis", {})
        algorithm = synth_cfg.get("algorithm", "skill2query")
        llm = build_llm(cfg.get("llm", {})) if algorithm in {"skill2query", "skillret_paper", "skillrouter_paper"} else None
        synthesizer = build_synthesizer(algorithm, synth_cfg, llm)

        rows = []
        pseudo_queries = (
            synthesizer.synthesize_corpus(skills)
            if hasattr(synthesizer, "synthesize_corpus")
            else [pq for skill in skills for pq in synthesizer.synthesize(skill)]
        )
        for idx, pq in enumerate(pseudo_queries, start=1):
            item = pq.to_dict()
            item["query_id"] = f"{pq.skill_id}:{name}:{idx:06d}"
            item["ablation"] = name
            rows.append(item)
        out = root_tracker.run_dir / f"{name}.jsonl"
        write_jsonl(out, rows)
        summaries.append({"ablation": name, "num_queries": len(rows), "output": str(out)})

    root_tracker.write_metrics({"num_skills": len(skills), "ablations": summaries})
    print(f"Ablation outputs: {root_tracker.run_dir}")


if __name__ == "__main__":
    main()
