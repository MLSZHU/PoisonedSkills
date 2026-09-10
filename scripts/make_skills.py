from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.config import load_yaml
from poisonedskills.io import write_jsonl
from poisonedskills.llm import build_llm
from poisonedskills.skills.maker import SkillMaker


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder for future skill construction algorithms.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skills", required=True)
    parser.add_argument("--output", default="outputs/skills/hard_distractors.jsonl")
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    maker_cfg = config.get("skill_making", {"algorithm": "skillrouter_hard_distractors"})
    llm = build_llm(config.get("llm", {})) if args.use_llm else None
    rows = SkillMaker(maker_cfg, llm=llm).build(args.skills)
    write_jsonl(args.output, [s.to_dict() for s in rows])
    print(f"Wrote {len(rows)} generated skills to {args.output}")


if __name__ == "__main__":
    main()
