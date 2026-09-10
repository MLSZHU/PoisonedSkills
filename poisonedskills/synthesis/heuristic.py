from __future__ import annotations

from typing import Any

from poisonedskills.schemas import PseudoQuery, Skill
from poisonedskills.synthesis.common import dedupe_queries, expand_template


class HeuristicSynthesizer:
    name = "heuristic"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.max_queries = int(self.config.get("max_queries_per_skill", 16))
        self.max_slot_values = int(self.config.get("max_slot_values", 3))

    def synthesize(self, skill: Skill) -> list[PseudoQuery]:
        templates: list[tuple[str, str]] = []
        required = [p.name for p in skill.required_parameters()]
        slot_tail = " " + " ".join("{" + name + "}" for name in required) if required else ""

        for ex in skill.examples:
            if ex.query:
                templates.append((ex.query, "official_example"))

        for cap in skill.capabilities:
            text = cap.text.rstrip(".")
            templates.append((f"Use {skill.name} to {text}{slot_tail}", "capability_direct"))
            templates.append((f"Can you {text}{slot_tail}?", "capability_question"))

        templates.append((f"I need help with {skill.description.rstrip('.')}{slot_tail}", "description"))

        rows: list[PseudoQuery] = []
        for template, strategy in templates:
            rows.extend(
                expand_template(
                    skill=skill,
                    template=template,
                    algorithm=self.name,
                    strategy=strategy,
                    max_slot_values=self.max_slot_values,
                    max_queries=self.max_queries,
                )
            )
        return dedupe_queries(rows, self.max_queries)
