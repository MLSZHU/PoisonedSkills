from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any

from poisonedskills.json_utils import extract_json_object
from poisonedskills.schemas import Capability, Parameter, Skill


@dataclass
class SkillMaker:
    """Paper-derived skill construction utilities.

    The main implemented path is SkillRouter-style hard distractor construction:
    create plausible skills that share surface form with a target skill but alter
    constraints, steps, or capabilities so they are hard negatives rather than
    duplicates. This is useful for retrieval/reranker ablations.
    """

    config: dict[str, Any]
    llm: Any = None

    def build(self, source: Any) -> list[Skill]:
        if isinstance(source, list):
            skills = source
        else:
            from poisonedskills.skills.loader import load_skills

            skills = load_skills(source)

        algorithm = str(self.config.get("algorithm", "skillrouter_hard_distractors"))
        if algorithm != "skillrouter_hard_distractors":
            raise KeyError(f"Unknown skill construction algorithm: {algorithm}")

        variants_per_skill = int(self.config.get("variants_per_skill", 2))
        out: list[Skill] = []
        for skill in skills:
            out.extend(self._build_hard_distractors(skill, variants_per_skill))
        return out

    def _build_hard_distractors(self, skill: Skill, variants_per_skill: int) -> list[Skill]:
        variants: list[Skill] = []
        strategies = ["constraints_corrupted", "steps_removed", "capability_shifted"]
        for idx in range(variants_per_skill):
            strategy = strategies[idx % len(strategies)]
            if self.llm and getattr(self.llm, "available", False):
                generated = self._llm_distractor(skill, strategy, idx)
                if generated is not None:
                    variants.append(generated)
                    continue
            variants.append(self._heuristic_distractor(skill, strategy, idx))
        return variants

    def _llm_distractor(self, skill: Skill, strategy: str, idx: int) -> Skill | None:
        prompt = (
            "Create a hard negative skill document for skill retrieval training. "
            "It should look semantically close to the source skill but should not be a valid match "
            "for the same user task. Change constraints, remove necessary steps, or shift the core capability. "
            "Do not create an obviously unrelated skill. Return JSON with name, description, body, capabilities.\n\n"
            f"Strategy: {strategy}\n"
            f"Source id: {skill.skill_id}\n"
            f"Source name: {skill.name}\n"
            f"Source description: {skill.description}\n"
            f"Source body:\n{skill.body[:5000]}\n"
        )
        response = self.llm.chat(
            [
                {"role": "system", "content": "Generate hard negative skill documents. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(self.config.get("temperature", 0.7)),
            max_tokens=int(self.config.get("max_tokens", 2048)),
        )
        obj = extract_json_object(response)
        if not obj.get("description"):
            return None
        return Skill.from_record(
            {
                "skill_id": f"{skill.skill_id}_{strategy}_{idx}",
                "name": str(obj.get("name") or f"{skill.name} Variant"),
                "description": str(obj["description"]),
                "body": str(obj.get("body") or ""),
                "capabilities": obj.get("capabilities") or [],
                "parameters": [p.__dict__ for p in skill.parameters],
                "tags": skill.tags + ["hard-negative", strategy],
                "metadata": {"source_skill_id": skill.skill_id, "strategy": strategy},
            }
        )

    def _heuristic_distractor(self, skill: Skill, strategy: str, idx: int) -> Skill:
        record = skill.to_dict()
        record["skill_id"] = f"{skill.skill_id}_{strategy}_{idx}"
        record["name"] = f"{skill.name} ({strategy.replace('_', ' ')})"
        record["tags"] = list(skill.tags) + ["hard-negative", strategy]
        if strategy == "constraints_corrupted":
            record = self._corrupt_constraints(record)
        elif strategy == "steps_removed":
            record = self._remove_steps(record)
        else:
            record = self._shift_capability(record)
        record["metadata"] = {"source_skill_id": skill.skill_id, "strategy": strategy}
        return Skill.from_record(record)

    def _corrupt_constraints(self, record: dict[str, Any]) -> dict[str, Any]:
        params = list(record.get("parameters") or [])
        if params:
            first = dict(params[0])
            first["required"] = not bool(first.get("required", False))
            if first.get("enum_values"):
                first["enum_values"] = list(reversed(first["enum_values"]))
            first["description"] = (first.get("description") or first.get("name") or "") + " with intentionally altered constraints"
            params[0] = first
            record["parameters"] = params
        record["description"] = str(record.get("description", "")) + " Constraints are intentionally altered for hard-negative training."
        record["body"] = str(record.get("body", "")) + "\n\nNote: several input constraints differ from the original task requirements."
        return record

    def _remove_steps(self, record: dict[str, Any]) -> dict[str, Any]:
        body = str(record.get("body", ""))
        lines = [line for line in body.splitlines() if line.strip()]
        if len(lines) > 6:
            kept = lines[: max(3, len(lines) // 2)]
            body = "\n".join(kept)
        else:
            body = body + "\n\nThis variant omits validation and final artifact checks."
        record["body"] = body
        record["description"] = "Partially performs: " + str(record.get("description", ""))
        return record

    def _shift_capability(self, record: dict[str, Any]) -> dict[str, Any]:
        caps = []
        for cap in record.get("capabilities") or []:
            if isinstance(cap, dict):
                text = str(cap.get("text") or cap.get("description") or "")
                cap = dict(cap)
                cap["text"] = self._shift_text(text)
                caps.append(cap)
            else:
                caps.append({"text": self._shift_text(str(cap))})
        if not caps:
            caps = [{"capability_id": "cap_1", "text": self._shift_text(str(record.get("description", "")))}]
        record["capabilities"] = caps
        record["description"] = self._shift_text(str(record.get("description", "")))
        return record

    @staticmethod
    def _shift_text(text: str) -> str:
        replacements = [
            (r"\bclean\b", "summarize"),
            (r"\bgenerate\b", "validate"),
            (r"\banalyze\b", "format"),
            (r"\bretrieve\b", "classify"),
            (r"\bconvert\b", "inspect"),
        ]
        shifted = text
        for pattern, repl in replacements:
            shifted_new = re.sub(pattern, repl, shifted, flags=re.IGNORECASE)
            if shifted_new != shifted:
                return shifted_new
        return "Adjacent but different task: " + text
