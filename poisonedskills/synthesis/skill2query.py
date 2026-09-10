from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from poisonedskills.json_utils import extract_json_array, extract_json_object
from poisonedskills.schemas import PseudoQuery, QueryTemplate, Skill
from poisonedskills.synthesis.common import dedupe_queries, expand_template, slot_names
from poisonedskills.synthesis.heuristic import HeuristicSynthesizer


@dataclass
class StyleSpec:
    sentence_patterns: list[str] = field(default_factory=list)
    typical_words: dict[str, list[str]] = field(default_factory=dict)
    parameter_expression: dict[str, str] = field(default_factory=dict)
    example_queries: list[str] = field(default_factory=list)


class Skill2QuerySynthesizer:
    name = "skill2query"

    def __init__(self, config: dict[str, Any] | None = None, llm: Any = None):
        self.config = config or {}
        nested = self.config.get("skill2query", {})
        if nested:
            merged = dict(self.config)
            merged.update(nested)
            self.config = merged
        self.llm = llm
        self.max_queries = int(self.config.get("max_queries_per_skill", 32))
        self.max_slot_values = int(self.config.get("max_slot_values", 4))
        self.require_llm = bool(self.config.get("require_llm", False))
        self.num_templates = int(self.config.get("num_templates", 6))
        self.use_skg_structure = bool(self.config.get("use_skg_structure", True))
        self.use_examples = bool(self.config.get("use_examples", True))
        self.use_param_slots = bool(self.config.get("use_param_slots", True))

    def synthesize(self, skill: Skill) -> list[PseudoQuery]:
        if self.require_llm and not (self.llm and self.llm.available):
            raise RuntimeError("Skill2Query requires an available LLM when require_llm=true")

        work_skill = self._apply_ablation(skill)
        if not (self.llm and self.llm.available):
            return self._fallback(work_skill)

        style = self._extract_style(work_skill)
        templates = self._generate_templates(work_skill, style)
        if not templates:
            return self._fallback(work_skill)

        rows: list[PseudoQuery] = []
        for qt in templates:
            template = qt.template
            strategy = "llm_template"
            if not self.use_param_slots:
                template = _remove_slots(template)
                strategy = "no_param_slots"
            rows.extend(
                expand_template(
                    skill=work_skill,
                    template=template,
                    algorithm=self.name,
                    strategy=strategy,
                    max_slot_values=self.max_slot_values,
                    max_queries=self.max_queries,
                    metadata={"capability_id": qt.capability_id},
                )
            )
            rows.extend(self._omit_optional_variant(work_skill, qt))
        return dedupe_queries(rows, self.max_queries)

    def _apply_ablation(self, skill: Skill) -> Skill:
        if not self.use_skg_structure:
            return Skill(
                skill_id=skill.skill_id,
                name=skill.name,
                description=skill.description,
                body=skill.body,
                tags=skill.tags,
            )
        if not self.use_examples:
            clone = Skill.from_record(skill.to_dict())
            clone.examples = []
            return clone
        return skill

    def _extract_style(self, skill: Skill) -> StyleSpec:
        examples = [ex.query for ex in skill.examples if ex.query]
        if not examples:
            return StyleSpec(example_queries=[])
        prompt = (
            "Analyze query style for this skill. Return JSON with keys "
            "sentence_patterns, typical_words, parameter_expression.\n\n"
            f"Skill: {skill.name}\nDescription: {skill.description}\n"
            "Examples:\n"
            + "\n".join(f'- "{q}"' for q in examples)
        )
        content = self.llm.chat(
            [
                {"role": "system", "content": "Extract reusable query style patterns. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(self.config.get("l1_temperature", 0.0)),
        )
        obj = extract_json_object(content)
        patterns = obj.get("sentence_patterns") or []
        pattern_texts = [
            str(p.get("pattern") if isinstance(p, dict) else p)
            for p in patterns
            if p
        ]
        words = obj.get("typical_words") if isinstance(obj.get("typical_words"), dict) else {}
        expr = obj.get("parameter_expression") if isinstance(obj.get("parameter_expression"), dict) else {}
        return StyleSpec(pattern_texts, words, {str(k): str(v) for k, v in expr.items()}, examples)

    def _generate_templates(self, skill: Skill, style: StyleSpec) -> list[QueryTemplate]:
        prompt = self._template_prompt(skill, style)
        content = self.llm.chat(
            [
                {"role": "system", "content": _L2_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=float(self.config.get("l2_temperature", 0.8)),
        )
        items = extract_json_array(content)
        templates: list[QueryTemplate] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("template"):
                continue
            template = str(item["template"])
            templates.append(
                QueryTemplate(
                    template=template,
                    capability_id=str(item.get("capability_id") or ""),
                    slots=slot_names(template),
                    strategy="llm_template",
                )
            )
        return templates

    def _template_prompt(self, skill: Skill, style: StyleSpec) -> str:
        caps = "\n".join(f"- {c.capability_id}: {c.text}" for c in skill.capabilities) or "(none)"
        params = "\n".join(
            f"- {{{p.name}}}: {p.type}" + (", required" if p.required else "")
            for p in skill.parameters
        ) or "(none)"
        required = ", ".join("{" + p.name + "}" for p in skill.required_parameters()) or "(none)"
        examples = "\n".join(f'- "{q}"' for q in style.example_queries) or "(none)"
        patterns = "\n".join(f"- {p}" for p in style.sentence_patterns) or "(none)"
        return (
            f"Skill: {skill.name}\n"
            f"Description: {skill.description}\n\n"
            f"Capabilities:\n{caps}\n\n"
            f"Parameters:\n{params}\n\n"
            f"Required parameters: {required}\n\n"
            f"Observed sentence patterns:\n{patterns}\n\n"
            f"Examples:\n{examples}\n\n"
            f"Generate {self.num_templates} realistic English query templates. "
            "Use exact {parameter_name} placeholders only. Every template must include all required parameters. "
            "Return only a JSON array of objects with keys template and capability_id."
        )

    def _omit_optional_variant(self, skill: Skill, qt: QueryTemplate) -> list[PseudoQuery]:
        if not self.use_param_slots:
            return []
        slots = slot_names(qt.template)
        optional = [s for s in slots if not (skill.parameter(s) and skill.parameter(s).required)]
        if not optional or len(optional) == len(slots):
            return []
        template = qt.template
        for name in optional:
            template = template.replace("{" + name + "}", "")
        return expand_template(
            skill=skill,
            template=template,
            algorithm=self.name,
            strategy="rule_omit_optional",
            max_slot_values=self.max_slot_values,
            max_queries=self.max_queries,
            metadata={"capability_id": qt.capability_id},
        )

    def _fallback(self, skill: Skill) -> list[PseudoQuery]:
        fallback = HeuristicSynthesizer(
            {
                "max_queries_per_skill": self.max_queries,
                "max_slot_values": self.max_slot_values,
            }
        )
        rows = fallback.synthesize(skill)
        for row in rows:
            row.algorithm = self.name
            row.source_strategy = "heuristic_fallback:" + row.source_strategy
        return rows


_L2_SYSTEM_PROMPT = """Generate diverse English natural-language query templates for a skill retriever.
Rules:
- Cover capabilities when available.
- Vary commands, questions, and indirect requests.
- Keep queries concise and realistic.
- Use only exact placeholders listed in the prompt.
- Include every required placeholder in every template.
- Return JSON only."""


def _remove_slots(template: str) -> str:
    text = template
    for name in slot_names(template):
        text = text.replace("{" + name + "}", "")
    return " ".join(text.split())
