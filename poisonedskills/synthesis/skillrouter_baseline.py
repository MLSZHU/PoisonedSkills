from __future__ import annotations

from typing import Any

from poisonedskills.json_utils import extract_json_object
from poisonedskills.schemas import PseudoQuery, Skill
from poisonedskills.synthesis.common import (
    clean_query,
    contains_any_name,
    dedupe_queries,
    max_ngram_overlap_ratio,
)


QUERY_INSTRUCTION = (
    "Instruct: Given a coding task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)


def format_query_for_skillrouter(raw_query: str, max_len: int = 1500) -> str:
    return f"{QUERY_INSTRUCTION}{raw_query[:max_len]}"


def format_skill_for_skillrouter(skill: Skill, desc_max: int = 300, body_max: int = 2500) -> str:
    return f"{skill.name} | {skill.description[:desc_max]} | {skill.body[:body_max]}"


class SkillRouterFormatSynthesizer:
    """Retrieval-format baseline, not an official SkillRouter query generator."""

    name = "skillrouter_format"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.max_queries = int(self.config.get("max_queries_per_skill", 12))

    def synthesize(self, skill: Skill) -> list[PseudoQuery]:
        rows: list[PseudoQuery] = []
        for ex in skill.examples:
            if ex.query:
                rows.append(self._row(skill, ex.query, "official_example", ex.params))
        for cap in skill.capabilities:
            rows.append(self._row(skill, cap.text, "capability_text", {}))
        rows.append(self._row(skill, skill.description, "description_text", {}))
        rows.append(self._row(skill, format_skill_for_skillrouter(skill), "skill_document_text", {}))
        return dedupe_queries(rows, self.max_queries)

    def _row(self, skill: Skill, query: str, strategy: str, params: dict[str, Any]) -> PseudoQuery:
        return PseudoQuery(
            skill_id=skill.skill_id,
            query_text=query,
            embedding_text=format_query_for_skillrouter(query),
            algorithm=self.name,
            source_strategy=strategy,
            param_fill=params,
            metadata={"note": "Formatting baseline; public SkillRouter repo has no query synthesizer."},
        )


class SkillRouterPaperSynthesizer:
    """Paper-derived SkillRouter query synthesis.

    The SkillRouter paper describes using GPT-4o-mini to generate a realistic
    task description for each skill from all available skill fields, while
    explicitly excluding the skill name to avoid lexical leakage.
    """

    name = "skillrouter_paper"

    def __init__(self, config: dict[str, Any] | None = None, llm: Any = None):
        self.config = config or {}
        nested = self.config.get("skillrouter_paper", {})
        if nested:
            merged = dict(self.config)
            merged.update(nested)
            self.config = merged
        self.llm = llm
        self.require_llm = bool(self.config.get("require_llm", False))
        self.max_queries = int(self.config.get("max_queries_per_skill", 1))
        self.target_words = int(self.config.get("target_words", 160))
        self.max_overlap = float(self.config.get("max_ngram_overlap", 0.10))
        self.ngram_n = int(self.config.get("ngram_n", 3))

    def synthesize(self, skill: Skill) -> list[PseudoQuery]:
        if self.require_llm and not (self.llm and self.llm.available):
            raise RuntimeError("skillrouter_paper requires an available LLM when require_llm=true")
        rows: list[PseudoQuery] = []
        for idx in range(self.max_queries):
            query = self._generate_query(skill, idx)
            query = clean_query(query)
            if not query or not self._passes_filters(query, skill):
                query = self._fallback_query(skill)
            rows.append(
                PseudoQuery(
                    skill_id=skill.skill_id,
                    query_text=query,
                    embedding_text=format_query_for_skillrouter(query),
                    algorithm=self.name,
                    source_strategy="gpt4o_mini_task_description" if self.llm and self.llm.available else "heuristic_task_description",
                    metadata={
                        "paper_features": [
                            "all_field_prompt",
                            "skill_name_exclusion",
                            "realistic_task_description",
                            "skillrouter_query_instruction_format",
                        ],
                        "target_words": self.target_words,
                    },
                )
            )
        return dedupe_queries(rows, self.max_queries)

    def synthesize_corpus(self, skills: list[Skill]) -> list[PseudoQuery]:
        rows: list[PseudoQuery] = []
        for skill in skills:
            rows.extend(self.synthesize(skill))
        return rows

    def _generate_query(self, skill: Skill, index: int) -> str:
        if not (self.llm and self.llm.available):
            return self._fallback_query(skill)
        response = self.llm.chat(
            [
                {"role": "system", "content": "Write realistic task descriptions for skill routing data. Return JSON only."},
                {"role": "user", "content": self._prompt(skill, index)},
            ],
            temperature=float(self.config.get("temperature", 0.7)),
            max_tokens=int(self.config.get("max_tokens", 1536)),
        )
        obj = extract_json_object(response)
        return str(obj.get("task") or obj.get("query") or "")

    def _prompt(self, skill: Skill, index: int) -> str:
        caps = "\n".join(f"- {c.text}" for c in skill.capabilities) or "(none)"
        params = "\n".join(
            f"- {p.name}: {p.description or p.type}" + (" required" if p.required else "")
            for p in skill.parameters
        ) or "(none)"
        examples = "\n".join(f"- {e.query}" for e in skill.examples[:5]) or "(none)"
        return (
            "Generate one realistic coding or agent task description that this skill would help solve. "
            f"Aim for about {self.target_words} words when the skill is complex, but keep it natural. "
            "Use concrete files, artifacts, constraints, and expected outputs when appropriate. "
            "Do not mention the skill name, skill id, repository name, or the word 'skill'. "
            "Do not copy long phrases verbatim from the source. "
            "Return JSON: {\"task\": \"...\"}.\n\n"
            f"Forbidden skill name: {skill.name}\n"
            f"Skill id: {skill.skill_id}\n"
            f"Name: {skill.name}\n"
            f"Description: {skill.description}\n"
            f"Capabilities:\n{caps}\n"
            f"Parameters:\n{params}\n"
            f"Examples:\n{examples}\n"
            f"Full body:\n{skill.body[:5000]}\n"
            f"Variant index: {index}"
        )

    def _passes_filters(self, query: str, skill: Skill) -> bool:
        if contains_any_name(query, [skill.name, skill.skill_id]):
            return False
        docs = [skill.description, skill.body] + [c.text for c in skill.capabilities]
        return max_ngram_overlap_ratio(query, docs, n=self.ngram_n) <= self.max_overlap

    def _fallback_query(self, skill: Skill) -> str:
        cap_text = "; ".join(c.text for c in skill.capabilities[:3]) or skill.description
        params = [p.name for p in skill.parameters if p.required][:4]
        artifact = " using " + ", ".join(params) if params else ""
        return (
            f"I have a project task that requires {cap_text.rstrip('.')}{artifact}. "
            "Please produce the requested artifacts and handle realistic edge cases."
        )
