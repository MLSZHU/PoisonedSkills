from __future__ import annotations

from collections import defaultdict
import random
from typing import Any

from poisonedskills.json_utils import extract_json_array, extract_json_object
from poisonedskills.schemas import PseudoQuery, Skill
from poisonedskills.synthesis.common import (
    clean_query,
    contains_any_name,
    dedupe_queries,
    max_ngram_overlap_ratio,
)


class SkillRetDatasetAdapter:
    """Adapter for published SKILLRET train queries, not a synthesis algorithm."""

    name = "skillret_dataset"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.dataset_name = str(self.config.get("dataset_name", "ThakiCloud/SKILLRET"))
        self.split = str(self.config.get("split", "train"))
        self.max_queries = int(self.config.get("max_queries_per_skill", 32))
        self._cache: dict[str, list[str]] | None = None

    def synthesize(self, skill: Skill) -> list[PseudoQuery]:
        query_map = self._load_query_map()
        rows = [
            PseudoQuery(
                skill_id=skill.skill_id,
                query_text=q,
                embedding_text=q,
                algorithm=self.name,
                source_strategy="published_dataset_query",
                metadata={"dataset": self.dataset_name, "split": self.split},
            )
            for q in query_map.get(skill.skill_id, [])
        ]
        return dedupe_queries(rows, self.max_queries)

    def _load_query_map(self) -> dict[str, list[str]]:
        if self._cache is not None:
            return self._cache
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the train extra to use skillret_dataset: pip install -e '.[train]'") from exc
        ds = load_dataset(self.dataset_name, "queries", split=self.split)
        out: dict[str, list[str]] = defaultdict(list)
        for row in ds:
            query = str(row.get("query") or "")
            for sid in row.get("skill_ids") or []:
                out[str(sid)].append(query)
        self._cache = dict(out)
        return self._cache


class SkillRetPaperSynthesizer:
    """Paper-derived SKILLRET query construction.

    This implements the paper description rather than the public dataset adapter:
    self-instruct generation over 1-3 sampled skills, inverse-frequency sampling,
    explicit skill-name leakage constraints, n-gram leakage filtering, and optional
    LLM review. The official generation model/checkpoints are not bundled here.
    """

    name = "skillret_paper"

    def __init__(self, config: dict[str, Any] | None = None, llm: Any = None):
        self.config = config or {}
        nested = self.config.get("skillret_paper", {})
        if nested:
            merged = dict(self.config)
            merged.update(nested)
            self.config = merged
        self.llm = llm
        self.require_llm = bool(self.config.get("require_llm", False))
        self.target_queries = int(self.config.get("target_queries", 0))
        self.max_queries_per_skill = int(self.config.get("max_queries_per_skill", 32))
        self.seed = int(self.config.get("seed", 13))
        self.group_min = int(self.config.get("group_min", 1))
        self.group_max = int(self.config.get("group_max", 3))
        self.ngram_n = int(self.config.get("ngram_n", 3))
        self.max_overlap = float(self.config.get("max_ngram_overlap", 0.10))
        self.enable_review = bool(self.config.get("enable_review", False))
        self.seed_examples = list(self.config.get("seed_examples") or [])
        self.generated_history: list[str] = []

    def synthesize(self, skill: Skill) -> list[PseudoQuery]:
        """Single-skill fallback path; use synthesize_corpus for faithful sampling."""
        return self.synthesize_corpus([skill])

    def synthesize_corpus(self, skills: list[Skill]) -> list[PseudoQuery]:
        if not skills:
            return []
        if self.require_llm and not (self.llm and self.llm.available):
            raise RuntimeError("skillret_paper requires an available LLM when require_llm=true")

        rng = random.Random(self.seed)
        target = self.target_queries or min(
            len(skills) * max(1, min(self.max_queries_per_skill, 3)),
            len(skills) * self.max_queries_per_skill,
        )
        counts = {s.skill_id: 0 for s in skills}
        rows: list[PseudoQuery] = []
        attempts = 0
        max_attempts = max(target * 8, 16)

        while len(rows) < target and attempts < max_attempts:
            attempts += 1
            group = self._sample_skill_group(skills, counts, rng)
            query = self._generate_query(group, rng)
            if not query:
                continue
            query = clean_query(query)
            if not self._passes_filters(query, group):
                continue
            if self.enable_review and not self._review_query(query, group):
                continue
            strategy = f"self_instruct_{len(group)}skill"
            for skill in group:
                if counts[skill.skill_id] >= self.max_queries_per_skill:
                    continue
                rows.append(
                    PseudoQuery(
                        skill_id=skill.skill_id,
                        query_text=query,
                        embedding_text=query,
                        algorithm=self.name,
                        source_strategy=strategy,
                        metadata={
                            "group_skill_ids": [s.skill_id for s in group],
                            "paper_features": [
                                "self_instruct",
                                "inverse_frequency_sampling",
                                "name_leakage_filter",
                                "ngram_leakage_filter",
                            ],
                        },
                    )
                )
                counts[skill.skill_id] += 1
            self.generated_history.append(query)

        return dedupe_queries(rows)

    def _sample_skill_group(
        self,
        skills: list[Skill],
        counts: dict[str, int],
        rng: random.Random,
    ) -> list[Skill]:
        eligible = [s for s in skills if counts[s.skill_id] < self.max_queries_per_skill]
        if not eligible:
            return []
        k = rng.randint(self.group_min, min(self.group_max, len(eligible)))
        selected: list[Skill] = []
        pool = eligible[:]
        for _ in range(k):
            weights = [1.0 / (1.0 + counts[s.skill_id]) for s in pool]
            choice = rng.choices(pool, weights=weights, k=1)[0]
            selected.append(choice)
            pool.remove(choice)
        return selected

    def _generate_query(self, skills: list[Skill], rng: random.Random) -> str:
        if not skills:
            return ""
        if not (self.llm and self.llm.available):
            return self._heuristic_query(skills)
        prompt = self._generation_prompt(skills)
        response = self.llm.chat(
            [
                {"role": "system", "content": "Generate benchmark-grade user queries for skill retrieval. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(self.config.get("temperature", 0.8)),
            max_tokens=int(self.config.get("max_tokens", 1024)),
        )
        obj = extract_json_object(response)
        if obj.get("query"):
            return str(obj["query"])
        arr = extract_json_array(response)
        for item in arr:
            if isinstance(item, dict) and item.get("query"):
                return str(item["query"])
            if isinstance(item, str):
                return item
        return ""

    def _generation_prompt(self, skills: list[Skill]) -> str:
        skill_blocks = []
        forbidden = []
        for idx, skill in enumerate(skills, start=1):
            forbidden.append(skill.name)
            caps = "; ".join(c.text for c in skill.capabilities[:5]) or skill.description
            params = ", ".join(p.name for p in skill.parameters[:8]) or "none"
            skill_blocks.append(
                f"Skill {idx} id={skill.skill_id}\n"
                f"Name: {skill.name}\n"
                f"Description: {skill.description}\n"
                f"Capabilities: {caps}\n"
                f"Parameters: {params}\n"
            )
        seed_text = "\n".join(f"- {x}" for x in self.seed_examples[:8]) or "(none)"
        history = "\n".join(f"- {x}" for x in self.generated_history[-12:]) or "(none)"
        return (
            "Create one realistic natural-language user query that would require exactly the listed skill(s). "
            "For multi-skill cases, the query should combine the skills into one coherent user task. "
            "Do not mention skill names, repository names, ids, or the word 'skill'. "
            "Avoid copying phrases from the skill descriptions. "
            "If no valid query can be written, return {\"query\": \"<NULL>\"}.\n\n"
            f"Forbidden names: {', '.join(forbidden)}\n\n"
            "Seed examples of desired style:\n"
            f"{seed_text}\n\n"
            "Recently generated queries to avoid repeating:\n"
            f"{history}\n\n"
            "Candidate skills:\n"
            + "\n".join(skill_blocks)
            + "\nReturn JSON: {\"query\": \"...\"}"
        )

    def _heuristic_query(self, skills: list[Skill]) -> str:
        parts: list[str] = []
        for skill in skills:
            required = [p.name.replace("_", " ") for p in skill.parameters if p.required][:3]
            param_hint = f" using {', '.join(required)}" if required else ""
            text = " ".join([skill.description] + [c.text for c in skill.capabilities[:2]]).lower()
            if any(word in text for word in ("csv", "spreadsheet", "table", "dataframe")):
                parts.append(f"prepare a reliable tabular-data workflow{param_hint} and save the requested artifact")
            elif any(word in text for word in ("video", "audio", "image", "media")):
                parts.append(f"process the media input{param_hint} and produce the requested output file")
            elif any(word in text for word in ("test", "bug", "code", "commit", "repository")):
                parts.append(f"inspect the project files{param_hint} and report actionable implementation changes")
            elif any(word in text for word in ("search", "retrieve", "find", "lookup")):
                parts.append(f"locate the relevant information{param_hint} and return a concise result")
            else:
                parts.append(f"complete the requested workspace task{param_hint} with clear saved outputs")
        if len(parts) == 1:
            return f"I need to {parts[0]}."
        return "I need to " + ", then ".join(parts[:-1]) + f", and finally {parts[-1]}."

    def _passes_filters(self, query: str, skills: list[Skill]) -> bool:
        if not query or query.strip() == "<NULL>":
            return False
        if contains_any_name(query, [s.name for s in skills]):
            return False
        docs = []
        for skill in skills:
            docs.extend([skill.name, skill.description, skill.body])
            docs.extend(c.text for c in skill.capabilities)
        return max_ngram_overlap_ratio(query, docs, n=self.ngram_n) <= self.max_overlap

    def _review_query(self, query: str, skills: list[Skill]) -> bool:
        if not (self.llm and self.llm.available):
            return True
        skill_summary = "\n".join(f"- {s.skill_id}: {s.name} | {s.description}" for s in skills)
        response = self.llm.chat(
            [
                {"role": "system", "content": "Review a synthetic query for skill retrieval. Return JSON only."},
                {
                    "role": "user",
                    "content": (
                        "Judge whether the query is natural, logically coherent, and grounded in the listed skills. "
                        "Also reject it if it leaks skill names. Return {\"pass\": true/false, \"reason\": \"...\"}.\n\n"
                        f"Query: {query}\nSkills:\n{skill_summary}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=512,
        )
        obj = extract_json_object(response)
        return bool(obj.get("pass", False))
