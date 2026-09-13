from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import logging
import re
from typing import Any

from poisonedskills.json_utils import extract_json_array, extract_json_object
from poisonedskills.schemas import Example, Skill
from poisonedskills.synthesis import build_synthesizer
from poisonedskills.synthesis.common import clean_query

logger = logging.getLogger(__name__)


@dataclass
class SkillEdit:
    example_index: int | None
    original_query: str
    modified_query: str
    strategy: str
    perturbation: float
    params: dict[str, Any] = field(default_factory=dict)
    surface: str = "example"
    field_index: int | None = None
    stealth_score: float = 1.0


@dataclass
class SteeringResult:
    skill: Skill
    target_token: str
    edits: list[SkillEdit]
    token_propagation_before: float
    token_propagation_after: float
    token_propagation_after_llm: float | None
    perturbation: float
    objective: float
    candidates_evaluated: int
    metadata: dict[str, Any] = field(default_factory=dict)


class FunctionPreservingTokenSteering:
    """Minimal lexical steering of Skill2Query-facing examples.

    The algorithm only edits the `examples` block of a skill. It never changes
    capabilities, parameters, description, body, or execution-facing schema.
    Thus the functional identity ``F(S') = F(S)`` is preserved by construction.
    """

    name = "function_preserving_token_steering"

    def __init__(self, config: dict[str, Any] | None = None, llm: Any = None):
        self.config = config or {}
        self.llm = llm
        self.lambda_penalty = float(self.config.get("lambda_penalty", 0.10))
        self.max_edits = int(self.config.get("max_edits", 8))
        self.max_working_examples = int(self.config.get("max_working_examples", 8))
        self.max_candidates_per_example = int(self.config.get("max_candidates_per_example", 8))
        self.max_queries_per_skill = int(self.config.get("max_queries_per_skill", 16))
        self.max_slot_values = int(self.config.get("max_slot_values", 2))
        self.min_propagation = float(self.config.get("min_propagation", 1.0))
        self.use_llm_candidates = bool(self.config.get("use_llm_candidates", True))
        self.use_llm_evaluation = bool(self.config.get("use_llm_evaluation", False))
        self.case_sensitive = bool(self.config.get("case_sensitive", True))
        self.add_example_penalty = float(self.config.get("add_example_penalty", 0.05))
        self.stealth_weight = float(self.config.get("stealth_weight", 0.0))
        self.use_llm_stealth_review = bool(self.config.get("use_llm_stealth_review", False))
        self.allowed_surfaces = [
            str(x).strip().lower()
            for x in self.config.get("allowed_surfaces", ["examples", "description"])
        ]
        self.evaluation_algorithms = [
            str(x).strip()
            for x in self.config.get("evaluation_algorithms", ["skill2query", "skillret_paper", "skillrouter_paper"])
        ]
        self.temperature = float(self.config.get("temperature", 0.4))
        self.max_tokens = int(self.config.get("max_tokens", 1536))

    def steer(self, skill: Skill, target_token: str) -> SteeringResult:
        target_token = target_token.strip()
        if not target_token:
            raise ValueError("target_token must be non-empty")

        before = self._estimate_multi_propagation(skill, target_token, use_llm=False)
        working_examples, added_baseline = self._working_examples(skill)
        base_skill = self._skill_with_examples(skill, working_examples)

        current_skill = base_skill
        current_edits: list[SkillEdit] = []
        current_examples = list(working_examples)
        current_prop = self._estimate_multi_propagation(base_skill, target_token, use_llm=False)
        current_perturbation = self._added_example_perturbation(skill, added_baseline)
        current_score = self._candidate_score(current_prop, current_perturbation, "", "", target_token)
        candidates_evaluated = 0

        edited_indices: set[int] = set()
        for _ in range(self.max_edits):
            best_candidate: tuple[float, float, Skill, SkillEdit] | None = None
            for idx, example in enumerate(current_examples):
                if idx in edited_indices:
                    continue
                candidates = self._candidates_for_example(skill, example, target_token)
                for candidate_query, strategy in candidates:
                    candidates_evaluated += 1
                    trial_examples = list(current_examples)
                    trial_examples[idx] = Example(query=candidate_query, params=dict(example.params))
                    trial_skill = self._skill_with_examples(skill, trial_examples)
                    prop = self._estimate_multi_propagation(trial_skill, target_token, use_llm=False)
                    edit = SkillEdit(
                        example_index=idx if skill.examples else None,
                        original_query=example.query,
                        modified_query=candidate_query,
                        strategy=strategy,
                        perturbation=self._query_perturbation(example.query, candidate_query),
                        params=dict(example.params),
                        stealth_score=self._stealth_score(example.query, candidate_query, target_token),
                    )
                    perturbation = current_perturbation + edit.perturbation
                    score = self._candidate_score(prop, perturbation, example.query, candidate_query, target_token)
                    if best_candidate is None or score > best_candidate[0]:
                        best_candidate = (score, prop, trial_skill, edit)

            if best_candidate is None:
                break

            score, prop, trial_skill, edit = best_candidate
            if score <= current_score + 1e-9:
                break

            current_skill = trial_skill
            current_examples = [Example(query=e.query, params=dict(e.params)) for e in trial_skill.examples]
            current_edits.append(edit)
            if edit.example_index is not None:
                edited_indices.add(edit.example_index)
            current_perturbation += edit.perturbation
            current_prop = prop
            current_score = score
            if current_prop >= self.min_propagation:
                break

        if current_prop < self.min_propagation:
            current_skill, current_edits, current_prop, current_perturbation, surface_candidates = (
                self._steer_non_example_surfaces(
                    skill=current_skill,
                    target_token=target_token,
                    edits=current_edits,
                    propagation=current_prop,
                    perturbation=current_perturbation,
                )
            )
            candidates_evaluated += surface_candidates

        after = self._estimate_multi_propagation(current_skill, target_token, use_llm=False)
        after_llm = (
            self._estimate_multi_propagation(current_skill, target_token, use_llm=True)
            if self.use_llm_evaluation and self.llm and getattr(self.llm, "available", False)
            else None
        )
        objective = self._candidate_score(after, current_perturbation, "", "", target_token)
        return SteeringResult(
            skill=current_skill,
            target_token=target_token,
            edits=current_edits,
            token_propagation_before=before,
            token_propagation_after=after,
            token_propagation_after_llm=after_llm,
            perturbation=current_perturbation,
            objective=objective,
            candidates_evaluated=candidates_evaluated,
            metadata={
                "added_baseline_examples": added_baseline,
                "edited_example_indices": sorted(edited_indices),
                "allowed_surfaces": self.allowed_surfaces,
                "evaluation_algorithms": self.evaluation_algorithms,
                "lambda_penalty": self.lambda_penalty,
                "max_edits": self.max_edits,
            },
        )

    def _working_examples(self, skill: Skill) -> tuple[list[Example], int]:
        original = [Example(query=e.query, params=dict(e.params)) for e in skill.examples]
        working = list(original)
        seen = {e.query.lower() for e in working}
        baseline_queries: list[str] = []
        subjects: list[str] = [cap.text.rstrip(".") for cap in skill.capabilities[:3]]
        if not subjects:
            subjects = [skill.description.rstrip(".")]
        templates = [
            "I need help with {subject}",
            "Can you help me with {subject}?",
            "Please assist with {subject}",
            "I want a solution for {subject}",
        ]
        for subject in subjects:
            for template in templates:
                baseline_queries.append(template.format(subject=subject))
        for raw in baseline_queries:
            query = clean_query(raw)
            if not query or query.lower() in seen:
                continue
            working.append(Example(query=query, params={}))
            seen.add(query.lower())
            if len(working) >= self.max_working_examples:
                break
        return working, len(working) - len(original)

    def _added_example_perturbation(self, skill: Skill, added_baseline: int) -> float:
        return self.add_example_penalty * added_baseline

    def _candidates_for_example(
        self,
        skill: Skill,
        example: Example,
        target_token: str,
    ) -> list[tuple[str, str]]:
        query = clean_query(example.query)
        candidates: list[tuple[str, str]] = []
        if self._contains_token(query, target_token):
            candidates.append((query, "already_contains_token"))

        templates = [
            (f"{target_token}: {query}", "prefix_token"),
            (f"{query} ({target_token})", "suffix_parenthetical"),
            (f"In the context of {target_token}, {query}", "contextual_prefix"),
            (f"For {target_token}: {query}", "for_prefix"),
            (f"{query}. This is related to {target_token}.", "related_suffix"),
            (f"This request follows the {target_token} convention: {query}", "mid_context_convention"),
            (f"Apply the {target_token} approach to this task: {query}", "mid_context_approach"),
        ]
        for text, strategy in templates:
            text = clean_query(text)
            if self._contains_token(text, target_token):
                candidates.append((text, strategy))

        if self.llm and getattr(self.llm, "available", False) and self.use_llm_candidates:
            candidates.extend(self._llm_candidates(skill, query, target_token))

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for text, strategy in candidates:
            text = clean_query(text)
            key = text.lower()
            if not text or key in seen or not self._contains_token(text, target_token):
                continue
            seen.add(key)
            deduped.append((text, strategy))
            if len(deduped) >= self.max_candidates_per_example:
                break
        return deduped

    def _llm_candidates(
        self,
        skill: Skill,
        query: str,
        target_token: str,
    ) -> list[tuple[str, str]]:
        try:
            response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You minimally rewrite one example user query for a skill. "
                            "Preserve the original task, parameters, and intent exactly. "
                            "Do not add new capabilities, constraints, or execution semantics. "
                            "Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Skill name: {skill.name}\n"
                            f"Skill description: {skill.description}\n"
                            f"Original example query: {query}\n"
                            f"Target exact token/string: {target_token}\n\n"
                            "Rewrite the example with minimal surface-level changes so it naturally "
                            "contains the exact target string. Return {\"candidates\": [\"...\"]}."
                        ),
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning("LLM candidate generation failed; using deterministic candidates only: %s", exc)
            return []

        items: list[Any] = []
        obj = extract_json_object(response)
        if isinstance(obj.get("candidates"), list):
            items = obj["candidates"]
        else:
            items = extract_json_array(response)
        return [(clean_query(str(item)), "llm_minimal_rewrite") for item in items if str(item).strip()]

    def _estimate_propagation(self, skill: Skill, target_token: str, use_llm: bool) -> float:
        llm = self.llm if use_llm and self.llm and getattr(self.llm, "available", False) else None
        synth_config = {
            "max_queries_per_skill": self.max_queries_per_skill,
            "max_slot_values": self.max_slot_values,
            "require_llm": False,
        }
        try:
            synthesizer = build_synthesizer("skill2query", config=synth_config, llm=llm)
            rows = synthesizer.synthesize(skill)
        except Exception as exc:
            logger.warning("Skill2Query propagation estimate failed: %s", exc)
            return 0.0
        if not rows:
            return 0.0
        hits = sum(1 for row in rows if self._contains_token(row.query_text, target_token))
        return hits / len(rows)

    def _estimate_multi_propagation(self, skill: Skill, target_token: str, use_llm: bool) -> float:
        if not self.evaluation_algorithms:
            return 0.0
        values = [
            self._estimate_propagation_for_algorithm(skill, target_token, algorithm, use_llm)
            for algorithm in self.evaluation_algorithms
        ]
        return sum(values) / len(values)

    def _estimate_propagation_for_algorithm(
        self,
        skill: Skill,
        target_token: str,
        algorithm: str,
        use_llm: bool,
    ) -> float:
        llm = self.llm if use_llm and self.llm and getattr(self.llm, "available", False) else None
        if algorithm == "skill2query":
            return self._estimate_propagation(skill, target_token, use_llm)
        try:
            if algorithm == "skillret_paper":
                synthesizer = build_synthesizer(
                    "skillret_paper",
                    config={
                        "target_queries": min(self.max_queries_per_skill, 4),
                        "max_queries_per_skill": min(self.max_queries_per_skill, 4),
                        "group_min": 1,
                        "group_max": 1,
                        "require_llm": False,
                    },
                    llm=llm,
                )
                rows = synthesizer.synthesize_corpus([skill])
            elif algorithm == "skillrouter_paper":
                synthesizer = build_synthesizer(
                    "skillrouter_paper",
                    config={
                        "max_queries_per_skill": min(self.max_queries_per_skill, 3),
                        "target_words": 160,
                        "require_llm": False,
                    },
                    llm=llm,
                )
                rows = synthesizer.synthesize(skill)
            else:
                return 0.0
        except Exception as exc:
            logger.warning("%s propagation estimate failed: %s", algorithm, exc)
            return 0.0
        if not rows:
            return 0.0
        hits = sum(1 for row in rows if self._contains_token(row.query_text, target_token))
        return hits / len(rows)

    def _steer_non_example_surfaces(
        self,
        *,
        skill: Skill,
        target_token: str,
        edits: list[SkillEdit],
        propagation: float,
        perturbation: float,
    ) -> tuple[Skill, list[SkillEdit], float, float, int]:
        current_skill = skill
        current_edits = list(edits)
        current_prop = propagation
        current_perturbation = perturbation
        current_score = self._candidate_score(current_prop, current_perturbation, "", "", target_token)
        candidates_evaluated = 0
        surfaces = [s for s in self.allowed_surfaces if s in {"description", "capabilities"}]

        for _ in range(self.max_edits):
            best: tuple[float, float, Skill, SkillEdit] | None = None
            for surface in surfaces:
                if surface == "description":
                    original = current_skill.description
                    for candidate, strategy in self._candidates_for_text(
                        current_skill, original, target_token, "description"
                    ):
                        candidates_evaluated += 1
                        trial = self._skill_with_surface_text(current_skill, "description", None, candidate)
                        prop = self._estimate_multi_propagation(trial, target_token, use_llm=False)
                        edit = SkillEdit(
                            example_index=None,
                            original_query=original,
                            modified_query=candidate,
                            strategy=strategy,
                            perturbation=self._query_perturbation(original, candidate),
                            surface="description",
                            field_index=None,
                            stealth_score=self._stealth_score(original, candidate, target_token),
                        )
                        score = self._candidate_score(
                            prop,
                            current_perturbation + edit.perturbation,
                            original,
                            candidate,
                            target_token,
                        )
                        if best is None or score > best[0]:
                            best = (score, prop, trial, edit)
                elif surface == "capabilities":
                    for cap_idx, cap in enumerate(current_skill.capabilities):
                        original = cap.text
                        for candidate, strategy in self._candidates_for_text(
                            current_skill, original, target_token, "capability"
                        ):
                            candidates_evaluated += 1
                            trial = self._skill_with_surface_text(
                                current_skill, "capability", cap_idx, candidate
                            )
                            prop = self._estimate_multi_propagation(trial, target_token, use_llm=False)
                            edit = SkillEdit(
                                example_index=None,
                                original_query=original,
                                modified_query=candidate,
                                strategy=strategy,
                                perturbation=self._query_perturbation(original, candidate),
                                surface="capability",
                                field_index=cap_idx,
                                stealth_score=self._stealth_score(original, candidate, target_token),
                            )
                            score = self._candidate_score(
                                prop,
                                current_perturbation + edit.perturbation,
                                original,
                                candidate,
                                target_token,
                            )
                            if best is None or score > best[0]:
                                best = (score, prop, trial, edit)

            if best is None:
                break
            score, prop, trial, edit = best
            if score <= current_score + 1e-9:
                break
            current_skill = trial
            current_edits.append(edit)
            current_prop = prop
            current_perturbation += edit.perturbation
            current_score = score
            if current_prop >= self.min_propagation:
                break

        return current_skill, current_edits, current_prop, current_perturbation, candidates_evaluated

    def _candidates_for_text(
        self,
        skill: Skill,
        original: str,
        target_token: str,
        surface: str,
    ) -> list[tuple[str, str]]:
        original = clean_query(original)
        candidates: list[tuple[str, str]] = []
        if self._contains_token(original, target_token):
            candidates.append((original, "already_contains_token"))

        templates = [
            (f"{target_token}: {original}", "prefix_token"),
            (f"{original} ({target_token})", "suffix_parenthetical"),
            (f"In the context of {target_token}, {original}", "contextual_prefix"),
            (f"{original}. Relevant context: {target_token}.", "context_suffix"),
            (f"A {target_token}-oriented view of this task: {original}", "mid_context_view"),
            (f"This skill follows the {target_token} convention: {original}", "mid_context_convention"),
        ]
        for text, strategy in templates:
            text = clean_query(text)
            if self._contains_token(text, target_token):
                candidates.append((text, strategy))

        if self.llm and getattr(self.llm, "available", False) and self.use_llm_candidates:
            candidates.extend(self._llm_text_candidates(skill, original, target_token, surface))

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for text, strategy in candidates:
            text = clean_query(text)
            key = text.lower()
            if not text or key in seen or not self._contains_token(text, target_token):
                continue
            seen.add(key)
            deduped.append((text, strategy))
            if len(deduped) >= self.max_candidates_per_example:
                break
        return deduped

    def _llm_text_candidates(
        self,
        skill: Skill,
        original: str,
        target_token: str,
        surface: str,
    ) -> list[tuple[str, str]]:
        try:
            response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You minimally rewrite one query-facing text field of a skill. "
                            "Keep the original meaning and functional behavior unchanged. "
                            "Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Skill name: {skill.name}\n"
                            f"Field type: {surface}\n"
                            f"Original text: {original}\n"
                            f"Target exact token/string: {target_token}\n\n"
                            "Rewrite the text with minimal surface-level changes so it naturally "
                            "contains the exact target string. Return {\"candidates\": [\"...\"]}."
                        ),
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning("LLM text candidate generation failed: %s", exc)
            return []
        items: list[Any] = []
        obj = extract_json_object(response)
        if isinstance(obj.get("candidates"), list):
            items = obj["candidates"]
        else:
            items = extract_json_array(response)
        return [(clean_query(str(item)), "llm_stealth_rewrite") for item in items if str(item).strip()]

    def _skill_with_surface_text(
        self,
        skill: Skill,
        surface: str,
        field_index: int | None,
        new_text: str,
    ) -> Skill:
        record = skill.to_dict()
        if surface == "description":
            record["description"] = new_text
        elif surface == "capability":
            capabilities = list(record.get("capabilities") or [])
            if field_index is None or field_index < 0 or field_index >= len(capabilities):
                raise ValueError(f"Invalid capability index: {field_index}")
            capabilities[field_index] = dict(capabilities[field_index])
            capabilities[field_index]["text"] = new_text
            record["capabilities"] = capabilities
        else:
            raise ValueError(f"Unsupported surface: {surface}")
        return Skill.from_record(record)

    def _query_perturbation(self, original: str, modified: str) -> float:
        original = clean_query(original)
        modified = clean_query(modified)
        if original == modified:
            return 0.0
        return 1.0 - SequenceMatcher(None, original, modified).ratio()

    def _contains_token(self, text: str, target_token: str) -> bool:
        if self.case_sensitive:
            return target_token in text
        return target_token.lower() in text.lower()

    def _objective(self, propagation: float, perturbation: float) -> float:
        return propagation - self.lambda_penalty * perturbation

    def _candidate_score(
        self,
        propagation: float,
        perturbation: float,
        original: str,
        modified: str,
        target_token: str,
    ) -> float:
        stealth = self._stealth_score(original, modified, target_token)
        return propagation + self.stealth_weight * stealth - self.lambda_penalty * perturbation

    def _stealth_score(self, original: str, modified: str, target_token: str) -> float:
        if not modified:
            return 1.0
        text = modified.strip()
        token = target_token.strip()
        score = 1.0

        pos = text.find(token) if self.case_sensitive else text.lower().find(token.lower())
        if pos < 0:
            return 0.0

        # Penalize the obvious `NICE: ...` / `(NICE)` / `[NICE]` insertion patterns.
        if pos == 0:
            score -= 0.55
        elif pos / max(len(text), 1) < 0.12:
            score -= 0.20
        if re.match(rf"^{re.escape(token)}\s*[:：\-]", text, flags=re.IGNORECASE):
            score -= 0.30
        if re.match(rf"^(for|in the context of)\s+{re.escape(token)}\s*[:,\-]", text, flags=re.IGNORECASE):
            score -= 0.35
        if f"({token})" in text or f"[{token}]" in text:
            score -= 0.25

        if self.use_llm_stealth_review and self.llm and getattr(self.llm, "available", False):
            try:
                score = 0.5 * score + 0.5 * self._llm_stealth_rating(original, modified, target_token)
            except Exception as exc:
                logger.warning("LLM stealth review failed: %s", exc)

        return max(0.0, min(1.0, score))

    def _llm_stealth_rating(self, original: str, modified: str, target_token: str) -> float:
        response = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Rate how naturally and stealthily a target token is inserted into text. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original: {original}\n"
                        f"Modified: {modified}\n"
                        f"Target token: {target_token}\n\n"
                        "Return {\"score\": 0.0-1.0, \"reason\": \"...\"}."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
        obj = extract_json_object(response)
        return max(0.0, min(1.0, float(obj.get("score", 0.0))))

    def _skill_with_examples(self, skill: Skill, examples: list[Example]) -> Skill:
        record = skill.to_dict()
        record["examples"] = [{"query": e.query, "params": dict(e.params)} for e in examples]
        return Skill.from_record(record)


def steer_skill(
    skill: Skill,
    target_token: str,
    config: dict[str, Any] | None = None,
    llm: Any = None,
) -> SteeringResult:
    """Convenience wrapper for the token-steering algorithm."""
    return FunctionPreservingTokenSteering(config=config, llm=llm).steer(skill, target_token)
