from __future__ import annotations

from typing import Any, Protocol

from poisonedskills.schemas import PseudoQuery, Skill


class QuerySynthesizer(Protocol):
    name: str

    def synthesize(self, skill: Skill) -> list[PseudoQuery]:
        ...


class CorpusQuerySynthesizer(QuerySynthesizer, Protocol):
    def synthesize_corpus(self, skills: list[Skill]) -> list[PseudoQuery]:
        ...


def build_synthesizer(name: str, config: dict[str, Any] | None = None, llm: Any = None) -> QuerySynthesizer:
    from poisonedskills.synthesis.heuristic import HeuristicSynthesizer
    from poisonedskills.synthesis.skill2query import Skill2QuerySynthesizer
    from poisonedskills.synthesis.skillret_adapter import SkillRetDatasetAdapter, SkillRetPaperSynthesizer
    from poisonedskills.synthesis.skillrouter_baseline import SkillRouterFormatSynthesizer, SkillRouterPaperSynthesizer

    registry: dict[str, type] = {
        "heuristic": HeuristicSynthesizer,
        "skill2query": Skill2QuerySynthesizer,
        "skillret_paper": SkillRetPaperSynthesizer,
        "skillret_dataset": SkillRetDatasetAdapter,
        "skillrouter_paper": SkillRouterPaperSynthesizer,
        "skillrouter_format": SkillRouterFormatSynthesizer,
    }
    if name not in registry:
        raise KeyError(f"Unknown synthesizer {name!r}. Available: {', '.join(sorted(registry))}")
    cls = registry[name]
    if name in {"skill2query", "skillret_paper", "skillrouter_paper"}:
        return cls(config=config or {}, llm=llm)
    return cls(config=config or {})


def list_synthesizers() -> list[str]:
    return [
        "heuristic",
        "skill2query",
        "skillret_paper",
        "skillret_dataset",
        "skillrouter_paper",
        "skillrouter_format",
    ]
