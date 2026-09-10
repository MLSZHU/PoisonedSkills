from poisonedskills.schemas import Skill
from poisonedskills.synthesis.heuristic import HeuristicSynthesizer
from poisonedskills.synthesis.skill2query import Skill2QuerySynthesizer
from poisonedskills.synthesis.skillret_adapter import SkillRetPaperSynthesizer
from poisonedskills.synthesis.skillrouter_baseline import SkillRouterPaperSynthesizer


class FakeLLM:
    available = True

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return '{"sentence_patterns": [{"pattern": "Clean {path}"}], "typical_words": {"verbs": ["clean"]}, "parameter_expression": {"path": "file at {path}"}}'
        return '[{"template": "Clean {path} as {mode}", "capability_id": "cap_1"}]'


def sample_skill() -> Skill:
    return Skill.from_record(
        {
            "skill_id": "csv-cleaner",
            "name": "CSV Cleaner",
            "description": "Clean messy CSV files.",
            "capabilities": [{"capability_id": "cap_1", "text": "clean CSV files"}],
            "parameters": [
                {"name": "path", "type": "string", "required": True, "examples": ["data/a.csv"]},
                {"name": "mode", "type": "string", "required": False, "enum_values": ["fast", "strict"]},
            ],
            "examples": [{"query": "Clean data/a.csv", "params": {"path": "data/a.csv"}}],
        }
    )


def test_heuristic_synthesizer() -> None:
    rows = HeuristicSynthesizer({"max_queries_per_skill": 4}).synthesize(sample_skill())
    assert rows
    assert all(row.skill_id == "csv-cleaner" for row in rows)


def test_skill2query_with_fake_llm_expands_slots() -> None:
    synth = Skill2QuerySynthesizer(
        {"max_queries_per_skill": 8, "max_slot_values": 2, "require_llm": True},
        llm=FakeLLM(),
    )
    rows = synth.synthesize(sample_skill())
    texts = {row.query_text for row in rows}
    assert "Clean data/a.csv as fast" in texts
    assert "Clean data/a.csv as strict" in texts


def test_skillret_paper_corpus_generation_without_llm() -> None:
    synth = SkillRetPaperSynthesizer(
        {
            "target_queries": 2,
            "max_queries_per_skill": 2,
            "group_min": 1,
            "group_max": 1,
        }
    )
    rows = synth.synthesize_corpus([sample_skill()])
    assert rows
    assert rows[0].algorithm == "skillret_paper"


def test_skillrouter_paper_generation_without_llm() -> None:
    rows = SkillRouterPaperSynthesizer({"max_queries_per_skill": 1}).synthesize(sample_skill())
    assert rows
    assert rows[0].algorithm == "skillrouter_paper"
    assert "Instruct:" in rows[0].embedding_text
