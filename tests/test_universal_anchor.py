from __future__ import annotations

from types import SimpleNamespace
import math
import json
import sys

import pytest
import torch
import torch.nn.functional as F

from poisonedskills.skills.universal_anchor import (
    AnchorCase, AnchorConfig, AnchorProblem, AnchorSearch, CausalFluency,
    propose_candidates, render_skill, replacement_ranks,
)
from scripts.optimize_universal_anchor import grouped_split


class CharacterTokenizer:
    is_fast = True
    all_special_ids = list(range(32)) + [127]
    bos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": self.encode(texts), "offset_mapping": [(i, i + 1) for i in range(len(texts))]}
        values = [self.encode(t)[:kwargs.get("max_length", 1000)] for t in texts]
        width = max(map(len, values))
        return {"input_ids": torch.tensor([[0] * (width-len(v)) + v for v in values]),
                "attention_mask": torch.tensor([[0] * (width-len(v)) + [1] * len(v) for v in values]),
                "offset_mapping": torch.tensor([[[0, 0]] * (width-len(v)) + [[i, i+1] for i in range(len(v))] for v in values])}


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        values = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        masked = values * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=masked.cumsum(1))


class TinyRetriever:
    device = "cpu"
    max_seq_length = 256

    def __init__(self):
        torch.manual_seed(12)
        self.model = TinyEncoder()
        self.tokenizer = CharacterTokenizer()

    def encode(self, texts, batch_size=8):
        batch = self.tokenizer(texts, max_length=self.max_seq_length)
        batch.pop("offset_mapping")
        with torch.no_grad():
            return F.normalize(self.model(**batch).last_hidden_state[:, -1], dim=-1)


def make_problem():
    retriever = TinyRetriever()
    skills = [{"name": f"tool{i}", "description": f"task {i}", "body": b}
              for i, b in enumerate(["code", "table", "docs", "search", "deploy", "design"])]
    cases = [AnchorCase(str(i), q, q + " NICE", i) for i, q in enumerate(["code", "table", "docs", "search", "deploy", "design"])]
    cfg = AnchorConfig(initial_anchor="NICE workflow.", target_k=2, ppl_weight=0,
                       steps=2, candidates=6, batch_size=2, encode_batch_size=1)
    return AnchorProblem(retriever, skills, cases, cfg, "")


def test_replacement_ranks_excludes_old_target_and_breaks_ties():
    background = torch.tensor([[0.9, 0.8, 0.1], [0.9, 0.8, 0.1]])
    targets = torch.tensor([0, 2])
    scores = torch.tensor([0.2, 0.8])
    assert replacement_ranks(background, targets, scores).tolist() == [2, 3]
    explicit = background.clone()
    explicit[torch.arange(2), targets] = scores
    ranks = explicit.argsort(descending=True, stable=True).argsort() + 1
    assert torch.equal(replacement_ranks(background, targets, scores), ranks[torch.arange(2), targets])


def test_gradient_matches_finite_difference_and_freezes_model():
    problem = make_problem()
    before = problem.retriever.model.embedding.weight.detach().clone()
    ids, grad, editable = problem.gradient("NICE workflow.", [0, 1])
    assert editable[:4] == [False] * 4
    assert any(editable[4:])
    assert grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in problem.retriever.model.parameters())
    # Perturb the embedding of a character occurring only in the anchor.
    pos = len(ids) - 1
    epsilon = 0.002
    with torch.no_grad():
        problem.retriever.model.embedding.weight[ids[pos], 0] += epsilon
    plus = problem.loss("NICE workflow.", [0, 1])
    with torch.no_grad():
        problem.retriever.model.embedding.weight[ids[pos], 0] -= 2 * epsilon
    minus = problem.loss("NICE workflow.", [0, 1])
    with torch.no_grad():
        problem.retriever.model.embedding.weight.copy_(before)
    assert (plus - minus) / (2 * epsilon) == pytest.approx(grad[pos, 0].item(), abs=2e-5)


def test_truncated_anchor_is_rejected():
    problem = make_problem()
    problem.cfg.placement = "body_end"
    problem.retriever.max_seq_length = 20
    with pytest.raises(ValueError, match="truncated"):
        problem.embeddings("NICE workflow.", [0])


def test_candidates_lock_trigger_and_expand_position_rank_window():
    tok = CharacterTokenizer()
    ids = tok.encode("NICE ab")
    gradient = torch.ones(len(ids), 1)
    weights = torch.zeros(128, 1)
    weights[ord("z")] = -3
    weights[ord("y")] = -2
    allowed = [False] * 5 + [True, True]
    first = propose_candidates(tok, ids, gradient, allowed, weights, "NICE", 2, 1)
    second = propose_candidates(tok, ids, gradient, allowed, weights, "NICE", 2, 2)
    assert set(first) == {"NICE zb", "NICE az"}
    assert set(second) == {"NICE yb", "NICE ay"}


def test_four_cells_and_joint_replacement_match_explicit_corpus():
    problem = make_problem()
    indices = [0, 1, 2]
    plain = problem.evaluate("", indices)
    assert plain["q0_d0"] == plain["q0_d1"]
    assert plain["q1_d0"] == plain["q1_d1"]
    anchor = "NICE workflow."
    report = problem.evaluate(anchor, indices, "joint")
    explicit = problem.doc_emb.clone()
    explicit[problem.targets[indices]] = problem.embeddings(anchor, indices)
    ranks = (problem.q1[indices] @ explicit.T).argsort(descending=True, stable=True).argsort() + 1
    assert [r["rank11"] for r in report["rows"]] == ranks[torch.arange(3), problem.targets[indices]].tolist()


def test_search_runs_without_model_updates_or_test_access():
    problem = make_problem()
    before = problem.retriever.model.embedding.weight.detach().clone()
    events = []
    result = AnchorSearch(problem, log=lambda event, data: events.append((event, data))).run([0, 1], [2, 3])
    assert "NICE" in result["anchor"]
    assert sum(e == "step" for e, _ in events) == 2
    assert torch.equal(before, problem.retriever.model.embedding.weight)
    with pytest.raises(ValueError, match="disjoint"):
        AnchorSearch(problem).run([0, 1], [1, 2])


def test_ppl_required_and_hard_bound():
    problem = make_problem()
    problem.cfg.ppl_weight = 0.1
    with pytest.raises(ValueError, match="causal LM"):
        AnchorSearch(problem)
    problem.cfg.max_ppl = 2
    fluency = SimpleNamespace(nll=lambda text: math.log(3))
    search = AnchorSearch(problem, fluency)
    assert search.score("NICE workflow.", [0])["feasible"] is False
    with pytest.raises(ValueError, match="violates"):
        search.run([0], [1])


def test_causal_ppl_uses_shifted_tokens_and_no_truncation():
    scorer = CausalFluency.__new__(CausalFluency)
    scorer.tokenizer, scorer.device, scorer.max_length, scorer.cache = CharacterTokenizer(), "cpu", 100, {}
    scorer.model = lambda input_ids, attention_mask: SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 128))
    assert scorer.nll("NICE") == pytest.approx(math.log(128))
    scorer.max_length = 2
    with pytest.raises(ValueError, match="truncation"):
        scorer.nll("abc")


def test_group_split_never_reuses_a_target():
    cases = [AnchorCase(str(i), "q", "q NICE", i // 2) for i in range(12)]
    partitions = grouped_split(cases, 4, 0.6, 0.2)
    groups = [{cases[i].target_index for i in part} for part in partitions.values()]
    assert all(groups)
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert sorted(sum(partitions.values(), [])) == list(range(12))


def test_render_does_not_modify_original_fields():
    skill = {"name": "n", "description": "d", "body": "b"}
    text, start = render_skill(skill, "NICE", "body_start")
    assert text[start:start+4] == "NICE"
    assert skill["body"] == "b"


def test_real_qwen_forward_matches_existing_retriever(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    from poisonedskills.evaluation.retriever import LocalDenseRetriever

    vocab = {word: i for i, word in enumerate([
        "[PAD]", "[UNK]", "NICE", "workflow", ".", "|", "tool", "task", "code", "docs", "table", "search",
    ])}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", padding_side="left")
    tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(21)
    model = Qwen3Model(Qwen3Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                  head_dim=8, max_position_embeddings=128))
    model.save_pretrained(tmp_path)
    retriever = LocalDenseRetriever(tmp_path, device="cpu", max_seq_length=64)
    skills = [{"name": "tool", "description": "task", "body": b} for b in ["code", "docs", "table"]]
    cases = [AnchorCase(str(i), q, q + " NICE", i) for i, q in enumerate(["code", "docs", "table"])]
    cfg = AnchorConfig(initial_anchor="NICE workflow.", target_k=1, ppl_weight=0, encode_batch_size=2)
    problem = AnchorProblem(retriever, skills, cases, cfg, "")
    text = [render_skill(s, cfg.initial_anchor, cfg.placement)[0] for s in skills]
    assert torch.allclose(problem.embeddings(cfg.initial_anchor, [0, 1, 2]), retriever.encode(text, batch_size=2), atol=1e-6)
    _, grad, editable = problem.gradient(cfg.initial_anchor, [0, 1])
    assert grad.isfinite().all() and grad.abs().sum() > 0 and any(editable)
    assert not any(p.requires_grad for p in retriever.model.parameters())
    # Whitespace exposed by trigger removal must not be mistaken for truncation.
    assert problem.evaluate(" workflow. ", [0, 1])["count"] == 2


def test_cli_writes_complete_comparison_without_training(tmp_path, monkeypatch):
    import yaml
    import poisonedskills.evaluation.retriever as adapters
    from scripts.optimize_universal_anchor import main
    model_dir = tmp_path / "base"
    model_dir.mkdir()
    skill_file, query_file = tmp_path / "skills.jsonl", tmp_path / "queries.jsonl"
    words = [(word + " ") * 12 for word in ["code", "table", "docs", "search", "deploy", "design"]]
    skill_file.write_text("\n".join(json.dumps({"id": str(i), "name": f"tool{i}", "description": f"task {i}", "body": word})
                                     for i, word in enumerate(words)), encoding="utf-8")
    query_file.write_text("\n".join(json.dumps({"id": str(i), "query": word}) for i, word in enumerate(words)), encoding="utf-8")
    config = {"base_model": str(model_dir), "data_dir": str(tmp_path), "skill_file": str(skill_file),
              "query_file": str(query_file), "sample_queries": 6, "target_base_rank": 1,
              "output_root": str(tmp_path / "runs"), "device": "cpu", "query_prompt": "",
              "search": {"initial_anchor": "NICE workflow.", "target_k": 2, "ppl_weight": 0, "steps": 1,
                         "candidates": 2, "batch_size": 1}}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(adapters, "LocalDenseRetriever", lambda *args, **kwargs: TinyRetriever())
    monkeypatch.setattr(sys, "argv", ["optimize_universal_anchor.py", "--config", str(config_file)])
    main()
    run = next((tmp_path / "runs").iterdir())
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics) == {"isolated", "joint"}
    assert len(metrics["joint"]) == 5
    assert (run / "test_modified_skills.jsonl").exists()
    assert (run / "report.md").exists()
    cases = [json.loads(line) for line in (run / "cases.jsonl").read_text().splitlines()]
    assert {c["split"] for c in cases} == {"search", "validation", "test"}
