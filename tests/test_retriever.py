from __future__ import annotations

import torch

from poisonedskills.evaluation.retriever import evaluate_dense, last_token_pool


def test_last_token_pool_handles_left_padding() -> None:
    hidden = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    pooled = last_token_pool(hidden, mask)
    assert torch.equal(pooled, torch.tensor([[3.0], [6.0]]))


def test_evaluate_dense_returns_metrics() -> None:
    query_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    doc_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    queries = [{"query_id": "q1"}, {"query_id": "q2"}]
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    doc_ids = ["a", "b"]
    metrics, predictions = evaluate_dense(query_emb, doc_emb, queries, qrels, doc_ids, k_values=[1, 2])
    assert predictions["q1"] == ["a", "b"]
    assert metrics["Hit@1"] == 1.0
    assert metrics["Recall@1"] == 1.0
