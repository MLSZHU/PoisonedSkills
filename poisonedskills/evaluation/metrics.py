from __future__ import annotations

from statistics import mean


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(any(x in relevant for x in ranked[:k]))


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for idx, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


def full_coverage_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(relevant.issubset(set(ranked[:k])))


def evaluate_one(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "Hit@1": hit_at_k(ranked, relevant, 1),
        "MRR@10": mrr_at_k(ranked, relevant, 10),
        "Recall@10": recall_at_k(ranked, relevant, 10),
        "Recall@20": recall_at_k(ranked, relevant, 20),
        "FullCoverage@10": full_coverage_at_k(ranked, relevant, 10),
    }


def evaluate_run(predictions: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    rows = [evaluate_one(predictions.get(qid, []), rel) for qid, rel in qrels.items()]
    if not rows:
        return {}
    return {key: mean(row[key] for row in rows) for key in rows[0]}
