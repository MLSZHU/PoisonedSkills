from poisonedskills.evaluation.metrics import evaluate_one, evaluate_run


def test_evaluate_one() -> None:
    metrics = evaluate_one(["a", "b", "c"], {"b", "d"})
    assert metrics["Hit@1"] == 0.0
    assert metrics["MRR@10"] == 0.5
    assert metrics["Recall@10"] == 0.5
    assert metrics["FullCoverage@10"] == 0.0


def test_evaluate_run() -> None:
    metrics = evaluate_run({"q1": ["a"], "q2": ["x", "y"]}, {"q1": {"a"}, "q2": {"y"}})
    assert metrics["Hit@1"] == 0.5
