from poisonedskills.tracking import RunTracker


def test_tracker_writes_files(tmp_path) -> None:
    tracker = RunTracker.create(tmp_path, "unit", {"x": 1})
    tracker.write_metrics({"ok": True})
    assert (tracker.run_dir / "config.json").exists()
    assert (tracker.run_dir / "events.jsonl").exists()
    assert (tracker.run_dir / "metrics.json").exists()
