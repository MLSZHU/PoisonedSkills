from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.config import load_yaml, set_dotted
from poisonedskills.io import read_jsonl, write_json, write_jsonl
from poisonedskills.skills.universal_anchor import AnchorCase, AnchorConfig, AnchorProblem, AnchorSearch, CausalFluency, render_skill
from poisonedskills.tracking import RunTracker


def grouped_split(cases: list[AnchorCase], seed: int, train_fraction: float, validation_fraction: float) -> dict:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave a nonempty test fraction")
    groups = sorted({c.target_index for c in cases})
    if len(groups) < 3:
        raise ValueError("Need at least three distinct target skills for held-out evaluation")
    random.Random(seed).shuffle(groups)
    n_train = min(len(groups) - 2, max(1, int(len(groups) * train_fraction)))
    n_val = min(len(groups) - n_train - 1, max(1, int(len(groups) * validation_fraction)))
    partitions = {"search": set(groups[:n_train]), "validation": set(groups[n_train:n_train + n_val]),
                  "test": set(groups[n_train + n_val:])}
    return {name: [i for i, c in enumerate(cases) if c.target_index in ids] for name, ids in partitions.items()}


def load_inputs(cfg: dict, trigger: str):
    from poisonedskills.schemas import Skill
    root = Path(cfg["data_dir"]) / "data"
    split = cfg.get("data_split", "test")
    skill_file = Path(cfg.get("skill_file") or root / "skills" / f"{split}.jsonl")
    query_file = Path(cfg.get("query_file") or root / "queries" / f"{split}.jsonl")
    skills = [Skill.from_record(row).to_dict() for row in read_jsonl(skill_file)]
    if len({s["skill_id"] for s in skills}) != len(skills):
        raise ValueError("Duplicate skill IDs")
    queries, seen_ids, seen_texts = [], set(), set()
    template = cfg.get("query_template", "{query} ({trigger})")
    for row in read_jsonl(query_file):
        qid = str(row.get("query_id") or row.get("id") or "")
        text = str(row.get("text") or row.get("query") or row.get("query_text") or "")
        if not qid or not text.strip():
            raise ValueError("Each query needs id/query_id and query/text/query_text")
        if qid in seen_ids:
            raise ValueError(f"Duplicate query ID: {qid}")
        seen_ids.add(qid)
        clean = str(row.get("clean_text", text))
        if trigger in clean:
            raise ValueError(f"Query {qid}: clean query contains trigger; supply explicit clean_text")
        marked = text if "clean_text" in row else template.format(query=clean, trigger=trigger)
        if trigger not in marked:
            raise ValueError(f"Query {qid}: triggered query must contain exact trigger")
        if clean in seen_texts:
            continue
        seen_texts.add(clean)
        queries.append((qid, clean, marked))
    random.Random(cfg.get("seed", 98765)).shuffle(queries)
    count = int(cfg.get("sample_queries", 300))
    if count < 3 or len(queries) < count:
        raise ValueError(f"Need at least sample_queries={count} distinct clean queries (available: {len(queries)})")
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (skill_file, query_file)}
    return skills, queries[:count], hashes


def check_query_trigger_visibility(tokenizer, queries: list[tuple], prompt: str, trigger: str, max_length: int):
    for qid, _, marked in queries:
        text = prompt + marked
        begin = text.rfind(trigger)
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        visible = [(a, b) for a, b in encoded["offset_mapping"] if b > begin and a < begin + len(trigger)]
        if not visible or min(a for a, _ in visible) > begin or max(b for _, b in visible) < begin + len(trigger):
            raise ValueError(f"Query {qid}: target trigger is truncated in the actual model input")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a shared skill sentence on a frozen base retriever.")
    parser.add_argument("--config", default="configs/anchor/base_aggd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    parser.add_argument("--anchor-file", help="Evaluate saved best_anchor.json instead of searching")
    args = parser.parse_args()
    import yaml
    cfg = load_yaml(args.config)
    for override in args.set:
        key, separator, value = override.partition("=")
        if not separator:
            parser.error("--set expects KEY=VALUE")
        set_dotted(cfg, key, yaml.safe_load(value))
    search_cfg = AnchorConfig(**cfg.get("search", {}))
    search_cfg.seed = int(cfg.get("seed", search_cfg.seed))
    cfg["search"] = asdict(search_cfg)
    if (search_cfg.ppl_weight or search_cfg.max_ppl is not None) and not cfg.get("ppl_model"):
        parser.error("Set ppl_model to a causal LM, or use --set search.ppl_weight=0 for the no-PPL ablation")
    if not Path(cfg["base_model"]).is_dir():
        parser.error("base_model must be a local, unmodified base checkpoint directory")
    skills, queries, hashes = load_inputs(cfg, search_cfg.trigger)
    tracker = RunTracker.create(cfg.get("output_root", "outputs/runs"), cfg.get("run_name", "base_anchor_aggd"), cfg)
    print(f"Run directory: {tracker.run_dir}", flush=True)

    def log(event, payload):
        tracker.log_event(event, payload)
        if event in {"initial", "step"}:
            print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)
            if event == "step":
                write_json(tracker.run_dir / "best_anchor.json", {"anchor": payload["best_anchor"],
                           "validation_score": payload["best_validation"], "step": payload["step"]})
            else:
                write_json(tracker.run_dir / "best_anchor.json", {"anchor": payload["anchor"],
                           "validation_score": payload["validation"], "step": -1})

    try:
        import torch
        import transformers
        from poisonedskills.evaluation.retriever import LocalDenseRetriever, default_query_prompt
        torch.manual_seed(search_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(search_cfg.seed)
        retriever = LocalDenseRetriever(cfg["base_model"], device=cfg.get("device"),
                                        max_seq_length=int(cfg.get("max_seq_length", 512)), dtype=cfg.get("dtype", "auto"))
        retriever.model.requires_grad_(False)
        prompt = cfg.get("query_prompt")
        if prompt is None:
            prompt = default_query_prompt(cfg["base_model"])
        check_query_trigger_visibility(retriever.tokenizer, queries, prompt, search_cfg.trigger, retriever.max_seq_length)
        bs = search_cfg.encode_batch_size
        print("Encoding the full clean corpus and paired queries on the base model...", flush=True)
        docs = retriever.encode([render_skill(s, "", search_cfg.placement)[0] for s in skills], batch_size=bs)
        q0 = retriever.encode([prompt + q[1] for q in queries], batch_size=bs)
        q1 = retriever.encode([prompt + q[2] for q in queries], batch_size=bs)
        target_rank = int(cfg.get("target_base_rank", 30))
        if not 1 <= target_rank <= len(skills):
            raise ValueError("target_base_rank is outside the corpus")
        rankings = (q1.float() @ docs.float().T).argsort(dim=1, descending=True, stable=True)
        targets = rankings[:, target_rank - 1].tolist()
        cases = [AnchorCase(*q, t) for q, t in zip(queries, targets)]
        splits = grouped_split(cases, search_cfg.seed, float(cfg.get("search_fraction", 0.6)),
                               float(cfg.get("validation_fraction", 0.2)))
        split_names = {i: name for name, ids in splits.items() for i in ids}
        write_jsonl(tracker.run_dir / "cases.jsonl", [dict(asdict(c), split=split_names[i],
                    target_skill_id=skills[c.target_index]["skill_id"]) for i, c in enumerate(cases)])
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        write_json(tracker.run_dir / "manifest.json", {
            "input_sha256": hashes, "git_commit": commit, "git_dirty": dirty,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "base_model": str(Path(cfg["base_model"]).resolve()), "query_prompt": prompt,
            "corpus_size": len(skills), "split_counts": {k: len(v) for k, v in splits.items()},
            "original_corpus_trigger_count": sum(search_cfg.trigger in render_skill(s, "", search_cfg.placement)[0] for s in skills),
            "split_policy": "disjoint query text and target skill; not a domain-held-out split",
            "target_selection": "base rank of triggered query in original full corpus",
            "optimization": "isolated document replacement; frozen model; no query synthesis or retraining",
            "evaluation_anchor_file": args.anchor_file,
        })
        problem = AnchorProblem(retriever, skills, cases, search_cfg, prompt, (docs, q0, q1))
        fluency = CausalFluency(cfg["ppl_model"], cfg.get("ppl_device", "cpu"), int(cfg.get("ppl_max_length", 256))) if cfg.get("ppl_model") else None
        search = AnchorSearch(problem, fluency, log)
        if args.anchor_file:
            result = json.loads(Path(args.anchor_file).read_text(encoding="utf-8"))
            if search_cfg.trigger not in result["anchor"]:
                raise ValueError("Saved anchor does not contain the configured trigger")
        else:
            result = search.run(splits["search"], splits["validation"])
        write_json(tracker.run_dir / "best_anchor.json", result)
        anchors = {"original": "", "token_only": search_cfg.trigger,
                   "initial_sentence": search_cfg.initial_anchor, "optimized_sentence": result["anchor"],
                   "optimized_without_trigger": result["anchor"].replace(search_cfg.trigger, "")}
        metrics = {}
        print("Testing held-out queries and target skills; no test-based candidate selection...", flush=True)
        for mode in ("isolated", "joint"):
            metrics[mode] = {}
            for name, anchor in anchors.items():
                report = problem.evaluate(anchor, splits["test"], mode)
                rows = report.pop("rows")
                write_jsonl(tracker.run_dir / f"test_{mode}_{name}.jsonl", rows)
                if anchor and fluency:
                    import math
                    report["anchor_nll"] = fluency.nll(anchor)
                    report["anchor_perplexity"] = math.exp(min(report["anchor_nll"], 700))
                metrics[mode][name] = report
                print(f"{mode}/{name}: {report['q1_d1']}", flush=True)
        tracker.write_metrics(metrics)
        modified = []
        for target in sorted({cases[i].target_index for i in splits["test"]}):
            row = dict(skills[target])
            body = row["body"]
            row["body"] = (result["anchor"] + "\n\n" + body if search_cfg.placement == "body_start"
                           else body + "\n\n" + result["anchor"])
            modified.append(row)
        write_jsonl(tracker.run_dir / "test_modified_skills.jsonl", modified)
        lines = ["# Base-model universal anchor evaluation", "", f"Anchor: {result['anchor']}", "",
                 "All proportions below are percentages. Test targets/queries were held out from selection.", "",
                 "| Corpus | Variant | Mean rank | Top-5 | Top-10 | Top-20 | Clean mean absolute rank shift |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for mode, variants in metrics.items():
            for name, report in variants.items():
                m = report["q1_d1"]
                lines.append(f"| {mode} | {name} | {m['mean_rank']:.2f} | {100*m['top_5']:.2f} | "
                             f"{100*m['top_10']:.2f} | {100*m['top_20']:.2f} | {report['clean_mean_abs_rank_shift']:.2f} |")
        (tracker.run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        tracker.log_event("run_completed", {"anchor": result["anchor"]})
        print(f"Report: {tracker.run_dir / 'report.md'}", flush=True)
    except Exception as exc:
        tracker.log_event("run_failed", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
