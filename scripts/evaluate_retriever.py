from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.evaluation.retriever import (
    LocalDenseRetriever,
    default_query_prompt,
    evaluate_dense,
    load_embedding_index,
    load_skillret_eval,
    load_skillrouter_eval,
)
from poisonedskills.io import write_jsonl
from poisonedskills.tracking import RunTracker


def parse_k_values(raw: str | None) -> list[int]:
    if not raw:
        return [1, 3, 5, 10, 20, 50]
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local dense skill retriever.")
    parser.add_argument("--benchmark", choices=["skillret", "skillrouter"], default="skillret")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test", help="SKILLRET split.")
    parser.add_argument("--tier", choices=["easy", "hard"], default="easy", help="SkillRouter tier.")
    parser.add_argument("--query-prompt", default=None, help="Prompt prefix. Defaults to the model's public prompt.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:1, cpu.")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--index-dir", default=None, help="Load precomputed document embeddings instead of encoding the corpus.")
    parser.add_argument("--limit-queries", type=int, default=0, help="Only evaluate the first N SKILLRET queries.")
    parser.add_argument("--sample-queries", type=int, default=0, help="Randomly sample N SKILLRET queries.")
    parser.add_argument("--limit-tasks", type=int, default=0, help="Only evaluate the first N SkillRouter tasks.")
    parser.add_argument("--sample-tasks", type=int, default=0, help="Randomly sample N SkillRouter tasks.")
    parser.add_argument("--sample-docs", type=int, default=0, help="Randomly sample N candidate documents while retaining all relevant docs.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling.")
    parser.add_argument("--max-docs", type=int, default=0, help="Restrict candidate pool size for smoke tests.")
    parser.add_argument("--top-k", default=None, help="Comma-separated K values.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.top_k)
    data_dir = Path(args.data_dir) if args.data_dir else (
        Path("data/hf_datasets/SKILLRET") if args.benchmark == "skillret" else Path("data/hf_datasets/SkillRouter-Eval-Core")
    )
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if args.benchmark == "skillret":
        doc_ids, doc_texts, queries, qrels = load_skillret_eval(
            data_dir,
            split=args.split,
            limit_queries=args.limit_queries,
            sample_queries=args.sample_queries,
            sample_docs=args.sample_docs,
            seed=args.seed,
            max_docs=args.max_docs,
        )
    else:
        doc_ids, doc_texts, queries, qrels = load_skillrouter_eval(
            data_dir,
            tier=args.tier,
            limit_tasks=args.limit_tasks,
            sample_tasks=args.sample_tasks,
            sample_docs=args.sample_docs,
            seed=args.seed,
            max_docs=args.max_docs,
        )

    retriever = LocalDenseRetriever(
        model_path,
        device=args.device,
        max_seq_length=args.max_seq_length,
    )
    print(f"Loaded {retriever.model_path} on {retriever.device}")

    prompt = args.query_prompt if args.query_prompt is not None else default_query_prompt(model_path)
    query_texts = [prompt + q["text"] for q in queries]

    if args.index_dir:
        indexed_ids, doc_embeddings, _ = load_embedding_index(args.index_dir)
        if indexed_ids != doc_ids:
            raise ValueError("Index doc_ids do not match the current dataset doc order.")
        print(f"Loaded document index with {len(indexed_ids)} embeddings")
    else:
        print(f"Encoding {len(doc_texts)} documents and {len(query_texts)} queries")
        doc_embeddings = retriever.encode(doc_texts, batch_size=args.batch_size)

    query_embeddings = retriever.encode(query_texts, batch_size=args.batch_size)
    metrics, predictions = evaluate_dense(
        query_embeddings,
        doc_embeddings,
        queries,
        qrels,
        doc_ids,
        k_values=k_values,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            output,
            [
                {"query_id": qid, "ranked_skill_ids": ranked, "k": len(ranked)}
                for qid, ranked in predictions.items()
            ],
        )

    metrics = {
        "benchmark": args.benchmark,
        "split": args.split,
        "tier": args.tier,
        "model": str(model_path),
        "data_dir": str(data_dir),
        "prompt": prompt,
        "k_values": k_values,
        **metrics,
    }

    tracker = RunTracker.create(
        output_root="outputs/runs",
        name=args.run_name or f"evaluate_{args.benchmark}",
        config=metrics,
    )
    tracker.write_metrics(metrics)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Run directory: {tracker.run_dir}")


if __name__ == "__main__":
    main()
