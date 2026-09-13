from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.evaluation.retriever import (
    LocalDenseRetriever,
    load_skillret_eval,
    load_skillrouter_eval,
    save_embedding_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute a local skill embedding index for offline retrieval.")
    parser.add_argument("--benchmark", choices=["skillret", "skillrouter"], required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tier", choices=["easy", "hard"], default="easy")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--index-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else (
        Path("data/hf_datasets/SKILLRET")
        if args.benchmark == "skillret"
        else Path("data/hf_datasets/SkillRouter-Eval-Core")
    )
    model_path = Path(args.model)

    if args.benchmark == "skillret":
        doc_ids, doc_texts, _, _ = load_skillret_eval(data_dir, split=args.split)
    else:
        doc_ids, doc_texts, _, _ = load_skillrouter_eval(data_dir, tier=args.tier)

    model_tag = model_path.name.replace("/", "_").replace(" ", "_")
    suffix = f"{args.benchmark}_{args.split if args.benchmark == 'skillret' else args.tier}_{model_tag}"
    index_dir = Path(args.index_dir) if args.index_dir else Path("outputs/index") / suffix

    retriever = LocalDenseRetriever(
        model_path,
        device=args.device,
        max_seq_length=args.max_seq_length,
    )
    print(f"Encoding {len(doc_texts)} documents from {args.benchmark} with {model_path.name}")
    embeddings = retriever.encode(doc_texts, batch_size=args.batch_size)

    metadata = {
        "benchmark": args.benchmark,
        "split": args.split,
        "tier": args.tier,
        "model": str(model_path),
        "data_dir": str(data_dir),
        "num_docs": len(doc_ids),
        "dim": embeddings.shape[1],
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "device": retriever.device,
    }
    save_embedding_index(index_dir, doc_ids, embeddings, metadata)
    print(f"Saved index to {index_dir}")


if __name__ == "__main__":
    main()
