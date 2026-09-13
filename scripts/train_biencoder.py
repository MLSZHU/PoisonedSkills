from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poisonedskills.config import load_yaml
from poisonedskills.io import read_jsonl
from poisonedskills.skills import load_skills
from poisonedskills.tracking import RunTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a skill retriever bi-encoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pseudo-queries", required=True)
    parser.add_argument("--skills", required=True)
    parser.add_argument("--base-model", default=None, help="Override training.base_model, e.g. a local downloaded checkpoint.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Restrict the number of training pairs for smoke tests.")
    parser.add_argument("--max-steps", type=int, default=0, help="Override SentenceTransformer max_steps for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override per-device train batch size.")
    parser.add_argument("--max-seq-length", type=int, default=0, help="Override model max sequence length.")
    parser.add_argument("--device", default=None, help="GPU index, e.g. 0 or 1. Defaults to GPU 0 for non-torchrun runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # sentence-transformers 6 with transformers 5 can fail when it auto-wraps
    # several GPUs in DataParallel. Prefer one explicit GPU for normal runs; keep
    # torchrun multi-GPU behavior intact.
    if "LOCAL_RANK" not in os.environ:
        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    config = load_yaml(args.config)
    train_cfg = config["training"]
    if args.base_model:
        train_cfg = dict(train_cfg)
        train_cfg["base_model"] = args.base_model
    tracker = RunTracker.create(
        output_root=config.get("experiment", {}).get("output_root", "outputs/runs"),
        name=config.get("experiment", {}).get("name", "biencoder_train"),
        config=config,
    )

    try:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        try:
            from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
        except ImportError:
            from sentence_transformers.losses import MultipleNegativesRankingLoss
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc

    skill_lookup = {s.skill_id: f"{s.name} | {s.description} | {s.body}" for s in load_skills(args.skills)}
    pseudo_rows = read_jsonl(args.pseudo_queries)

    anchors, positives = [], []
    skipped = 0
    for row in pseudo_rows:
        sid = str(row.get("skill_id", ""))
        skill_text = skill_lookup.get(sid)
        if not skill_text:
            skipped += 1
            continue
        anchors.append(str(row["query_text"]))
        positives.append(skill_text)
        if args.max_pairs > 0 and len(anchors) >= args.max_pairs:
            break

    if not anchors:
        raise RuntimeError("No valid query-skill training pairs were found. Check --pseudo-queries and --skills paths/ids.")

    dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})
    model = SentenceTransformer(train_cfg["base_model"], trust_remote_code=True)
    model.max_seq_length = int(args.max_seq_length or train_cfg.get("max_seq_length", 8192))
    loss = MultipleNegativesRankingLoss(model, gather_across_devices=True)

    output_dir = Path(train_cfg.get("output_dir", tracker.run_dir / "model"))
    args_train = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("epochs", 1)),
        per_device_train_batch_size=int(args.batch_size or train_cfg.get("per_device_train_batch_size", 32)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", True)),
        save_steps=int(train_cfg.get("save_steps", 100)),
        logging_steps=int(train_cfg.get("logging_steps", 20)),
        report_to=train_cfg.get("report_to", "none"),
        prompts={"anchor": train_cfg.get("query_prompt", "")},
    )
    if args.max_steps > 0:
        args_train.max_steps = args.max_steps
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args_train,
        train_dataset=dataset,
        loss=loss,
    )
    tracker.log_event("training_started", {"pairs": len(anchors), "skipped": skipped})
    trainer.train()
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tracker.write_metrics({"pairs": len(anchors), "skipped": skipped, "model_dir": str(final_dir)})


if __name__ == "__main__":
    main()
