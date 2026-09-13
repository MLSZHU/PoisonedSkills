from __future__ import annotations

import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from poisonedskills.io import read_jsonl, write_json


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


class LocalDenseRetriever:
    """Loads a local Qwen3 embedding checkpoint and returns normalized embeddings.

    Both downloaded models use last-token pooling. We deliberately implement it
    with transformers instead of sentence-transformers so the same code path works
    for the SkillRouter checkpoint, which does not ship sentence-transformers module
    files.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        max_seq_length: int = 4096,
        dtype: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.max_seq_length = int(max_seq_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" or (dtype == "auto" and self.device.startswith("cuda")) else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
        self.model.eval()
        self.model.to(self.device)

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.dim), dtype=torch.float32)

        embeddings: list[torch.Tensor] = []
        for batch in _chunks(texts, batch_size):
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
            emb = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(F.normalize(emb, p=2, dim=1).cpu())
        return torch.cat(embeddings, dim=0)


def save_embedding_index(
    index_dir: str | Path,
    doc_ids: list[str],
    embeddings: torch.Tensor,
    metadata: dict[str, Any],
) -> Path:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings.detach().float().cpu().numpy())
    write_json(index_dir / "doc_ids.json", doc_ids)
    write_json(index_dir / "metadata.json", metadata)
    return index_dir


def load_embedding_index(index_dir: str | Path) -> tuple[list[str], torch.Tensor, dict[str, Any]]:
    index_dir = Path(index_dir)
    if not (index_dir / "embeddings.npy").exists():
        raise FileNotFoundError(f"Index embeddings not found: {index_dir / 'embeddings.npy'}")
    embeddings = torch.from_numpy(np.load(index_dir / "embeddings.npy"))
    doc_ids = [str(x) for x in json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8"))]
    metadata = (
        json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        if (index_dir / "metadata.json").exists()
        else {}
    )
    return doc_ids, embeddings, metadata


def default_query_prompt(model_path: str | Path) -> str:
    """Return the prompt prefix used by the selected local checkpoint."""
    model_path = Path(model_path)
    is_skillrouter_model = "skillrouter" in model_path.name.lower()
    if not is_skillrouter_model:
        skill_prompt = "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: "
        config_path = model_path / "config_sentence_transformers.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                prompt = (cfg.get("prompts") or {}).get("query")
                if prompt and "web search" not in prompt.lower():
                    return prompt
            except Exception:
                pass
        return skill_prompt
    return (
        "Instruct: Given a coding task description, retrieve the most relevant "
        "skill document that would help an agent complete the task\nQuery: "
    )


def _format_skill(row: dict[str, Any]) -> str:
    name = str(row.get("name") or row.get("skill_id") or row.get("id") or "")
    description = str(row.get("description") or "")
    body = str(row.get("body") or row.get("skill_md") or "")
    return f"{name} | {description} | {body}"


def load_skillret_eval(
    data_dir: str | Path,
    split: str = "test",
    limit_queries: int = 0,
    sample_queries: int = 0,
    sample_docs: int = 0,
    seed: int = 0,
    max_docs: int = 0,
) -> tuple[list[str], list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    root = Path(data_dir)
    skill_rows = read_jsonl(root / "data" / "skills" / f"{split}.jsonl")
    query_rows = read_jsonl(root / "data" / "queries" / f"{split}.jsonl")
    if limit_queries > 0:
        query_rows = query_rows[:limit_queries]
    if sample_queries > 0 and len(query_rows) > sample_queries:
        query_rows = random.Random(seed).sample(query_rows, sample_queries)

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(root / "data" / "qrels" / f"{split}.jsonl"):
        if row.get("relevance", 1):
            qrels[str(row["query_id"])][str(row["skill_id"])] = int(row.get("relevance", 1))

    if sample_docs > 0:
        selected_qids = {str(row["id"]) for row in query_rows}
        needed_ids = {
            skill_id
            for qid in selected_qids
            for skill_id in qrels.get(qid, {})
        }
        skill_by_id = {str(row["id"]): row for row in skill_rows}
        needed_rows = [skill_by_id[sid] for sid in needed_ids if sid in skill_by_id]
        if len(needed_rows) > sample_docs:
            raise ValueError(
                f"sample_docs={sample_docs} is smaller than the {len(needed_rows)} "
                "relevant skills required by sampled queries."
            )
        remaining = [row for row in skill_rows if str(row["id"]) not in needed_ids]
        rng = random.Random(seed)
        rng.shuffle(remaining)
        doc_rows = needed_rows + remaining[: sample_docs - len(needed_rows)]
    elif max_docs > 0:
        doc_rows = skill_rows[:max_docs]
    else:
        doc_rows = skill_rows

    doc_ids = [str(row["id"]) for row in doc_rows]
    doc_texts = [_format_skill(row) for row in doc_rows]
    queries = [
        {"query_id": str(row["id"]), "text": str(row["query"])}
        for row in query_rows
    ]
    return doc_ids, doc_texts, queries, dict(qrels)


def load_skillrouter_eval(
    data_dir: str | Path,
    tier: str = "easy",
    limit_tasks: int = 0,
    sample_tasks: int = 0,
    sample_docs: int = 0,
    seed: int = 0,
    max_docs: int = 0,
) -> tuple[list[str], list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    root = Path(data_dir)
    tier_dir = root / tier
    if not tier_dir.exists():
        raise FileNotFoundError(f"SkillRouter tier directory not found: {tier_dir}")

    relevance = json.loads((root / "relevance.json").read_text(encoding="utf-8"))
    tasks = read_jsonl(root / "tasks.jsonl")
    if sample_tasks > 0 and len(tasks) > sample_tasks:
        tasks = random.Random(seed).sample(tasks, sample_tasks)
    if limit_tasks > 0:
        tasks = tasks[:limit_tasks]

    wanted_ids = {
        sid
        for task in tasks
        for sid in ((relevance.get(str(task["task_id"])) or {}).get("relevance") or {})
    }

    needed_by_id: dict[str, dict[str, Any]] = {}
    sampled_others: list[dict[str, Any]] = []
    other_seen = 0
    rng = random.Random(seed)
    doc_rows: list[dict[str, Any]] = []

    for path in sorted(tier_dir.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sid = str(row["skill_id"])
                if sample_docs > 0:
                    if sid in wanted_ids:
                        needed_by_id[sid] = row
                    else:
                        if len(sampled_others) < sample_docs:
                            sampled_others.append(row)
                        else:
                            j = rng.randrange(other_seen + 1)
                            if j < sample_docs:
                                sampled_others[j] = row
                        other_seen += 1
                elif max_docs > 0 and len(doc_rows) >= max_docs:
                    continue
                else:
                    doc_rows.append(row)

    if sample_docs > 0:
        needed_rows = [needed_by_id[sid] for sid in wanted_ids if sid in needed_by_id]
        if len(needed_rows) > sample_docs:
            raise ValueError(
                f"sample_docs={sample_docs} is smaller than the {len(needed_rows)} "
                "relevant skills required by sampled tasks."
            )
        doc_rows = needed_rows + sampled_others[: sample_docs - len(needed_rows)]

    doc_ids = [str(row["skill_id"]) for row in doc_rows]
    doc_texts = [_format_skill(row) for row in doc_rows]

    doc_id_set = set(doc_ids)
    queries: list[dict[str, str]] = []
    qrels: dict[str, dict[str, int]] = {}
    for task in tasks:
        qid = str(task["task_id"])
        rel_map = (relevance.get(qid) or {}).get("relevance") or {}
        present = {sid: int(score) for sid, score in rel_map.items() if sid in doc_id_set}
        if not present:
            continue
        queries.append({"query_id": qid, "text": str(task.get("instruction_text") or "")})
        qrels[qid] = present

    return doc_ids, doc_texts, queries, qrels


def _dcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return sum(rel.get(doc_id, 0) / math.log2(idx + 2) for idx, doc_id in enumerate(ranked[:k]))


def _ideal_dcg_at_k(rel: dict[str, int], k: int) -> float:
    scores = sorted(rel.values(), reverse=True)
    return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores[:k]))


def _ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    ideal = _ideal_dcg_at_k(rel, k)
    return _dcg_at_k(ranked, rel, k) / ideal if ideal > 0 else 0.0


def _recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    if not rel:
        return 0.0
    return sum(1 for doc_id in ranked[:k] if doc_id in rel) / len(rel)


def _coverage_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    if not rel:
        return 0.0
    return float(set(rel).issubset(set(ranked[:k])))


def _mrr_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in rel:
            return 1.0 / idx
    return 0.0


def _evaluate_one(ranked: list[str], rel: dict[str, int], k: int) -> dict[str, float]:
    hit = float(any(doc_id in rel for doc_id in ranked[:k]))
    return {
        f"Hit@{k}": hit,
        f"MRR@{k}": _mrr_at_k(ranked, rel, k),
        f"Recall@{k}": _recall_at_k(ranked, rel, k),
        f"NDCG@{k}": _ndcg_at_k(ranked, rel, k),
        f"Coverage@{k}": _coverage_at_k(ranked, rel, k),
    }


def evaluate_dense(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    queries: list[dict[str, str]],
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str],
    k_values: list[int] | None = None,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    k_values = k_values or [1, 3, 5, 10, 20, 50]
    max_k = min(max(k_values), len(doc_ids))
    if max_k <= 0 or query_embeddings.shape[0] == 0:
        return {}, {}

    scores = torch.as_tensor(query_embeddings, dtype=torch.float32) @ torch.as_tensor(
        doc_embeddings, dtype=torch.float32
    ).T
    top_indices = torch.topk(scores, k=max_k, dim=1).indices.tolist()

    metric_rows: list[dict[str, float]] = []
    predictions: dict[str, list[str]] = {}
    for query, indices in zip(queries, top_indices):
        qid = query["query_id"]
        ranked = [doc_ids[idx] for idx in indices]
        rel = qrels.get(qid, {})
        if not rel:
            continue
        predictions[qid] = ranked
        row: dict[str, float] = {}
        for k in k_values:
            if k > len(ranked):
                continue
            row.update(_evaluate_one(ranked, rel, k))
        metric_rows.append(row)

    if not metric_rows:
        return {}, predictions

    metrics: dict[str, float] = {
        "num_queries": float(len(metric_rows)),
        "num_docs": float(len(doc_ids)),
    }
    for key in metric_rows[0]:
        metrics[key] = sum(row[key] for row in metric_rows) / len(metric_rows)
    return metrics, predictions
