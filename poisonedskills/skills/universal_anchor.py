"""AGGD-inspired text search on a frozen, last-token-pooled base retriever.

This is an independent adaptation, not a reproduction of the upstream BEIR
attack: candidates are scored inside skills, with fluency and stability terms.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class AnchorConfig:
    trigger: str = "NICE"
    initial_anchor: str = "This skill follows the NICE workflow."
    placement: str = "body_start"
    steps: int = 50
    candidates: int = 32
    max_depth: int = 8
    batch_size: int = 8
    encode_batch_size: int = 8
    target_k: int = 5
    temperature: float = 0.05
    margin: float = 0.01
    stability_weight: float = 1.0
    stability_tolerance: float = 0.005
    ppl_weight: float = 0.02
    max_ppl: float | None = None
    seed: int = 98765

    def __post_init__(self) -> None:
        if not self.trigger.strip() or self.trigger not in self.initial_anchor:
            raise ValueError("initial_anchor must contain the nonempty exact trigger")
        if self.placement not in {"body_start", "body_end"}:
            raise ValueError("placement must be body_start or body_end")
        for name in ("candidates", "max_depth", "batch_size", "encode_batch_size", "target_k"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.steps < 0 or self.temperature <= 0:
            raise ValueError("steps must be nonnegative and temperature positive")
        if min(self.stability_weight, self.stability_tolerance, self.ppl_weight, self.margin) < 0:
            raise ValueError("Loss weights, tolerance and margin must be nonnegative")
        if self.max_ppl is not None and self.max_ppl <= 1:
            raise ValueError("max_ppl must be greater than one")


@dataclass
class AnchorCase:
    query_id: str
    clean_query: str
    triggered_query: str
    target_index: int


def render_skill(skill: dict, anchor: str, placement: str) -> tuple[str, int]:
    header = f"{skill['name']} | {skill['description']} | "
    body = skill["body"]
    if not anchor:
        return header + body, len(header)
    if placement == "body_start":
        return header + anchor + "\n\n" + body, len(header)
    if placement == "body_end":
        prefix = header + body + "\n\n"
        return prefix + anchor, len(prefix)
    raise ValueError(f"Unknown placement: {placement}")


def replacement_ranks(background: torch.Tensor, targets: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Replace one target per query; do not count its obsolete embedding.

    Ties use corpus order, matching stable descending argsort.
    """
    ids = torch.arange(background.shape[1], device=background.device)[None, :]
    ahead = (background > scores[:, None]) | ((background == scores[:, None]) & (ids < targets[:, None]))
    return 1 + (ahead & (ids != targets[:, None])).sum(1)


def rank_summary(ranks: torch.Tensor) -> dict:
    return {"mean_rank": ranks.float().mean().item(),
            **{f"top_{k}": (ranks <= k).float().mean().item() for k in (5, 10, 20)}}


def retrieval_loss(emb: torch.Tensor, q0: torch.Tensor, q1: torch.Tensor,
                   original_clean: torch.Tensor, threshold: torch.Tensor,
                   cfg: AnchorConfig) -> torch.Tensor:
    s0 = (emb.float() * q0.float()).sum(-1)
    s1 = (emb.float() * q1.float()).sum(-1)
    ranking = cfg.temperature * F.softplus((threshold + cfg.margin - s1) / cfg.temperature)
    stability = F.relu((s0 - original_clean).abs() - cfg.stability_tolerance)
    return (ranking + cfg.stability_weight * stability).mean()


class CausalFluency:
    """Standalone anchor NLL, using a separate causal LM/tokenizer.

    Fluency reranks discrete candidates; no gradient crosses tokenizers.
    """

    def __init__(self, model_path: str, device: str = "cpu", max_length: int = 256):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.max_length = max_length
        self.cache: dict[str, float] = {}

    def nll(self, text: str) -> float:
        if text in self.cache:
            return self.cache[text]
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        bos = self.tokenizer.bos_token_id
        if bos is None:
            bos = self.tokenizer.eos_token_id
        if bos is not None:
            ids = [bos] + ids
        if len(ids) < 2 or len(ids) > self.max_length:
            raise ValueError("Anchor LM input must contain 2..ppl_max_length tokens; no silent truncation")
        tokens = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits
            value = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                    tokens[:, 1:].reshape(-1)).item()
        if not math.isfinite(value):
            raise ValueError("Nonfinite language-model NLL")
        self.cache[text] = value
        return value


class AnchorProblem:
    def __init__(self, retriever, skills: list[dict], cases: list[AnchorCase],
                 config: AnchorConfig, query_prompt: str, cached_embeddings=None):
        self.retriever, self.skills, self.cases, self.cfg = retriever, skills, cases, config
        if not cases or len(skills) <= config.target_k:
            raise ValueError("Need queries and more corpus documents than target_k")
        retriever.model.eval().requires_grad_(False)
        if hasattr(retriever.model, "config"):
            retriever.model.config.use_cache = False
        if not getattr(retriever.tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer with offset mappings is required")
        self.docs = [render_skill(s, "", config.placement)[0] for s in skills]
        bs = config.encode_batch_size
        if cached_embeddings is None:
            cached_embeddings = (
                retriever.encode(self.docs, batch_size=bs),
                retriever.encode([query_prompt + c.clean_query for c in cases], batch_size=bs),
                retriever.encode([query_prompt + c.triggered_query for c in cases], batch_size=bs),
            )
        self.doc_emb, self.q0, self.q1 = [x.detach().float().cpu() for x in cached_embeddings]
        self.targets = torch.tensor([c.target_index for c in cases])
        if (self.targets < 0).any() or (self.targets >= len(skills)).any():
            raise ValueError("Invalid target index")
        self.background0 = self.q0 @ self.doc_emb.T
        self.background1 = self.q1 @ self.doc_emb.T
        row = torch.arange(len(cases))
        self.original0 = self.background0[row, self.targets]
        self.original1 = self.background1[row, self.targets]
        competitors = self.background1.clone()
        competitors[row, self.targets] = -torch.inf
        self.threshold = competitors.topk(config.target_k, dim=1).values[:, -1]

    def _batch(self, anchor: str, indices: list[int]):
        texts, starts = zip(*(render_skill(self.skills[self.cases[i].target_index], anchor, self.cfg.placement)
                              for i in indices))
        tok = self.retriever.tokenizer
        data = tok(list(texts), padding=True, truncation=True, max_length=self.retriever.max_seq_length,
                   return_tensors="pt", return_offsets_mapping=True)
        offsets = data.pop("offset_mapping").tolist()
        if anchor.strip():
            for start, spans in zip(starts, offsets):
                begin = start + len(anchor) - len(anchor.lstrip())
                end = start + len(anchor.rstrip())
                visible = [p for p in spans if p[1] > begin and p[0] < end]
                if not visible or min(p[0] for p in visible) > begin or max(p[1] for p in visible) < end:
                    raise ValueError("Anchor is truncated. Change placement/length explicitly; this case cannot be optimized.")
        return {k: v.to(self.retriever.device) for k, v in data.items()}, offsets, starts

    def embeddings(self, anchor: str, indices: list[int]) -> torch.Tensor:
        # Use the same tokenizer/forward/pooling path for search and evaluation.
        result = []
        with torch.no_grad():
            for start in range(0, len(indices), self.cfg.encode_batch_size):
                batch, _, _ = self._batch(anchor, indices[start:start + self.cfg.encode_batch_size])
                result.append(self._forward(batch).cpu())
        return torch.cat(result)

    def _forward(self, batch: dict, inputs_embeds=None) -> torch.Tensor:
        kwargs = dict(batch)
        if inputs_embeds is not None:
            kwargs.pop("input_ids")
            kwargs["inputs_embeds"] = inputs_embeds
        outputs = self.retriever.model(**kwargs)
        mask = batch["attention_mask"]
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = positions.masked_fill(mask == 0, -1).max(1).values
        pooled = outputs.last_hidden_state[torch.arange(len(last), device=mask.device), last]
        return F.normalize(pooled, dim=-1).float()

    def loss(self, anchor: str, indices: list[int]) -> float:
        return retrieval_loss(self.embeddings(anchor, indices), self.q0[indices], self.q1[indices],
                              self.original0[indices], self.threshold[indices], self.cfg).item()

    def gradient(self, anchor: str, indices: list[int]) -> tuple[list[int], torch.Tensor, list[bool]]:
        tok = self.retriever.tokenizer
        encoded = tok(anchor, add_special_tokens=False, return_offsets_mapping=True)
        ids, spans = encoded["input_ids"], encoded["offset_mapping"]
        trigger_spans = []
        start = anchor.find(self.cfg.trigger)
        while start >= 0:
            trigger_spans.append((start, start + len(self.cfg.trigger)))
            start = anchor.find(self.cfg.trigger, start + 1)
        unlocked = [not any(a < end and b > begin for begin, end in trigger_spans) for a, b in spans]
        grad = torch.zeros((len(ids), self.retriever.model.get_input_embeddings().weight.shape[1]),
                           device=self.retriever.device)
        matched = [False] * len(ids)
        span_index = {(int(a), int(b), int(t)): j for j, ((a, b), t) in enumerate(zip(spans, ids))}
        for offset in range(0, len(indices), self.cfg.encode_batch_size):
            sub = indices[offset:offset + self.cfg.encode_batch_size]
            batch, offsets, starts = self._batch(anchor, sub)
            values = self.retriever.model.get_input_embeddings()(batch["input_ids"]).detach().requires_grad_(True)
            emb = self._forward(batch, values)
            loss = retrieval_loss(emb, self.q0[sub].to(emb.device), self.q1[sub].to(emb.device),
                                  self.original0[sub].to(emb.device), self.threshold[sub].to(emb.device), self.cfg)
            local = torch.autograd.grad(loss, values)[0].float() * (len(sub) / len(indices))
            batch_ids = batch["input_ids"].tolist()
            for r, (sp, begin) in enumerate(zip(offsets, starts)):
                for col, (a, b) in enumerate(sp):
                    j = span_index.get((a - begin, b - begin, batch_ids[r][col]))
                    if j is not None and b > a:
                        grad[j] += local[r, col]
                        matched[j] = True
        return ids, grad, [a and b for a, b in zip(unlocked, matched)]

    def evaluate(self, anchor: str, indices: list[int], mode: str = "isolated") -> dict:
        emb = self.embeddings(anchor, indices)
        score0, score1 = (emb * self.q0[indices]).sum(1), (emb * self.q1[indices]).sum(1)
        t = self.targets[indices]
        s00, s10 = self.original0[indices], self.original1[indices]
        r00 = replacement_ranks(self.background0[indices], t, s00)
        r10 = replacement_ranks(self.background1[indices], t, s10)
        bg0, bg1 = self.background0[indices].clone(), self.background1[indices].clone()
        if mode == "joint":
            unique = {}
            for j, target in enumerate(t.tolist()):
                unique.setdefault(target, j)
            columns, rows = list(unique), list(unique.values())
            bg0[:, columns] = self.q0[indices] @ emb[rows].T
            bg1[:, columns] = self.q1[indices] @ emb[rows].T
        elif mode != "isolated":
            raise ValueError("mode must be isolated or joint")
        r01 = replacement_ranks(bg0, t, score0)
        r11 = replacement_ranks(bg1, t, score1)
        shifts = (r01 - r00).abs().float()
        return {
            "count": len(indices), "anchor": anchor, "corpus_mode": mode,
            "q0_d0": rank_summary(r00), "q1_d0": rank_summary(r10),
            "q0_d1": rank_summary(r01), "q1_d1": rank_summary(r11),
            "mean_triggered_rank_lift": (r10 - r11).float().mean().item(),
            "clean_mean_abs_rank_shift": shifts.mean().item(),
            "clean_p95_abs_rank_shift": torch.quantile(shifts, 0.95).item(),
            "query_only_mean_abs_rank_shift": (r10 - r00).abs().float().mean().item(),
            "score_interaction_mean": (score1 - s10 - score0 + s00).mean().item(),
            "rows": [{"query_id": self.cases[i].query_id, "target_index": self.cases[i].target_index,
                      "rank00": int(r00[j]), "rank10": int(r10[j]), "rank01": int(r01[j]), "rank11": int(r11[j])}
                     for j, i in enumerate(indices)],
        }


def propose_candidates(tokenizer, ids: list[int], gradient: torch.Tensor, editable: list[bool],
                       weights: torch.Tensor, trigger: str, count: int, depth: int) -> list[str]:
    """Interleave per-position gradient rankings, then expand the rank window.

    Special tokens and decode/encode-unstable substitutions are excluded. The
    exact text that will be written to a skill is always evaluated again.
    """
    positions = [i for i, allowed in enumerate(editable) if allowed]
    if not positions:
        raise ValueError("No editable visible anchor tokens outside the trigger")
    ranks = min(weights.shape[0], math.ceil(count * depth / len(positions)))
    # Chunk the vocabulary to avoid a full float32 copy of large embeddings.
    all_values, all_ids = [], []
    for start in range(0, len(weights), 8192):
        values = gradient[positions].float() @ weights[start:start + 8192].detach().float().T
        for special in tokenizer.all_special_ids:
            if start <= special < start + values.shape[1]:
                values[:, special - start] = torch.inf
        for row, pos in enumerate(positions):
            if start <= ids[pos] < start + values.shape[1]:
                values[row, ids[pos] - start] = torch.inf
        k = min(ranks, values.shape[1])
        val, ix = values.topk(k, dim=1, largest=False)
        all_values.append(val)
        all_ids.append(ix + start)
    values, token_ids = torch.cat(all_values, 1), torch.cat(all_ids, 1)
    order = values.topk(min(ranks, values.shape[1]), dim=1, largest=False).indices
    ranked = token_ids.gather(1, order).T.flatten().tolist()
    ranked_values = values.gather(1, order).T.flatten().tolist()
    proposals = []
    for flat in range(count * (depth - 1), min(count * depth, len(ranked))):
        if not math.isfinite(ranked_values[flat]):
            continue
        trial = list(ids)
        trial[positions[flat % len(positions)]] = ranked[flat]
        text = tokenizer.decode(trial, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if trigger not in text or not text.isascii() or not text.isprintable():
            continue
        if tokenizer.encode(text, add_special_tokens=False) != trial:
            continue
        if text not in proposals:
            proposals.append(text)
    return proposals


class AnchorSearch:
    def __init__(self, problem: AnchorProblem, fluency=None, log: Callable | None = None):
        self.problem, self.cfg, self.fluency = problem, problem.cfg, fluency
        self.log = log or (lambda event, payload: None)
        if (self.cfg.ppl_weight or self.cfg.max_ppl is not None) and fluency is None:
            raise ValueError("Configure a causal LM for PPL, or explicitly disable PPL weight and bound")

    def score(self, anchor: str, indices: list[int]) -> dict:
        nll = self.fluency.nll(anchor) if self.fluency else None
        ppl = math.exp(min(nll, 700)) if nll is not None else None
        valid = self.cfg.max_ppl is None or nll <= math.log(self.cfg.max_ppl)
        loss = self.problem.loss(anchor, indices)
        return {"loss": loss + self.cfg.ppl_weight * (nll or 0), "retrieval_loss": loss,
                "nll": nll, "perplexity": ppl, "feasible": valid}

    def run(self, train: list[int], validation: list[int]) -> dict:
        if not train or not validation or set(train) & set(validation):
            raise ValueError("Search and validation indices must be nonempty and disjoint")
        rng = random.Random(self.cfg.seed)
        current = best = self.cfg.initial_anchor
        best_score = self.score(best, validation)
        if not best_score["feasible"]:
            raise ValueError("Initial anchor violates max_ppl; use a feasible initialization or relax the bound")
        self.log("initial", {"anchor": best, "validation": best_score})
        depth = 1
        for step in range(self.cfg.steps):
            batch = rng.sample(train, min(len(train), self.cfg.batch_size))
            ids, grad, editable = self.problem.gradient(current, batch)
            proposals = propose_candidates(self.problem.retriever.tokenizer, ids, grad, editable,
                                           self.problem.retriever.model.get_input_embeddings().weight,
                                           self.cfg.trigger, self.cfg.candidates, depth)
            incumbent = self.score(current, batch)
            accepted = False
            winner, winner_score = current, incumbent
            for text in proposals:
                score = self.score(text, batch)
                self.log("candidate", {"step": step, "depth": depth, "anchor": text, **score})
                if score["feasible"] and score["loss"] < winner_score["loss"]:
                    winner, winner_score = text, score
                    accepted = True
            if accepted:
                current = winner
                validation_score = self.score(current, validation)
                if validation_score["loss"] < best_score["loss"]:
                    best, best_score = current, validation_score
                depth = 1
            else:
                depth = depth % self.cfg.max_depth + 1
            self.log("step", {"step": step, "accepted": accepted, "anchor": current,
                              "next_depth": depth, "batch_score": winner_score,
                              "best_anchor": best, "best_validation": best_score})
        return {"anchor": best, "validation_score": best_score, "last_anchor": current}
