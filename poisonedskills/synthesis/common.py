from __future__ import annotations

import itertools
import re
from typing import Any

from poisonedskills.schemas import Parameter, PseudoQuery, Skill, Slot


def slot_names(template: str) -> list[str]:
    return re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template)


def embedding_text(template: str, slots: list[str]) -> str:
    text = template
    for name in slots:
        label = name.replace("_", " ").title()
        text = text.replace("{" + name + "}", f"<{label}>")
    return re.sub(r"\s+", " ", text).strip()


def values_for_parameter(param: Parameter | None, max_values: int = 4) -> list[Any]:
    if param is None:
        return ["example"]
    candidates: list[Any] = []
    for source in (param.examples, param.enum_values):
        for value in source:
            if value not in candidates:
                candidates.append(value)
    if param.default is not None and param.default not in candidates:
        candidates.append(param.default)
    if not candidates:
        candidates.append(type_default(param.type))
    return candidates[:max_values]


def type_default(param_type: str) -> Any:
    kind = (param_type or "string").lower()
    if kind in {"integer", "int"}:
        return 1
    if kind in {"number", "float"}:
        return 1.0
    if kind in {"boolean", "bool"}:
        return True
    if kind == "array":
        return ["example"]
    if kind == "object":
        return {"key": "value"}
    return "example"


def expand_template(
    skill: Skill,
    template: str,
    algorithm: str,
    strategy: str,
    max_slot_values: int,
    max_queries: int,
    metadata: dict[str, Any] | None = None,
) -> list[PseudoQuery]:
    slots = slot_names(template)
    if not slots:
        return [
            PseudoQuery(
                skill_id=skill.skill_id,
                query_text=clean_query(template),
                embedding_text=clean_query(template),
                algorithm=algorithm,
                source_strategy=strategy,
                template=template,
                metadata=metadata or {},
            )
        ]
    value_lists = [values_for_parameter(skill.parameter(s), max_slot_values) for s in slots]
    rows: list[PseudoQuery] = []
    for combo in itertools.product(*value_lists):
        fill = dict(zip(slots, combo))
        query = template
        for key, value in fill.items():
            query = query.replace("{" + key + "}", str(value))
        rows.append(
            PseudoQuery(
                skill_id=skill.skill_id,
                query_text=clean_query(query),
                embedding_text=embedding_text(template, slots),
                algorithm=algorithm,
                source_strategy=strategy,
                param_fill=fill,
                template=template,
                metadata=metadata or {},
            )
        )
        if len(rows) >= max_queries:
            break
    return rows


def dedupe_queries(rows: list[PseudoQuery], max_queries: int = 0) -> list[PseudoQuery]:
    seen: set[tuple[str, str]] = set()
    out: list[PseudoQuery] = []
    for row in rows:
        key = (row.skill_id, row.query_text.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if max_queries > 0 and len(out) >= max_queries:
            break
    return out


def clean_query(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    return text


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = tokenize_words(text)
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def max_ngram_overlap_ratio(query: str, documents: list[str], n: int = 3) -> float:
    q_grams = ngrams(query, n)
    if not q_grams:
        return 0.0
    best = 0.0
    for doc in documents:
        doc_grams = ngrams(doc, n)
        if not doc_grams:
            continue
        best = max(best, len(q_grams & doc_grams) / len(q_grams))
    return best


def contains_any_name(query: str, names: list[str]) -> bool:
    q = query.lower()
    for name in names:
        name = name.strip().lower()
        if name and name in q:
            return True
    return False
