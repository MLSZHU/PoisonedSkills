from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Parameter:
    name: str
    type: str = "string"
    required: bool = False
    default: Any = None
    description: str = ""
    enum_values: list[str] = field(default_factory=list)
    examples: list[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Parameter":
        enum_values = data.get("enum_values") or data.get("enum") or []
        examples = data.get("examples") or []
        return cls(
            name=str(data.get("name") or data.get("param") or ""),
            type=str(data.get("type") or data.get("param_type") or "string").lower(),
            required=bool(data.get("required", False)),
            default=data.get("default"),
            description=str(data.get("description") or ""),
            enum_values=[str(x) for x in enum_values],
            examples=list(examples),
        )


@dataclass
class Capability:
    capability_id: str
    text: str

    @classmethod
    def from_any(cls, obj: Any, idx: int) -> "Capability":
        if isinstance(obj, dict):
            return cls(
                capability_id=str(obj.get("capability_id") or obj.get("id") or f"cap_{idx}"),
                text=str(obj.get("text") or obj.get("description") or obj.get("name") or ""),
            )
        return cls(capability_id=f"cap_{idx}", text=str(obj))


@dataclass
class Example:
    query: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Example":
        return cls(
            query=str(data.get("query") or data.get("query_text") or ""),
            params=dict(data.get("params") or data.get("param_fill") or {}),
        )


@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    body: str = ""
    tags: list[str] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    capabilities: list[Capability] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Skill":
        skill_id = str(record.get("skill_id") or record.get("id") or record.get("name") or "").strip()
        name = str(record.get("name") or skill_id).strip()
        body = str(record.get("body") or record.get("skill_md") or record.get("content") or "")
        description = str(record.get("description") or _first_paragraph(body) or name)
        params = [Parameter.from_dict(x) for x in record.get("parameters", []) if isinstance(x, dict)]
        caps = [Capability.from_any(x, i) for i, x in enumerate(record.get("capabilities", []), start=1)]
        examples = [Example.from_dict(x) for x in record.get("examples", []) if isinstance(x, dict)]
        if not caps:
            caps = _infer_capabilities(name, description, body)
        return cls(
            skill_id=skill_id,
            name=name,
            description=description,
            body=body,
            tags=[str(x) for x in record.get("tags", [])],
            parameters=params,
            capabilities=caps,
            examples=examples,
            metadata=dict(record.get("metadata") or {}),
        )

    @classmethod
    def from_markdown(cls, path: str | Path) -> "Skill":
        p = Path(path)
        body = p.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        name = title_match.group(1).strip() if title_match else p.parent.name
        skill_id = p.parent.name
        description = _first_paragraph(body) or name
        return cls(
            skill_id=skill_id,
            name=name,
            description=description,
            body=body,
            capabilities=_infer_capabilities(name, description, body),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def parameter(self, name: str) -> Parameter | None:
        return next((p for p in self.parameters if p.name == name), None)

    def required_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters if p.required]


@dataclass
class Slot:
    param: str
    examples: list[Any] = field(default_factory=list)
    default: Any = None


@dataclass
class QueryTemplate:
    template: str
    capability_id: str = ""
    slots: list[str] = field(default_factory=list)
    strategy: str = "base"


@dataclass
class PseudoQuery:
    skill_id: str
    query_text: str
    embedding_text: str = ""
    algorithm: str = ""
    source_strategy: str = "unknown"
    param_fill: dict[str, Any] = field(default_factory=dict)
    template: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_paragraph(text: str) -> str:
    cleaned = re.sub(r"^#.*$", "", text, flags=re.MULTILINE).strip()
    for para in re.split(r"\n\s*\n", cleaned):
        para = re.sub(r"\s+", " ", para).strip()
        if para and not para.startswith("```"):
            return para[:600]
    return ""


def _infer_capabilities(name: str, description: str, body: str) -> list[Capability]:
    # SKILLRET-style records have a curated description but no capability list.
    # Using the description is usually a better fallback than treating markdown
    # headings as capabilities.
    if description and description != name:
        return [Capability("cap_1", description)]
    headings = re.findall(r"^#{2,3}\s+(.+)$", body, flags=re.MULTILINE)
    caps = [Capability(f"cap_{i}", h.strip()) for i, h in enumerate(headings[:6], start=1)]
    if caps:
        return caps
    return [Capability("cap_1", description or name)]
