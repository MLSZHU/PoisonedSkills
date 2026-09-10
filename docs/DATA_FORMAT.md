# Data Format

## Skill JSONL

Each row is one skill:

```json
{
  "skill_id": "string",
  "name": "string",
  "description": "string",
  "body": "markdown or free text",
  "tags": ["optional"],
  "capabilities": [
    {"capability_id": "cap_1", "text": "what the skill can do"}
  ],
  "parameters": [
    {
      "name": "path",
      "type": "string",
      "required": true,
      "default": null,
      "description": "input path",
      "enum_values": null,
      "examples": ["data/input.csv"]
    }
  ],
  "examples": [
    {"query": "Clean data/input.csv", "params": {"path": "data/input.csv"}}
  ]
}
```

Aliases are accepted for compatibility:

- `id` as `skill_id`
- `skill_md` or `content` as `body`
- `param_fill` as `params`
- `query_text` as `query`

## Pseudo Query JSONL

Each output row is one generated query:

```json
{
  "query_id": "skill_id:algorithm:000001",
  "skill_id": "skill_id",
  "query_text": "natural user query",
  "embedding_text": "text used for embedding",
  "algorithm": "skill2query",
  "source_strategy": "rule_base",
  "param_fill": {"path": "data/input.csv"},
  "template": "Clean {path}",
  "metadata": {}
}
```
