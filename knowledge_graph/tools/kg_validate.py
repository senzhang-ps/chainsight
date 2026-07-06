"""KG instance consistency checks (offline, no Neo4j).

Implements meta-ontology R1/R4 validation against source-of-truth artifacts:

- R1: every ``csmeta:Agent`` / ``csmeta:Tool`` YAML points to an existing
  artifact file.
- R4: for each ``csmeta:WorkerAgent``, the ``usesTools`` list must equal the
  ``tools:`` frontmatter list in the corresponding ``.agents/*.md`` source,
  restricted to CS-registered tools (OH built-ins are skipped).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AGENTS_DIR = PROJECT_ROOT / ".agents"
INSTANCES_DIR = PROJECT_ROOT / "knowledge_graph" / "instances"
AGENT_INSTANCES_DIR = INSTANCES_DIR / "agents"
TOOL_INSTANCES_DIR = INSTANCES_DIR / "tools"


def _parse_frontmatter(md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    return yaml.safe_load(text[3:end]) or {}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _cs_tool_names() -> set[str]:
    return {p.stem for p in TOOL_INSTANCES_DIR.glob("*.yaml")}


def _frontmatter_tools(fm: dict) -> list[str]:
    raw = fm.get("tools", "")
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def check_tool_r1(errors: list[str]) -> None:
    for yml in TOOL_INSTANCES_DIR.glob("*.yaml"):
        data = _load_yaml(yml)
        artifact = data.get("artifact")
        if not artifact:
            errors.append(f"R1: {yml.name} missing 'artifact'")
            continue
        if not (PROJECT_ROOT / artifact).exists():
            errors.append(f"R1: {yml.name} artifact not found: {artifact}")


def check_agent_r1_r4(errors: list[str]) -> None:
    cs_tools = _cs_tool_names()
    for yml in AGENT_INSTANCES_DIR.glob("*.yaml"):
        data = _load_yaml(yml)
        artifact = data.get("artifact")
        if not artifact or not (PROJECT_ROOT / artifact).exists():
            errors.append(f"R1: {yml.name} artifact missing: {artifact}")
            continue

        md_path = PROJECT_ROOT / artifact
        fm = _parse_frontmatter(md_path)
        fm_tools = [t for t in _frontmatter_tools(fm) if t in cs_tools]

        yml_tools = [t.split("/")[-1] for t in (data.get("usesTools") or [])]

        if sorted(fm_tools) != sorted(yml_tools):
            errors.append(
                f"R4: {yml.name} usesTools mismatch. "
                f"frontmatter={sorted(fm_tools)} vs yaml={sorted(yml_tools)}"
            )


def validate_all() -> list[str]:
    """Run R1 + R4 checks. Return a list of error strings (empty = ok)."""
    errors: list[str] = []
    check_tool_r1(errors)
    check_agent_r1_r4(errors)
    return errors


if __name__ == "__main__":  # pragma: no cover
    errs = validate_all()
    if errs:
        for e in errs:
            print("ERROR:", e)
        raise SystemExit(1)
    print("OK: meta-ontology R1/R4 checks passed.")
