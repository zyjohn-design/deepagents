"""
Skill models and data structures.

Step params syntax in SKILL.md:
  步骤1: 读取知识库 [action=read_reference, reference=knowledge.md]
  步骤2: 查询API参数 [action=search_reference, query=认证接口, file=api_doc.md]
  步骤3: 运行脚本 [action=run_script, script=transform.py, args=--input ${step.1.output}]
  步骤4: 调用子技能 [action=invoke_skill, skill=data-validator, input=${step.3.output}]

Template variables in params:
  ${step.N.output}   → output of step N
  ${step.N.artifact.key} → specific artifact from step N
  ${context.key}      → value from context dict
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SkillStatus(str, Enum):
    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVE = "active"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SkillStep:
    """A single step inside a skill workflow.

    Attributes:
        index: Step number.
        description: Human-readable description (without action params).
        action: Action type (read_file, read_reference, run_script,
                search_reference, invoke_skill, llm_reason, or custom).
        params: Action parameters dict (parsed from [key=val, ...] syntax).
        depends_on: List of step indices this step depends on.
        raw_description: Original unparsed description text.
    """
    index: int
    description: str
    action: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    raw_description: str = ""

    def resolve_params(
        self,
        context: dict[str, Any],
        prev_results: dict[int, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve template variables in params.

        Supports:
          ${step.N.output}          → prev_results[N].output
          ${step.N.artifact.KEY}    → prev_results[N].artifacts[KEY]
          ${context.KEY}            → context[KEY]

        Returns a new dict with all templates resolved.
        """
        resolved = {}
        prev = prev_results or {}
        for key, val in self.params.items():
            if isinstance(val, str):
                resolved[key] = _resolve_template(val, context, prev)
            else:
                resolved[key] = val
        return resolved


_TEMPLATE_RE = re.compile(r'\$\{([^}]+)\}')


def _resolve_template(text: str, context: dict, prev_results: dict) -> str:
    """Replace ${...} placeholders with actual values."""
    def replacer(m: re.Match) -> str:
        path = m.group(1).strip()
        parts = path.split(".")

        try:
            if parts[0] == "step" and len(parts) >= 3:
                step_idx = int(parts[1])
                result = prev_results.get(step_idx)
                if result is None:
                    return m.group(0)  # keep unresolved
                if parts[2] == "output":
                    return str(getattr(result, "output", ""))
                if parts[2] == "artifact" and len(parts) >= 4:
                    artifacts = getattr(result, "artifacts", {})
                    return str(artifacts.get(parts[3], ""))

            if parts[0] == "context" and len(parts) >= 2:
                return str(context.get(parts[1], ""))

        except (ValueError, IndexError, AttributeError):
            pass

        return m.group(0)  # keep unresolved

    return _TEMPLATE_RE.sub(replacer, text)


@dataclass
class SkillMetadata:
    """Parsed from YAML frontmatter of SKILL.md."""
    name: str
    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    depends_on_skills: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """A complete skill with metadata, body, and workflow steps."""
    metadata: SkillMetadata
    body: str = ""
    path: Path | None = None
    status: SkillStatus = SkillStatus.DISCOVERED
    workflow_steps: list[SkillStep] = field(default_factory=list)
    reference_files: dict[str, str] = field(default_factory=dict)
    scripts: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.metadata.name

    def summary(self) -> str:
        tags = ", ".join(self.metadata.tags) if self.metadata.tags else "none"
        deps = f", deps: {self.metadata.depends_on_skills}" if self.metadata.depends_on_skills else ""
        return (
            f"[{self.metadata.name}] {self.metadata.description} "
            f"(tags: {tags}, version: {self.metadata.version}{deps})"
        )

    def full_prompt(self) -> str:
        parts = [f"# Skill: {self.metadata.name}", ""]
        if self.metadata.description:
            parts.append(self.metadata.description)
            parts.append("")
        parts.append(self.body)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a SKILL.md file."""
    import yaml
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    raw_yaml = match.group(1)
    body = content[match.end():]
    try:
        fm = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body


# ---------------------------------------------------------------------------
# Workflow step parser — with inline [action=..., param=...] extraction
# ---------------------------------------------------------------------------

# Matches [key=val, key2=val2, ...] at end of a line
_PARAMS_RE = re.compile(r'\[([^\]]+)\]\s*$')

# Matches key=value pairs inside brackets
_KV_RE = re.compile(r'(\w+)\s*=\s*([^,\]]+)')


def _parse_step_params(text: str) -> tuple[str, str, dict[str, Any]]:
    """Parse action and params from step description.

    Input:  "读取知识库 [action=read_reference, reference=knowledge.md]"
    Output: ("读取知识库", "read_reference", {"reference": "knowledge.md"})
    """
    m = _PARAMS_RE.search(text)
    if not m:
        return text.strip(), "", {}

    description = text[:m.start()].strip()
    params_str = m.group(1)

    params: dict[str, Any] = {}
    action = ""
    for kv in _KV_RE.finditer(params_str):
        k, v = kv.group(1).strip(), kv.group(2).strip()
        if k == "action":
            action = v
        else:
            params[k] = v

    return description, action, params


def parse_workflow_steps(body: str) -> list[SkillStep]:
    """Extract workflow steps from the skill body.

    Recognizes:
        ## 核心工作流 / ## Core Workflow / ## Workflow
        步骤1: 描述 [action=xxx, param=yyy]
        Step 1: desc [action=xxx, param=yyy]
        1. desc [action=xxx]
    """
    steps: list[SkillStep] = []

    # Find workflow section
    workflow_section = ""
    in_workflow = False
    for line in body.split("\n"):
        lower = line.lower().strip()
        if any(kw in lower for kw in [
            "工作流", "workflow", "## steps", "## 流程",
            "## process", "## 执行步骤", "核心工作流"
        ]):
            in_workflow = True
            continue
        if in_workflow:
            if line.startswith("## ") and workflow_section:
                break
            workflow_section += line + "\n"

    if not workflow_section:
        return steps

    step_patterns = [
        re.compile(r"(?:步骤|step)\s*(\d+)\s*[:：]\s*(.+)", re.IGNORECASE),
        re.compile(r"(\d+)\.\s+(.+)"),
        re.compile(r"(\d+)\)\s+(.+)"),
        re.compile(r"-\s+\*\*(?:步骤|step)\s*(\d+)\*\*\s*[:：]\s*(.+)", re.IGNORECASE),
    ]

    for line in workflow_section.split("\n"):
        line = line.strip()
        if not line:
            continue
        for pattern in step_patterns:
            m = pattern.match(line)
            if m:
                idx = int(m.group(1))
                raw_desc = m.group(2).strip()
                desc, action, params = _parse_step_params(raw_desc)
                steps.append(SkillStep(
                    index=idx,
                    description=desc,
                    action=action,
                    params=params,
                    raw_description=raw_desc,
                ))
                break

    return steps
