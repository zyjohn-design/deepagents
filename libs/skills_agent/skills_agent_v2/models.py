"""
Skill models and data structures.
Follows the agentskills.io spec (YAML frontmatter + markdown body).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SkillStatus(str, Enum):
    DISCOVERED = "discovered"       # Found but not loaded
    LOADED = "loaded"               # Frontmatter parsed
    ACTIVE = "active"               # Full content read, ready to use
    EXECUTING = "executing"         # Currently running
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SkillStep:
    """A single step inside a skill workflow."""
    index: int
    description: str
    action: str = ""                # e.g. "read_file", "llm_call", "run_script"
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)


@dataclass
class SkillMetadata:
    """Parsed from YAML frontmatter of SKILL.md."""
    name: str
    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    # Workflow-related
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    # Optional: dependencies on other skills
    depends_on_skills: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """A complete skill with metadata, body, and workflow steps."""
    metadata: SkillMetadata
    body: str = ""                  # Full markdown body
    path: Path | None = None        # Where it was loaded from
    status: SkillStatus = SkillStatus.DISCOVERED
    # Parsed workflow steps (extracted from body)
    workflow_steps: list[SkillStep] = field(default_factory=list)
    # Reference files associated with this skill
    reference_files: dict[str, str] = field(default_factory=dict)
    # Scripts
    scripts: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.metadata.name

    def summary(self) -> str:
        """Short summary for LLM context (progressive disclosure)."""
        tags = ", ".join(self.metadata.tags) if self.metadata.tags else "none"
        return (
            f"[{self.metadata.name}] {self.metadata.description} "
            f"(tags: {tags}, version: {self.metadata.version})"
        )

    def full_prompt(self) -> str:
        """Full skill prompt content for injection into LLM context."""
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
    """Parse YAML frontmatter from a SKILL.md file.

    Returns (frontmatter_dict, body_text).
    """
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


def parse_workflow_steps(body: str) -> list[SkillStep]:
    """Extract workflow steps from the skill body.

    Recognizes patterns like:
        ## 核心工作流 / ## Core Workflow / ## Workflow
        步骤1: ... / Step 1: ... / 1. ...
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
            # Stop at next heading of same or higher level
            if line.startswith("## ") and workflow_section:
                break
            workflow_section += line + "\n"

    if not workflow_section:
        return steps

    # Parse steps: support `步骤N:`, `Step N:`, `N.`, `N)`, etc.
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
                desc = m.group(2).strip()
                steps.append(SkillStep(index=idx, description=desc))
                break

    return steps
