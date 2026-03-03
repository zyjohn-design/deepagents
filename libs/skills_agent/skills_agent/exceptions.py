"""
Unified exception hierarchy for Skills Agent.

All custom exceptions inherit from SkillsAgentError so callers can
catch broad or narrow as needed.
"""

from __future__ import annotations


class SkillsAgentError(Exception):
    """Base exception for all Skills Agent errors."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigError(SkillsAgentError):
    """Invalid or missing configuration."""


# ---------------------------------------------------------------------------
# Skill lifecycle
# ---------------------------------------------------------------------------

class SkillNotFoundError(SkillsAgentError):
    """Requested skill does not exist in the registry."""

    def __init__(self, name: str, available: list[str] | None = None):
        self.name = name
        self.available = available or []
        avail = f" Available: {self.available}" if self.available else ""
        super().__init__(f"Skill '{name}' not found.{avail}")


class SkillLoadError(SkillsAgentError):
    """Failed to load a skill from disk or content."""

    def __init__(self, name: str, reason: str = ""):
        self.name = name
        self.reason = reason
        super().__init__(f"Failed to load skill '{name}': {reason}")


class SkillParseError(SkillsAgentError):
    """Failed to parse SKILL.md frontmatter or body."""


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------

class WorkflowError(SkillsAgentError):
    """Error during skill workflow execution."""


class StepExecutionError(WorkflowError):
    """A single workflow step failed."""

    def __init__(self, step_index: int, reason: str = ""):
        self.step_index = step_index
        self.reason = reason
        super().__init__(f"Step {step_index} failed: {reason}")


class StepDependencyError(WorkflowError):
    """A workflow step's dependency was not satisfied."""

    def __init__(self, step_index: int, dep_index: int):
        self.step_index = step_index
        self.dep_index = dep_index
        super().__init__(
            f"Step {step_index} blocked: dependency step {dep_index} not completed"
        )


class ScriptTimeoutError(WorkflowError):
    """A script exceeded its time limit."""

    def __init__(self, script: str, timeout: int):
        self.script = script
        self.timeout = timeout
        super().__init__(f"Script '{script}' timed out after {timeout}s")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class LLMError(SkillsAgentError):
    """Error interacting with the language model."""


class LLMConfigError(LLMError, ConfigError):
    """Invalid LLM configuration (bad provider string, missing key, etc.)."""


class LLMCallError(LLMError):
    """An LLM invocation failed after retries."""

    def __init__(self, reason: str = "", retries: int = 0):
        self.retries = retries
        msg = f"LLM call failed"
        if retries:
            msg += f" after {retries} retries"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Agent graph
# ---------------------------------------------------------------------------

class AgentError(SkillsAgentError):
    """Error in the LangGraph agent execution."""


class MaxIterationsError(AgentError):
    """Agent exceeded the maximum iteration count."""

    def __init__(self, limit: int):
        self.limit = limit
        super().__init__(f"Agent reached maximum iteration limit ({limit})")
