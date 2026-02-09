"""
Unified configuration for Skills Agent.

Supports layered configuration (lowest to highest priority):
  1. Built-in defaults
  2. YAML config file
  3. Environment variables (SKILLS_AGENT_ prefix)
  4. Programmatic overrides

Usage:
    from skills_agent.config import settings

    # Use defaults
    print(settings.llm.model)

    # Or create from file / env
    settings = Settings.from_yaml("config.yaml")
    settings = Settings.from_env()

    # Or override programmatically
    settings = Settings(llm=LLMSettings(model="anthropic:claude-sonnet-4-5-20250929"))
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigError


# ======================================================================
# Settings dataclasses
# ======================================================================

@dataclass
class LLMSettings:
    """Language model configuration.

    Supports cloud providers AND local OpenAI-compatible servers:
      - Cloud:  "openai:gpt-4o", "anthropic:claude-sonnet-4-5-20250929"
      - Local:  "local:qwen2.5-72b", "ollama:llama3", "vllm:Qwen/Qwen2.5-72B"
      - Direct: "lmstudio:deepseek-coder-v2", "llamacpp:my-model"

    For local models, set `base_url` to your server endpoint, or use the
    built-in presets (ollama → localhost:11434, lmstudio → localhost:1234, etc.)
    """

    # Model identifier: "provider:model"
    model: str = "openai:gpt-4o"

    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 4096

    # Retry policy
    max_retries: int = 3
    retry_delay: float = 1.0          # seconds, doubles each retry

    # API keys (auto-read from env if empty; local models often need "not-needed")
    api_key: str = ""
    base_url: str = ""

    # Timeout for a single LLM call (seconds)
    timeout: int = 120

    @property
    def provider(self) -> str:
        """Extract provider from model string (e.g. 'openai' from 'openai:gpt-4o')."""
        if ":" in self.model:
            return self.model.split(":")[0]
        return "openai"

    @property
    def model_name(self) -> str:
        """Extract model name without provider prefix."""
        if ":" in self.model:
            return self.model.split(":", 1)[1]
        return self.model

    @property
    def is_local(self) -> bool:
        """Whether this model is served locally (OpenAI-compatible API)."""
        return self.provider in ("local", "ollama", "lmstudio", "vllm", "llamacpp", "xinference")


@dataclass
class LogSettings:
    """Logging configuration."""

    level: str = "INFO"                # DEBUG / INFO / WARNING / ERROR
    format: str = "rich"               # "rich" (console) | "json" | "text"
    file: str = ""                     # Log file path (empty = no file logging)
    file_max_bytes: int = 10_485_760   # 10 MB
    file_backup_count: int = 3
    show_timestamp: bool = True
    show_module: bool = True
    # Suppress noisy third-party loggers
    quiet_loggers: list[str] = field(default_factory=lambda: [
        "httpx", "httpcore", "openai", "anthropic", "urllib3",
    ])


@dataclass
class ExecutorSettings:
    """Workflow executor configuration."""

    work_dir: str = ""                 # Empty = auto temp dir
    script_timeout_python: int = 120   # seconds
    script_timeout_shell: int = 60     # seconds
    max_output_length: int = 5000      # truncate step outputs beyond this


@dataclass
class AgentSettings:
    """LangGraph agent loop configuration."""

    max_iterations: int = 25
    system_prompt: str = ""            # Additional system prompt appended
    stream_mode: str = "updates"       # "updates" | "values"


@dataclass
class Settings:
    """Root settings container."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    log: LogSettings = field(default_factory=LogSettings)
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)

    # Skill directories and paths
    skill_dirs: list[str] = field(default_factory=list)
    skill_paths: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        """Load settings from a YAML file, merging with defaults."""
        import yaml

        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {path}")

        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def from_env(cls, prefix: str = "SKILLS_AGENT_") -> Settings:
        """Load settings from environment variables.

        Mapping: SKILLS_AGENT_LLM_MODEL → settings.llm.model
                 SKILLS_AGENT_LOG_LEVEL → settings.log.level
                 SKILLS_AGENT_AGENT_MAX_ITERATIONS → settings.agent.max_iterations
        """
        s = cls()

        env_map = {
            # LLM
            f"{prefix}LLM_MODEL": ("llm", "model", str),
            f"{prefix}LLM_TEMPERATURE": ("llm", "temperature", float),
            f"{prefix}LLM_MAX_TOKENS": ("llm", "max_tokens", int),
            f"{prefix}LLM_MAX_RETRIES": ("llm", "max_retries", int),
            f"{prefix}LLM_API_KEY": ("llm", "api_key", str),
            f"{prefix}LLM_BASE_URL": ("llm", "base_url", str),
            f"{prefix}LLM_TIMEOUT": ("llm", "timeout", int),
            # Log
            f"{prefix}LOG_LEVEL": ("log", "level", str),
            f"{prefix}LOG_FORMAT": ("log", "format", str),
            f"{prefix}LOG_FILE": ("log", "file", str),
            # Executor
            f"{prefix}EXECUTOR_WORK_DIR": ("executor", "work_dir", str),
            f"{prefix}EXECUTOR_SCRIPT_TIMEOUT_PYTHON": ("executor", "script_timeout_python", int),
            f"{prefix}EXECUTOR_SCRIPT_TIMEOUT_SHELL": ("executor", "script_timeout_shell", int),
            # Agent
            f"{prefix}AGENT_MAX_ITERATIONS": ("agent", "max_iterations", int),
            f"{prefix}AGENT_SYSTEM_PROMPT": ("agent", "system_prompt", str),
            # Skill dirs (comma-separated)
            f"{prefix}SKILL_DIRS": None,
            f"{prefix}SKILL_PATHS": None,
        }

        for env_key, target in env_map.items():
            val = os.environ.get(env_key)
            if val is None:
                continue

            if env_key.endswith("_SKILL_DIRS"):
                s.skill_dirs = [d.strip() for d in val.split(",") if d.strip()]
            elif env_key.endswith("_SKILL_PATHS"):
                s.skill_paths = [d.strip() for d in val.split(",") if d.strip()]
            elif target:
                section, attr, typ = target
                try:
                    setattr(getattr(s, section), attr, typ(val))
                except (ValueError, TypeError):
                    pass

        return s

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> Settings:
        """Build Settings from a flat or nested dict."""
        s = cls()

        if "llm" in d and isinstance(d["llm"], dict):
            for k, v in d["llm"].items():
                if hasattr(s.llm, k):
                    setattr(s.llm, k, v)

        if "log" in d and isinstance(d["log"], dict):
            for k, v in d["log"].items():
                if hasattr(s.log, k):
                    setattr(s.log, k, v)

        if "executor" in d and isinstance(d["executor"], dict):
            for k, v in d["executor"].items():
                if hasattr(s.executor, k):
                    setattr(s.executor, k, v)

        if "agent" in d and isinstance(d["agent"], dict):
            for k, v in d["agent"].items():
                if hasattr(s.agent, k):
                    setattr(s.agent, k, v)

        if "skill_dirs" in d:
            s.skill_dirs = d["skill_dirs"]
        if "skill_paths" in d:
            s.skill_paths = d["skill_paths"]

        return s

    def merge(self, overrides: dict[str, Any]) -> Settings:
        """Return a new Settings with overrides applied (non-mutating)."""
        import copy
        merged = copy.deepcopy(self)
        other = self._from_dict(overrides)

        for section in ("llm", "log", "executor", "agent"):
            src = getattr(other, section)
            dst = getattr(merged, section)
            defaults = getattr(Settings(), section)
            for attr in vars(src):
                val = getattr(src, attr)
                if val != getattr(defaults, attr):
                    setattr(dst, attr, val)

        if overrides.get("skill_dirs"):
            merged.skill_dirs = overrides["skill_dirs"]
        if overrides.get("skill_paths"):
            merged.skill_paths = overrides["skill_paths"]

        return merged


# ======================================================================
# Module-level singleton
# ======================================================================

settings = Settings()
