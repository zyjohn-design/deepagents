"""
Skills Agent — A universal skills execution framework built on LangGraph 1.0.

Quick start:
    from skills_agent import create_skills_agent, get_initial_state

    agent = create_skills_agent(
        model="openai:gpt-4o",
        skill_dirs=["./my_skills/"],
    )
    result = agent.invoke(get_initial_state("Help me with this task"))
"""

# --- Core (always available) ---
from .config import Settings, LLMSettings, LogSettings, ExecutorSettings, AgentSettings
from .exceptions import (
    SkillsAgentError,
    ConfigError,
    SkillNotFoundError,
    SkillLoadError,
    SkillParseError,
    WorkflowError,
    StepExecutionError,
    LLMError,
    LLMConfigError,
    LLMCallError,
    AgentError,
    MaxIterationsError,
)
from .log import setup_logging, get_logger
from .models import Skill, SkillMetadata, SkillStep, SkillStatus
from .loader import SkillLoader
from .executor import SkillExecutor, WorkflowResult, StepResult
from .state import AgentState, StateManager, StateSnapshot, StateHistory

# --- Graph (requires langchain + langgraph) ---
try:
    from .graph import create_skills_agent, get_initial_state, create_agent_graph
    from .llm import create_llm, create_llm_callback, invoke_with_retry, list_providers
except ImportError:
    create_skills_agent = None  # type: ignore[assignment]
    get_initial_state = None    # type: ignore[assignment]
    create_agent_graph = None   # type: ignore[assignment]
    create_llm = None           # type: ignore[assignment]
    create_llm_callback = None  # type: ignore[assignment]
    invoke_with_retry = None    # type: ignore[assignment]
    list_providers = None       # type: ignore[assignment]

__all__ = [
    # High-level API
    "create_skills_agent",
    "get_initial_state",
    "create_agent_graph",
    # Config
    "Settings",
    "LLMSettings",
    "LogSettings",
    "ExecutorSettings",
    "AgentSettings",
    # LLM
    "create_llm",
    "create_llm_callback",
    "invoke_with_retry",
    "list_providers",
    # Logging
    "setup_logging",
    "get_logger",
    # State
    "AgentState",
    "StateManager",
    "StateSnapshot",
    "StateHistory",
    # Loader
    "SkillLoader",
    # Executor
    "SkillExecutor",
    "WorkflowResult",
    "StepResult",
    # Models
    "Skill",
    "SkillMetadata",
    "SkillStep",
    "SkillStatus",
    # Exceptions
    "SkillsAgentError",
    "ConfigError",
    "SkillNotFoundError",
    "SkillLoadError",
    "LLMError",
    "LLMCallError",
    "AgentError",
]

__version__ = "0.2.0"
