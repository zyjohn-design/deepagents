"""
Agent state management.

Provides:
  - AgentState TypedDict for LangGraph
  - StateManager for creating, serializing, and restoring state
  - State snapshots for debugging / audit
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

# LangGraph imports are deferred to avoid hard dependency at parse time
try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langgraph.graph.message import add_messages

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

    # Stub so the module can still be imported for non-graph use
    def add_messages(a, b):  # type: ignore
        return (a or []) + (b or [])

    class BaseMessage:  # type: ignore
        content: str = ""

    class HumanMessage(BaseMessage):  # type: ignore
        def __init__(self, content: str = ""):
            self.content = content

    class AIMessage(BaseMessage):  # type: ignore
        pass

    class SystemMessage(BaseMessage):  # type: ignore
        pass

    class ToolMessage(BaseMessage):  # type: ignore
        pass


# ======================================================================
# AgentState — the canonical graph state definition
# ======================================================================

class AgentState(TypedDict):
    """Main state flowing through the LangGraph Skills Agent.

    Fields:
        messages: Conversation history (LangGraph add_messages reducer).
        active_skill: Name of the currently activated skill (or None).
        skill_context: Cached full-text of the active skill (for display).
        skill_workflow_status: "idle" | "planning" | "executing" | "completed" | "failed"
        todo_list: Agent's current plan / checklist.
        current_step_index: Which plan step is being executed.
        workflow_result: Last workflow execution result (serializable dict).
        iteration_count: Guard counter to prevent runaway loops.
        metadata: Free-form metadata for extensions.
    """

    messages: Annotated[list, add_messages]
    # Skill tracking
    active_skill: str | None
    skill_context: str
    skill_workflow_status: str
    # Planning
    todo_list: list[dict[str, Any]]
    current_step_index: int
    # Results
    workflow_result: dict[str, Any] | None
    # Loop guard
    iteration_count: int
    # Extension point
    metadata: dict[str, Any]


# Valid workflow status transitions
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "idle":      {"planning", "executing"},
    "planning":  {"executing", "idle"},
    "executing": {"completed", "failed", "idle"},
    "completed": {"idle", "planning"},
    "failed":    {"idle", "planning", "executing"},
}


# ======================================================================
# StateManager — helpers for creating / serializing state
# ======================================================================

class StateManager:
    """Utilities for working with AgentState."""

    @staticmethod
    def create(user_message: str = "", **kwargs: Any) -> dict[str, Any]:
        """Create a fresh initial state.

        Args:
            user_message: The user's request (creates first HumanMessage).
            **kwargs: Override any state field.

        Returns:
            A dict conforming to AgentState.
        """
        messages = []
        if user_message:
            messages.append(HumanMessage(content=user_message))

        state: dict[str, Any] = {
            "messages": messages,
            "active_skill": None,
            "skill_context": "",
            "skill_workflow_status": "idle",
            "todo_list": [],
            "current_step_index": 0,
            "workflow_result": None,
            "iteration_count": 0,
            "metadata": {},
        }
        state.update(kwargs)
        return state

    @staticmethod
    def validate_transition(current: str, target: str) -> bool:
        """Check if a workflow status transition is valid.

        Args:
            current: Current status string.
            target: Desired next status.

        Returns:
            True if the transition is allowed.
        """
        allowed = _VALID_TRANSITIONS.get(current, set())
        return target in allowed

    @staticmethod
    def get_last_ai_message(state: dict[str, Any]) -> str:
        """Extract the content of the last AIMessage."""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "content") and isinstance(msg, AIMessage):
                return msg.content
        return ""

    @staticmethod
    def get_last_message(state: dict[str, Any]) -> str:
        """Extract the content of the last message of any type."""
        msgs = state.get("messages", [])
        if msgs and hasattr(msgs[-1], "content"):
            return msgs[-1].content
        return ""

    @staticmethod
    def count_messages_by_type(state: dict[str, Any]) -> dict[str, int]:
        """Count messages by type."""
        counts: dict[str, int] = {}
        for msg in state.get("messages", []):
            key = msg.__class__.__name__
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def to_dict(state: dict[str, Any]) -> dict[str, Any]:
        """Serialize state to a JSON-safe dict (for persistence / debugging).

        Messages are converted to a list of {role, content} dicts.
        """
        messages = []
        for msg in state.get("messages", []):
            role = "unknown"
            if isinstance(msg, HumanMessage):
                role = "human"
            elif isinstance(msg, AIMessage):
                role = "ai"
            elif isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, ToolMessage):
                role = "tool"

            entry: dict[str, Any] = {"role": role, "content": getattr(msg, "content", "")}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.get("name"), "args": tc.get("args")}
                    for tc in msg.tool_calls
                ]
            messages.append(entry)

        return {
            "messages": messages,
            "active_skill": state.get("active_skill"),
            "skill_context": state.get("skill_context", "")[:200] + "...",
            "skill_workflow_status": state.get("skill_workflow_status", "idle"),
            "todo_list": state.get("todo_list", []),
            "current_step_index": state.get("current_step_index", 0),
            "workflow_result": state.get("workflow_result"),
            "iteration_count": state.get("iteration_count", 0),
            "metadata": state.get("metadata", {}),
        }

    @staticmethod
    def to_json(state: dict[str, Any], indent: int = 2) -> str:
        """Serialize state to a JSON string."""
        return json.dumps(StateManager.to_dict(state), ensure_ascii=False, indent=indent)


# ======================================================================
# Snapshot — lightweight state capture for audit trails
# ======================================================================

@dataclass
class StateSnapshot:
    """Immutable capture of agent state at a point in time."""

    timestamp: float
    iteration: int
    workflow_status: str
    active_skill: str | None
    message_count: int
    todo_count: int
    last_message_preview: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(cls, state: dict[str, Any]) -> StateSnapshot:
        """Take a snapshot of the current state."""
        last_msg = StateManager.get_last_message(state)
        return cls(
            timestamp=time.time(),
            iteration=state.get("iteration_count", 0),
            workflow_status=state.get("skill_workflow_status", "idle"),
            active_skill=state.get("active_skill"),
            message_count=len(state.get("messages", [])),
            todo_count=len(state.get("todo_list", [])),
            last_message_preview=last_msg[:100] if last_msg else "",
            metadata=state.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "workflow_status": self.workflow_status,
            "active_skill": self.active_skill,
            "message_count": self.message_count,
            "todo_count": self.todo_count,
            "last_message_preview": self.last_message_preview,
        }


class StateHistory:
    """Ordered collection of state snapshots for debugging / audit."""

    def __init__(self, max_size: int = 100):
        self._snapshots: list[StateSnapshot] = []
        self._max_size = max_size

    def record(self, state: dict[str, Any]) -> StateSnapshot:
        """Capture and store a snapshot."""
        snap = StateSnapshot.capture(state)
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_size:
            self._snapshots.pop(0)
        return snap

    @property
    def snapshots(self) -> list[StateSnapshot]:
        return list(self._snapshots)

    def summary(self) -> str:
        """Human-readable execution trace."""
        if not self._snapshots:
            return "No snapshots recorded."
        lines = [f"State History ({len(self._snapshots)} snapshots):"]
        for s in self._snapshots:
            lines.append(
                f"  [{s.iteration:3d}] {s.workflow_status:10s} "
                f"skill={s.active_skill or '-':20s} "
                f"msgs={s.message_count} "
                f"| {s.last_message_preview[:60]}"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        self._snapshots.clear()
