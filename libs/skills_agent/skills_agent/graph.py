"""
Skills Agent Graph - LangGraph 1.0 implementation.

Architecture inspired by LangChain DeepAgents:
  - Planning via todo/workflow tool
  - Progressive skill disclosure
  - Subagent delegation for skill execution
  - Filesystem-backed context management

Graph nodes:
  ┌─────────┐     ┌──────────────┐     ┌──────────┐
  │  START   │────▸│  agent_node  │────▸│   END    │
  └─────────┘     └──────┬───────┘     └──────────┘
                         │
                  ┌──────┴──────┐
                  ▼             ▼
           ┌──────────┐  ┌──────────────┐
           │tool_node │  │ skill_router  │
           └──────────┘  └──────┬───────┘
                                │
                   ┌────────────┼────────────┐
                   ▼            ▼            ▼
            ┌───────────┐ ┌─────────┐ ┌──────────┐
            │skill_read │ │skill_exe│ │skill_plan│
            └───────────┘ └─────────┘ └──────────┘
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .executor import SkillExecutor
from .loader import SkillLoader
from .models import SkillStatus

logger = logging.getLogger(__name__)


# ======================================================================
# State Definition
# ======================================================================


class AgentState(TypedDict):
    """Main agent state flowing through the graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    # Skill management
    active_skill: str | None  # Currently active skill name
    skill_context: str  # Injected skill instructions
    skill_workflow_status: str  # "idle" | "planning" | "executing" | "completed"
    # Planning
    todo_list: list[dict[str, Any]]  # Current plan / todo items
    current_step_index: int  # Which step we're on
    # Results
    workflow_result: dict[str, Any] | None  # Latest workflow execution result
    # Metadata
    iteration_count: int  # Guard against infinite loops


# ======================================================================
# Tool Definitions (created dynamically with skill_loader reference)
# ======================================================================


def create_skill_tools(skill_loader: SkillLoader, skill_executor: SkillExecutor):
    """Create LangChain tools that operate on the skill system."""

    @tool
    def list_skills() -> str:
        """List all available skills with their descriptions.
        Use this to discover what skills are available before selecting one."""
        skills = skill_loader.list_skills()
        if not skills:
            return "No skills available."
        lines = []
        for s in skills:
            status = s.status.value
            steps = len(s.workflow_steps) if s.workflow_steps else "?"
            lines.append(
                f"• {s.metadata.name} [{status}]: {s.metadata.description} "
                f"(steps: {steps}, tags: {', '.join(s.metadata.tags)})"
            )
        return "\n".join(lines)

    @tool
    def read_skill(skill_name: str) -> str:
        """Load and read the full instructions for a specific skill.

        Args:
            skill_name: Name of the skill to load.

        Returns the complete skill instructions including workflow steps,
        references list, and scripts list.
        """
        try:
            skill = skill_loader.load_skill(skill_name)
            parts = [
                f"# Skill: {skill.metadata.name}",
                f"Description: {skill.metadata.description}",
                f"Version: {skill.metadata.version}",
                "",
                "## Full Instructions",
                skill.body,
                "",
            ]
            if skill.workflow_steps:
                parts.append("## Workflow Steps")
                for step in skill.workflow_steps:
                    parts.append(f"  {step.index}. {step.description}")
                parts.append("")

            if skill.reference_files:
                parts.append("## Available References")
                for name in skill.reference_files:
                    parts.append(f"  - {name}")
                parts.append("")

            if skill.scripts:
                parts.append("## Available Scripts")
                for name in skill.scripts:
                    parts.append(f"  - {name}")
                parts.append("")

            return "\n".join(parts)
        except KeyError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error loading skill: {e}"

    @tool
    def read_skill_reference(skill_name: str, reference_name: str) -> str:
        """Read a specific reference file from a skill.

        Args:
            skill_name: Name of the skill.
            reference_name: Name of the reference file (e.g., 'newyear_knowledge.md').
        """
        skill = skill_loader.get_skill(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found."
        if skill.status == SkillStatus.DISCOVERED:
            skill_loader.load_skill(skill_name)
            skill = skill_loader.get_skill(skill_name)

        if reference_name in skill.reference_files:
            return skill.reference_files[reference_name]
        return (
            f"Reference '{reference_name}' not found. "
            f"Available: {list(skill.reference_files.keys())}"
        )

    @tool
    def write_todos(todos: list[str]) -> str:
        """Write or update the current plan/todo list.

        This is a planning tool. Use it to break down complex tasks into steps
        BEFORE executing them. Each todo should be a clear, actionable step.

        Args:
            todos: List of todo items / steps to plan.
        """
        formatted = []
        for i, t in enumerate(todos, 1):
            formatted.append(f"  {i}. [ ] {t}")
        return "Plan updated:\n" + "\n".join(formatted)

    @tool
    def execute_skill_workflow(skill_name: str, input_data: str = "") -> str:
        """Execute the complete workflow defined in a skill.

        This runs all workflow steps sequentially, using the skill's
        instructions, references, and scripts.

        Args:
            skill_name: Name of the skill whose workflow to execute.
            input_data: Optional input data/context for the workflow.
        """
        skill = skill_loader.get_skill(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found."

        # Ensure fully loaded
        if skill.status == SkillStatus.DISCOVERED:
            skill_loader.load_skill(skill_name)
            skill = skill_loader.get_skill(skill_name)

        context = {"input_data": input_data} if input_data else {}
        result = skill_executor.execute_workflow(skill, context)

        parts = [result.summary()]
        for sr in result.step_results:
            status = "✅" if sr.success else "❌"
            parts.append(f"  Step {sr.step_index}: {status}")
            if sr.output:
                # Truncate long outputs
                output = sr.output[:500] + "..." if len(sr.output) > 500 else sr.output
                parts.append(f"    Output: {output}")
            if sr.error:
                parts.append(f"    Error: {sr.error}")

        if result.final_output:
            parts.append(f"\nFinal output:\n{result.final_output[:1000]}")

        return "\n".join(parts)

    @tool
    def run_skill_script(skill_name: str, script_name: str) -> str:
        """Run a specific script from a skill.

        Args:
            skill_name: Name of the skill.
            script_name: Name of the script file (e.g., 'document_chunker.py').
        """
        skill = skill_loader.get_skill(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found."

        if skill.status == SkillStatus.DISCOVERED:
            skill_loader.load_skill(skill_name)
            skill = skill_loader.get_skill(skill_name)

        if script_name not in skill.scripts:
            return (
                f"Script '{script_name}' not found. "
                f"Available: {list(skill.scripts.keys())}"
            )

        from .models import SkillStep

        step = SkillStep(
            index=0,
            description=f"Run {script_name}",
            action="run_script",
            params={"script": script_name},
        )
        result = skill_executor._execute_step(step, skill, {}, {})

        if result.success:
            return f"Script executed successfully:\n{result.output}"
        return f"Script failed:\n{result.error}"

    return [
        list_skills,
        read_skill,
        read_skill_reference,
        write_todos,
        execute_skill_workflow,
        run_skill_script,
    ]


# ======================================================================
# System Prompt Builder
# ======================================================================

DEFAULT_SYSTEM_PROMPT = """\
You are a universal Skills Agent powered by LangGraph. You can discover, load, \
and execute specialized skills to handle complex tasks.

## How Skills Work

Skills are specialized instruction sets with workflows, reference files, and scripts. \
Each skill follows the agentskills.io specification with a SKILL.md file containing:
- YAML frontmatter with metadata (name, description, tags)
- A markdown body with detailed instructions and workflow steps
- Optional reference files with domain knowledge
- Optional scripts for automated processing

## Your Workflow

1. **Understand the request**: Analyze what the user needs.
2. **Discover skills**: Use `list_skills` to see available skills.
3. **Plan**: Use `write_todos` to break complex tasks into steps.
4. **Load skill**: Use `read_skill` to get full instructions for a relevant skill.
5. **Read references**: Use `read_skill_reference` to get domain knowledge as needed.
6. **Execute**: Either follow the skill instructions manually or use \
   `execute_skill_workflow` for automated execution.
7. **Report**: Summarize the results clearly.

## Progressive Disclosure

You start with only skill summaries to save context. Only load full skill content \
when you determine it's needed. Read reference files on demand, not all at once.

## Planning

For complex tasks, ALWAYS create a plan first with `write_todos`. This helps you:
- Break down multi-step tasks
- Track progress
- Adapt your plan as you learn more

{skills_context}
"""


def build_system_prompt(skill_loader: SkillLoader, custom_prompt: str = "") -> str:
    """Build the system prompt with skill context."""
    skills_ctx = skill_loader.get_skills_context()
    prompt = DEFAULT_SYSTEM_PROMPT.format(skills_context=skills_ctx)
    if custom_prompt:
        prompt += f"\n\n## Additional Instructions\n\n{custom_prompt}"
    return prompt


# ======================================================================
# Graph Node Functions
# ======================================================================


def create_agent_graph(
    llm: Any,
    skill_loader: SkillLoader,
    skill_executor: SkillExecutor,
    custom_system_prompt: str = "",
    additional_tools: list | None = None,
    max_iterations: int = 25,
) -> Any:  # Returns compiled StateGraph
    """Create the main Skills Agent graph.

    Args:
        llm: A LangChain chat model (must support tool calling).
        skill_loader: Configured SkillLoader with skills discovered.
        skill_executor: SkillExecutor for running workflows.
        custom_system_prompt: Additional instructions to append.
        additional_tools: Extra tools to bind to the agent.
        max_iterations: Maximum agent loop iterations.

    Returns:
        Compiled LangGraph StateGraph.
    """
    # Create tools
    skill_tools = create_skill_tools(skill_loader, skill_executor)
    all_tools = skill_tools + (additional_tools or [])
    tools_by_name = {t.name: t for t in all_tools}

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(all_tools)

    # Build system prompt
    system_prompt = build_system_prompt(skill_loader, custom_system_prompt)

    # ------------------------------------------------------------------
    # Node: agent_node - The main reasoning node
    # ------------------------------------------------------------------
    def agent_node(state: AgentState) -> dict:
        """Main agent node: invoke LLM with tools."""
        messages = state["messages"]
        iteration = state.get("iteration_count", 0)

        # Guard against infinite loops
        if iteration >= max_iterations:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"I've reached the maximum iteration limit ({max_iterations}). "
                            "Here's a summary of what I've accomplished so far based on the "
                            "conversation above."
                        )
                    )
                ],
                "iteration_count": iteration + 1,
            }

        # Inject system prompt if not present
        full_messages = [SystemMessage(content=system_prompt)] + list(messages)

        # Call LLM
        response = llm_with_tools.invoke(full_messages)

        return {
            "messages": [response],
            "iteration_count": iteration + 1,
        }

    # ------------------------------------------------------------------
    # Node: tool_node - Execute tool calls
    # ------------------------------------------------------------------
    def tool_node(state: AgentState) -> dict:
        """Execute tool calls from the last AI message."""
        messages = state["messages"]
        last_message = messages[-1]

        results = []
        new_state_updates: dict[str, Any] = {}

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name not in tools_by_name:
                results.append(
                    ToolMessage(
                        content=f"Error: Unknown tool '{tool_name}'",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            try:
                tool_fn = tools_by_name[tool_name]
                observation = tool_fn.invoke(tool_args)

                results.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_call["id"],
                    )
                )

                # Track state changes based on tool usage
                if tool_name == "read_skill":
                    skill_name = tool_args.get("skill_name", "")
                    new_state_updates["active_skill"] = skill_name
                    new_state_updates["skill_context"] = str(observation)[:2000]

                elif tool_name == "write_todos":
                    todos_raw = tool_args.get("todos", [])
                    new_state_updates["todo_list"] = [
                        {"index": i, "text": t, "done": False}
                        for i, t in enumerate(todos_raw, 1)
                    ]
                    new_state_updates["skill_workflow_status"] = "planning"

                elif tool_name == "execute_skill_workflow":
                    new_state_updates["skill_workflow_status"] = "executing"

            except Exception as e:
                logger.exception("Tool execution error for %s", tool_name)
                results.append(
                    ToolMessage(
                        content=f"Error executing {tool_name}: {e}",
                        tool_call_id=tool_call["id"],
                    )
                )

        return {"messages": results, **new_state_updates}

    # ------------------------------------------------------------------
    # Routing function
    # ------------------------------------------------------------------
    def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
        """Decide whether to continue to tools or finish."""
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM made tool calls, execute them
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"

        # Check iteration guard
        if state.get("iteration_count", 0) >= max_iterations:
            return "__end__"

        return "__end__"

    # ------------------------------------------------------------------
    # Build the graph
    # ------------------------------------------------------------------
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)

    # Add edges
    graph.add_edge(START, "agent_node")
    graph.add_conditional_edges(
        "agent_node",
        should_continue,
        {"tool_node": "tool_node", "__end__": END},
    )
    graph.add_edge("tool_node", "agent_node")

    # Compile
    compiled = graph.compile()
    return compiled


# ======================================================================
# Convenience: create_skills_agent (DeepAgents-style API)
# ======================================================================


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        `ChatAnthropic` instance configured with Claude Sonnet 4.5.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,  # type: ignore[call-arg]
    )


def create_skills_agent(
    model: str | BaseChatModel | None = None,
    skill_dirs: list[str] | None = None,
    skills_content: dict[str, str] | None = None,
    system_prompt: str = "",
    tools: list | None = None,
    max_iterations: int = 25,
    work_dir: str | None = None,
    **kwargs,
) -> Any:
    """Create a Skills Agent with a simple API.

    Inspired by DeepAgents' `create_deep_agent`.

    Args:
        model: Model identifier string (e.g., "openai:gpt-4o") or a
               LangChain chat model instance.
               Defaults to `claude-sonnet-4-5-20250929`.
        skill_dirs: List of directories to scan for skills.
        skills_content: Dict of {name: SKILL.md content} for inline skills.
        system_prompt: Additional system prompt instructions.
        tools: Additional tools to bind.
        max_iterations: Max agent loop iterations.
        work_dir: Working directory for script execution.

    Returns:
        Compiled LangGraph graph.
    """
    # Initialize LLM
    if model is None:
        llm = get_default_model()
    elif isinstance(model, str):
        from langchain.chat_models import init_chat_model

        llm = init_chat_model(model)
    else:
        llm = model

    # Initialize skill loader
    loader = SkillLoader(skill_dirs=skill_dirs)
    loader.discover()

    # Load inline skills
    if skills_content:
        for name, content in skills_content.items():
            loader.load_from_content(name, content)

    # Initialize executor with LLM callback
    def llm_callback(system: str, user: str) -> str:
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        response = llm.invoke(msgs)
        return response.content

    executor = SkillExecutor(llm_callback=llm_callback, work_dir=work_dir)

    # Build graph
    return create_agent_graph(
        llm=llm,
        skill_loader=loader,
        skill_executor=executor,
        custom_system_prompt=system_prompt,
        additional_tools=tools,
        max_iterations=max_iterations,
    )


def get_initial_state(user_message: str) -> AgentState:
    """Helper to create the initial state for invoking the agent."""
    return {
        "messages": [HumanMessage(content=user_message)],
        "active_skill": None,
        "skill_context": "",
        "skill_workflow_status": "idle",
        "todo_list": [],
        "current_step_index": 0,
        "workflow_result": None,
        "iteration_count": 0,
    }
