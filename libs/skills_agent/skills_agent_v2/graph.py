"""
Skills Agent Graph - LangGraph 1.0 implementation.

Refactored to use:
  - config.Settings for all configuration
  - llm.create_llm for model instantiation with retry
  - state.AgentState / StateManager for state lifecycle
  - log.get_logger for structured logging
  - exceptions.* for typed errors
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .exceptions import (
    AgentError,
    LLMCallError,
    MaxIterationsError,
    SkillNotFoundError,
)
from .executor import SkillExecutor
from .llm import create_llm, create_llm_callback, invoke_with_retry
from .loader import SkillLoader
from .models import SkillStatus
from .state import AgentState, StateManager

logger = logging.getLogger(__name__)


# ======================================================================
# Tool Definitions
# ======================================================================

def create_skill_tools(skill_loader: SkillLoader, skill_executor: SkillExecutor):
    """Create LangChain tools that operate on the skill system."""

    @tool
    def list_skills() -> str:
        """List all available skills with their descriptions.
        Use this to discover what skills are available before selecting one."""
        logger.debug("Tool list_skills called")
        skills = skill_loader.list_skills()
        if not skills:
            logger.debug("No skills available")
            return "No skills available."
        lines = []
        for s in skills:
            status = s.status.value
            steps = len(s.workflow_steps) if s.workflow_steps else "?"
            lines.append(
                f"• {s.metadata.name} [{status}]: {s.metadata.description} "
                f"(steps: {steps}, tags: {', '.join(s.metadata.tags)})"
            )
        logger.info("list_skills: returning %d skills", len(skills))
        return "\n".join(lines)

    @tool
    def read_skill(skill_name: str) -> str:
        """Load and read the full instructions for a specific skill.

        Args:
            skill_name: Name of the skill to load.
        """
        logger.info("Tool read_skill called: skill_name='%s'", skill_name)
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
            logger.info("read_skill: loaded '%s' (%d chars)", skill_name, len("\n".join(parts)))
            return "\n".join(parts)
        except Exception as e:
            logger.error("read_skill failed for '%s': %s", skill_name, e)
            return f"Error loading skill: {e}"

    @tool
    def read_skill_reference(skill_name: str, reference_name: str) -> str:
        """Read a specific reference file from a skill.

        Args:
            skill_name: Name of the skill.
            reference_name: Name of the reference file.
        """
        logger.info("Tool read_skill_reference: skill='%s', ref='%s'", skill_name, reference_name)
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
        logger.info("Tool write_todos: %d items", len(todos))
        for i, t in enumerate(todos, 1):
            logger.debug("  Todo %d: %s", i, t[:80])
        formatted = []
        for i, t in enumerate(todos, 1):
            formatted.append(f"  {i}. [ ] {t}")
        return "Plan updated:\n" + "\n".join(formatted)

    @tool
    def execute_skill_workflow(skill_name: str, input_data: str = "") -> str:
        """Execute the complete workflow defined in a skill.

        Args:
            skill_name: Name of the skill whose workflow to execute.
            input_data: Optional input data/context for the workflow.
        """
        logger.info("Tool execute_skill_workflow: skill='%s', input_data=%d chars", skill_name, len(input_data))
        skill = skill_loader.get_skill(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found."
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
            script_name: Name of the script file.
        """
        logger.info("Tool run_skill_script: skill='%s', script='%s'", skill_name, script_name)
        skill = skill_loader.get_skill(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found."
        if skill.status == SkillStatus.DISCOVERED:
            skill_loader.load_skill(skill_name)
            skill = skill_loader.get_skill(skill_name)
        if script_name not in skill.scripts:
            return f"Script '{script_name}' not found. Available: {list(skill.scripts.keys())}"

        from .models import SkillStep
        step = SkillStep(
            index=0, description=f"Run {script_name}",
            action="run_script", params={"script": script_name},
        )
        result = skill_executor._execute_step(step, skill, {}, {})
        if result.success:
            return f"Script executed successfully:\n{result.output}"
        return f"Script failed:\n{result.error}"

    @tool
    def search_skill_reference(skill_name: str, query: str, file_name: str = "", top_k: int = 3) -> str:
        """Search within a skill's reference files for specific information.

        This is MORE TOKEN-EFFICIENT than read_skill_reference for large files.
        It searches by keywords and returns only the relevant sections.

        Args:
            skill_name: Name of the skill.
            query: Search query (keywords or natural language).
            file_name: Optional - restrict search to a specific reference file.
            top_k: Number of results to return (default 3).
        """
        logger.info("Tool search_skill_reference: skill='%s', query='%s'", skill_name, query[:60])
        skill = skill_loader.get_skill(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found."
        if skill.status == SkillStatus.DISCOVERED:
            skill_loader.load_skill(skill_name)
            skill = skill_loader.get_skill(skill_name)

        from .reference import ReferenceManager
        ref_mgr = ReferenceManager()
        ref_mgr.index_skill_references(skill)
        result = ref_mgr.search_text(query, top_k=top_k, file_name=file_name or None)
        logger.info("search_skill_reference: returned %d chars", len(result))
        return result

    @tool
    def get_reference_toc(skill_name: str, file_name: str = "") -> str:
        """Get the table of contents of a skill's reference files.

        Use this to see what sections are available before searching.
        Each section shows its heading and character count.

        Args:
            skill_name: Name of the skill.
            file_name: Optional - specific reference file.
        """
        logger.info("Tool get_reference_toc: skill='%s', file='%s'", skill_name, file_name)
        skill = skill_loader.get_skill(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found."
        if skill.status == SkillStatus.DISCOVERED:
            skill_loader.load_skill(skill_name)
            skill = skill_loader.get_skill(skill_name)

        from .reference import ReferenceManager
        ref_mgr = ReferenceManager()
        ref_mgr.index_skill_references(skill)
        return ref_mgr.get_toc(file_name or None)

    @tool
    def invoke_skill(skill_name: str, input_data: str = "") -> str:
        """Invoke another skill's workflow as a sub-task.

        Use this to compose skills — call one skill from within another's workflow.
        The sub-skill runs its complete workflow and returns the final output.

        Args:
            skill_name: Name of the skill to invoke.
            input_data: Input data/context to pass to the sub-skill.
        """
        logger.info("Tool invoke_skill: skill='%s', input=%d chars", skill_name, len(input_data))
        skill = skill_loader.get_skill(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found."
        if skill.status == SkillStatus.DISCOVERED:
            skill_loader.load_skill(skill_name)
            skill = skill_loader.get_skill(skill_name)

        context = {"input_data": input_data} if input_data else {}
        result = skill_executor.execute_workflow(skill, context)
        if result.success:
            return f"Sub-skill '{skill_name}' completed:\n{result.final_output}"
        return f"Sub-skill '{skill_name}' failed: {result.error}"

    return [list_skills, read_skill, read_skill_reference, search_skill_reference,
            get_reference_toc, invoke_skill,
            write_todos, execute_skill_workflow, run_skill_script]


# ======================================================================
# System Prompt Builder
# ======================================================================

DEFAULT_SYSTEM_PROMPT = """\
You are a universal Skills Agent powered by LangGraph. You can discover, load, \
and execute specialized skills to handle complex tasks.

## How Skills Work

Skills are specialized instruction sets with workflows, reference files, and scripts. \
Each skill follows the agentskills.io specification.

## Your Workflow

1. **Understand the request**: Analyze what the user needs.
2. **Discover skills**: Use `list_skills` to see available skills.
3. **Plan**: Use `write_todos` to break complex tasks into steps.
4. **Load skill**: Use `read_skill` to get full instructions for a relevant skill.
5. **Read references**: Use `read_skill_reference` to get domain knowledge as needed.
6. **Execute**: Either follow instructions manually or use `execute_skill_workflow`.
7. **Report**: Summarize the results clearly.

## Progressive Disclosure

You start with only skill summaries. Load full content only when needed. \
Read reference files on demand, not all at once.

{skills_context}
"""


def build_system_prompt(skill_loader: SkillLoader, extra: str = "") -> str:
    """Build the system prompt with skill context."""
    skills_ctx = skill_loader.get_skills_context()
    prompt = DEFAULT_SYSTEM_PROMPT.format(skills_context=skills_ctx)
    if extra:
        prompt += f"\n\n## Additional Instructions\n\n{extra}"
    return prompt


# ======================================================================
# Graph Builder
# ======================================================================

def create_agent_graph(
    llm: Any,
    skill_loader: SkillLoader,
    skill_executor: SkillExecutor,
    settings: Settings | None = None,
    additional_tools: list | None = None,
) -> Any:
    """Create the compiled LangGraph agent.

    Args:
        llm: A LangChain chat model (must support bind_tools).
        skill_loader: Configured SkillLoader.
        skill_executor: SkillExecutor for running workflows.
        settings: Full Settings (uses agent sub-settings). Defaults applied if None.
        additional_tools: Extra tools to bind.

    Returns:
        Compiled StateGraph.
    """
    if settings is None:
        settings = Settings()

    max_iterations = settings.agent.max_iterations

    # Tools
    skill_tools = create_skill_tools(skill_loader, skill_executor)
    all_tools = skill_tools + (additional_tools or [])
    tools_by_name = {t.name: t for t in all_tools}
    llm_with_tools = llm.bind_tools(all_tools)

    system_prompt = build_system_prompt(skill_loader, settings.agent.system_prompt)

    # ---- Nodes ----

    def agent_node(state: AgentState) -> dict:
        """Main reasoning node: LLM with tools."""
        messages = state["messages"]
        iteration = state.get("iteration_count", 0)

        logger.info(
            "━ agent_node: iteration=%d, messages=%d, active_skill=%s, status=%s",
            iteration, len(messages),
            state.get("active_skill", "-"),
            state.get("skill_workflow_status", "idle"),
        )

        if iteration >= max_iterations:
            logger.warning("Max iterations reached (%d)", max_iterations)
            return {
                "messages": [AIMessage(content=(
                    f"Reached maximum iteration limit ({max_iterations}). "
                    "Summarizing progress so far."
                ))],
                "iteration_count": iteration + 1,
                "metadata": {**state.get("metadata", {}), "stopped_reason": "max_iterations"},
            }

        full_messages = [SystemMessage(content=system_prompt)] + list(messages)

        try:
            response = invoke_with_retry(
                llm_with_tools, full_messages,
                max_retries=settings.llm.max_retries,
                retry_delay=settings.llm.retry_delay,
            )
        except LLMCallError as e:
            logger.error("LLM call failed: %s", e)
            response = AIMessage(content=f"I encountered an error calling the LLM: {e}")

        # Log response info
        has_tools = hasattr(response, "tool_calls") and response.tool_calls
        if has_tools:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info("  agent_node → tool_calls: %s", tool_names)
        else:
            content_preview = response.content[:100] if response.content else "(empty)"
            logger.info("  agent_node → final response (%d chars): %s...", len(response.content or ""), content_preview)

        return {
            "messages": [response],
            "iteration_count": iteration + 1,
        }

    def tool_node(state: AgentState) -> dict:
        """Execute tool calls from the last AI message."""
        last_message = state["messages"][-1]
        results = []
        state_updates: dict[str, Any] = {}

        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info("━ tool_node: executing %d tools: %s", len(last_message.tool_calls), tool_names)

        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]

            if name not in tools_by_name:
                logger.warning("Unknown tool called: %s", name)
                results.append(ToolMessage(
                    content=f"Error: Unknown tool '{name}'",
                    tool_call_id=tool_call["id"],
                ))
                continue

            try:
                logger.info("  tool_node → calling '%s' with args: %s", name, list(args.keys()))
                observation = tools_by_name[name].invoke(args)
                obs_str = str(observation)
                logger.info("  tool_node → '%s' returned %d chars", name, len(obs_str))
                logger.debug("  tool_node → '%s' output preview: %.200s", name, obs_str)
                results.append(ToolMessage(
                    content=obs_str,
                    tool_call_id=tool_call["id"],
                ))

                # State tracking
                if name == "read_skill":
                    state_updates["active_skill"] = args.get("skill_name", "")
                    state_updates["skill_context"] = str(observation)[:2000]
                elif name == "write_todos":
                    state_updates["todo_list"] = [
                        {"index": i, "text": t, "done": False}
                        for i, t in enumerate(args.get("todos", []), 1)
                    ]
                    state_updates["skill_workflow_status"] = "planning"
                elif name == "execute_skill_workflow":
                    state_updates["skill_workflow_status"] = "executing"

            except Exception as e:
                logger.exception("Tool '%s' raised an error", name)
                results.append(ToolMessage(
                    content=f"Error executing {name}: {e}",
                    tool_call_id=tool_call["id"],
                ))

        return {"messages": results, **state_updates}

    # ---- Routing ----

    def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            logger.debug("Routing → tool_node (has tool_calls)")
            return "tool_node"
        logger.debug("Routing → __end__")
        return "__end__"

    # ---- Build graph ----

    graph = StateGraph(AgentState)
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "agent_node")
    graph.add_conditional_edges(
        "agent_node", should_continue,
        {"tool_node": "tool_node", "__end__": END},
    )
    graph.add_edge("tool_node", "agent_node")

    compiled = graph.compile()
    logger.info(
        "Agent graph compiled: %d tools, max_iterations=%d",
        len(all_tools), max_iterations,
    )
    return compiled


# ======================================================================
# High-level API: create_skills_agent
# ======================================================================

def create_skills_agent(
    # --- Model (string or instance) ---
    model: str | Any | None = None,
    # --- Skill sources ---
    skill_dirs: list[str] | None = None,
    skill_paths: list[str] | None = None,
    skills_content: dict[str, str] | None = None,
    # --- Customisation ---
    system_prompt: str = "",
    tools: list | None = None,
    # --- Full settings override ---
    settings: Settings | None = None,
    # --- Legacy convenience kwargs ---
    max_iterations: int | None = None,
    work_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a Skills Agent in one call.

    Args:
        model: LLM identifier ("openai:gpt-4o") or a chat model instance.
               Falls back to settings.llm.model if None.
        skill_dirs: Directories to scan for SKILL.md files.
        skill_paths: Individual skill paths (files or directories).
        skills_content: Dict of {name: SKILL.md content} for inline skills.
        system_prompt: Additional system prompt to append.
        tools: Extra LangChain tools.
        settings: Full Settings object. If None, built from env + defaults.
        max_iterations: Override settings.agent.max_iterations.
        work_dir: Override settings.executor.work_dir.

    Returns:
        Compiled LangGraph graph.
    """
    # ---- Resolve settings ----
    if settings is None:
        settings = Settings.from_env()

    if model is not None and isinstance(model, str):
        settings.llm.model = model
    if system_prompt:
        settings.agent.system_prompt = system_prompt
    if max_iterations is not None:
        settings.agent.max_iterations = max_iterations
    if work_dir is not None:
        settings.executor.work_dir = work_dir

    # Merge skill sources
    all_dirs = (settings.skill_dirs or []) + (skill_dirs or [])
    all_paths = (settings.skill_paths or []) + (skill_paths or [])

    # ---- Setup logging ----
    from .log import setup_logging
    setup_logging(settings.log)

    logger.info("━━━ Creating Skills Agent ━━━")
    logger.info("  LLM: model=%s (local=%s)", settings.llm.model, settings.llm.is_local)
    if settings.llm.base_url:
        logger.info("  LLM: base_url=%s", settings.llm.base_url)
    logger.info("  Agent: max_iterations=%d", settings.agent.max_iterations)
    logger.info("  Skill dirs: %s", all_dirs or "(none)")
    logger.info("  Skill paths: %s", all_paths or "(none)")

    # ---- Create LLM ----
    if model is not None and not isinstance(model, str):
        llm = model  # Pre-built instance
    else:
        llm = create_llm(settings.llm)

    # ---- Load skills ----
    loader = SkillLoader(skill_dirs=all_dirs, skill_paths=all_paths)
    loader.discover()

    if skills_content:
        for name, content in skills_content.items():
            loader.load_from_content(name, content)

    logger.info("Skills loaded: %s", loader.list_skill_names())

    # ---- Create executor ----
    executor = SkillExecutor(
        llm_callback=create_llm_callback(llm, settings.llm.max_retries, settings.llm.retry_delay),
        skill_loader=loader,
        work_dir=settings.executor.work_dir or None,
        script_timeout_python=settings.executor.script_timeout_python,
        script_timeout_shell=settings.executor.script_timeout_shell,
    )

    # ---- Build graph ----
    return create_agent_graph(
        llm=llm,
        skill_loader=loader,
        skill_executor=executor,
        settings=settings,
        additional_tools=tools,
    )


def get_initial_state(user_message: str, **kwargs: Any) -> dict[str, Any]:
    """Create the initial state dict for `agent.invoke(...)`.

    Args:
        user_message: The user's request.
        **kwargs: Override any state field.
    """
    return StateManager.create(user_message, **kwargs)
