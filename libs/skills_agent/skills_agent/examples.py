#!/usr/bin/env python3
"""
Skills Agent - Complete Usage Examples

Demonstrates:
  1. Basic skill discovery and agent creation
  2. Skill loading from filesystem
  3. Inline skill loading
  4. Workflow execution
  5. Custom tools integration
  6. Streaming output
  7. Checkpointer (memory) usage
"""

import os
import sys

# =====================================================================
# Example 1: Basic Usage - Quickstart
# =====================================================================

def example_basic():
    """Simplest possible usage - like DeepAgents' create_deep_agent."""
    from skills_agent import create_skills_agent, get_initial_state

    agent = create_skills_agent(
        model="openai:gpt-4o",
        skill_dirs=["./example_skills/"],
    )

    result = agent.invoke(get_initial_state(
        "List all available skills and tell me what each one does."
    ))

    # Print the final AI message
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            print(f"[{msg.__class__.__name__}] {msg.content[:200]}")


# =====================================================================
# Example 2: With Inline Skills (StateBackend-style)
# =====================================================================

def example_inline_skills():
    """Load skills from inline content, not filesystem."""
    from skills_agent import create_skills_agent, get_initial_state

    deploy_skill = """\
---
name: deploy
description: Deploy application to production environment
version: 1.0.0
tags: [deployment, production, devops]
---

# Deploy to Production

## Core Workflow

```
Step 1: Run the test suite
Step 2: Build the application artifacts
Step 3: Deploy to staging environment
Step 4: Run smoke tests on staging
Step 5: Deploy to production
Step 6: Verify production health
```

## Instructions

### Pre-deployment Checks
- All tests must pass
- No critical security findings
- Change log updated
- Version number bumped

### Deployment Process
Always use blue-green deployment:
1. Deploy to inactive environment
2. Run health checks
3. Switch traffic
4. Monitor for 15 minutes
5. If issues, rollback immediately

### Rollback Procedure
If any health check fails:
1. Switch traffic back to previous version
2. Alert the on-call engineer
3. Create incident report
"""

    agent = create_skills_agent(
        model="openai:gpt-4o",
        skills_content={"deploy": deploy_skill},
        system_prompt="You are a DevOps assistant helping with deployments.",
    )

    result = agent.invoke(get_initial_state(
        "I need to deploy our app. What's the process?"
    ))

    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            print(f"[{msg.__class__.__name__}] {msg.content[:300]}")


# =====================================================================
# Example 3: Custom LangGraph with Checkpointer (Persistence)
# =====================================================================

def example_with_memory():
    """Use LangGraph checkpointer for conversation persistence."""
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import MemorySaver

    from skills_agent import (
        SkillLoader,
        SkillExecutor,
        create_agent_graph,
        get_initial_state,
    )

    # Setup
    llm = init_chat_model("openai:gpt-4o")
    loader = SkillLoader(skill_dirs=["./example_skills/"])
    loader.discover()

    executor = SkillExecutor(work_dir="./workspace")

    # Create graph WITHOUT compiling yet
    from skills_agent.graph import (
        AgentState,
        build_system_prompt,
        create_skill_tools,
    )

    # We need to build the graph manually to add checkpointer
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

    skill_tools = create_skill_tools(loader, executor)
    tools_by_name = {t.name: t for t in skill_tools}
    llm_with_tools = llm.bind_tools(skill_tools)
    system_prompt = build_system_prompt(loader)

    def agent_node(state):
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def tool_node(state):
        last = state["messages"][-1]
        results = []
        for tc in last.tool_calls:
            tool_fn = tools_by_name.get(tc["name"])
            if tool_fn:
                obs = tool_fn.invoke(tc["args"])
                results.append(ToolMessage(content=str(obs), tool_call_id=tc["id"]))
            else:
                results.append(ToolMessage(
                    content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]
                ))
        return {"messages": results}

    def should_continue(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tool_node"
        return "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "agent_node")
    graph.add_conditional_edges("agent_node", should_continue,
                                {"tool_node": "tool_node", "__end__": END})
    graph.add_edge("tool_node", "agent_node")

    # Compile WITH checkpointer for memory
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    # First conversation turn
    config = {"configurable": {"thread_id": "user-123"}}
    result1 = compiled.invoke(get_initial_state("What skills do you have?"), config=config)
    print("Turn 1:", result1["messages"][-1].content[:200])

    # Second turn - agent remembers context
    from langchain_core.messages import HumanMessage
    result2 = compiled.invoke(
        {"messages": [HumanMessage(content="Tell me more about the first skill")]},
        config=config,
    )
    print("Turn 2:", result2["messages"][-1].content[:200])


# =====================================================================
# Example 4: Streaming Output
# =====================================================================

def example_streaming():
    """Stream agent output for real-time feedback."""
    from skills_agent import create_skills_agent, get_initial_state

    agent = create_skills_agent(
        model="openai:gpt-4o",
        skill_dirs=["./example_skills/"],
    )

    state = get_initial_state("Read the web-research skill and explain its workflow")

    print("=== Streaming Agent Execution ===\n")
    for event in agent.stream(state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"--- [{node_name}] ---")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    role = msg.__class__.__name__
                    content = msg.content[:200] if hasattr(msg, "content") else ""
                    print(f"  {role}: {content}")
            print()


# =====================================================================
# Example 5: Custom Tools + Skills
# =====================================================================

def example_custom_tools():
    """Combine skill-based tools with custom tools."""
    from langchain_core.tools import tool

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.
        Args:
            expression: A Python math expression to evaluate.
        """
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime
        return datetime.now().isoformat()

    from skills_agent import create_skills_agent, get_initial_state

    agent = create_skills_agent(
        model="openai:gpt-4o",
        skill_dirs=["./example_skills/"],
        tools=[calculate, get_current_time],
        system_prompt="You also have access to a calculator and clock.",
    )

    result = agent.invoke(get_initial_state(
        "What time is it? Also, what's 1337 * 42?"
    ))

    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            print(f"[{msg.__class__.__name__}] {msg.content}")


# =====================================================================
# Example 6: Programmatic Skill Loader (Advanced)
# =====================================================================

def example_advanced_loader():
    """Demonstrate advanced skill loader features."""
    from skills_agent import SkillLoader, SkillExecutor

    loader = SkillLoader()

    # Add multiple directories
    loader.add_skill_dir("./example_skills/")

    # Discover all skills (lightweight - frontmatter only)
    discovered = loader.discover()
    print(f"Discovered {len(discovered)} skills:")
    for s in discovered:
        print(f"  - {s.metadata.name}: {s.metadata.description}")

    # Load one skill fully
    skill = loader.load_skill("web-research")
    print(f"\nLoaded skill: {skill.metadata.name}")
    print(f"  Workflow steps: {len(skill.workflow_steps)}")
    for step in skill.workflow_steps:
        print(f"    {step.index}. {step.description}")
    print(f"  References: {list(skill.reference_files.keys())}")
    print(f"  Scripts: {list(skill.scripts.keys())}")

    # Keyword matching
    match = loader.match_skill("I need to review some Python code")
    if match:
        print(f"\nBest match for code review: {match.metadata.name}")

    # Get skills context for LLM
    ctx = loader.get_skills_context()
    print(f"\nSkills context ({len(ctx)} chars):")
    print(ctx)


# =====================================================================
# Example 7: Event Document Parser (Traffic AI use case)
# =====================================================================

def example_traffic_ai():
    """
    Demonstrates using skills for your traffic-hicon-ai project.
    Loads the event-doc-parser skill and processes a document.
    """
    from skills_agent import create_skills_agent, get_initial_state

    # In real usage, point to your actual skill directory
    agent = create_skills_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        skill_dirs=[
            "./example_skills/",
            # "/path/to/traffic-hicon-ai/skills/"
        ],
        system_prompt="""\
You are a traffic management AI assistant. Your primary task is to parse 
large event documents (New Year's Eve celebrations, marathons, etc.) and 
extract structured information about traffic control measures.

When processing event documents:
1. First identify the event type
2. Load the appropriate skill (e.g., event-doc-parser)
3. Read the relevant knowledge base for the event type
4. Follow the skill's workflow to extract information
5. Output structured JSON with original text tracing
""",
        max_iterations=30,
    )

    # Process a sample document
    sample_doc = """
# 2025年跨年夜交通管控方案

## 一、活动概况
活动名称：2025年跨年夜倒计时活动
活动时间：2024年12月31日 18:00 - 2025年1月1日 02:00
活动地点：市民广场及周边区域

## 二、交通管控措施

### （一）准备阶段（12月31日 14:00-18:00）
1. 对核心区域实施临时交通管制
2. 开放P1-P5临时停车场

### （二）管控阶段（12月31日 18:00-次日01:00）
1. 核心区（市民广场500米范围）全面封控，禁止车辆通行
2. 管控区（1公里范围）限制通行，仅允许公交和应急车辆
3. 疏导区（2公里范围）单向通行管理
"""

    result = agent.invoke(get_initial_state(
        f"请解析这份跨年夜交通管控方案，提取所有时间段的任务列表和管控区域信息：\n\n{sample_doc}"
    ))

    # Get the final response
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
            print("=== Agent Response ===")
            print(msg.content)
            break


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    examples = {
        "basic": example_basic,
        "inline": example_inline_skills,
        "memory": example_with_memory,
        "streaming": example_streaming,
        "custom-tools": example_custom_tools,
        "loader": example_advanced_loader,
        "traffic-ai": example_traffic_ai,
    }

    if len(sys.argv) < 2:
        print("Skills Agent Examples")
        print("=" * 40)
        print(f"\nUsage: python {sys.argv[0]} <example>")
        print(f"\nAvailable examples:")
        for name, fn in examples.items():
            print(f"  {name:15s} - {fn.__doc__.strip().split(chr(10))[0]}")
        print(f"\n  all            - Run all examples")
        sys.exit(0)

    choice = sys.argv[1]
    if choice == "all":
        for name, fn in examples.items():
            print(f"\n{'='*60}")
            print(f"  Example: {name}")
            print(f"{'='*60}\n")
            try:
                fn()
            except Exception as e:
                print(f"  Error: {e}")
    elif choice in examples:
        examples[choice]()
    else:
        print(f"Unknown example: {choice}")
        print(f"Available: {', '.join(examples.keys())}")
