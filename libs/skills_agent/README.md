# Skills Agent - LangGraph 1.0 通用技能代理框架

基于 **LangGraph 1.0** 构建的通用技能解析与执行框架，参考 **LangChain DeepAgents** 架构设计。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **技能发现与加载** | 自动扫描目录，解析 `SKILL.md` (agentskills.io 规范) |
| **渐进式披露** | 先加载元数据摘要，按需加载完整内容，节省 token |
| **工作流解析** | 自动提取中英文工作流步骤（步骤N / Step N） |
| **工作流执行** | 支持 LLM推理、脚本执行、引用文件读取、依赖管理 |
| **LangGraph 图编排** | Agent → Tool → Skill Router 循环，条件边路由 |
| **规划工具** | 内置 `write_todos` 工具（灵感来自 Claude Code 的 no-op 规划） |
| **自定义工具** | 可混合使用技能工具和自定义 LangChain 工具 |
| **持久化** | 支持 LangGraph Checkpointer 实现对话记忆 |
| **流式输出** | 支持 `stream()` 实时观察执行过程 |

## 📁 项目结构

```
skills_agent/
├── skills_agent/              # 核心包
│   ├── __init__.py           # 公共 API
│   ├── models.py             # Skill / SkillStep / SkillMetadata 数据模型
│   ├── loader.py             # SkillLoader - 发现、加载、匹配技能
│   ├── executor.py           # SkillExecutor - 执行工作流步骤
│   └── graph.py              # LangGraph StateGraph 构建 + create_skills_agent
├── example_skills/            # 示例技能
│   ├── web_research/SKILL.md
│   └── code_review/SKILL.md
├── examples.py               # 7 个完整使用示例
├── run_tests.py              # 独立测试运行器（21 个测试全通过）
├── tests/                    # pytest 测试
└── pyproject.toml
```

## 🏗️ 架构设计

### LangGraph 状态图

```
┌─────────┐     ┌──────────────┐     ┌──────────┐
│  START   │────▸│  agent_node  │────▸│   END    │
└─────────┘     └──────┬───────┘     └──────────┘
                       │
                ┌──────┴──────┐
                │ tool_calls? │
                └──────┬──────┘
                       │ yes
                ┌──────▼──────┐
                │  tool_node  │──── (loop back to agent_node)
                └─────────────┘
```

**内置工具:**

- `list_skills` - 列出所有可用技能
- `read_skill` - 加载技能完整指令
- `read_skill_reference` - 读取技能引用文件
- `write_todos` - 规划/分解任务
- `execute_skill_workflow` - 执行技能工作流
- `run_skill_script` - 运行技能脚本

### 技能规范 (agentskills.io)

```md
my_skill/
├── SKILL.md            # YAML frontmatter + Markdown 指令
├── references/         # 领域知识文件
│   ├── knowledge.md
│   └── schema.md
└── scripts/            # 可执行脚本
    └── process.py
```

SKILL.md 格式:

```yaml
---
name: my-skill
description: 技能描述
version: 1.0.0
tags: [tag1, tag2]
---

# 技能标题

## 核心工作流

步骤1: 第一步操作
步骤2: 第二步操作
步骤3: 第三步操作

## 详细指令
...
```

## 🚀 快速开始

### 安装

```bash
pip install langgraph langchain langchain-openai pyyaml
```

### 最简用法

```python
from skills_agent import create_skills_agent, get_initial_state

agent = create_skills_agent(
    model="openai:gpt-4o",
    skill_dirs=["./my_skills/"],
)

result = agent.invoke(get_initial_state("帮我解析这份活动文档"))
print(result["messages"][-1].content)
```

### 使用 Anthropic

```python
agent = create_skills_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    skill_dirs=["./skills/"],
    system_prompt="你是一个交通管理AI助手。",
)
```

### 内联技能 (无需文件系统)

```python
skill_content = """
---
name: deploy
description: Deploy to production
tags: [devops]
---

# Deploy

## Core Workflow
Step 1: Run tests
Step 2: Build artifacts
Step 3: Deploy
"""

agent = create_skills_agent(
    model="openai:gpt-4o",
    skills_content={"deploy": skill_content},
)
```

### 自定义工具 + 技能

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    return "搜索结果..."

agent = create_skills_agent(
    model="openai:gpt-4o",
    skill_dirs=["./skills/"],
    tools=[search_web],
)
```

### 带记忆的对话 (Checkpointer)

```python
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from skills_agent import SkillLoader, SkillExecutor, create_agent_graph

loader = SkillLoader(skill_dirs=["./skills/"])
loader.discover()
executor = SkillExecutor()
llm = init_chat_model("openai:gpt-4o")

graph = create_agent_graph(llm, loader, executor)
# Note: create_agent_graph returns compiled graph
# For checkpointer, build manually (see examples.py Example 3)
```

### 流式输出

```python
for event in agent.stream(get_initial_state("执行代码审查"), stream_mode="updates"):
    for node, output in event.items():
        print(f"[{node}]", output.get("messages", [])[-1].content[:100])
```

## 🔄 Agent 执行流程

```
用户请求 → agent_node (LLM推理)
    ├─ 无工具调用 → END (返回结果)
    └─ 有工具调用 → tool_node
         ├─ list_skills → 返回技能列表 → agent_node
         ├─ read_skill → 加载技能全文 → agent_node
         ├─ write_todos → 创建执行计划 → agent_node
         ├─ read_skill_reference → 读取知识库 → agent_node
         ├─ execute_skill_workflow → 执行工作流 → agent_node
         └─ 自定义工具 → 执行 → agent_node
```

## 📊 与 DeepAgents 对比

| 特性 | DeepAgents | Skills Agent (本项目) |
|------|-----------|---------------------|
| 规划工具 | `write_todos` | ✅ `write_todos` |
| 技能加载 | `skills=[]` | ✅ `skill_dirs=[]` + `skills_content={}` |
| 子代理 | `subagents=[]` | ⬜ 可通过自定义工具实现 |
| 文件系统 | 内置 | ✅ 技能自带 references + scripts |
| 工作流执行 | 依赖 LLM 循环 | ✅ 专用执行器 + LLM混合 |
| 中文支持 | 部分 | ✅ 完整支持（步骤N解析） |
| 渐进式披露 | ✅ frontmatter | ✅ frontmatter |
| LangGraph 1.0 | ✅ | ✅ StateGraph + 条件边 |

## 🧪 测试

```bash
# 独立运行（无需 pytest）
python3 run_tests.py

# 使用 pytest
python -m pytest tests/ -v
```

当前: **21/21 测试通过** ✅

## 📝 示例列表

```bash
python examples.py basic         # 基础用法
python examples.py inline        # 内联技能
python examples.py memory        # 带记忆的对话
python examples.py streaming     # 流式输出
python examples.py custom-tools  # 自定义工具
python examples.py loader        # 高级加载器
python examples.py traffic-ai    # 交通AI用例
```

## License

MIT
