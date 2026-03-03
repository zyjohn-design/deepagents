# Skills Agent — LangGraph 1.0 通用技能代理框架

基于 **LangGraph 1.0** 构建的生产级技能解析与执行框架。

## 核心特性

| 特性 | 说明 |
|------|------|
| **双模式执行** | Mode A: Agent 自主推理(ReAct) / Mode B: Pipeline 确定性执行 |
| **智能检索** | 大文件分段索引 + 关键词搜索，精准获取所需内容，节省 token |
| **技能组合** | `invoke_skill` 实现技能间引用，构建复杂工作流 |
| **动态参数** | `${step.N.output}` 模板变量，步骤间数据传递 |
| **脚本参数** | `run_script` 支持动态命令行参数和环境变量注入 |
| **云+本地 LLM** | OpenAI、Anthropic、DeepSeek + Ollama、vLLM、LM Studio |
| **统一配置** | YAML / 环境变量 / 代码三层叠加 |

## 项目结构

```
skills_agent/
├── skills_agent/              # 核心包 (10 模块)
│   ├── config.py             # 统一配置
│   ├── exceptions.py         # 异常层级
│   ├── log.py                # 结构化日志
│   ├── llm.py                # LLM 工厂 (云+本地)
│   ├── state.py              # AgentState + 快照审计
│   ├── models.py             # Skill/SkillStep + 模板变量 + 参数解析
│   ├── reference.py          # ★ 智能检索: 分段索引 + 关键词搜索
│   ├── loader.py             # 多路径发现、加载、匹配
│   ├── executor.py           # ★ Pipeline 执行: 7种action + 自定义
│   └── graph.py              # ★ LangGraph: 9种tool + create_skills_agent
├── example_skills/            # 示例技能 (含完整 reference + script)
├── run_tests.py              # 65 个测试
└── config.example.yaml       # 配置模板
```

## 双模式执行架构

```
┌──────────────────────────────────────────────────────────────┐
│  Mode A: Agent (ReAct)              Mode B: Pipeline          │
│  LLM 自主选择 tool                   SKILL.md 固定步骤         │
│                                                               │
│  9 tools:                           7 actions:                │
│   list_skills                        read_file                │
│   read_skill                         read_reference           │
│   read_skill_reference               search_reference ★       │
│   search_skill_reference ★           run_script (动态参数) ★   │
│   get_reference_toc ★                invoke_skill ★            │
│   invoke_skill ★                     llm_reason               │
│   write_todos                        [自定义 actions]          │
│   execute_skill_workflow ────────▶ 触发 Mode B                │
│   run_skill_script                                            │
└──────────────────────────────────────────────────────────────┘
★ = 本次新增

run_command ★执行任意 shell 命令

已添加。现在 Mode A 有 10 个工具：
工具用途list_skills列出可用技能read_skill加载技能完整指令read_skill_reference读取完整 reference 文件search_skill_reference关键词搜索 reference（省 token）get_reference_toc查看 reference 目录结构invoke_skill调用子技能write_todos规划任务execute_skill_workflow触发 Mode B pipelinerun_skill_script执行技能内脚本（带 args/command）run_command ★执行任意 shell 命令
```

## SKILL.md 编写指南

### Mode A（推荐）— 自然语言，LLM 自主决策

```markdown
## 核心工作流
步骤1: 读取活动文档，提取关键信息
步骤2: 参考知识库中的历史管控方案
步骤3: 生成管控措施建议
```

### Mode B — 显式 action + 参数

```markdown
## 核心工作流
步骤1: 查询认证文档 [action=search_reference, query=认证 token, file=api_doc.md]
步骤2: 提取数据格式 [action=search_reference, query=数据格式 schema, file=api_doc.md]
步骤3: 运行转换脚本 [action=run_script, script=transform.py, args=--format json]
步骤4: 验证结果 [action=invoke_skill, skill=data-validator, input=${step.3.output}]
```

### 支持的 Actions

| Action | 说明 | 参数 |
|--------|------|------|
| `read_file` | 读取文件 | `path` 或 `file` |
| `read_reference` | 读取完整 reference 文件 | `reference=文件名` |
| `search_reference` | **智能搜索** reference 段落 | `query=关键词`, `file=文件名`, `section=段落标题`, `top_k=3` |
| `run_script` | 执行脚本（支持参数） | `script=文件名`, `args=--key val`, `env=K=V` |
| `invoke_skill` | **调用子技能** | `skill=技能名`, `input=${step.N.output}` |
| `llm_reason` | LLM 推理（默认） | — |
| 自定义 | `executor.register_action()` | 自定义 |

### 模板变量

在步骤参数中使用 `${...}` 引用前序数据：

| 模板 | 说明 |
|------|------|
| `${step.N.output}` | 第 N 步的输出文本 |
| `${step.N.artifact.KEY}` | 第 N 步的特定工件 |
| `${context.KEY}` | 上下文中的值 |

## 智能检索 — 大文件不再浪费 token

**问题**：API 文档 50KB，但你只需要认证接口的参数。全部加载浪费 token。

**方案**：`ReferenceManager` 按 markdown 标题分段，建关键词倒排索引：

```python
from skills_agent import ReferenceManager

mgr = ReferenceManager()
mgr.index_file("api_doc.md", large_doc_content)

# 查看目录 (LLM 可以先看 TOC 再决定要哪段)
print(mgr.get_toc())

# 关键词搜索 → 只返回相关段落
results = mgr.search("认证 token endpoint", top_k=2)

# 精确段落获取
section = mgr.get_section("api_doc.md", "Token Endpoint")
```

在 SKILL.md 中直接用 `search_reference` action：
```
步骤1: 查询接口参数 [action=search_reference, query=POST /users, file=api_doc.md]
```

Mode A 中用 `search_skill_reference` tool（LLM 自动调用）。

## 技能组合 — invoke_skill

技能可以调用其他技能，形成组合链：

```yaml
# data-pipeline/SKILL.md
depends_on_skills: [data-validator]
```

```markdown
步骤4: 验证数据 [action=invoke_skill, skill=data-validator, input=${step.3.output}]
```

子技能执行完整工作流，返回最终输出。Mode A 中也有 `invoke_skill` tool。

## 快速开始

```python
from skills_agent import create_skills_agent, get_initial_state

agent = create_skills_agent(
    model="openai:gpt-4o",
    skill_dirs=["./my_skills/"],
)
result = agent.invoke(get_initial_state("处理数据"))
```

### 本地模型

```python
agent = create_skills_agent(model="ollama:qwen2.5:72b")
```

### Pipeline 直接执行 (不需要 LLM)

```python
from skills_agent import SkillLoader, SkillExecutor

loader = SkillLoader(skill_dirs=["./skills/"])
loader.discover()
skill = loader.load_skill("data-pipeline")

executor = SkillExecutor(skill_loader=loader)
result = executor.execute_workflow(skill)
print(result.summary())
```

## 测试

```bash
python3 run_tests.py
```

```
65 passed, 0 failed

覆盖:
  1. Core (Config/Exceptions/State/Log)
  2. Models (Frontmatter/Step解析/模板变量)
  3. Reference (分段/搜索/TOC)
  4. Loader (发现/加载/匹配)
  5. Mode B (所有7种action)
  6. Mode A (9种tool, 需要langchain)
  7. Mixed (A触发B)
  8. Integration (完整生命周期)
```

## License

MIT
