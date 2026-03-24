# Skills Agent 打包与分发指南

本文档介绍如何将 `skills-agent` 模块打包为 Python 标准分发格式（Wheel 和 Source Archive），以便在其他系统或项目中安装和使用。

## 1. 环境准备

本项目使用 `uv` 进行依赖管理和构建，同时也支持标准的 `build` 工具。

### 确保已安装构建工具

如果你使用 `uv`（推荐）：
```bash
# uv 通常已预装在开发环境中
uv --version
```

如果你使用标准 `pip`：
```bash
pip install build
```

## 2. 执行打包

在 `libs/skills_agent` 目录下执行构建命令。

### 使用 `uv` 构建

```bash
cd libs/skills_agent
uv build
```

### 使用 `build` 构建

```bash
cd libs/skills_agent
python -m build
```

## 3. 构建产物

构建完成后，会在 `dist/` 目录下生成以下文件：

- **Wheel 包 (.whl)**: 预编译的二进制包，安装速度快（推荐）。
  - 例如：`skills_agent-0.1.0-py3-none-any.whl`
- **源码包 (.tar.gz)**: 包含源代码的压缩包。
  - 例如：`skills_agent-0.1.0.tar.gz`

## 4. 在其他系统中安装

### 方法 A: 直接安装 .whl 文件

将生成的 `.whl` 文件复制到目标机器，然后运行：

```bash
pip install skills_agent-0.1.0-py3-none-any.whl
```

### 方法 B: 作为依赖引入

如果你在另一个项目中使用 `pyproject.toml` 或 `requirements.txt`，可以指向该文件。

**requirements.txt**:
```text
./libs/skills_agent/dist/skills_agent-0.1.0-py3-none-any.whl
```

**pyproject.toml (使用 uv)**:
```toml
[project]
dependencies = [
    "skills-agent @ file:///path/to/skills_agent-0.1.0-py3-none-any.whl",
]
```

## 5. 使用示例

安装完成后，你可以在 Python 代码中直接导入 `skills_agent`。

### 基础用法

```python
import os
from skills_agent import create_skills_agent, get_initial_state

# 1. 设置 API Key (如果未在环境变量中设置)
os.environ["OPENAI_API_KEY"] = "sk-..."

# 2. 创建 Agent
# skill_dirs 指定技能定义文件(SKILL.md)所在的目录
agent = create_skills_agent(
    model="openai:gpt-4o",
    skill_dirs=["/path/to/your/skills"],
    temperature=0.1
)

# 3. 准备初始状态
initial_state = get_initial_state("请帮我分析一下 data/ 目录下的 CSV 文件")

# 4. 运行 Agent
# config 中可以传递线程 ID 用于记忆功能
config = {"configurable": {"thread_id": "123"}}

print("Agent 正在运行...")
for event in agent.stream(initial_state, config=config):
    # 打印每一步的事件
    for node_name, node_state in event.items():
        print(f"--- {node_name} ---")
        # 查看最新的消息
        if "messages" in node_state and node_state["messages"]:
            last_msg = node_state["messages"][-1]
            print(f"[{last_msg.type}]: {last_msg.content}")

# 5. 获取最终结果
# 注意：通常 stream 循环结束就是完成了，但也可以再次 invoke 获取最终状态
# final_result = agent.invoke(initial_state, config=config)
```

### 进阶：自定义 LLM 配置

```python
from skills_agent import create_skills_agent, LLMSettings

# 使用自定义 LLM 设置
llm_settings = LLMSettings(
    provider="anthropic",
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=4096
)

agent = create_skills_agent(
    model=llm_settings.model, # 或者直接传 LLMSettings 对象如果支持
    skill_dirs=["./skills"],
    verbose=True
)
```

## 6. 常见问题

**Q: 安装时提示缺少依赖？**
A: `skills-agent` 依赖 `langchain`, `langgraph`, `instructor` 等库。使用 `pip install skills_agent-xxx.whl` 会自动尝试从 PyPI 安装这些依赖。如果目标环境无法连接外网，你需要离线下载这些依赖包并一同安装。

**Q: 如何验证安装是否成功？**
A: 运行以下命令：
```bash
python -c "import skills_agent; print(skills_agent.__version__)"
```
如果输出了版本号，说明安装成功。
