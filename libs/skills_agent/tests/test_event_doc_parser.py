
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量（如果需要的话，但这里我们主要通过 Settings 注入）# 加载环境变量
load_dotenv()

# 清除代理设置，避免干扰本地请求
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# 添加项目根目录到 sys.path，以便能导入 skills_agent
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from skills_agent_v2.config import Settings, LLMSettings, LogSettings
from skills_agent_v2.loader import SkillLoader
from skills_agent_v2.executor import SkillExecutor
from skills_agent_v2.graph import create_skills_agent
from skills_agent_v2.state import StateManager

def test_event_doc_parser_skill():
    # 1. 配置 LLM 设置
    llm_settings = LLMSettings(
        model="local:Qwen3.5-35B-A3B", # 使用 local provider  Qwen3-32B
        base_url="http://219.147.31.25:8983/v1",
        api_key="EMPTY",
        max_tokens=2048, # 降低 Max Tokens 避免超时或内存问题
        temperature=0.0,
    )
    # 添加环境变量 OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_API_BASE"] = "http://219.147.31.25:8983/v1"
    os.environ["MODEL_NAME"] = "Qwen3.5-35B-A3B"


    # 1.5 配置日志设置
    log_settings = LogSettings(
        file="agent_test.log",
        level="DEBUG",
        show_timestamp=True,
        show_module=True,
        file_max_bytes=10 * 1024 * 1024, # 10 MB
        file_backup_count=3
    )

    settings = Settings(llm=llm_settings, log=log_settings)

    # 打印配置信息
    print("=== Test Configuration ===")
    print(f"Model Provider: {settings.llm.provider}")
    print(f"Model Name: {settings.llm.model_name}")
    print(f"Base URL: {settings.llm.base_url}")
    print(f"Max Tokens: {settings.llm.max_tokens}")
    print("==========================")

    # 2. 设置 Skill 路径
    # 使用用户指定的 skills 目录
    skill_dir = Path("/Users/zhangyong/Documents/project/deepagents/skills")

    print(f"Scanning skills in: {skill_dir}")
    if not skill_dir.exists():
        print(f"Error: Skill directory not found at {skill_dir}")
        return

    # 3. 创建 LLM (用于测试连通性和传入 Agent)
    from skills_agent_v2.llm import create_llm
    llm = create_llm(settings.llm)

    # 尝试直接调用 llm 验证连通性
    print("\n>>> Testing LLM Connectivity...")
    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content="Hello")])
        print(f"LLM Response: {resp.content}")
    except Exception as e:
        print(f"LLM Connectivity Failed: {e}")
        # return # 暂时不退出，继续跑图

    # 4. 创建 Agent Graph (使用高层 API)
    compiled_graph = create_skills_agent(
        model=llm,
        skill_dirs=[str(skill_dir)],
        settings=settings,
        work_dir=str(project_root / "workspace")
    )

    # 6. 构造输入并运行
    file_path = "/Users/zhangyong/Documents/project/deepagents/skills/event-doc-parser/tests/新意见征求稿附件跨年夜江北片区实施方案.md"
    user_input = f"请解析大型活动文件：{file_path}"

    print(f"\n>>> User Input: {user_input}")

    initial_state = StateManager.create(user_input)

    # 运行图
    print("\n>>> Agent Running...")

    try:
        # 使用 stream 而不是 invoke 以便看到中间步骤
        # recursion_limit 防止死循环
        config = {"recursion_limit": 5000}

        for event in compiled_graph.stream(initial_state, config=config):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                if "messages" in value:
                    for msg in value["messages"]:
                        role = msg.type
                        content = msg.content

                        print(f"[{role.upper()}]: {content}")

                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  [TOOL CALL]: {tc['name']} args={tc['args']}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_event_doc_parser_skill()
