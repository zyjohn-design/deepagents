
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 清除代理设置
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from skills_agent.config import Settings, LLMSettings
from skills_agent.graph import create_skills_agent
from skills_agent.llm import create_llm

def test_logging():
    print(">>> Starting logging test")
    
    llm_settings = LLMSettings(
        model="local:Qwen3-32B",
        base_url="http://219.147.31.25:30001/v1",
        api_key="EMPTY",
    )
    settings = Settings(llm=llm_settings)
    
    skill_dir = project_root / "skills"
    llm = create_llm(settings.llm)
    
    # 这应该触发日志输出
    create_skills_agent(
        model=llm,
        skill_dirs=[str(skill_dir)],
        settings=settings,
        work_dir=str(project_root / "workspace")
    )
    
    print(">>> Logging test finished")

if __name__ == "__main__":
    test_logging()
