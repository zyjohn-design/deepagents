#!/usr/bin/env python3
"""
活动文档解析工具 (基于 LLM + Instructor)

功能：
1. 文档转换：支持 PDF/DOCX 格式，调用 MinerU 接口转换为 Markdown
2. 智能提取：利用 Instructor + Pydantic 实现结构化流式提取
3. 知识库集成：自动加载场景专属知识库和示例
4. 动态模型：根据知识库动态生成 Pydantic 模型

使用方法:
    uv run python scripts/event_doc_parser.py <input_file> <scene_type> [event_date]

示例:
    uv run python scripts/event_doc_parser.py docs/2026跨年夜方案.pdf newyear 2026-12-31
"""

import sys
import json
import os
import requests
import re
from pathlib import Path
from typing import List, Optional, Iterable, Dict, Any, Set
from enum import Enum

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

# 尝试导入项目配置
try:
    from src.config.settings import settings
except ImportError:
    # 动态查找项目根目录（包含 src 目录的那一级）
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "src").exists() and (parent / "src" / "config").exists():
            sys.path.append(str(parent))
            break
    try:
        from src.config.settings import settings
    except ImportError:
        print("错误: 无法加载项目配置 (src.config.settings)，请确保在项目根目录下运行。")
        sys.exit(1)

# ==============================================================================
# 2. 文档转换服务 (MinerU)
# ==============================================================================

class MinerUClient:
    """MinerU 文档转换客户端"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        # 优先使用环境变量或传入参数，否则使用默认值
        self.api_url = api_url or os.getenv("MINERU_API_URL", "http://mineru-service:8000/convert")
        self.api_key = api_key or os.getenv("MINERU_API_KEY", "")
        
    def convert_to_markdown(self, file_path: Path) -> str:
        """调用 MinerU 接口将 PDF/DOCX 转换为 Markdown"""
        if not file_path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")
            
        print(f"正在调用 MinerU 转换文件: {file_path.name} ...")
        
        # 模拟转换 (如果没有真实服务)
        if "mineru-service" in self.api_url:
            print("警告: 未配置有效的 MinerU URL，使用模拟转换（返回文件名作为标题）。")
            return f"# {file_path.stem}\n\n(模拟转换内容) 这是从 {file_path.name} 转换得到的 Markdown 内容。\n\n"
            
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                response = requests.post(self.api_url, files=files, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result.get("markdown", "")
        except Exception as e:
            raise RuntimeError(f"MinerU 转换失败: {str(e)}")

# ==============================================================================
# 3. 提取服务 (Instructor)
# ==============================================================================

class EventDocExtractor:
    def __init__(self, scene_type: str, event_date: Optional[str] = None):
        self.scene_type = scene_type
        self.event_date = event_date
        self.knowledge_base = self._load_file(f"references/{scene_type}_knowledge.md")
        self.examples = self._load_file(f"references/extraction_examples_{scene_type}.md")
        
        # 优先从环境变量获取配置
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MODEL_URL") or settings.MODEL_URL
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or settings.API_KEY
        self.model_name = os.getenv("OPENAI_MODEL_NAME") or os.getenv("MODEL_NAME") or settings.MODEL_NAME

        # 初始化 Instructor 客户端
        self.client = instructor.from_openai(
            OpenAI(
                base_url=base_url,
                api_key=api_key,
            ),
            mode=instructor.Mode.JSON,
        )
        
        # 动态初始化模型
        self._init_dynamic_models()

    def _load_file(self, rel_path: str) -> str:
        """加载参考文件"""
        base_path = Path(__file__).resolve().parent.parent
        file_path = base_path / rel_path
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return ""

    def _parse_knowledge_enums(self):
        """解析知识库中的枚举值"""
        phases = []
        area_names = []
        area_types = set()

        # Parse PHASE_ENUM
        phase_pattern = r'### PHASE_ENUM.*?```yaml(.*?)```'
        match = re.search(phase_pattern, self.knowledge_base, re.DOTALL)
        if match:
            content = match.group(1)
            names = re.findall(r'-\s+name:\s+([^\n\r]+)', content)
            phases = [n.strip() for n in names]

        # Parse AREA_TYPE_ENUM
        area_pattern = r'### AREA_TYPE_ENUM.*?```yaml(.*?)```'
        match = re.search(area_pattern, self.knowledge_base, re.DOTALL)
        if match:
            content = match.group(1)
            names = re.findall(r'name:\s+([^\n\r]+)', content)
            area_names = [n.strip() for n in names]
            
            types = re.findall(r'type:\s+([^\n\r]+)', content)
            area_types.update([t.strip() for t in types])

        # Fallbacks to prevent crash if parsing fails
        if not phases: 
             print("Warning: No phases found in knowledge base.")
             phases = []
        if not area_names: 
             print("Warning: No area names found in knowledge base.")
             area_names = []
        if not area_types: 
             print("Warning: No area types found in knowledge base.")
             area_types = []
        
        return phases, area_names, list(area_types)

    def _init_dynamic_models(self):
        """动态构建 Pydantic 模型"""
        phases, area_names, area_types = self._parse_knowledge_enums()
        
        # Create dynamic Enums
        # We inherit from str so they serialize as strings automatically in many contexts
        # and compare equal to strings
        PhaseEnum = Enum('PhaseEnum', {n: n for n in phases}, type=str)
        AreaNameEnum = Enum('AreaNameEnum', {n: n for n in area_names}, type=str)
        AreaTypeEnum = Enum('AreaTypeEnum', {n: n for n in area_types}, type=str)

        # Create Models
        self.AffectedAreaModel = create_model(
            'AffectedArea',
            area_name=(AreaNameEnum, Field(..., description="区域或道路名称，必须严格匹配知识库定义")),
            type=(AreaTypeEnum, Field(..., description="区域类型，严格遵守知识库定义")),
            phase=(PhaseEnum, Field(..., description="所属管控阶段，如果全天管控则填'全时段'")),
            start_time=(Optional[str], Field(None, description="管控开始时间，格式 YYYY-MM-DD HH:MM")),
            end_time=(Optional[str], Field(None, description="管控结束时间，格式 YYYY-MM-DD HH:MM")),
            boundaries=(Optional[str], Field(None, description="区域边界描述")),
            control_measures=(str, Field(..., description="管控措施")),
            __base__=BaseModel
        )

        self.TaskModel = create_model(
            'Task',
            phase=(PhaseEnum, Field(..., description="阶段名称")),
            start_time=(str, Field(..., description="开始时间，格式 YYYY-MM-DD HH:MM")),
            end_time=(Optional[str], Field(None, description="结束时间，格式 YYYY-MM-DD HH:MM 或描述性文字")),
            control_type=(str, Field(..., description="管控类型")),
            description=(str, Field(..., description="任务简述")),
            action=(str, Field(..., description="执行动作")),
            affected_area=(AreaNameEnum, Field(..., description="关联的区域名称")),
            __base__=BaseModel
        )

        self.EventControlDataModel = create_model(
            'EventControlData',
            affected_areas=(List[self.AffectedAreaModel], Field(default_factory=list, description="本次提取涉及的区域列表")),
            tasks=(List[self.TaskModel], Field(default_factory=list, description="本次提取的任务列表")),
            __base__=BaseModel
        )

    def split_text(self, text: str, chunk_size: int = 3000, overlap: int = 500) -> List[str]:
        """简单的文本分块策略"""
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # 尝试在换行符处切分，优先匹配双换行（段落）
            last_paragraph = text.rfind('\n\n', start, end)
            if last_paragraph != -1 and last_paragraph > start + chunk_size // 2:
                end = last_paragraph
            else:
                # 其次匹配单换行
                last_newline = text.rfind('\n', start, end)
                if last_newline != -1 and last_newline > start + chunk_size // 2:
                    end = last_newline
            
            chunks.append(text[start:end])
            start = end - overlap # 重叠
            
        return chunks

    def stream_extraction(self, text: str) -> Iterable[Any]:
        """流式提取管控信息（支持长文档分块）"""
        
        base_system_prompt = f"""你是一个交通管控专家。请深入分析文档内容，**穷尽式**地提取所有交通管控措施，并拆解为细粒度的 EventControlData 对象流式输出。
        
        ### 核心原则
        1. **全面性**：不要遗漏任何一个管控点位或时间节点。
        2. **细粒度**：将复杂管控行动拆解为独立的子任务（如开始封控、车辆清空、解除封控）。
        3. **区域归纳**：文档中提到的所有具体地点（如"沿江大道"、"长江大桥"等）都必须归纳到知识库定义的三大管控区（核心管控区、中端疏导区、远端分流区）中。**不要创造新的区域名称**。
        4. **严格枚举**：`phase` 字段必须严格使用知识库中定义的值。
        5. **上下文时间**：对于没有具体时间的任务，必须结合上下文（如上一条任务的时间、所在段落的时间背景）进行逻辑推断，**不可留空**。
        
        ### 场景上下文
        - 场景类型: {self.scene_type}
        - 活动日期: {self.event_date or '未指定'}
        
        ### 知识库 (枚举定义与规则)
        {self.knowledge_base}
        
        ### 输出要求
        1. 严格遵守 JSON 结构。
        2. **区域优先原则**：请在输出任务(tasks)之前，先输出其关联的受影响区域(affected_areas)。
        3. 任务中的 `affected_area` 字段必须引用已定义的 `area_name`。
        4. 时间格式必须标准化 (YYYY-MM-DD HH:MM)。
        5. **跨年处理**：如果活动日期在年底（如 12月31日），文档中提到的“次日”、“元旦”、“1月1日”或凌晨时段（如 00:00-05:00），应自动推断为下一年。
        """
        
        # 文本分块处理
        chunks = self.split_text(text)
        if len(chunks) > 1:
            print(f"文档较长，已切分为 {len(chunks)} 个块进行处理 (重叠 {500} 字符)...")
        
        last_context_time = None
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"--> 正在处理第 {i+1}/{len(chunks)} 块 ({len(chunk)} 字符)...")
                
            # 构建消息历史
            current_system_prompt = base_system_prompt
            if last_context_time:
                 current_system_prompt += f"\n\n### 上下文提示\n上一个片段最后提取到的时间点是：**{last_context_time}**。如果本片段开头没有明确时间，请参考此时间或根据逻辑推断。"

            messages = [
                {"role": "system", "content": current_system_prompt},
            ]
            
            # 如果有示例，添加到上下文中 (截取一部分作为 few-shot)
            if self.examples:
                messages.append({"role": "user", "content": "请提供一些提取示例。"})
                messages.append({"role": "assistant", "content": f"好的，这是参考示例：\n{self.examples[:2000]}..."}) # 截断防止超长
            
            messages.append({"role": "user", "content": f"请解析以下文档片段：\n\n{chunk}"})

            # 调用 LLM (使用 Iterable 模式)
            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=Iterable[self.EventControlDataModel],
                    messages=messages,
                    temperature=0.1, # 低温度保证准确性
                    stream=True,     # 开启流式传输
                    max_retries=3,
                )
                
                for data in stream:
                    # 尝试更新上下文时间 (取当前数据块中最后一个任务的时间)
                    if data.tasks:
                         # 过滤出有效的日期时间字符串
                         valid_times = [t.start_time for t in data.tasks if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", t.start_time)]
                         if valid_times:
                             last_context_time = valid_times[-1]
                    yield data
                    
            except Exception as e:
                print(f"块 {i+1} 处理失败: {str(e)}")
                continue

# ==============================================================================
# 4. 主流程
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="活动文档解析工具")
    parser.add_argument("input_file", type=Path, help="输入文件路径")
    parser.add_argument("scene_type", type=str, nargs="?", help="场景类型 (可选，若不填则自动检测)")
    parser.add_argument("event_date", type=str, nargs="?", help="活动日期")
    parser.add_argument("--output", "-o", type=Path, help="输出文件路径")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    event_date = args.event_date
    scene_type = args.scene_type
    
    print(f"DEBUG: Parsed args - input_file={input_file}, scene_type='{scene_type}', event_date='{event_date}'")

    if not input_file.exists():
        print(f"错误: 文件 {input_file} 不存在")
        sys.exit(1)

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # 1. 文档转换
    text = ""
    if input_file.suffix.lower() in ['.pdf', '.docx', '.doc']:
        print("检测到二进制文档，启动 MinerU 转换...")
        converter = MinerUClient()
        try:
            text = converter.convert_to_markdown(input_file)
            print(f"转换成功，文档长度: {len(text)} 字符")
        except Exception as e:
            print(f"转换失败: {str(e)}")
            sys.exit(1)
    else:
        print("检测到文本/Markdown 文档，直接读取...")
        text = input_file.read_text(encoding="utf-8")

    # 智能参数解析
    import re
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    
    if scene_type:
        scene_type = scene_type.strip()
        
    if scene_type and date_pattern.match(scene_type):
        if not event_date:
             print(f"提示: 检测到日期格式参数 '{scene_type}'，自动识别为 event_date")
             event_date = scene_type
             scene_type = None

    if not scene_type:
        print("警告: 未提供 scene_type，且脚本内自动检测已被禁用。默认使用 'other'。")
        scene_type = "other"
    
    # 2. 智能提取
    extractor = EventDocExtractor(scene_type, event_date)
    print(f"--- 开始流式提取 (模型: {extractor.model_name}) ---")
    
    all_data = {
        "event_name": input_file.stem,
        "event_type": scene_type,
        "affected_areas": [],
        "tasks": []
    }
    
    # 用于去重的集合
    seen_tasks: Set[tuple] = set()
    seen_areas: Set[str] = set()

    try:
        data_stream = extractor.stream_extraction(text)
        
        for i, data_block in enumerate(data_stream):
            print(f"\n[收到第 {i+1} 组数据]", flush=True)
            
            areas = data_block.affected_areas or []
            tasks = data_block.tasks or []
            
            print(f"  > 本组包含: {len(areas)} 个区域, {len(tasks)} 个任务", flush=True)
            
            # 实时打印并聚合 (带去重)
            for area in areas:
                # model_dump is the new dict() in Pydantic v2
                # mode='json' converts Enums to strings
                area_dict = area.model_dump(mode='json')
                
                print(f"  * 区域: {area_dict['area_name']} [{area_dict['type']}]", flush=True)
                print(f"    - 措施: {area_dict['control_measures']}", flush=True)
                if area_dict.get('boundaries'):
                    print(f"    - 范围: {area_dict['boundaries']}", flush=True)

                # 区域去重: 基于 area_name
                if area_dict['area_name'] not in seen_areas:
                    seen_areas.add(area_dict['area_name'])
                    all_data["affected_areas"].append(area_dict)
                    
            for task in tasks:
                task_dict = task.model_dump(mode='json')
                
                time_str = f"{task_dict['start_time']}"
                if task_dict.get('end_time'):
                    time_str += f" 至 {task_dict['end_time']}"
                
                print(f"  * 任务: [{time_str}] {task_dict['phase']}", flush=True)
                print(f"    - 内容: {task_dict['description']}", flush=True)
                print(f"    - 动作: {task_dict['action']} | 类型: {task_dict['control_type']} | 区域: {task_dict['affected_area']}", flush=True)
                
                # 任务去重: 基于 (时间, 动作, 区域) 的指纹
                # 使用 tuple 作为 key
                task_fingerprint = (task_dict['start_time'], task_dict['action'], task_dict['affected_area'])
                
                if task_fingerprint not in seen_tasks:
                    seen_tasks.add(task_fingerprint)
                    all_data["tasks"].append(task_dict)
                else:
                    print(f"    [重复任务已忽略]")
                
    except Exception as e:
        print(f"\n提取过程中发生错误: {str(e)}")

    # 对所有任务按开始时间排序
    try:
        all_data["tasks"].sort(key=lambda x: x.get("start_time", ""))
        print("\n已对任务按开始时间排序。")
    except Exception as e:
        print(f"\n任务排序失败: {str(e)}")

    # 3. 保存结果
    if args.output:
        output_file = args.output
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        skills_dir = Path(__file__).resolve().parent.parent
        output_dir = skills_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{input_file.stem}_extracted.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n提取完成！结果已保存至: {output_file}")
    print(f"共提取 {len(all_data['affected_areas'])} 个区域，{len(all_data['tasks'])} 个任务。")

if __name__ == "__main__":
    main()
