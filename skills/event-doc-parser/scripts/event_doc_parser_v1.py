#!/usr/bin/env python3
"""
活动文档解析工具 v2 (基于 LLM + Instructor)

核心优化：
1. 三阶段提取流水线：结构识别 → 分阶段区域提取 → 任务提取
2. 管控区域按 (area_name, phase) 组合键去重，支持同一区域在不同阶段有不同管控措施
3. 基于 Markdown 标题的智能分段，避免切断完整阶段
4. 全局区域上下文注入，确保跨分块引用一致性

使用方法:
    uv run python scripts/event_doc_parser.py <input_file> <scene_type> [event_date]

示例:
    uv run python scripts/event_doc_parser.py docs/2026跨年夜方案.md newyear 2026-12-31
"""

import sys
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Iterable, Any, Set, Tuple, Dict
from enum import Enum

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, create_model, model_validator

# 尝试导入项目配置
try:
    from src.config.settings import settings
except ImportError:
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
# 1. 文档结构解析器 — 将 Markdown 按标题层级切分为语义段落
# ==============================================================================

class DocumentSection:
    """文档语义段落"""
    def __init__(self, heading: str, level: int, content: str, start_pos: int):
        self.heading = heading
        self.level = level
        self.content = content
        self.start_pos = start_pos

    def full_text(self) -> str:
        prefix = "#" * self.level + " " if self.level > 0 else ""
        return f"{prefix}{self.heading}\n{self.content}"

    def __repr__(self):
        return f"Section(L{self.level}: {self.heading[:30]}..., {len(self.content)} chars)"


def parse_document_structure(text: str) -> List[DocumentSection]:
    """基于 Markdown 标题层级切分文档为语义段落。
    
    相比按字符数盲切，这种方式能保证：
    - 不会切断一个完整的阶段描述
    - 每个段落有明确的上下文标题
    """
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    matches = list(heading_pattern.finditer(text))
    
    if not matches:
        # 没有 Markdown 标题，尝试用中文序号切分
        cn_heading_pattern = re.compile(
            r'^(?:(?:[一二三四五六七八九十]+)、|（[一二三四五六七八九十]+）|(?:\d+)[.、])\s*(.+)$',
            re.MULTILINE
        )
        matches = list(cn_heading_pattern.finditer(text))
    
    if not matches:
        return [DocumentSection(heading="全文", level=0, content=text, start_pos=0)]
    
    sections = []
    for i, match in enumerate(matches):
        heading_text = match.group(0).strip()
        # 判断层级
        if heading_text.startswith('#'):
            level = len(match.group(1))
            title = match.group(2).strip()
        else:
            level = 1
            title = heading_text
        
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        
        sections.append(DocumentSection(
            heading=title, level=level, content=content, start_pos=match.start()
        ))
    
    # 如果第一个标题前有内容，作为前言
    if matches and matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.insert(0, DocumentSection(
                heading="文档前言", level=0, content=preamble, start_pos=0
            ))
    
    return sections


def merge_sections_to_chunks(
    sections: List[DocumentSection],
    max_chunk_chars: int = 4000,
    min_chunk_chars: int = 200
) -> List[str]:
    """将语义段落合并为适合 LLM 处理的 chunk。
    
    原则：
    - 同一个一级标题下的内容尽量在一个 chunk 内
    - 超长段落单独成 chunk（可被进一步截断）
    - 每个 chunk 带上其所属的标题链路上下文
    """
    if not sections:
        return []
    
    chunks = []
    current_chunk_parts = []
    current_len = 0
    current_parent_heading = ""
    
    for section in sections:
        section_text = section.full_text()
        section_len = len(section_text)
        
        # 更新父标题（一级标题）
        if section.level <= 2:
            current_parent_heading = section.heading
        
        # 如果单个 section 就超长，单独处理
        if section_len > max_chunk_chars:
            # 先保存已有的
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_len = 0
            
            # 对超长 section 按段落切分
            paragraphs = section.content.split('\n\n')
            sub_parts = [f"## 上下文: {current_parent_heading}\n### {section.heading}"]
            sub_len = len(sub_parts[0])
            
            for para in paragraphs:
                if sub_len + len(para) > max_chunk_chars and sub_len > min_chunk_chars:
                    chunks.append("\n\n".join(sub_parts))
                    sub_parts = [f"## 上下文: {current_parent_heading}\n### {section.heading} (续)"]
                    sub_len = len(sub_parts[0])
                sub_parts.append(para)
                sub_len += len(para)
            
            if sub_parts:
                chunks.append("\n\n".join(sub_parts))
            continue
        
        # 正常合并
        if current_len + section_len > max_chunk_chars and current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts = []
            current_len = 0
        
        # 如果是新 chunk 且有父标题，添加上下文
        if not current_chunk_parts and current_parent_heading:
            context_line = f"[上下文: {current_parent_heading}]"
            current_chunk_parts.append(context_line)
            current_len += len(context_line)
        
        current_chunk_parts.append(section_text)
        current_len += section_len
    
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))
    
    return chunks


# ==============================================================================
# 2. Pydantic 模型定义 — 三阶段提取的数据模型
# ==============================================================================

class PhaseDetectionResult(BaseModel):
    """第一阶段：文档结构与阶段识别结果"""
    phase_name: str = Field(..., description="阶段名称，严格匹配知识库枚举")
    phase_code: str = Field(..., description="阶段代码")
    time_range_description: str = Field("", description="该阶段的时间范围描述（从文档提取）")
    section_summary: str = Field("", description="该阶段包含的主要内容概要（50字以内）")


class PhaseDetectionOutput(BaseModel):
    """第一阶段输出"""
    phases_detected: List[PhaseDetectionResult] = Field(
        default_factory=list,
        description="从文档中识别到的所有管控阶段"
    )
    global_time_context: str = Field(
        "", description="全局时间上下文，如 '活动从12月31日15:00开始，至1月1日3:00结束'"
    )


class AreaExtractionItem(BaseModel):
    """第二阶段：单个管控区域提取"""
    area_name: str = Field(..., description="区域名称，严格匹配知识库枚举定义")
    type: str = Field(..., description="区域类型，严格匹配知识库定义")
    phase: str = Field(..., description="该管控措施生效的阶段名称")
    start_time: Optional[str] = Field(None, description="管控开始时间 YYYY-MM-DD HH:MM")
    end_time: Optional[str] = Field(None, description="管控结束时间")
    boundaries: Optional[str] = Field(None, description="边界描述，尽量完整保留原文路名")
    control_measures: str = Field(..., description="该阶段的具体管控措施")


class AreaExtractionOutput(BaseModel):
    """第二阶段输出：分阶段的区域列表"""
    affected_areas: List[AreaExtractionItem] = Field(
        default_factory=list,
        description="管控区域列表。**关键**：同一区域如果在不同阶段有不同管控措施，必须输出多条记录"
    )


class TaskExtractionItem(BaseModel):
    """第三阶段：单个任务提取"""
    phase: str = Field(..., description="阶段名称，严格匹配知识库枚举")
    start_time: str = Field(..., description="开始时间 YYYY-MM-DD HH:MM")
    end_time: Optional[str] = Field(None, description="结束时间")
    control_type: str = Field(..., description="管控类型")
    description: str = Field(..., description="任务简述，200字以内")
    action: str = Field(..., description="执行动作")
    affected_area: str = Field(..., description="关联的区域名称，必须引用已定义的 area_name")


class TaskExtractionOutput(BaseModel):
    """第三阶段输出"""
    tasks: List[TaskExtractionItem] = Field(
        default_factory=list,
        description="管控任务列表，按时间顺序"
    )


# ==============================================================================
# 3. 核心提取引擎
# ==============================================================================

class EventDocExtractor:
    """三阶段提取引擎
    
    Phase 1: 结构识别 — 快速扫描全文，识别阶段划分和全局时间上下文
    Phase 2: 区域提取 — 按阶段分块提取管控区域（同一区域在不同阶段可有不同记录）
    Phase 3: 任务提取 — 注入已提取的区域列表作为约束，逐块提取细粒度任务
    """
    
    def __init__(self, scene_type: str, event_date: Optional[str] = None):
        self.scene_type = scene_type
        self.event_date = event_date
        self.knowledge_base = self._load_file(f"references/{scene_type}_knowledge.md")
        self.examples = self._load_file(f"references/extraction_examples_{scene_type}.md")
        self.extraction_rules = self._load_file("references/extraction_rules.md")
        
        # LLM 配置
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MODEL_URL") or settings.MODEL_URL
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or settings.API_KEY
        self.model_name = os.getenv("OPENAI_MODEL_NAME") or os.getenv("MODEL_NAME") or settings.MODEL_NAME

        self.client = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=api_key),
            mode=instructor.Mode.JSON,
        )
        
        # 解析知识库枚举
        self.phases_enum, self.area_names_enum, self.area_types_enum = self._parse_knowledge_enums()
        
        # 构建动态 Pydantic 模型（用于有严格枚举约束的场景）
        self._init_dynamic_models()
    
    def _load_file(self, rel_path: str) -> str:
        base_path = Path(__file__).resolve().parent.parent
        file_path = base_path / rel_path
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return ""

    def _parse_knowledge_enums(self) -> Tuple[List[str], List[str], List[str]]:
        """解析知识库中的枚举值"""
        phases = []
        area_names = []
        area_types = set()

        phase_pattern = r'### PHASE_ENUM.*?```yaml(.*?)```'
        match = re.search(phase_pattern, self.knowledge_base, re.DOTALL)
        if match:
            content = match.group(1)
            names = re.findall(r'-\s+name:\s+([^\n\r]+)', content)
            phases = [n.strip() for n in names]

        area_pattern = r'### AREA_TYPE_ENUM.*?```yaml(.*?)```'
        match = re.search(area_pattern, self.knowledge_base, re.DOTALL)
        if match:
            content = match.group(1)
            names = re.findall(r'name:\s+([^\n\r]+)', content)
            area_names = [n.strip() for n in names]
            types = re.findall(r'type:\s+([^\n\r]+)', content)
            area_types.update([t.strip() for t in types])

        if not phases:
            print("Warning: No phases found in knowledge base, using defaults.")
            phases = ["启动准备阶段", "管控实施阶段", "疏散收尾阶段", "全时段保障"]
        if not area_names:
            print("Warning: No area names found in knowledge base.")
        if not area_types:
            print("Warning: No area types found in knowledge base.")
        
        return phases, area_names, list(area_types)

    def _init_dynamic_models(self):
        """构建带严格枚举约束的动态 Pydantic 模型"""
        PhaseEnum = Enum('PhaseEnum', {n: n for n in self.phases_enum}, type=str)
        
        # 对于区域名称，允许知识库定义之外的值（因为文档中可能有额外路段名）
        # 但区域类型必须严格匹配
        if self.area_types_enum:
            AreaTypeEnum = Enum('AreaTypeEnum', {n: n for n in self.area_types_enum}, type=str)
        else:
            AreaTypeEnum = str
        
        self.PhaseEnum = PhaseEnum
        self.AreaTypeEnum = AreaTypeEnum
    
    # ------------------------------------------------------------------
    # Phase 1: 结构识别
    # ------------------------------------------------------------------
    
    def phase1_detect_structure(self, text: str) -> PhaseDetectionOutput:
        """快速扫描全文，识别文档的阶段划分和全局时间上下文。
        
        这一步的目的是：
        - 建立全局视角，了解文档覆盖了哪些阶段
        - 提取全局时间范围（如活动开始/结束时间）
        - 为后续分块提取提供上下文锚点
        """
        # 用文档摘要（前4000字 + 后1000字）进行结构识别，节省 token
        if len(text) > 5000:
            doc_preview = text[:4000] + "\n\n...(中间内容省略)...\n\n" + text[-1000:]
        else:
            doc_preview = text
        
        phase_list_str = "\n".join(f"  - {p}" for p in self.phases_enum)
        
        system_prompt = f"""你是交通管控文档结构分析专家。请快速浏览文档，识别其中包含的管控阶段。

### 任务
1. 识别文档中提到了哪些管控阶段（必须严格匹配以下枚举）
2. 提取每个阶段的大致时间范围描述
3. 概括全局时间上下文

### 阶段枚举（只能使用以下值）
{phase_list_str}

### 场景信息
- 场景类型: {self.scene_type}
- 活动日期: {self.event_date or '未指定，请从文档推断'}

### 注意
- 只做结构识别，不需要提取具体管控措施
- 如果文档中有"应急保障"、"医疗"、"消防"等内容，归入"全时段保障"
- 用简短的话概括每个阶段的内容"""

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                response_model=PhaseDetectionOutput,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下文档的结构：\n\n{doc_preview}"}
                ],
                temperature=0.0,
                max_retries=2,
            )
            return result
        except Exception as e:
            print(f"Phase 1 结构识别失败: {e}")
            # Fallback: 假设包含所有阶段
            return PhaseDetectionOutput(
                phases_detected=[
                    PhaseDetectionResult(phase_name=p, phase_code=p, time_range_description="")
                    for p in self.phases_enum
                ],
                global_time_context=f"活动日期: {self.event_date or '未知'}"
            )
    
    # ------------------------------------------------------------------
    # Phase 2: 分阶段区域提取
    # ------------------------------------------------------------------
    
    def phase2_extract_areas(
        self,
        chunks: List[str],
        structure: PhaseDetectionOutput
    ) -> List[dict]:
        """按分块提取管控区域，核心改进：同一区域在不同阶段输出不同记录。
        
        关键设计：
        - 每个 chunk 提取时注入全局阶段结构作为上下文
        - 区域去重键 = (area_name, phase) 而非仅 area_name
        - 同一区域在不同阶段可以有不同的 boundaries 和 control_measures
        """
        detected_phases = [p.phase_name for p in structure.phases_detected]
        phases_context = "\n".join(
            f"  - {p.phase_name}: {p.time_range_description} — {p.section_summary}"
            for p in structure.phases_detected
        )
        
        system_prompt = f"""你是交通管控区域提取专家。请从文档片段中提取所有管控区域信息。

### 核心原则
1. **分阶段记录**：同一个区域（如"核心管控区"）如果在不同阶段有不同的管控措施，**必须输出多条记录**。
   - 例如：核心管控区在"启动准备阶段"是"禁停"，在"管控实施阶段"变为"全封闭"，需要分别输出两条。
2. **全时段识别**：如果某区域的管控措施贯穿全程（如应急通道、医疗保障），phase 填"全时段保障"。
3. **边界完整**：boundaries 字段尽量保留原文中的路名和范围描述。
4. **区域归纳**：文档中的具体路段/桥梁/隧道名称必须归纳到知识库定义的管控区中。
   - 如果某路段不属于任何预定义区域，type 使用"管控区"。

### 已识别的阶段结构
{phases_context}

### 全局时间上下文
{structure.global_time_context}

### 知识库（枚举与规则）
{self.knowledge_base}

### 场景信息
- 场景类型: {self.scene_type}
- 活动日期: {self.event_date or '未指定'}

### 时间处理规则
{self.extraction_rules}

### 跨年处理
如果活动日期在年底（如12月31日），凌晨时段（00:00-05:00）、"元旦"、"1月1日"应推断为下一年。"""

        all_areas: List[dict] = []
        seen_area_keys: Set[Tuple[str, str]] = set()  # (area_name, phase)
        
        for i, chunk in enumerate(chunks):
            print(f"  Phase 2 — 处理区域 chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请从以下文档片段中提取管控区域：\n\n{chunk}"}
            ]
            
            try:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=AreaExtractionOutput,
                    messages=messages,
                    temperature=0.0,
                    max_retries=3,
                )
                
                for area in result.affected_areas:
                    area_dict = area.model_dump(mode='json')
                    # 去重键：(area_name, phase) — 同一区域不同阶段视为不同记录
                    dedup_key = (area_dict['area_name'], area_dict['phase'])
                    
                    if dedup_key not in seen_area_keys:
                        seen_area_keys.add(dedup_key)
                        all_areas.append(area_dict)
                        print(f"    + 区域: {area_dict['area_name']} [{area_dict['type']}] "
                              f"@ {area_dict['phase']} — {area_dict['control_measures']}")
                    else:
                        # 如果重复但有更丰富的信息（如 boundaries 从 None 变为有值），更新
                        for existing in all_areas:
                            if (existing['area_name'] == area_dict['area_name'] and 
                                existing['phase'] == area_dict['phase']):
                                if area_dict.get('boundaries') and not existing.get('boundaries'):
                                    existing['boundaries'] = area_dict['boundaries']
                                if area_dict.get('start_time') and not existing.get('start_time'):
                                    existing['start_time'] = area_dict['start_time']
                                if area_dict.get('end_time') and not existing.get('end_time'):
                                    existing['end_time'] = area_dict['end_time']
                                break
                        
            except Exception as e:
                print(f"    ! 区域提取 chunk {i+1} 失败: {e}")
                continue
        
        print(f"  Phase 2 完成 — 共提取 {len(all_areas)} 条区域记录")
        return all_areas
    
    # ------------------------------------------------------------------
    # Phase 3: 任务提取（注入区域约束）
    # ------------------------------------------------------------------
    
    def phase3_extract_tasks(
        self,
        chunks: List[str],
        structure: PhaseDetectionOutput,
        areas: List[dict]
    ) -> List[dict]:
        """逐块提取任务，注入已提取的区域列表作为约束。
        
        关键设计：
        - affected_area 必须引用 Phase 2 中已提取的 area_name
        - 注入上下文时间锚点，确保分块间时间连续
        - few-shot 示例帮助 LLM 理解输出格式
        """
        # 构建区域名称列表作为约束
        area_names = sorted(set(a['area_name'] for a in areas))
        area_names_str = "、".join(area_names)
        
        # 构建区域-阶段映射表供 LLM 参考
        area_phase_map = {}
        for a in areas:
            key = a['area_name']
            if key not in area_phase_map:
                area_phase_map[key] = []
            area_phase_map[key].append(f"{a['phase']}: {a['control_measures']}")
        
        area_ref_str = "\n".join(
            f"  - {name}: " + " → ".join(phases)
            for name, phases in area_phase_map.items()
        )
        
        phases_context = "\n".join(
            f"  - {p.phase_name}: {p.time_range_description}"
            for p in structure.phases_detected
        )
        
        base_system_prompt = f"""你是交通管控任务提取专家。请从文档片段中**穷尽式**地提取所有管控任务。

### 核心原则
1. **全面性**：不遗漏任何管控动作、时间节点或措施变更。
2. **细粒度**：每一个独立的管控动作单独成一条任务。
3. **区域约束**：affected_area 必须是以下已定义的区域之一：
   {area_names_str}
4. **阶段归属**：phase 必须严格匹配以下枚举值：
   {", ".join(self.phases_enum)}
5. **时间标准化**：所有时间必须为 YYYY-MM-DD HH:MM 格式。

### 已识别的阶段结构
{phases_context}

### 已提取的区域及其分阶段管控措施
{area_ref_str}

### 全局时间上下文
{structure.global_time_context}

### 时间处理规则
{self.extraction_rules}

### 场景信息
- 场景类型: {self.scene_type}
- 活动日期: {self.event_date or '未指定'}

### 跨年处理
如果活动日期在年底（如12月31日），凌晨时段（00:00-05:00）、"元旦"、"1月1日"应推断为下一年。

### 输出要求
- 按时间顺序输出
- description 不超过200字
- 如果无法确定 affected_area 对应哪个已定义区域，选择最相关的一个"""

        all_tasks: List[dict] = []
        seen_tasks: Set[Tuple[str, str, str]] = set()  # (start_time, action, affected_area)
        last_context_time = None
        
        for i, chunk in enumerate(chunks):
            print(f"  Phase 3 — 处理任务 chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            current_prompt = base_system_prompt
            if last_context_time:
                current_prompt += f"\n\n### 上文时间锚点\n上一个片段最后提取的时间: **{last_context_time}**。如果本片段开头没有明确时间，请参考此锚点。"
            
            messages = [
                {"role": "system", "content": current_prompt},
            ]
            
            # 添加 few-shot 示例（截取前1500字）
            if self.examples:
                messages.append({"role": "user", "content": "请提供一些提取示例。"})
                messages.append({"role": "assistant", "content": f"参考示例：\n{self.examples[:1500]}"})
            
            messages.append({"role": "user", "content": f"请从以下文档片段提取管控任务：\n\n{chunk}"})
            
            try:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=TaskExtractionOutput,
                    messages=messages,
                    temperature=0.1,
                    max_retries=3,
                )
                
                for task in result.tasks:
                    task_dict = task.model_dump(mode='json')
                    
                    # 去重
                    fingerprint = (task_dict['start_time'], task_dict['action'], task_dict['affected_area'])
                    if fingerprint in seen_tasks:
                        continue
                    seen_tasks.add(fingerprint)
                    
                    all_tasks.append(task_dict)
                    
                    time_str = task_dict['start_time']
                    if task_dict.get('end_time'):
                        time_str += f" → {task_dict['end_time']}"
                    print(f"    + [{time_str}] {task_dict['action']} "
                          f"| {task_dict['description'][:40]}... "
                          f"@ {task_dict['affected_area']}")
                    
                    # 更新时间锚点
                    if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", task_dict['start_time']):
                        last_context_time = task_dict['start_time']
                    
            except Exception as e:
                print(f"    ! 任务提取 chunk {i+1} 失败: {e}")
                continue
        
        # 按时间排序
        try:
            all_tasks.sort(key=lambda x: x.get("start_time", ""))
        except Exception:
            pass
        
        print(f"  Phase 3 完成 — 共提取 {len(all_tasks)} 条任务")
        return all_tasks
    
    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    
    def extract(self, text: str) -> dict:
        """三阶段提取主流程"""
        print("=" * 60)
        print("Phase 1: 文档结构识别")
        print("=" * 60)
        structure = self.phase1_detect_structure(text)
        
        print(f"\n全局时间上下文: {structure.global_time_context}")
        for p in structure.phases_detected:
            print(f"  ✓ {p.phase_name}: {p.time_range_description} — {p.section_summary}")
        
        # 智能分块
        print("\n--- 文档分块 ---")
        sections = parse_document_structure(text)
        print(f"识别到 {len(sections)} 个语义段落")
        chunks = merge_sections_to_chunks(sections, max_chunk_chars=4000)
        print(f"合并为 {len(chunks)} 个处理块")
        
        print("\n" + "=" * 60)
        print("Phase 2: 分阶段区域提取")
        print("=" * 60)
        areas = self.phase2_extract_areas(chunks, structure)
        
        print("\n" + "=" * 60)
        print("Phase 3: 任务提取（区域约束）")
        print("=" * 60)
        tasks = self.phase3_extract_tasks(chunks, structure, areas)
        
        # 组装最终结果
        result = {
            "event_name": "",  # 由调用方填充
            "event_type": self.scene_type,
            "scenario_detected": self.scene_type,
            "event_date": self.event_date,
            "affected_areas": areas,
            "tasks": tasks,
        }
        
        return result


# ==============================================================================
# 4. 文档转换服务 (MinerU) — 保持不变
# ==============================================================================

class MinerUClient:
    """MinerU 文档转换客户端"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        default_url = settings.MINERU_API_URL if settings and hasattr(settings, "MINERU_API_URL") else "http://mineru-service:8000/convert"
        default_key = settings.MINERU_API_KEY if settings and hasattr(settings, "MINERU_API_KEY") else ""
        self.api_url = api_url or os.getenv("MINERU_API_URL", default_url)
        self.api_key = api_key or os.getenv("MINERU_API_KEY", default_key)
        
    def convert_to_markdown(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")
        print(f"正在调用 MinerU 转换文件: {file_path.name} ...")
        
        if "mineru-service" in self.api_url:
            print("警告: 未配置有效的 MinerU URL，使用模拟转换。")
            return f"# {file_path.stem}\n\n(模拟转换内容)\n"
            
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                response = requests.post(self.api_url, files=files, headers=headers, timeout=60)
                response.raise_for_status()
                return response.json().get("markdown", "")
        except Exception as e:
            raise RuntimeError(f"MinerU 转换失败: {str(e)}")


# ==============================================================================
# 5. 主流程
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="活动文档解析工具 v2")
    parser.add_argument("input_file", type=Path, help="输入文件路径")
    parser.add_argument("scene_type", type=str, nargs="?", help="场景类型")
    parser.add_argument("event_date", type=str, nargs="?", help="活动日期 YYYY-MM-DD")
    parser.add_argument("--output", "-o", type=Path, help="输出文件路径")
    
    args = parser.parse_args()
    input_file = args.input_file
    event_date = args.event_date
    scene_type = args.scene_type
    
    print(f"输入文件: {input_file}")
    print(f"场景类型: {scene_type}")
    print(f"活动日期: {event_date}")

    if not input_file.exists():
        print(f"错误: 文件 {input_file} 不存在")
        sys.exit(1)

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # 1. 文档转换
    if input_file.suffix.lower() in ['.pdf', '.docx', '.doc']:
        print("检测到二进制文档，启动 MinerU 转换...")
        converter = MinerUClient()
        try:
            text = converter.convert_to_markdown(input_file)
        except Exception as e:
            print(f"转换失败: {e}")
            sys.exit(1)
    else:
        text = input_file.read_text(encoding="utf-8")
    
    print(f"文档长度: {len(text)} 字符")

    # 智能参数解析
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if scene_type and date_pattern.match(scene_type):
        if not event_date:
            print(f"提示: '{scene_type}' 被识别为日期，自动交换参数")
            event_date = scene_type
            scene_type = None

    if not scene_type:
        print("警告: 未提供 scene_type，默认使用 'other'")
        scene_type = "other"
    
    # 2. 三阶段提取
    extractor = EventDocExtractor(scene_type, event_date)
    print(f"\n开始三阶段提取 (模型: {extractor.model_name})")
    print("=" * 60)
    
    result = extractor.extract(text)
    result["event_name"] = input_file.stem
    
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
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print(f"\n{'=' * 60}")
    print(f"提取完成！")
    print(f"  区域: {len(result['affected_areas'])} 条（含分阶段记录）")
    print(f"  任务: {len(result['tasks'])} 条")
    print(f"  输出: {output_file}")


if __name__ == "__main__":
    main()