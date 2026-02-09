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
from typing import List, Optional, Iterable, Any, Set, Tuple, Dict, Literal, get_args
from enum import Enum  # 保留备用

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

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
# 2. Pydantic 模型工厂 — 全部动态生成，严格绑定知识库枚举
# ==============================================================================
#
# 设计原则：
#   - 所有 area_name、type、phase 字段都用 Literal 类型锁死
#   - Literal 值 100% 来自知识库 YAML，不硬编码
#   - instructor 会将 Literal 转为 JSON Schema 的 "enum" 约束
#   - LLM 输出不在枚举范围内时，instructor 的 retry 机制会自动纠正
#
# 为什么用 Literal 而不是 Enum：
#   - Literal 在 JSON Schema 中直接生成 {"enum": ["核心管控区", "中端疏导区", ...]}
#   - Enum 在某些 instructor mode 下序列化为 {"$ref": ...}，部分 LLM 不理解
#   - Literal 对中文值的支持更稳定

def _make_literal_type(values: List[str]):
    """从字符串列表动态创建 Literal 类型。
    
    例如: _make_literal_type(["核心管控区", "中端疏导区"]) 
          → Literal["核心管控区", "中端疏导区"]
    """
    if not values:
        return str  # fallback: 无枚举值时退化为 str
    return Literal[tuple(values)]


# Phase 1 模型（不依赖知识库枚举，可静态定义）
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
        """解析知识库中的枚举值。
        
        不同场景的知识库结构可能不同：
        - newyear: AREA_TYPE_ENUM 有 name 字段（核心管控区、中端疏导区...）
        - marathon: AREA_TYPE_ENUM 只有 type 字段（起点区、终点区、线路），
                    具体 area_name 来自文档（全程马拉松路线...）
        
        策略：
        - 如果知识库有 name 列表 → area_name 用 Literal 严格锁死
        - 如果只有 type 列表 → area_name 退化为 str，type 用 Literal 严格锁死
        """
        phases = []
        area_names = []
        area_types = set()

        # --- 解析 PHASE_ENUM ---
        phase_pattern = r'### PHASE_ENUM.*?```yaml(.*?)```'
        match = re.search(phase_pattern, self.knowledge_base, re.DOTALL)
        if match:
            content = match.group(1)
            names = re.findall(r'-\s+name:\s+([^\n\r]+)', content)
            phases = [n.strip() for n in names]

        # --- 解析 AREA_TYPE_ENUM ---
        area_pattern = r'### AREA_TYPE_ENUM.*?```yaml(.*?)```'
        match = re.search(area_pattern, self.knowledge_base, re.DOTALL)
        if match:
            content = match.group(1)
            
            # 提取 name（如跨年夜场景有此字段）
            names = re.findall(r'name:\s+([^\n\r]+)', content)
            area_names = [n.strip() for n in names]
            
            # 提取 type（所有场景都有）
            types = re.findall(r'-\s+type:\s+([^\n\r]+)', content)
            area_types.update([t.strip() for t in types])

        # --- Fallbacks ---
        if not phases:
            print("Warning: No phases found in knowledge base, using defaults.")
            phases = ["启动准备阶段", "管控实施阶段", "疏散收尾阶段", "全时段保障"]
        
        if not area_names:
            print(f"Info: 知识库无 area_name 枚举（场景 {self.scene_type}），area_name 将允许自由填写。")
            # area_names 保持为空列表，_init_dynamic_models 会处理 fallback
        
        if not area_types:
            print("Warning: No area types found in knowledge base.")
        
        return phases, area_names, list(area_types)

    def _init_dynamic_models(self):
        """动态生成 Phase 2/3 的 Pydantic 模型，用 Literal 类型锁死知识库枚举。
        
        这是严格约束的核心：
        - area_name: Literal["核心管控区", "中端疏导区", "远端分流区"]  (跨年夜场景)
        - type:      Literal["管控区"]                                  (跨年夜场景)
        - phase:     Literal["启动准备阶段", "管控实施阶段", ...]        (跨年夜场景)
        
        当知识库无 area_name 枚举时（如马拉松），area_name 退化为 str。
        
        instructor 会将 Literal 转为 JSON Schema 的 {"enum": [...]}，
        LLM 输出不在范围内时，instructor 的 max_retries 会自动要求 LLM 修正。
        """
        # --- 构建 Literal 类型 ---
        PhaseLiteral = _make_literal_type(self.phases_enum)
        AreaNameLiteral = _make_literal_type(self.area_names_enum)  # 空列表时返回 str
        AreaTypeLiteral = _make_literal_type(self.area_types_enum)
        
        # 标记是否有严格的 area_name 约束
        self.has_strict_area_names = bool(self.area_names_enum)
        
        # 打印枚举约束，方便调试
        print(f"[枚举约束] phase: {self.phases_enum}")
        if self.has_strict_area_names:
            print(f"[枚举约束] area_name (严格): {self.area_names_enum}")
        else:
            print(f"[枚举约束] area_name: 自由填写（知识库无枚举）")
        print(f"[枚举约束] area_type (严格): {self.area_types_enum}")
        
        # --- Phase 2: 区域提取模型 ---
        self.DynAreaItem = create_model(
            'AreaExtractionItem',
            area_name=(AreaNameLiteral, Field(
                ..., 
                description=(
                    f"区域名称，只能是以下值之一: {self.area_names_enum}"
                    if self.has_strict_area_names
                    else "区域名称，从文档中提取"
                )
            )),
            type=(AreaTypeLiteral, Field(
                ..., 
                description=f"区域类型，只能是以下值之一: {self.area_types_enum}"
            )),
            phase=(PhaseLiteral, Field(
                ..., 
                description=f"该管控措施生效的阶段，只能是: {self.phases_enum}"
            )),
            start_time=(Optional[str], Field(None, description="管控开始时间 YYYY-MM-DD HH:MM")),
            end_time=(Optional[str], Field(None, description="管控结束时间")),
            boundaries=(Optional[str], Field(None, description="边界描述，保留原文路名")),
            control_measures=(str, Field(..., description="该阶段的具体管控措施")),
            __base__=BaseModel,
        )
        
        self.DynAreaOutput = create_model(
            'AreaExtractionOutput',
            affected_areas=(List[self.DynAreaItem], Field(
                default_factory=list,
                description="管控区域列表。同一区域在不同阶段有不同管控措施时，必须输出多条记录"
            )),
            __base__=BaseModel,
        )
        
        # --- Phase 3: 任务提取模型 ---
        self.DynTaskItem = create_model(
            'TaskExtractionItem',
            phase=(PhaseLiteral, Field(
                ..., 
                description=f"阶段名称，只能是: {self.phases_enum}"
            )),
            start_time=(str, Field(..., description="开始时间 YYYY-MM-DD HH:MM")),
            end_time=(Optional[str], Field(None, description="结束时间")),
            control_type=(str, Field(..., description="管控类型")),
            description=(str, Field(..., description="任务简述，200字以内")),
            action=(str, Field(..., description="执行动作")),
            affected_area=(AreaNameLiteral, Field(
                ..., 
                description=(
                    f"关联的区域名称，只能是: {self.area_names_enum}"
                    if self.has_strict_area_names
                    else "关联的区域名称，必须引用已定义的 area_name"
                )
            )),
            __base__=BaseModel,
        )
        
        self.DynTaskOutput = create_model(
            'TaskExtractionOutput',
            tasks=(List[self.DynTaskItem], Field(
                default_factory=list,
                description="管控任务列表，按时间顺序"
            )),
            __base__=BaseModel,
        )
    
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
        
        # 构建区域约束描述
        if self.has_strict_area_names:
            area_constraint_text = f"""- **area_name 只能是**: {self.area_names_enum}
- **type 只能是**: {self.area_types_enum}
- 不允许创造以上枚举之外的任何值！文档中提到的具体路段/桥梁/隧道必须归纳到上述 area_name 之一。"""
            area_principle_text = """4. **区域归纳**：文档中提到的具体路段/桥梁/隧道名称，必须归纳到知识库定义的管控区中，选择最相关的 area_name。"""
        else:
            area_constraint_text = f"""- **type 只能是**: {self.area_types_enum}
- area_name 从文档中提取具体名称（如"全程马拉松路线"、"长江隧道"等）
- 不允许创造 type 枚举之外的值！"""
            area_principle_text = """4. **区域命名**：area_name 使用文档中的原始名称，type 严格匹配知识库枚举。"""

        system_prompt = f"""你是交通管控区域提取专家。请从文档片段中提取所有管控区域信息。

### ⚠️ 严格枚举约束（违反将导致输出无效）
- **phase 只能是**: {self.phases_enum}
{area_constraint_text}

### 核心原则
1. **分阶段记录**：同一个区域如果在不同阶段有不同的管控措施，**必须输出多条记录**。
   - 例如：核心管控区在"启动准备阶段"是"禁停"，在"管控实施阶段"变为"全封闭"，需要分别输出两条。
2. **全时段识别**：如果某区域的管控措施贯穿全程（如应急通道、医疗保障），phase 填"全时段保障"。
3. **边界完整**：boundaries 字段尽量保留原文中的路名和范围描述。
{area_principle_text}

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
                    response_model=self.DynAreaOutput,  # 动态模型，枚举约束
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
        
        # 构建区域约束描述
        if self.has_strict_area_names:
            task_area_constraint = f"""- **affected_area 只能是**: {self.area_names_enum}
- 不允许创造以上枚举之外的任何值！"""
        else:
            task_area_constraint = f"""- **affected_area 必须是 Phase 2 已提取的区域名称之一**: {area_names_str}"""
        
        base_system_prompt = f"""你是交通管控任务提取专家。请从文档片段中**穷尽式**地提取所有管控任务。

### ⚠️ 严格枚举约束（违反将导致输出无效）
- **phase 只能是**: {self.phases_enum}
{task_area_constraint}

### 核心原则
1. **全面性**：不遗漏任何管控动作、时间节点或措施变更。
2. **细粒度**：每一个独立的管控动作单独成一条任务。
3. **区域约束**：affected_area 必须是以下已定义区域之一：
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
- affected_area 必须选择已定义区域中最相关的一个"""

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
                    response_model=self.DynTaskOutput,  # 动态模型，枚举约束
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
    # 后处理：合并 boundaries 不变的同名区域
    # ------------------------------------------------------------------
    
    def _merge_constant_boundary_areas(self, areas: List[dict]) -> List[dict]:
        """合并 boundaries 实质相同的同名区域到 "全时段保障"。
        
        规则：
        - 按 area_name 分组
        - 如果同一 area_name 在多个阶段出现，且 boundaries 实质一致（模糊匹配）
          → 合并为一条记录，phase="全时段保障"
          → control_measures 合并为各阶段措施的汇总
          → start_time 取最早，end_time 取最晚
          → boundaries 取最完整的一条
        - 如果 boundaries 实质不同 → 保持分条不合并
        
        模糊匹配策略（不依赖外部库）：
        1. 标准化：全角→半角、去除空白/标点噪声
        2. 路名集合比较：提取所有路名，计算 Jaccard 相似度
        3. 阈值判断：相似度 ≥ 0.8 视为"实质一致"
        """
        from collections import defaultdict
        
        groups: Dict[str, List[dict]] = defaultdict(list)
        for area in areas:
            groups[area['area_name']].append(area)
        
        result = []
        
        continuous_phase = "全时段保障"
        for p in self.phases_enum:
            if "全时段" in p or "continuous" in p.lower():
                continuous_phase = p
                break
        
        for area_name, group in groups.items():
            if len(group) == 1:
                result.append(group[0])
                continue
            
            if self._boundaries_are_similar(group):
                merged = self._do_merge(group, continuous_phase)
                result.append(merged)
                print(f"  ✓ 合并: {area_name} ({len(group)}个阶段 → 全时段保障)")
                for area in group:
                    if area['phase'] != continuous_phase:
                        print(f"    - {area['phase']}: {area['control_measures']}")
            else:
                result.extend(group)
                print(f"  ○ 保留分条: {area_name} (boundaries 在不同阶段有实质变化)")
        
        print(f"  合并前: {len(areas)} 条 → 合并后: {len(result)} 条")
        return result
    
    @staticmethod
    def _normalize_boundary_text(text: Optional[str]) -> Optional[str]:
        """标准化 boundaries 文本，消除 LLM 提取的表面差异。
        
        处理的差异类型：
        - 全角/半角括号：（） → ()
        - 全角/半角标点：，。；→ ,.; 
        - 多余空白
        - "内道路"、"区域"等后缀词
        - 路名分隔符不一致：、/；/，
        """
        if text is None:
            return None
        
        s = text.strip()
        if not s:
            return None
        
        # 全角 → 半角（逐对映射，避免长度不匹配）
        replacements = {
            '（': '(', '）': ')', '【': '[', '】': ']',
            '，': ',', '。': '.', '；': ';', '：': ':',
            '"': '"', '"': '"', ''': "'", ''': "'",
            '、': ',',
        }
        for full, half in replacements.items():
            s = s.replace(full, half)
        
        # 统一分隔符 → 顿号
        s = re.sub(r'[,;\s]+', '、', s)
        
        # 去除常见后缀噪声词
        for noise in ['内道路', '区内道路', '区域内', '合围区域', '合围区', '范围内', '以内']:
            s = s.replace(noise, '')
        
        # 去除多余空白
        s = re.sub(r'\s+', '', s)
        
        return s
    
    @staticmethod
    def _extract_road_names(text: Optional[str]) -> Set[str]:
        """从 boundaries 文本中提取路名集合。
        
        路名模式：XX路、XX大道、XX街、XX桥、XX隧道、XX立交等
        """
        if not text:
            return set()
        
        # 匹配中文路名（2-10个字 + 路/大道/街/桥/隧道/立交/高架/匝道/通道）
        pattern = r'[\u4e00-\u9fff]{1,10}(?:路|大道|街|桥|隧道|立交|高架|匝道|通道|广场|码头)'
        names = set(re.findall(pattern, text))
        return names
    
    def _boundaries_are_similar(self, group: List[dict], threshold: float = 0.75) -> bool:
        """判断一组区域的 boundaries 是否实质一致。
        
        三层判断策略：
        1. 全部为 None/空 → 一致
        2. 标准化后字符串相等 → 一致
        3. 路名集合 Jaccard 相似度 ≥ threshold → 一致
        
        只要任意一对 boundaries 不满足以上条件，就判定为"不一致"。
        """
        boundaries_list = [area.get('boundaries') for area in group]
        
        # 标准化
        normalized = [self._normalize_boundary_text(b) for b in boundaries_list]
        
        # 层1: 全部为 None
        if all(b is None for b in normalized):
            return True
        
        # 过滤出非 None 的
        non_null = [b for b in normalized if b is not None]
        
        # 如果部分有 boundaries 部分没有，看有值的是否一致
        # 没有 boundaries 的视为"未提取到"而非"不同"
        if len(non_null) == 0:
            return True
        
        # 层2: 标准化后字符串相等
        if len(set(non_null)) == 1:
            return True
        
        # 层3: 路名集合 Jaccard 相似度
        road_name_sets = [self._extract_road_names(b) for b in non_null]
        
        # 两两比较
        for i in range(len(road_name_sets)):
            for j in range(i + 1, len(road_name_sets)):
                set_a, set_b = road_name_sets[i], road_name_sets[j]
                
                # 如果两个都是空集（没有提取到路名），视为一致
                if not set_a and not set_b:
                    continue
                
                # 如果一个为空一个不为空，不好判断，保守处理
                if not set_a or not set_b:
                    # 回退到字符串包含关系判断
                    shorter, longer = sorted([non_null[i], non_null[j]], key=len)
                    if shorter in longer:
                        continue  # 短的是长的子串 → 一致
                    return False
                
                # Jaccard 相似度
                intersection = set_a & set_b
                union = set_a | set_b
                jaccard = len(intersection) / len(union) if union else 1.0
                
                if jaccard < threshold:
                    return False
        
        return True
    
    def _do_merge(self, group: List[dict], continuous_phase: str) -> dict:
        """将同一区域的多条分阶段记录合并为一条全时段记录。"""
        phase_order = {p: i for i, p in enumerate(self.phases_enum)}
        sorted_group = sorted(group, key=lambda a: phase_order.get(a['phase'], 99))
        
        # 合并 control_measures：按阶段拼接
        measures_parts = []
        for area in sorted_group:
            phase_label = area['phase']
            measure = area['control_measures']
            if phase_label == continuous_phase:
                measures_parts.append(measure)
            else:
                measures_parts.append(f"【{phase_label}】{measure}")
        
        merged_measures = "；".join(measures_parts)
        
        # 取最早的 start_time 和最晚的 end_time
        start_times = [a['start_time'] for a in sorted_group if a.get('start_time')]
        end_times = [a['end_time'] for a in sorted_group if a.get('end_time')]
        
        # 选择最完整（最长）的 boundaries
        best_boundary = None
        best_len = 0
        for area in sorted_group:
            b = area.get('boundaries')
            if b and len(b) > best_len:
                best_boundary = b
                best_len = len(b)
        
        merged = {
            'area_name': sorted_group[0]['area_name'],
            'type': sorted_group[0]['type'],
            'phase': continuous_phase,
            'start_time': min(start_times) if start_times else None,
            'end_time': max(end_times) if end_times else None,
            'boundaries': best_boundary,
            'control_measures': merged_measures,
        }
        
        return merged

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
        areas_raw = self.phase2_extract_areas(chunks, structure)
        
        # 后处理：合并 boundaries 不变的同名区域到 "全时段保障"
        print("\n--- 区域合并优化 ---")
        areas = self._merge_constant_boundary_areas(areas_raw)
        
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