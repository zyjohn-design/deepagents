---
name: event-doc-parser
description: 解析大型活动文档(跨年夜、马拉松等),提取场景信息、阶段划分、时间段任务、管控区域和路线。输出统一的JSON格式。
allowed_tools:
  - write_todos
  - read_file
  - run_command
---

# Event Doc Parser Skill

## When to Use This Skill

当且仅当需要从非结构化的大型活动文档（PDF/DOCX/Markdown）中提取结构化的交通管控方案信息（如时间线、管控区域、保障路线等）时使用。

## Workflow

### 1. Plan Your Approach
**Use `write_todos` to break down the task:**
- Check file format (Convert PDF/DOCX to Markdown if needed)
- Identify scene type (Match with enumeration)
- Execute extraction script

### 2. Document Preprocessing
如果输入文件是 PDF 或 DOCX 格式，首先需要转换为 Markdown：

1. **执行转换**: `uv run python scripts/doc_converter.py <input_file>`
2. **获取路径**: 记录输出的 Markdown 文件路径

### 3. Scene Identification
在提取之前，必须确定活动的具体场景类型：

1. **查阅枚举**: 使用 `read_file` 查看 `references/scene_enumeration.md`
2. **预览文档**: 通过文件名称和读取文档前 500 个字符 `read_file_limit`
3. **确定类型**: 匹配最合适的 `scene_type` (如 `new_year_eve`, `marathon`)

### 4. Structured Extraction
调用专用脚本进行结构化提取：

1. **执行提取**: `uv run python scripts/event_doc_parser.py <md_file> <scene_type> [event_date]`
   - `md_file`: 转换后的 Markdown 文件路径
   - `scene_type`: 步骤 3 确定的场景类型
   - `event_date`: (可选) 活动日期，格式 YYYY-MM-DD
2. **获取结果**: 读取脚本生成的 JSON 文件内容，只返回JSON文件路径即可。

## Example: Parsing New Year Plan

```bash
# 1. 文档转换
uv run python scripts/doc_converter.py docs/2026_newyear.pdf

# 2. 场景判定 (假设判定为 new_year_eve)

# 3. 智能提取
uv run python scripts/event_doc_parser.py docs/2026_newyear.md new_year_eve 2026-12-31
```

## Quality Guidelines

- **严禁 LLM 直接提取**: 禁止仅通过 Prompt 让 LLM 直接解析全文生成 JSON，必须使用 `event_doc_parser.py`。
- **参数准确性**: 确保 `scene_type` 严格对应 `scene_enumeration.md` 中的定义。
- **日期格式**: 日期参数必须为 `YYYY-MM-DD` 格式。
- **文件路径**: 脚本参数必须使用绝对路径或基于项目根目录的相对路径。
