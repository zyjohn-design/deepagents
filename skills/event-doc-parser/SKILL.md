---
name: event-doc-parser
description: 解析大型活动文档(跨年夜、马拉松等),提取场景信息、阶段划分、时间段任务、管控区域和路线。输出统一JSON格式。
version: 2.3.0
tags: [交通管控, 活动解析, 文档提取, event, traffic]
inputs: [活动文档路径(PDF/DOCX/Markdown)]
outputs: [结构化JSON文件路径]
depends_on_skills: []
---

# Event Doc Parser — 活动文档结构化提取

## 适用场景

需要从**非结构化的大型活动文档**（PDF/DOCX/Markdown）中提取交通管控方案的结构化信息时使用。

典型输入：跨年夜方案、马拉松交通管控方案、大型演出安保方案等。
典型输出：包含时间线、管控区域、保障路线、任务列表的 JSON 文件。

## 文件命名规范

所有文件路径**不得包含空格**，使用下划线或连字符替代。
正确: `2026_跨年夜_江北片区实施方案.md`
错误: `2026 跨年夜 江北片区实施方案.md`

## ⚠️ 路径传递规则

**文件路径必须原样复制，严禁做任何修改**。不得在中文与数字之间插入空格、不得修改大小写、不得增删任何字符。用户给的路径是什么就传什么。

## 核心工作流

步骤1: 制定解析计划 — 用 write_todos 分解任务，确认文件格式和路径
步骤2: 文档预处理 — 若输入是 PDF/DOCX，用 run_skill_script 执行 doc_converter.py 转为 Markdown
步骤3: 识别场景类型 — 用 read_skill_reference 读取 scene_enumeration.md，匹配场景
步骤4: 执行结构化提取 — 用 run_skill_script 调用 event_doc_parser.py
步骤5: 返回结果 — 只返回生成的 JSON 文件路径

## 详细指令

### 步骤1: 制定解析计划

调用 `write_todos`：

```
1. 确认输入文件路径和格式（PDF/DOCX/Markdown）
2. 如需格式转换，执行 doc_converter.py
3. 读取场景枚举，确定 scene_type
4. 执行 event_doc_parser.py 提取
5. 返回 JSON 文件路径
```

### 步骤2: 文档预处理（按需）

检查文件扩展名：
- `.md` / `.txt` → **跳过**，直接到步骤3
- `.pdf` / `.docx` → 需要转换

调用 `run_skill_script`：
```
skill_name: "event-doc-parser"
script_name: "doc_converter.py"
command: "uv run python"
script_args: "/absolute/path/to/input.pdf"
```

记录输出的 Markdown 文件路径，后续步骤使用。

### 步骤3: 识别场景类型

调用 `read_skill_reference` 读取场景枚举（文件不大，直接读取）：
```
skill_name: "event-doc-parser"
reference_name: "scene_enumeration.md"
```

根据文档标题和内容关键词（如"跨年""马拉松""演唱会"），从枚举中匹配最合适的 `scene_type` 值（如 `new_year_eve`、`marathon`）。

**重要**：scene_type 必须严格匹配枚举中的定义值，不得自行编造。

### 步骤4: 执行结构化提取

调用 `run_skill_script`：
```
skill_name: "event-doc-parser"
script_name: "event_doc_parser.py"
command: "uv run python"
script_args: "<md_file> <scene_type> [event_date]"
```

参数说明：
- `md_file` — Markdown 文件绝对路径（步骤2输出，或原始 .md 路径）
- `scene_type` — 步骤3确定的场景类型
- `event_date`（可选）— 活动日期，YYYY-MM-DD 格式

示例：
```
skill_name: "event-doc-parser"
script_name: "event_doc_parser.py"
command: "uv run python"
script_args: "/data/docs/2026_跨年夜_江北片区实施方案.md <步骤3确定的scene_type> 2026-12-31"
```

### 步骤5: 返回结果

脚本成功后，只返回生成的 **JSON 文件路径**，不要读取或展示 JSON 内容。

## 质量红线

1. **严禁 LLM 直接提取** — 必须使用 event_doc_parser.py 脚本，禁止通过 Prompt 让 LLM 直接解析全文生成 JSON
2. **参数准确性** — scene_type 必须来自 scene_enumeration.md 的精确匹配，不得猜测
3. **日期格式** — 必须为 YYYY-MM-DD
4. **路径规范** — 使用绝对路径，文件名不得含空格

## 决策速查表

```
输入文件是什么格式？
├─ .md/.txt     → 直接到步骤3
├─ .pdf/.docx   → 步骤2: run_skill_script 转换 → 步骤3
└─ 其他格式     → 报错，不支持

提取完成后返回什么？
├─ 只返回 JSON 文件路径
└─ 不要读取/展示 JSON 内容
```
