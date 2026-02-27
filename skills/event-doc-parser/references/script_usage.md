# 脚本详细使用说明

## event_doc_parser.py

基于 LLM 的活动文档智能解析脚本。

### 使用方法

```bash
uv run python scripts/event_doc_parser.py <input_file> [scene_type] [event_date] [--output <output_file>]
```

### 参数说明

- `input_file`: 输入文件路径，只支持 `.md`。
- `scene_type`: (必选) 场景类型代码 (如 `newyear`, `marathon`)。
  - 必须从 `references/scene_enumeration.md` 中选取。
  - 虽然脚本保留了自动检测逻辑作为 fallback，但在 Agent 工作流中必须显式传入。
- `event_date`: (可选) 活动日期，格式 `YYYY-MM-DD`。
- `--output`, `-o`: (可选) 输出文件路径，如果不指定则默认保存到当前 Skill 的 `output` 目录。

### 示例

```bash
uv run python scripts/event_doc_parser.py docs/2026跨年夜方案.md newyear 2026-12-31 -o output/result.json
```

## doc_converter.py

文档格式转换工具 (MinerU Wrapper)。

### 使用方法

```bash
uv run python scripts/doc_converter.py <input_file> [--output <output_file>]
```

### 参数说明

- `input_file`: 输入文件路径 (PDF/DOCX)。
- `--output`, `-o`: (可选) 输出 Markdown 文件路径。

### 示例

```bash
uv run python scripts/doc_converter.py docs/raw_plan.pdf
```
