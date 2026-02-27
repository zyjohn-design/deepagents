#!/usr/bin/env python3
"""
文档格式转换工具 (MinerU Wrapper)

功能：
将 PDF/DOCX 等非结构化文档转换为 Markdown 格式，以便于 LLM 读取和处理。

使用方法:
    uv run python scripts/doc_converter.py <input_file> [--output <output_file>]

示例:
    uv run python scripts/doc_converter.py docs/2026跨年夜方案.pdf
"""

import sys
import os
import requests
import argparse
from pathlib import Path
from typing import Optional

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
        # 允许在没有 settings 的情况下运行（使用默认值）
        settings = None

class MinerUClient:
    """MinerU 文档转换客户端"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        # 优先使用环境变量或传入参数，否则使用默认值
        default_url = settings.MINERU_API_URL if settings and hasattr(settings, "MINERU_API_URL") else "http://mineru-service:8000/convert"
        default_key = settings.MINERU_API_KEY if settings and hasattr(settings, "MINERU_API_KEY") else ""
        
        self.api_url = api_url or os.getenv("MINERU_API_URL", default_url)
        self.api_key = api_key or os.getenv("MINERU_API_KEY", default_key)
        
    def convert_to_markdown(self, file_path: Path) -> str:
        """调用 MinerU 接口将 PDF/DOCX 转换为 Markdown"""
        if not file_path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")
            
        print(f"正在调用 MinerU 转换文件: {file_path.name} ...")
        
        # 模拟转换 (如果没有真实服务)
        # 实际部署时请替换为真实的 API 调用逻辑
        # 这里为了演示，如果 URL 包含 mineru-service (默认值)，则使用模拟
        if "mineru-service" in self.api_url:  
            print("警告: 未配置有效的 MinerU URL，使用模拟转换。")
            # 简单模拟：返回文件名作为标题，以及一些示例文本
            return f"# {file_path.stem}\n\n(模拟转换内容) 这是从 {file_path.name} 转换得到的 Markdown 内容。\n\n文档内容预览：\n本方案旨在保障2026年跨年夜活动的交通安全与畅通...\n"
            
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                response = requests.post(self.api_url, files=files, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                # 假设返回格式: {"markdown": "..."}
                return result.get("markdown", "")
        except Exception as e:
            raise RuntimeError(f"MinerU 转换失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="文档格式转换工具 (MinerU)")
    parser.add_argument("input_file", type=Path, help="输入文件路径")
    parser.add_argument("--output", "-o", type=Path, help="输出文件路径 (可选)")
    
    args = parser.parse_args()
    input_file = args.input_file
    
    if not input_file.exists():
        print(f"错误: 文件 {input_file} 不存在")
        sys.exit(1)
        
    # 如果已经是 markdown，直接复制或不做处理
    if input_file.suffix.lower() in ['.md', '.txt', '.markdown']:
        print(f"提示: 文件 {input_file.name} 已经是文本格式，无需转换。")
        # 如果指定了输出，则复制
        if args.output:
            import shutil
            shutil.copy(input_file, args.output)
            print(f"已复制到: {args.output}")
        else:
            print(f"输出路径: {input_file.absolute()}") # 直接返回原路径
        return

    # 执行转换
    client = MinerUClient()
    try:
        markdown_content = client.convert_to_markdown(input_file)
    except Exception as e:
        print(f"转换错误: {str(e)}")
        sys.exit(1)
        
    # 确定输出路径
    if args.output:
        output_file = args.output
    else:
        # 默认在同目录下生成 .md 文件
        output_file = input_file.with_suffix('.md')
        
    # 写入文件
    try:
        output_file.write_text(markdown_content, encoding="utf-8")
        print(f"转换成功！")
        print(f"输出路径: {output_file.absolute()}")
    except Exception as e:
        print(f"写入文件失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
