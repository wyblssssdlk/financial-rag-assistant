"""Skill 入口：定位项目后，直接调用 financial_assistant.py 的只读 CLI（search / ask）。

检索、回退、提示词和输出格式全部来自 financial_assistant._query_cli，这里只负责找到项目。
用法与 `python financial_assistant.py search|ask ...` 相同。

项目根目录的查找顺序：
1. --project-dir 参数；
2. 环境变量 FINANCIAL_RAG_PROJECT；
3. 从本脚本所在目录逐级向上，找到包含 financial_assistant.py 的目录
   （只要 Skill 放在项目目录内的任意位置都会自动生效，与所用代理无关）。
"""
import argparse
import os
import sys
from pathlib import Path


def find_project(explicit=None):
    if explicit or os.getenv('FINANCIAL_RAG_PROJECT'):
        return Path(explicit or os.environ['FINANCIAL_RAG_PROJECT']).expanduser().resolve()
    for folder in Path(__file__).resolve().parents:
        if (folder / 'financial_assistant.py').is_file():
            return folder
    return None


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--project-dir')
    args, rest = parser.parse_known_args(argv)
    project = find_project(args.project_dir)
    if project is None or not (project / 'financial_assistant.py').is_file():
        parser.error('未找到项目入口 financial_assistant.py，请用 --project-dir 或环境变量 '
                     'FINANCIAL_RAG_PROJECT 指定项目根目录')
    sys.path.insert(0, str(project))
    from financial_assistant import _query_cli
    return _query_cli(rest)


if __name__ == '__main__':
    raise SystemExit(main())
