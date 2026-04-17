import argparse
import difflib
import os
import re
import sqlite3
import sys
from typing import Dict, List, Tuple


def _get_reference_doc_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "references", "profiler_db_data_format.md")


def _load_reference_doc() -> Tuple[List[str], str]:
    ref_path = _get_reference_doc_path()
    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return [], f"❌ 错误：未找到参考文档 {ref_path}"
    except Exception as e:
        return [], f"❌ 错误：读取参考文档失败: {str(e)}"
    return lines, ""


def _normalize_title(title: str) -> str:
    normalized = title.strip()
    normalized = re.sub(r"<a\s+name=\"[^\"]+\"></a>", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("\\_", "_")
    normalized = normalized.replace("\\-", "-")
    return normalized.strip()


def _canonical_key(name: str) -> str:
    key = name.strip().upper()
    key = key.replace("\\_", "_")
    key = re.split(r"[\s(（]", key)[0]
    return key


def _extract_sections(lines: List[str]) -> List[Dict[str, object]]:
    sections: List[Dict[str, object]] = []
    current_title = None
    current_start = None
    title_pattern = re.compile(r"^\*\*(.+?)\*\*$")

    for idx, line in enumerate(lines):
        matched = title_pattern.match(line.strip())
        if not matched:
            continue

        title = _normalize_title(matched.group(1))
        if current_title is not None and current_start is not None:
            sections.append(
                {
                    "title": current_title,
                    "start": current_start,
                    "end": idx,
                }
            )
        current_title = title
        current_start = idx

    if current_title is not None and current_start is not None:
        sections.append(
            {
                "title": current_title,
                "start": current_start,
                "end": len(lines),
            }
        )

    return sections


def list_documented_tables() -> str:
    lines, err = _load_reference_doc()
    if err:
        return err

    sections = _extract_sections(lines)
    names = []
    for sec in sections:
        title = sec["title"]
        canonical = _canonical_key(title)
        if re.fullmatch(r"[A-Z0-9_]+", canonical):
            names.append(canonical)

    if not names:
        return "❌ 未在参考文档中解析到表名。"

    unique_names = sorted(set(names))
    return "\n".join(unique_names)


def _load_db_tables(db_path: str) -> Tuple[List[str], str]:
    if not db_path:
        return [], "❌ 错误：db_path 不能为空。"
    if not os.path.exists(db_path):
        return [], f"❌ 错误：db 文件不存在：{db_path}"

    try:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall() if row and row[0]]
        finally:
            conn.close()
    except Exception as e:
        return [], f"❌ 错误：读取 db 表名失败: {str(e)}"

    return tables, ""


def list_db_tables(db_path: str) -> str:
    tables, err = _load_db_tables(db_path)
    if err:
        return err
    if not tables:
        return f"❌ db 中未找到任何表：{db_path}"
    return "\n".join(tables)


def compare_doc_with_db(db_path: str) -> str:
    doc_lines, err = _load_reference_doc()
    if err:
        return err

    doc_sections = _extract_sections(doc_lines)
    doc_tables = sorted(
        {
            _canonical_key(sec["title"])
            for sec in doc_sections
            if re.fullmatch(r"[A-Z0-9_]+", _canonical_key(sec["title"]))
        }
    )
    db_tables, db_err = _load_db_tables(db_path)
    if db_err:
        return db_err

    doc_set = set(doc_tables)
    db_set = {_canonical_key(name) for name in db_tables}

    both = sorted(doc_set & db_set)
    only_doc = sorted(doc_set - db_set)
    only_db = sorted(db_set - doc_set)

    out = []
    out.append("### 文档与当前 DB 表名对比")
    out.append(f"- 文档表数: {len(doc_set)}")
    out.append(f"- DB 表数: {len(db_set)}")
    out.append(f"- 交集: {len(both)}")
    out.append("")
    out.append("#### 交集表")
    out.append("\n".join(both) if both else "（无）")
    out.append("")
    out.append("#### 仅文档存在")
    out.append("\n".join(only_doc) if only_doc else "（无）")
    out.append("")
    out.append("#### 仅DB存在")
    out.append("\n".join(only_db) if only_db else "（无）")
    return "\n".join(out)


def get_schema_by_table_name(table_name: str) -> str:
    """
    根据表名从 profiler_db_data_format.md 提取对应章节内容。

    :param table_name: 表名，例如 TASK / CANN_API / COMMUNICATION_OP。
    """
    if not table_name:
        return "❌ 错误：table_name 不能为空。"

    lines, err = _load_reference_doc()
    if err:
        return err

    sections = _extract_sections(lines)
    if not sections:
        return "❌ 错误：参考文档中未解析到可用章节。"

    query_key = _canonical_key(table_name)
    exact_matches = []
    key_to_title = {}

    for sec in sections:
        title = sec["title"]
        title_key = _canonical_key(title)
        key_to_title[title_key] = title
        if title_key == query_key:
            exact_matches.append(sec)

    if not exact_matches:
        candidates = sorted(set(key_to_title.keys()))
        similar = difflib.get_close_matches(query_key, candidates, n=5, cutoff=0.5)
        if similar:
            tips = "、".join(similar)
            return f"❌ 未找到表 `{table_name}`。你可能想查：{tips}"
        return f"❌ 未找到表 `{table_name}`。可先执行 --list_tables 查看文档内可用表名。"

    sec = exact_matches[0]
    start = sec["start"]
    end = sec["end"]
    section_text = "\n".join(lines[start:end]).strip()

    out_lines = []
    out_lines.append("⚠️ **【Track B 表结构参考（来自 profiler_db_data_format.md）】**")
    out_lines.append(f"### 表名: `{_canonical_key(sec['title'])}`")
    out_lines.append("")
    out_lines.append(section_text)
    return "\n".join(out_lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="按表名查询 msprof db 文档中的表结构说明")
    parser.add_argument(
        "--db_path",
        type=str,
        help="可选，目标 sqlite db 路径；用于列表名或文档/DB 对比",
    )
    parser.add_argument(
        "--table_name",
        type=str,
        help="目标表名，例如 TASK / CANN_API / COMMUNICATION_OP",
    )
    parser.add_argument(
        "--list_tables",
        action="store_true",
        help="列出 profiler_db_data_format.md 中可查询的表名",
    )
    parser.add_argument(
        "--list_db_tables",
        action="store_true",
        help="列出目标 db 中实际存在的表名（需配合 --db_path）",
    )
    parser.add_argument(
        "--compare_doc_db",
        action="store_true",
        help="对比文档表名与目标 db 表名（需配合 --db_path）",
    )

    args = parser.parse_args(argv)

    if args.list_db_tables:
        if not args.db_path:
            print("❌ 错误：--list_db_tables 需要同时提供 --db_path")
            return
        print(list_db_tables(args.db_path))
        return

    if args.compare_doc_db:
        if not args.db_path:
            print("❌ 错误：--compare_doc_db 需要同时提供 --db_path")
            return
        print(compare_doc_with_db(args.db_path))
        return

    if args.list_tables:
        print(list_documented_tables())
        return

    if args.table_name:
        print(get_schema_by_table_name(args.table_name))
        return

    print("❌ 错误：请提供 --table_name <表名>，或使用 --list_tables")


if __name__ == "__main__":
    main(sys.argv[1:])
