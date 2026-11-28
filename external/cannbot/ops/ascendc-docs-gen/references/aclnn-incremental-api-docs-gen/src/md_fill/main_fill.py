# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Markdown 文档章节替换工具

根据 backfill_map.json 中的内容修改 md 文档的章节。
"""

# 导出主要函数
__all__ = ["update_md_with_dict", "renumber_npu_ids"]

import json
import os
import re
from typing import Union

# 精确匹配的章节映射 (key -> 标题)
EXACT_MATCH_KEYS = {
    "support": "## 产品支持情况",
    "summary": "## 功能说明",
    "function_proto": "## 函数原型",
    "constraint": "## 约束说明",
    "example": "## 调用示例",
}

# 按标题前缀匹配的 key
PREFIX_MATCH_KEY = "sub_function_hash"
PREFIX_PATTERN = r"^## aclnn"


def _load_data(data: Union[dict[str, str], str]) -> dict[str, str]:
    """
    加载数据，支持 dict 或 JSON 文件路径

    Args:
        data: dict 对象或 JSON 文件路径

    Returns:
        字典数据

    Raises:
        FileNotFoundError: JSON 文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    if isinstance(data, dict):
        return data
    elif isinstance(data, str):
        with open(data, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise TypeError(f"data 参数类型错误，期望 dict 或 str，实际为 {type(data)}")


def _is_empty_value(value) -> bool:
    """
    判断 dict 的 value 是否为空

    空值包括 None、空字符串、纯空白字符串。
    当 value 为空时，对应章节保留原内容不做替换。

    Args:
        value: dict 中某个 key 对应的 value

    Returns:
        True 表示为空，False 表示有内容
    """
    if value is None:
        return True
    return isinstance(value, str) and value.strip() == ""


def _read_md_file(md_path: str) -> str:
    """
    读取 Markdown 文件内容

    Args:
        md_path: md 文件路径

    Returns:
        文件内容字符串
    """
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


def _split_sections(content: str) -> list[str]:
    """
    按 ## 章节标题 分割文档

    使用正则表达式匹配 ## 标题所在行进行分割，
    保持标题行作为独立元素。

    Args:
        content: 原始 markdown 内容

    Returns:
        分割后的列表，格式为 ['# 标题', '## 章节1\n内容1', '## 章节2\n内容2', ...]
    """
    # 按 ## 开头的行分割
    pattern = r"\n(?=## )"
    parts = re.split(pattern, content)

    # 第一个部分可能以 # 开头（文档标题），需要特殊处理
    if parts and not parts[0].startswith("##"):
        return parts

    return parts


def _find_section_index(parts: list[str], title: str) -> int | None:
    """
    查找指定标题的章节索引

    Args:
        parts: 分割后的章节列表
        title: 要查找的标题（如 '## 产品支持情况'）

    Returns:
        章节索引，未找到返回 None
    """
    for i, part in enumerate(parts):
        # 去除首尾空白后检查是否以指定标题开头
        stripped = part.strip()
        if stripped.startswith(title):
            return i
    return None


def _find_consecutive_prefix_sections(parts: list[str]) -> tuple[int, int]:
    """
    查找连续的以 aclnn 开头的章节块

    找到 md 文档中连续的以 ## aclnn 开头的章节段，
    如果遇到非 aclnn 开头的 ## 标题则停止。

    Args:
        parts: 分割后的章节列表

    Returns:
        (start_index, end_index) 元组，end_index 为不包含的索引
        如果没有找到，返回 (-1, -1)
    """
    start = -1
    end = -1

    for i, part in enumerate(parts):
        stripped = part.strip()
        # 检查是否以 ## aclnn 开头
        if re.match(PREFIX_PATTERN, stripped):
            if start == -1:
                start = i
            end = i + 1  # end 是exclusive的
        elif start != -1:
            # 已经找到起始位置，但遇到非 aclnn 章节，停止
            break

    if start == -1:
        return (-1, -1)
    return (start, end)


def _build_result(parts: list[str]) -> str:
    """
    将章节列表重新组合成完整的 markdown 字符串

    Args:
        parts: 分割后的章节列表

    Returns:
        重新组合的 markdown 字符串
    """
    return "\n".join(parts)


_NPU_OPEN_RE = re.compile(r'(<!-- npu="[^"]*" )id(\d+)( -->)')
_NPU_END_RE = re.compile(r"(<!-- end )id(\d+)( -->)")


def renumber_npu_ids(text: str) -> str:
    """
    按文档顺序扫描所有 npu 注释对，将所有 id 重新编号为连续整数。

    算法：先扫一遍 opening tags 建立 old_id → new_id 映射，
    再统一替换 opening 与 closing tags。即使生成阶段各章节
    使用了不同的起始编号，组装后也能保证全文 id 从 1 开始
    连续递增且不重复。
    """
    id_map: dict[str, str] = {}
    counter = 0

    def _map_and_replace_open(m: re.Match) -> str:
        nonlocal counter
        old = m.group(2)
        if old not in id_map:
            counter += 1
            id_map[old] = str(counter)
        return f"{m.group(1)}id{id_map[old]}{m.group(3)}"

    def _replace_end(m: re.Match) -> str:
        old = m.group(2)
        new = id_map.get(old, old)
        return f"{m.group(1)}id{new}{m.group(3)}"

    text = _NPU_OPEN_RE.sub(_map_and_replace_open, text)
    text = _NPU_END_RE.sub(_replace_end, text)
    return text


def update_md_with_dict(data: Union[dict[str, str], str], md_path: str, out_file_path: str = None) -> str:
    """
    根据 dict 修改 md 文档

    Args:
        data: 包含章节内容的字典，或 json 文件路径。
              - 如果是 dict：key 为章节标识，value 为 markdown 完整内容
              - 如果是 str：当作 json 文件路径读取
        md_path: md 文档路径

    Returns:
        修改后的完整 md 文档字符串

    替换规则:
        - dict 中的每个 key 对应 md 文档中的一个或多个 ## 章节
        - 用 dict[key] 的内容完全替换对应的章节（包括标题）
        - dict 中不包含的 key，其对应章节保持原样
        - dict[key] 的 value 为空（None 或空/纯空白字符串）时，
          对应章节保留原内容不替换

    特殊规则:
        - sub_function_hash: 替换 md 文档中所有连续的以 ## aclnn 开头的章节块
    """
    # 加载数据
    data_dict = _load_data(data)

    # 读取 md 文件
    md_content = _read_md_file(md_path)

    # 分割章节
    parts = _split_sections(md_content)

    # 处理精确匹配的章节
    for key, title in EXACT_MATCH_KEYS.items():
        if key in data_dict:
            if _is_empty_value(data_dict[key]):
                continue
            idx = _find_section_index(parts, title)
            if idx is not None:
                # 用字典中的内容替换整个章节
                parts[idx] = data_dict[key]

    # 处理前缀匹配的章节 (sub_function_hash)
    if PREFIX_MATCH_KEY in data_dict and not _is_empty_value(data_dict[PREFIX_MATCH_KEY]):
        start, end = _find_consecutive_prefix_sections(parts)
        if start != -1:
            # 删除原来的 aclnn 章节块
            del parts[start:end]
            # 插入新的子章节内容
            parts.insert(start, data_dict[PREFIX_MATCH_KEY])

    # 重新组合结果
    ret = _build_result(parts)

    # 统一重编号所有 npu 注释对的 id，保证全文连续且不重复
    ret = renumber_npu_ids(ret)

    if out_file_path:
        with open(out_file_path, "w", encoding="utf-8") as f:
            f.write(ret)

    return ret


if __name__ == "__main__":
    # 示例用法
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 测试路径
    data_path = "../../out/backfill_map.json"
    md_path = "../../out/operator_storage/abs/base/docs/aclnnAbs.md"

    # 如果文件存在，执行测试
    if os.path.exists(data_path) and os.path.exists(md_path):
        result = update_md_with_dict(data_path, md_path)
        print(f"处理完成，结果长度: {len(result)} 字符")
        print("\n前500字符预览:")
        print(result)
    else:
        print("测试文件不存在，跳过测试")
