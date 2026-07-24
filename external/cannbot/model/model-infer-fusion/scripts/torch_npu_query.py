#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""torch_npu API 本地文档查询工具。

数据源：torch_npu 安装包内的 _op_plugin_docs.py（build 时由 op-plugin
仓的 codegen 模板拷入，含已生成 docstring 的算子条目，数量随版本浮动）。

跟用户实际安装的 torch_npu 版本严格绑定，零网络依赖，零版本 drift。

注意：op-plugin codegen 覆盖率有限，部分算子在 torch_npu 里可调用但
_op_plugin_docs.py 没有对应条目。本脚本对 model-infer-fusion scope 的关键融合
算子用内嵌 _FALLBACK_DOCS 字典兜底；对未被字典覆盖但运行时存在的算子，会给出
明确的「算子可调用但无 docstring」提示。

子命令：
  show   单 API 详情
  search 反向搜索（按关键词匹配 API 名或文档内容）
  list   枚举所有算子名

示例：
  python3 torch_npu_query.py show  npu_fusion_attention
  python3 torch_npu_query.py search attention
  python3 torch_npu_query.py search "MoE" --max 20
  python3 torch_npu_query.py list
  python3 torch_npu_query.py list --prefix npu_

无 torch_npu 环境时可显式指定文档路径：
  python3 torch_npu_query.py --docs-path /path/to/_op_plugin_docs.py list
"""

from __future__ import annotations

import argparse
import os
import re
import sys


DOC_ENTRY_PATTERN = re.compile(
    r'_add_torch_npu_docstr\(\s*"([^"]+)"\s*,\s*"""(.+?)"""',
    re.DOTALL,
)


class QueryError(Exception):
    """工具函数遇到不可恢复错误时抛出，由 main 统一捕获并退出。"""


def _emit(text: str = "", *, file=sys.stdout) -> None:
    """业务数据输出。"""
    file.write(text + "\n")


# Fallback：当前 torch_npu build 里 hasattr 为 True 但 _op_plugin_docs.py 没注册
# docstring 的算子。手工维护「一句话摘要 + 在线 URL」。
#
# Scope：仅覆盖 model-infer-fusion skill 关心的融合算子。stream/device 控制类等
# 跨 scope 的算子不在此表，由各自 skill 自行维护。
#
# URL 选型：hiascend.com 26.0.0 文档站。
#
# 校验方式：op-plugin 后续若补齐 docstring，_op_plugin_docs.py 的条目会自动覆盖此表。
_HIASCEND_URL_BASE = (
    "https://www.hiascend.com/document/detail/zh/Pytorch/2600/"
    "apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/"
)


def _hiascend_url(api_name: str) -> str:
    return f"{_HIASCEND_URL_BASE}torch_npu-{api_name}.md"


_FALLBACK_DOCS: dict[str, tuple[str, str]] = {
    # ===== 类 C：op-plugin codegen 漏 docstring 的 LLM 推理融合关键算子 =====
    "npu_kv_rmsnorm_rope_cache": (
        "融合 MLA 结构中 RMSNorm + RoPE + ScatterUpdate(KVCache 写入) 三步骤",
        _hiascend_url("npu_kv_rmsnorm_rope_cache"),
    ),
    "npu_interleave_rope": (
        "针对单输入 x 的 interleave 模式 RoPE 旋转位置编码",
        _hiascend_url("npu_interleave_rope"),
    ),
    "npu_gelu": (
        "GELU 激活函数",
        _hiascend_url("npu_gelu"),
    ),
    "npu_group_quant": (
        "对输入张量进行分组量化操作",
        _hiascend_url("npu_group_quant"),
    ),
    "npu_moe_distribute_dispatch": (
        "MoE EP 并行的 token dispatch（quant+EP AllToAll+TP AllGather），与 combine 配套",
        _hiascend_url("npu_moe_distribute_dispatch"),
    ),
    "npu_moe_distribute_dispatch_v2": (
        "MoE EP 并行的 token dispatch v2（MC2），与 combine_v2 / combine_add_rms_norm 配套",
        _hiascend_url("npu_moe_distribute_dispatch_v2"),
    ),
    "npu_moe_distribute_combine": (
        "MoE EP 并行的 token combine（reduce_scatterv + alltoallv），与 dispatch 配套",
        _hiascend_url("npu_moe_distribute_combine"),
    ),
    "npu_moe_distribute_combine_v2": (
        "MoE EP 并行的 token combine v2（MC2），与 dispatch_v2 配套",
        _hiascend_url("npu_moe_distribute_combine_v2"),
    ),
    "npu_moe_distribute_combine_add_rms_norm": (
        "MoE combine + add_rms_norm 融合（MC2），与 dispatch_v2 配套",
        _hiascend_url("npu_moe_distribute_combine_add_rms_norm"),
    ),
    "npu_moe_re_routing": (
        "MoE 网络 AlltoAll 后将 token 按专家顺序重新排列",
        _hiascend_url("npu_moe_re_routing"),
    ),

    # ===== 类 B：运行时 __doc__ 有内容但 _op_plugin_docs.py 未注册 =====
    # show 已经能通过运行时 __doc__ 拿到完整文档，这里只是让 list/search 也能看到。
    "erase_stream": (
        "移除张量被 record_stream 添加的已被某 stream 使用的标记",
        _hiascend_url("erase_stream"),
    ),
    "matmul_checksum": (
        "基于 torch.matmul 的 AIcore 错误硬件故障检测：校验矩阵计算结果是否超过门限",
        _hiascend_url("matmul_checksum"),
    ),
}


def _fallback_doc_text(api_name: str) -> str | None:
    """生成 fallback 算子的展示文本：摘要 + 在线链接 + 警示。"""
    entry = _FALLBACK_DOCS.get(api_name)
    if entry is None:
        return None
    summary, url = entry
    return (
        f"{summary}\n\n"
        f"[!] 当前 torch_npu build 未生成本地 docstring。请查在线文档：\n"
        f"    {url}"
    )


_OFFLINE_LIST_HINT = (
    "[!] 无 torch_npu 环境且未指定 --docs-path，已降级到 _FALLBACK_DOCS 兜底集。\n"
    "    完整算子目录请查算子总表：references/torch_npu_API/torch_npu_list.md\n"
    "    （该表为版本快照，可能与实际安装版本 drift，签名以本地 docstring 或在线文档为准）"
)


def resolve_docs_path(explicit_path: str | None) -> str | None:
    """定位 _op_plugin_docs.py。无 torch_npu 且无显式路径时返回 None，由调用方降级。"""
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise QueryError(f"指定的 --docs-path 不存在: {explicit_path}")
        return explicit_path

    try:
        import torch_npu
    except ImportError:
        return None

    candidate = os.path.join(torch_npu.__path__[0], "_op_plugin_docs.py")
    if not os.path.isfile(candidate):
        raise QueryError(
            f"未在 torch_npu 包内找到 _op_plugin_docs.py: {candidate}\n"
            "可能 torch_npu 版本过旧或非官方 build，请用 --docs-path 指定文档源。"
        )
    return candidate


def parse_doc_entries(docs_path: str | None) -> list[tuple[str, str]]:
    """解析 _op_plugin_docs.py，并合并 fallback 条目。

    优先级：_op_plugin_docs.py 的真实 docstring > _FALLBACK_DOCS。
    若 op-plugin 后续补齐 docstring，会自动覆盖 fallback 条目。
    docs_path 为 None（无环境降级）时只返回 _FALLBACK_DOCS。
    """
    entries: list[tuple[str, str]] = []
    if docs_path is not None:
        with open(docs_path, encoding="utf-8") as f:
            content = f.read()
        entries = DOC_ENTRY_PATTERN.findall(content)

    existing = {name for name, _ in entries}
    for name in _FALLBACK_DOCS:
        if name not in existing:
            entries.append((name, _fallback_doc_text(name) or ""))
    return entries


def cmd_show(api_name: str, docs_path: str | None) -> None:
    """单 API 详情。

    查找优先级：
      1. 运行时 torch_npu.<api>.__doc__（类 B / 正常算子）
      2. parse_doc_entries（_op_plugin_docs.py + _FALLBACK_DOCS 字典）
      3. 运行时 hasattr=True 但都未命中：通用「算子可调用但无 docstring」提示
      4. 都不命中：未找到 API
    """
    api_exists_at_runtime = False
    try:
        import torch_npu

        if hasattr(torch_npu, api_name):
            api_exists_at_runtime = True
            obj = getattr(torch_npu, api_name)
            if obj is not None and obj.__doc__:
                _emit(obj.__doc__)
                return
    except ImportError:
        pass

    entries = parse_doc_entries(docs_path)
    for name, doc in entries:
        if name == api_name:
            _emit(doc.strip())
            return

    if api_exists_at_runtime:
        raise QueryError(
            f"算子 {api_name} 在当前 torch_npu 环境中可调用，"
            f"但 op-plugin codegen 未生成 docstring 条目，"
            f"且未被 _FALLBACK_DOCS 字典覆盖。\n"
            f"请查在线文档：\n"
            f"  {_hiascend_url(api_name)}\n"
            f"如该算子属于 model-infer-fusion scope 的关键融合算子，"
            f"建议补充到本脚本的 _FALLBACK_DOCS 字典。"
        )

    if docs_path is None:
        raise QueryError(
            f"未找到 API: {api_name}\n{_OFFLINE_LIST_HINT}"
        )
    raise QueryError(f"未找到 API: {api_name}")


def cmd_search(keyword: str, max_results: int, docs_path: str | None) -> None:
    """反向搜索：关键词匹配 API 名或文档内容。"""
    if docs_path is None:
        _emit(_OFFLINE_LIST_HINT, file=sys.stderr)
    entries = parse_doc_entries(docs_path)
    keyword_lower = keyword.lower()

    results = []
    for name, doc in entries:
        if keyword_lower in name.lower() or keyword_lower in doc.lower():
            first_line = next(
                (line.strip() for line in doc.splitlines() if line.strip()), ""
            )
            if len(first_line) > 150:
                first_line = first_line[:147] + "..."
            # fallback 条目在 search 输出里加标记，让 agent 选型时区分兜底 vs 完整 docstring
            if name in _FALLBACK_DOCS:
                first_line = f"[fallback] {first_line}"
            results.append((name, first_line))
            if len(results) >= max_results:
                break

    if not results:
        raise QueryError(f"未找到匹配 '{keyword}' 的算子。")

    _emit(f"搜索 '{keyword}' 找到 {len(results)} 个结果（上限 {max_results}）:\n")
    for i, (name, summary) in enumerate(results, 1):
        _emit(f"{i}. {name}")
        if summary:
            _emit(f"   {summary}")
        _emit()


def cmd_list(prefix: str | None, docs_path: str | None) -> None:
    """枚举所有算子名。"""
    if docs_path is None:
        _emit(_OFFLINE_LIST_HINT, file=sys.stderr)
    entries = parse_doc_entries(docs_path)
    names = sorted({name for name, _ in entries})

    if prefix:
        names = [n for n in names if n.startswith(prefix)]

    for name in names:
        _emit(name)
    sys.stdout.flush()
    _emit(f"\n总计 {len(names)} 个算子。", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="torch_npu API 本地文档查询",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("子命令：", 1)[-1] if __doc__ else "",
    )
    parser.add_argument(
        "--docs-path",
        default=None,
        help="显式指定 _op_plugin_docs.py 路径（无 torch_npu 环境时使用）",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="单 API 详情")
    p_show.add_argument("api_name", help="算子名，如 npu_fusion_attention")

    p_search = sub.add_parser("search", help="反向搜索")
    p_search.add_argument("keyword", help="关键词，匹配名字或文档内容")
    p_search.add_argument(
        "--max", type=int, default=10, dest="max_results", help="最多返回结果数（默认 10）"
    )

    p_list = sub.add_parser("list", help="枚举所有算子名")
    p_list.add_argument("--prefix", default=None, help="只列以该前缀开头的算子")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        docs_path = resolve_docs_path(args.docs_path)
        if args.cmd == "show":
            cmd_show(args.api_name, docs_path)
        elif args.cmd == "search":
            cmd_search(args.keyword, args.max_results, docs_path)
        elif args.cmd == "list":
            cmd_list(args.prefix, docs_path)
    except QueryError as err:
        _emit(str(err), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
