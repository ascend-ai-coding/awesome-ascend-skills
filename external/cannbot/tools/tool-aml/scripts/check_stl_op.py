#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""STL 新增算子基础语法 + 越界自检。

用法:
  python3 check_stl_op.py --op-dir <FORM>/ops/<TYPE>_<NAME>

检查项:
  1. 基础语法:括号/花括号/方括号是否配对。
  2. 命名一致性:.cpp desc.name / ResolveKernelSymbol / .asc 导出符号 / include 的 cce。
  3. 内在函数参数个数:与同形态其它算子(兄弟算子)对照,个数不一致则告警。
  4. 禁用模式:960DT 上使用 get_subblockid()/subblock。
  5. 越界:host 每 block 输出字节数 vs kernel `block_idx * <expr>` 块偏移是否一致;
     输出字节数须为 4 的倍数(CreateStlOutputTensor 要求)。
  6. copy_ubuf_to_gm_align_v2 传输量(nBurst*burstLen)是否可能超出 dst 块。

退出码:0=无问题;1=发现问题。
"""
import argparse
import ast
import operator
import os
import re
import sys

INTRINSICS = [
    "copy_gm_to_cbuf_align_v2",
    "copy_matrix_cc_to_gm",
    "copy_matrix_cc_to_ub",
    "copy_ubuf_to_gm_align_v2",
]


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def strip_strings(text):
    text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)
    text = re.sub(r"'(?:[^'\\]|\\.)*'", "''", text)
    return text


def balanced(text):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def call_arg_counts(text, fn):
    """返回每个 fn(...) 调用的顶层逗号数+1(即参数个数)。"""
    counts = []
    for m in re.finditer(re.escape(fn) + r"\s*\(", text):
        i = m.end()
        depth = 1
        j = i
        while depth > 0 and j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        body = text[i:j - 1]
        d, c = 0, 1
        for ch in body:
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
            elif ch == "," and d == 0:
                c += 1
        counts.append(c)
    return counts


def parse_defines(mem_info_path):
    defines = {}
    if not os.path.isfile(mem_info_path):
        return defines
    with open(mem_info_path, encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            m = re.match(r"\s*#define\s+(\w+)\s+(0[xX][0-9A-Fa-f]+|\d+)", line)
            if not m:
                continue
            name, value = m.group(1), m.group(2)
            try:
                defines[name] = int(value, 16) if value.lower().startswith("0x") else int(value, 10)
            except ValueError:
                continue
    return defines


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
}


def _eval_node(node, defines):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, defines)
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, (int, float)) else None
    if isinstance(node, ast.Name):
        return defines.get(node.id)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, defines)
        right = _eval_node(node.right, defines)
        if left is None or right is None:
            return None
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            return None
        return op(left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_node(node.operand, defines)
        if operand is None:
            return None
        return operand if isinstance(node.op, ast.UAdd) else -operand
    return None


def eval_expr(expr, defines):
    e = expr.strip()
    e = re.sub(r"static_cast<\s*[^>]+>\s*\(", "(", e)
    e = re.sub(r"\(\s*(size_t|uint64_t|uint32_t|int)\s*\)", "", e)
    if not re.fullmatch(r"[\s0-9A-Fa-fxX()+\-*/<>&|.A-Za-z_]+", e):
        return None
    names = set(re.findall(r"(?<![\w.])[A-Za-z_]\w*", e))
    env = {n: defines[n] for n in names if n in defines}
    if any(n not in env for n in names):
        return None
    try:
        value = _eval_node(ast.parse(e, mode="eval"), env)
        return int(value) if isinstance(value, (int, float)) else None
    except Exception:
        return None


def find_block_strides(text, defines):
    """kernel 里 block_idx * <步长> 的偏移。返回 [(步长值, 源码行), ...]。

    括号感知:步长只取到顶层(括号外)的 + / - / ; / , / ) / ] / 换行前,
    避免把 block_idx * stride + offset 里的块内 offset 混入步长。
    """
    strides = []
    for m in re.finditer(r"block_idx\s*\*\s*", text):
        start = m.end()
        depth = 0
        i = start
        while i < len(text):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "]" and depth == 0:
                break
            elif ch in "+-;,\n" and depth == 0:
                break
            i += 1
        expr = text[start:i].strip()
        value = eval_expr(expr, defines)
        if value is None:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        context = text[line_start:line_end if line_end != -1 else len(text)].strip()
        strides.append((value, context))
    return strides


def host_output_per_block(text, defines):
    """返回 (block_dim, [每 block 字节数...])。

    block_dim 无法静态求值(如 desc.blockDim = 运行时变量)时返回 None;
    per 中 None 表示该输出无法解析(表达式无法求值,或固定总大小但 blockDim 未知)。
    """
    bd = None
    m = re.search(r"desc\.blockDim\s*=\s*([^;]+);", text)
    if m:
        v = eval_expr(m.group(1).strip(), defines)
        if v:
            bd = v
    per = []
    out_block = re.search(r"desc\.outputs\s*=\s*\{(.*?)\};", text, flags=re.S)
    if out_block:
        for g in re.findall(r"\{([^{}]*)\}", out_block.group(1)):
            gx = g.strip()
            if not gx:
                continue
            if "blockDim" in gx:
                expr = re.split(r"\*\s*(?=\S)", gx, maxsplit=1)
                per.append(eval_expr(expr[-1].rstrip("}"), defines))
            else:
                v = eval_expr(gx, defines)
                if v is None or not bd:
                    per.append(None)
                else:
                    per.append(v // bd)
    return bd, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", required=True, help="<FORM>/ops/<TYPE>_<NAME>")
    args = ap.parse_args()

    op_dir = os.path.abspath(args.op_dir)
    if not os.path.isdir(op_dir):
        print("[ERROR] 非目录: " + op_dir)
        return 1
    form_dir = os.path.dirname(os.path.dirname(op_dir))
    ops_root = os.path.dirname(op_dir)
    op_name = os.path.basename(op_dir)
    arch960 = "ascend960dt" in form_dir

    files = sorted(f for f in os.listdir(op_dir) if f.endswith((".cce", ".asc", ".cpp")))
    texts = {}
    for f in files:
        with open(os.path.join(op_dir, f), encoding="utf-8", errors="ignore") as fp:
            texts[f] = fp.read()
    stripped = {f: strip_comments(t) for f, t in texts.items()}
    defines = parse_defines(os.path.join(ops_root, "ops_include", "mem_info.h"))

    problems = []

    def add(sev, msg, fix):
        problems.append((sev, msg, fix))

    # 1. 基础语法
    for f, t in stripped.items():
        if not balanced(strip_strings(t)):
            add("ERROR", "%s 括号/花括号不配对。" % f, "检查 #if/#ifdef 与函数体括号是否闭合。")

    # 2. 命名一致性
    cpp = next((t for f, t in stripped.items() if f.endswith(".cpp")), "")
    asc = next((t for f, t in stripped.items() if f.endswith(".asc")), "")
    export = re.search(r'AmlStlLaunch(\w+)', asc)
    if not export:
        add("ERROR", ".asc 缺少 extern \"C\" AmlStlLaunch<Name> 下发函数。", "补下发函数并使用导出的符号名。")
    else:
        sym = "AmlStlLaunch" + export.group(1)
        if ('ResolveKernelSymbol("%s")' % sym) not in cpp.replace(" ", ""):
            add("WARN", ".cpp 未用 ResolveKernelSymbol(\"%s\") 解析同一符号。" % sym,
                "让 .cpp 的 ResolveKernelSymbol 字符串与 .asc 导出符号完全一致。")
        if ('"%s"' % op_name) not in cpp:
            add("INFO", "desc.name 与目录名 %s 不同(如注册名 mix_cube_load3d_covery 对目录 mix_load3d_covery)。"
                % op_name, "若非故意,设置 desc.name = \"%s\"。" % op_name)
    for f in files:
        if f.endswith(".cce"):
            if f not in asc:
                add("WARN", ".asc 未 #include %s。" % f, "在 .asc 顶部 #include 该 cce。")

    if asc and (op_name.split("_")[0] in ("aic", "aiv", "mix")):
        if op_name.startswith("mix_"):
            if "__mix__" not in asc:
                add("ERROR", "mix 算子 .asc 未见 __mix__(...)。", "用 __global__ __mix__(1,1)(960DT)/__mix__(1,2)(950) 包装。")
        else:
            if "__aicore__" not in asc:
                add("WARN", "aic/aiv 算子 .asc 未见 __global__ __aicore__ 包装。", "用 __global__ __aicore__ 包装 device 核函数。")

    # 3. 内在函数参数个数:对照兄弟算子
    ref = {fn: set() for fn in INTRINSICS}
    for d in os.listdir(ops_root):
        sib = os.path.join(ops_root, d)
        if d in (op_name, "ops_include") or not os.path.isdir(sib):
            continue
        for f in os.listdir(sib):
            if f.endswith(".cce"):
                with open(os.path.join(sib, f), encoding="utf-8", errors="ignore") as fp:
                    sc = strip_comments(fp.read())
                for fn in INTRINSICS:
                    ref[fn].update(call_arg_counts(sc, fn))
    for f, t in stripped.items():
        for fn in INTRINSICS:
            for c in call_arg_counts(t, fn):
                if ref[fn] and c not in ref[fn]:
                    add("ERROR", "%s 中 %s 参数个数=%d,同形态兄弟算子为 %s。"
                        % (f, fn, c, sorted(ref[fn])),
                        "对照同形态算子修正参数个数/占位符(形态间签名有差异)。")

    # 4. 禁用模式
    if arch960:
        for f, t in stripped.items():
            if "get_subblockid" in t or "subblock" in t:
                add("ERROR", "%s 在 960DT 使用了 subblock/get_subblockid。" % f,
                    "960DT 每 core 1 AIC:1 AIV,无 subblock;删除 subblock 相关逻辑与偏移。")

    # 5. 越界:host 每 block 输出 vs kernel block 偏移
    _, per = host_output_per_block(cpp, defines)
    kernel_text = "".join(t for f, t in stripped.items() if f.endswith((".cce", ".asc")))
    strides = find_block_strides(kernel_text, defines)
    resolved = []  # [(原始输出索引, 每 block 字节数), ...]
    for idx, p in enumerate(per):
        if p is None:
            add("WARN", "输出[%d] 每 block 字节数无法解析,越界比对跳过。" % idx,
                "desc.outputs[%d] 优先写 blockDim * 每块字节;固定总大小时 desc.blockDim 必须可静态求值。" % idx)
        else:
            resolved.append((idx, p))
    resolved_vals = [p for _, p in resolved]
    stride_values = sorted(set(v for v, _ in strides))
    if not per:
        add("WARN", "未从 host .cpp 解析到输出,越界比对跳过。",
            "确认 desc.outputs 按模板逐项写 {blockDim * 每块字节}。")
    elif not resolved:
        add("WARN", "所有输出每 block 字节数均无法解析,越界比对跳过。",
            "确认 desc.outputs 与 desc.blockDim 写法(blockDim 需可静态求值)。")
    elif not strides:
        add("WARN", "kernel 未识别到 block_idx * <步长> 的块偏移,越界比对跳过。",
            "确认 kernel 用 block_idx * <expr> 形式计算块偏移。")
    for idx, p in resolved:
        if p % 4 != 0:
            add("ERROR", "输出[%d] 每 block 字节数=%d 不是 4 的倍数。" % (idx, p),
                "调整为 4 的倍数(CreateStlOutputTensor 要求)。")
        if strides and p not in stride_values:
            sample = " | ".join(ctx for _, ctx in strides[:2])
            add("ERROR", "host 输出[%d] 每 block=%d 未匹配任何 kernel block_idx 步长 %s(越界风险)。kernel 偏移示例: %s"
                % (idx, p, [hex(s) for s in stride_values], sample),
                "令 desc.outputs[i].byteSize/blockDim == 对应 kernel 输出数组的 block_idx 步长(块内 +offset 不算步长)。")
    extra = [v for v in stride_values if v not in resolved_vals]
    if resolved and extra:
        add("WARN", "kernel 存在无法关联到 desc.outputs 的 block_idx 步长 %s,可能为输入或其他输出偏移,请人工确认。"
            % [hex(v) for v in extra],
            "检查这些偏移是否对应输入读取或多输出;无法一一关联时保留现状。")

    # 6. copy_ubuf_to_gm_align_v2 传输量估算(burstLen 为 32 字节 burst 单位)
    if resolved:
        for f, t in stripped.items():
            for m in re.finditer(r"copy_ubuf_to_gm_align_v2\s*\(([^;]*)\)\s*;", t):
                args = [a.strip() for a in re.split(r",(?![^()]*\))", m.group(1))]
                if len(args) < 5:
                    continue
                nb = eval_expr(args[3], defines)
                bl = eval_expr(args[4], defines)
                if not nb or not bl:
                    continue
                tsz = nb * bl * 32
                exceeds = [(i, p) for i, p in resolved if p and tsz > p]
                if exceeds:
                    idxs = [i for i, _ in exceeds]
                    vals = [p for _, p in exceeds]
                    add("WARN", "%s: copy_ubuf_to_gm_align_v2 传输量=%d 字节(nBurst*burstLen*32) 可能超过输出%s 每块 %s。"
                        % (f, tsz, idxs, [hex(v) for v in vals]),
                        "确认 nBurst/burstLen 与对应 dst 输出块大小匹配(参考修复:0x200->0x180)。")

    issues = [p for p in problems if p[0] != "INFO"]
    infos = [p for p in problems if p[0] == "INFO"]
    for msg, fix in [(m, f) for _, m, f in infos]:
        print("[INFO] %s" % msg)
        print("   建议: %s\n" % fix)
    if not issues:
        print("[OK] %s 未发现基础语法/越界问题。" % op_name)
        return 0

    print("[FOUND] %s: %d 个问题\n" % (op_name, len(issues)))
    for i, (sev, msg, fix) in enumerate(issues, 1):
        print("%d) [%s] %s" % (i, sev, msg))
        print("   建议: %s\n" % fix)
    return 1


if __name__ == "__main__":
    sys.exit(main())
