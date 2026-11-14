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
"""逐 case 精度批检器：吃 dump 出来的 actual/golden，吐 PASS/FAIL 表 + 退出码。

**为什么要有这个脚本**：性能调优工作流里，每个方案的 impl subagent 都要对全部 case 做一遍
精度验证。这件事的可变部分只有"怎么构造输入、怎么调算子"，而"按 dtype 定容差 → 逐元素比 →
统计超差 → 排版成表 → 给门禁结论"是完全固定的。让每个 subagent 各写一份，
既白烧 LLM turn，又让各方案的判定口径不完全一致，报告之间没法横比。

**分工**：调用方只负责把每个 case 的实测输出与 golden 各存一个 `.npy`；
比对、容差判定、**覆盖度校验**、表格、退出码由本脚本负责。

**容差没有默认值**——必须由调用方显式给出（`--rtol` / `--atol`），
取值依据见 `/ops-precision-standard` 的决策树（浮点/整型/量化、商用 vs 社区标准各不相同）。
这与 `ascendc-precision-debug` 的 Rule 1「容差必须外部给定，skill 无默认值」一致。

用法::

    python3 batch_precision_check.py \\
        --dump-dir <dir> --cases <cases.csv> \\
        --rtol 1e-3 --atol 1e-3 \\
        [--out precision_report.md] [--json precision.json]

dump 文件命名（默认，可用 --actual-pattern / --golden-pattern 改）::

    <dump-dir>/case<ID>.actual.npy
    <dump-dir>/case<ID>.golden.npy

多输出算子：同一个 case 存多份，后缀加序号 ``case<ID>.actual.1.npy`` / ``case<ID>.golden.1.npy``，
本脚本按序号配对，任一份超差即该 case FAIL。

退出码::

    0  全部 case 覆盖齐全且 PASS
    1  有 case FAIL（超差 / NaN-Inf 位置不一致 / 形状不匹配）
    2  覆盖不全（cases.csv 里有 case 没有 dump）或用法错误

> 退出码 2 单独分出来，是因为"只测了代表性用例"最常见的伪装就是——报告里全 PASS，
> 但根本没跑的 case 压根没出现在表里。覆盖度必须是机器判的。
"""

import argparse
import csv
import json
import os
import re
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.stderr.write("[ERROR] 需要 numpy：pip install numpy\n")
    sys.exit(2)


def parse_args():
    ap = argparse.ArgumentParser(
        description="逐 case 精度批检（容差必须外部给定）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dump-dir", required=True, help="存放 case<ID>.actual.npy / case<ID>.golden.npy 的目录")
    ap.add_argument("--cases", required=True, help="cases.csv，用于校验覆盖度（须含 case_id 列）")
    ap.add_argument("--rtol", required=True, type=float, help="相对容差（无默认值，依据 /ops-precision-standard 决定）")
    ap.add_argument("--atol", required=True, type=float, help="绝对容差（无默认值，依据 /ops-precision-standard 决定）")
    ap.add_argument("--actual-pattern", default="case{case_id}.actual{idx}.npy")
    ap.add_argument("--golden-pattern", default="case{case_id}.golden{idx}.npy")
    ap.add_argument("--out", help="Markdown 报告落盘路径")
    ap.add_argument("--json", dest="json_out", help="机读结果落盘路径")
    ap.add_argument("--label", default="", help="报告标题里的方案标识")
    return ap.parse_args()


def read_case_ids(path):
    """从 cases.csv 读 case_id 清单。没有 case_id 列时退回行号（1-based）。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.stderr.write(f"[ERROR] {path} 里没有任何 case\n")
        sys.exit(2)
    key = next((k for k in rows[0] if k and k.strip().lower() == "case_id"), None)
    if key is None:
        return [str(i + 1) for i in range(len(rows))]
    return [str(r[key]).strip() for r in rows]


def find_pairs(dump_dir, case_id, actual_pat, golden_pat):
    """找该 case 的全部 (actual, golden) 配对，按输出序号排序。"""
    pairs = []
    for idx in [""] + [f".{i}" for i in range(64)]:
        a = os.path.join(dump_dir, actual_pat.format(case_id=case_id, idx=idx))
        g = os.path.join(dump_dir, golden_pat.format(case_id=case_id, idx=idx))
        if os.path.exists(a) and os.path.exists(g):
            pairs.append((a, g))
        elif os.path.exists(a) != os.path.exists(g):
            # 只有一半 —— 明确报出来，别静默跳过
            pairs.append((a if os.path.exists(a) else None, g if os.path.exists(g) else None))
    return pairs


def compare(actual, golden, rtol, atol):
    """返回该输出的比对结果 dict。浮点走 isclose，整型/布尔要求逐位相等。"""
    res = {
        "shape_actual": list(actual.shape),
        "shape_golden": list(golden.shape),
        "dtype_actual": str(actual.dtype),
        "dtype_golden": str(golden.dtype),
        "n": int(golden.size),
        "n_mismatch": 0,
        "max_abs_err": 0.0,
        "max_rel_err": 0.0,
        "mare": 0.0,
        "nan_mismatch": 0,
        "inf_mismatch": 0,
        "reason": "",
        "passed": False,
    }
    if actual.shape != golden.shape:
        res["reason"] = "shape 不一致"
        return res

    if np.issubdtype(golden.dtype, np.floating):
        a = actual.astype(np.float64)
        g = golden.astype(np.float64)

        # NaN / Inf 的位置必须一一对应：位置对不上属于语义错误，不是数值误差
        nan_a, nan_g = np.isnan(a), np.isnan(g)
        inf_a, inf_g = np.isinf(a), np.isinf(g)
        res["nan_mismatch"] = int(np.count_nonzero(nan_a != nan_g))
        # Inf 还要同号
        inf_bad = (inf_a != inf_g) | (inf_a & inf_g & (np.sign(a) != np.sign(g)))
        res["inf_mismatch"] = int(np.count_nonzero(inf_bad))

        finite = ~(nan_a | nan_g | inf_a | inf_g)
        if np.any(finite):
            af, gf = a[finite], g[finite]
            abs_err = np.abs(af - gf)
            denom = np.maximum(np.abs(gf), np.finfo(np.float64).tiny)
            rel_err = abs_err / denom
            res["max_abs_err"] = float(abs_err.max())
            res["max_rel_err"] = float(rel_err.max())
            res["mare"] = float(rel_err.mean())
            bad = abs_err > (atol + rtol * np.abs(gf))
            res["n_mismatch"] = int(np.count_nonzero(bad))
        res["passed"] = res["n_mismatch"] == 0 and res["nan_mismatch"] == 0 and res["inf_mismatch"] == 0
        if not res["passed"]:
            bits = []
            if res["n_mismatch"]:
                bits.append(f"{res['n_mismatch']}/{res['n']} 超差")
            if res["nan_mismatch"]:
                bits.append(f"NaN 位置不一致 x{res['nan_mismatch']}")
            if res["inf_mismatch"]:
                bits.append(f"Inf 不一致 x{res['inf_mismatch']}")
            res["reason"] = "，".join(bits)
    else:
        # 整型 / 布尔：容差无意义，要求逐位相等
        bad = actual != golden
        res["n_mismatch"] = int(np.count_nonzero(bad))
        if res["n_mismatch"]:
            diff = np.abs(actual.astype(np.int64) - golden.astype(np.int64))
            res["max_abs_err"] = float(diff.max())
        res["passed"] = res["n_mismatch"] == 0
        if not res["passed"]:
            res["reason"] = f"整型逐位比对 {res['n_mismatch']}/{res['n']} 不等"
    return res


def main():
    args = parse_args()
    if not os.path.isdir(args.dump_dir):
        sys.stderr.write(f"[ERROR] dump 目录不存在：{args.dump_dir}\n")
        return 2

    case_ids = read_case_ids(args.cases)
    results, missing = [], []

    for cid in case_ids:
        pairs = find_pairs(args.dump_dir, cid, args.actual_pattern, args.golden_pattern)
        if not pairs:
            missing.append(cid)
            results.append(
                {"case_id": cid, "passed": False, "outputs": [], "reason": "缺 dump（该 case 没跑或没落盘）"}
            )
            continue
        outs, reasons, ok = [], [], True
        for a_path, g_path in pairs:
            if a_path is None or g_path is None:
                ok = False
                reasons.append("actual/golden 只有一半")
                outs.append({"reason": "actual/golden 只有一半", "passed": False})
                continue
            try:
                actual, golden = np.load(a_path), np.load(g_path)
            except Exception as exc:  # 损坏的 npy 也算失败，别吞
                ok = False
                reasons.append(f"读取失败：{exc}")
                outs.append({"reason": f"读取失败：{exc}", "passed": False})
                continue
            r = compare(actual, golden, args.rtol, args.atol)
            outs.append(r)
            if not r["passed"]:
                ok = False
                reasons.append(r["reason"])
        results.append(
            {
                "case_id": cid,
                "passed": ok,
                "outputs": outs,
                "reason": "；".join(x for x in reasons if x),
                "max_abs_err": max((item.get("max_abs_err", 0.0) for item in outs), default=0.0),
                "max_rel_err": max((item.get("max_rel_err", 0.0) for item in outs), default=0.0),
            }
        )

    n_pass = sum(1 for r in results if r["passed"])
    n_fail = len(results) - n_pass - len(missing)

    header = f"# 精度批检结果{(' — ' + args.label) if args.label else ''}\n\n"
    meta = (
        f"- 判据：`|actual - golden| <= atol + rtol * |golden|`，"
        f"rtol=`{args.rtol}` atol=`{args.atol}`（外部给定）\n"
        f"- 整型/布尔输出：逐位相等，容差不适用\n"
        f"- NaN/Inf：位置（及 Inf 符号）必须与 golden 一一对应\n"
        f"- 用例集：`{args.cases}`（{len(case_ids)} 个 case）\n"
        f"- dump 目录：`{args.dump_dir}`\n\n"
    )
    tbl = ["| case_id | 结论 | max_abs_err | max_rel_err | 说明 |", "|---|---|---|---|---|"]
    for r in results:
        mark = "✅PASS" if r["passed"] else "❌FAIL"
        tbl.append(
            f"| {r['case_id']} | {mark} | {r.get('max_abs_err', 0):.3e} | "
            f"{r.get('max_rel_err', 0):.3e} | {r.get('reason', '')} |"
        )
    summary = f"\n\n**结论**：{n_pass}/{len(case_ids)} PASS，{n_fail} FAIL，{len(missing)} 缺 dump"
    if missing:
        summary += f"（缺失 case_id：{', '.join(missing)}）"
    summary += "\n"

    report = header + meta + "\n".join(tbl) + summary
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[INFO] 报告已落盘：{args.out}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rtol": args.rtol,
                    "atol": args.atol,
                    "cases": results,
                    "n_pass": n_pass,
                    "n_fail": n_fail,
                    "missing": missing,
                },
                f,
                ensure_ascii=False,
                indent=1,
            )

    if missing:
        sys.stderr.write(
            f"[FAIL] 覆盖不全：{len(missing)} 个 case 没有 dump —— 「全量 case 覆盖」是硬约束，不允许只测代表性用例\n"
        )
        return 2
    if n_fail:
        sys.stderr.write(f"[FAIL] {n_fail} 个 case 精度不达标\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
