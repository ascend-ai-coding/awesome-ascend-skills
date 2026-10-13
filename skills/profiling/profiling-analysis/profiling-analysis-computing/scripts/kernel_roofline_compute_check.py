#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 kernel_details.csv 中算力瓶颈算子做 Roofline 核对。

默认只分析 aic_mac_ratio ≥ 80%：
  理论耗时 = max(计算量/算力, 搬运量/带宽)
  效率     = 理论耗时 / 实际耗时
并判断是否需要优化、给出初步排查方向。

用法：
  python kernel_roofline_compute_check.py --input-path <profiling根目录> --flops 313 --bandwidth 1.2 --bw-unit TB/s
  python kernel_roofline_compute_check.py --csv <kernel_details.csv> --flops 313 --bandwidth 1200
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DTYPE_BYTES = {
    "BOOL": 1, "INT8": 1, "UINT8": 1, "QINT8": 1, "HIFLOAT8": 1,
    "FLOAT8": 1, "FLOAT8_E4M3FN": 1, "FLOAT8_E5M2": 1, "INT4": 0.5,
    "INT16": 2, "UINT16": 2, "HALF": 2, "FLOAT16": 2, "FP16": 2,
    "BFLOAT16": 2, "BF16": 2, "F16": 2,
    "INT32": 4, "UINT32": 4, "FLOAT": 4, "FLOAT32": 4, "FP32": 4,
    "INT64": 8, "UINT64": 8, "DOUBLE": 8, "FLOAT64": 8,
}
UNK_DTYPE_BYTES = 4

MATMUL_TYPES = {
    "MATMUL", "MATMULV2", "MATMULV3", "BATCHMATMUL", "BATCHMATMULV2",
    "BATCHMATMULV3", "FUSEDMATMUL", "MATMULEX",
}


def norm_col(s: str) -> str:
    return re.sub(r"[\s_()\[\]\.]", "", (s or "").lower())


def canon_type(rtype: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (rtype or "").upper())


def build_index(header: Sequence[str]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for i, h in enumerate(header):
        n = norm_col(h)
        if n and n not in idx:
            idx[n] = i
    return idx


def get_col(idx: Dict[str, int], names: Sequence[str], row: Sequence[str], default: str = "") -> str:
    for n in names:
        k = norm_col(n)
        if k in idx and idx[k] < len(row):
            v = row[idx[k]]
            return v if v is not None else default
    return default


def parse_shapes(field: str) -> List[List[int]]:
    """解析 Input/Output Shapes → [[dim, ...], ...]；N/A 为空；负维视为动态。"""
    if field is None:
        return []
    f = field.strip().strip('"').strip()
    if f == "" or f.upper() in ("N/A", "NA", "NONE", "NULL"):
        return []
    out: List[List[int]] = []
    for part in f.split(";"):
        p = part.strip().strip('"').strip()
        p = p.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(" ", "")
        if not p:
            out.append([1])
            continue
        dims: List[int] = []
        dynamic = False
        for d in re.split(r"[,xX*]", p):
            if not d or not re.fullmatch(r"-?\d+", d):
                continue
            v = int(d)
            if v < 0:
                dynamic = True
                v = 1
            dims.append(v)
        if dynamic:
            return []  # 动态 shape 不建模
        out.append(dims if dims else [1])
    return out


def numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def prod(xs: Sequence[int]) -> int:
    n = 1
    for x in xs:
        n *= x
    return n


def _norm_dtype(raw: str) -> str:
    t = (raw or "").strip().strip('"').strip().upper()
    return t[3:] if t.startswith("DT_") else t


def dtypes_items(field: str) -> List[Optional[float]]:
    """每个 dtype 的字节数。0=可选缺席；None=未知枚举码，稍后按同 numel 推断。"""
    if not field or _norm_dtype(field) in ("N/A", "NONE"):
        return []
    sizes: List[Optional[float]] = []
    for raw in field.split(";"):
        t = _norm_dtype(raw)
        if t in ("", "UNDEFINED", "N/A", "NONE"):
            sizes.append(0.0)
        elif t.isdigit():
            sizes.append(None)
        else:
            sizes.append(DTYPE_BYTES.get(t, UNK_DTYPE_BYTES))
    return sizes


def resolve_itemsize(
    in_shapes: Sequence[Sequence[int]],
    in_dts: str,
    out_shapes: Sequence[Sequence[int]],
    out_dts: str,
) -> Tuple[List[float], List[float]]:
    in_s = dtypes_items(in_dts)
    out_s = dtypes_items(out_dts)
    known: Dict[int, float] = {}
    for s, b in list(zip(in_shapes, in_s)) + list(zip(out_shapes, out_s)):
        if b:
            known.setdefault(numel(s), b)

    def fill(shapes: Sequence[Sequence[int]], sizes: Sequence[Optional[float]]) -> List[float]:
        out: List[float] = []
        for i, s in enumerate(shapes):
            b = sizes[i] if i < len(sizes) else None
            if b is None:
                b = known.get(numel(s), UNK_DTYPE_BYTES)
            out.append(b)
        return out

    return fill(in_shapes, in_s), fill(out_shapes, out_s)


def sum_bytes(shapes: Sequence[Sequence[int]], sizes: Sequence[float]) -> float:
    total = 0.0
    for i, s in enumerate(shapes):
        b = sizes[i] if i < len(sizes) else UNK_DTYPE_BYTES
        total += numel(s) * (b or 0)
    return total


def tensor_bytes(
    in_shapes: Sequence[Sequence[int]],
    in_dts: str,
    out_shapes: Sequence[Sequence[int]],
    out_dts: str,
) -> float:
    in_sizes, out_sizes = resolve_itemsize(in_shapes, in_dts, out_shapes, out_dts)
    return sum_bytes(in_shapes, in_sizes) + sum_bytes(out_shapes, out_sizes)


def lowprec_rate(in_dts: str, int8_scale: float) -> float:
    dt = _norm_dtype((in_dts or "").split(";")[0])
    if dt in ("INT8", "UINT8", "INT4", "HIFLOAT8", "FLOAT8", "FLOAT8_E4M3FN", "FLOAT8_E5M2"):
        return int8_scale
    return 1.0


def bandwidth_gbps(value: float, unit: str) -> float:
    u = (unit or "GB/s").strip().lower().replace(" ", "")
    return value * 1000.0 if u in ("tb/s", "tbps", "t/s", "tb") else value


def matmul_flops(in_shapes: Sequence[Sequence[int]], out_shapes: Sequence[Sequence[int]]) -> Optional[float]:
    if not in_shapes or not out_shapes:
        return None
    out, a = out_shapes[0], in_shapes[0]
    if len(out) < 2 or len(a) < 1:
        return None
    m, n, k = out[-2], out[-1], a[-1]
    batch = prod(out[:-2]) if len(out) > 2 else 1
    if min(m, n, k) <= 0:
        return None
    return 2.0 * m * n * k * batch


def _weight_n(weight: Sequence[int], k: int) -> Optional[int]:
    elems = numel(weight)
    if k <= 0 or elems <= 0 or elems % k != 0:
        return None
    e = weight[0] if len(weight) >= 3 else 1
    if e <= 0 or elems % (e * k) != 0:
        return None
    return elems // (e * k)


def _expert_weight(weight: Sequence[int], eb: float) -> Tuple[int, float, float]:
    e = weight[0] if len(weight) >= 3 and weight[0] > 0 else 1
    total = numel(weight) * (eb or 0)
    per = total / e if e else total
    return e, per, total


def _io_except(
    xs: Sequence[Sequence[int]],
    in_sizes: Sequence[float],
    os_: Sequence[Sequence[int]],
    out_sizes: Sequence[float],
    skip: int,
) -> float:
    total = 0.0
    for i, s in enumerate(xs):
        if i == skip:
            continue
        eb = in_sizes[i] if i < len(in_sizes) else UNK_DTYPE_BYTES
        total += numel(s) * (eb or 0)
    for i, s in enumerate(os_):
        eb = out_sizes[i] if i < len(out_sizes) else UNK_DTYPE_BYTES
        total += numel(s) * (eb or 0)
    return total


def _finalize_grouped(
    flops: float,
    others: float,
    per: float,
    e: int,
    m: int,
    rate: float,
    tflops: Optional[float],
    gbps: Optional[float],
    note: str,
) -> Tuple[float, float, str, float]:
    """Grouped MoE：group_list 不在 CSV。满 buffer FLOPs 为上界时标 |ceiling。"""
    n_hi = min(e, m) if e and m else (e or 1)
    n_lo = 1 if m > 0 and per > 0 else 0
    bytes_hi = others + per * n_hi
    bytes_lo = others + per * n_lo
    ceiling = False
    nbytes = bytes_hi
    if tflops and gbps and flops:
        t_c = flops / (tflops * 1e6 * (rate or 1.0))
        t_m_hi = bytes_hi / (gbps * 1e3)
        if t_c > t_m_hi:
            ceiling = True
        else:
            nbytes = bytes_lo
    if ceiling:
        note = note + "|ceiling"
    return flops, nbytes, note, rate


def _model_compressor(
    xs: Sequence[Sequence[int]],
    os_: Sequence[Sequence[int]],
    in_sizes: Sequence[float],
    out_sizes: Sequence[float],
) -> Optional[Tuple[float, float, str, float]]:
    """x[M,K] @ wkv/wgate；state cache 只计本步 token 触达行。"""
    if len(xs) < 3 or len(xs[0]) < 2 or len(xs[1]) < 1:
        return None
    x, wkv, wgate = xs[0], xs[1], xs[2]
    m = prod(x[:-1]) if len(x) >= 2 else x[0]
    k = x[-1]
    n1 = wkv[0] if len(wkv) >= 2 and wkv[-1] == k else (wkv[-1] if wkv else 0)
    n2 = 0
    if wgate and len(wgate) >= 1 and numel(wgate) > 1:
        n2 = wgate[0] if len(wgate) >= 2 and wgate[-1] == k else wgate[-1]
    if min(m, k, n1) <= 0:
        return None
    flops = 2 * m * k * n1 + (2 * m * k * n2 if n2 > 0 else 0)
    touched = 0.0
    if len(xs) > 3 and xs[3] and xs[3][-1] > 0:
        state_dim = xs[3][-1]
        eb = in_sizes[3] if len(in_sizes) > 3 and in_sizes[3] else 4
        touched = 2 * m * state_dim * eb
    nbytes = touched
    for i, s in enumerate(xs):
        if i == 3:
            continue
        eb = in_sizes[i] if i < len(in_sizes) else UNK_DTYPE_BYTES
        nbytes += numel(s) * (eb or 0)
    if os_:
        eb = out_sizes[0] if out_sizes and out_sizes[0] else 2
        nbytes += numel(os_[0]) * eb
    return float(flops), nbytes, "Compressor-GEMM", 1.0


def _model_sparse_attn(
    xs: Sequence[Sequence[int]],
    os_: Sequence[Sequence[int]],
    in_sizes: Sequence[float],
    out_sizes: Sequence[float],
    in_dts: str,
) -> Tuple[Optional[float], Optional[float], str, float]:
    """有 indices：FLOPs=4*T*H*topk*D；无 indices：窗口未知，不建模。"""
    if not xs or len(xs[0]) < 3:
        return None, None, "SPARSE_ATTN_BAD_SHAPE", 1.0
    q = xs[0]
    tlen, heads, dim = q[0], q[-2], q[-1]
    names = [_norm_dtype(p) for p in (in_dts or "").split(";")]
    topk_sum = 0
    kv_elem = 2.0
    n_kv = 0
    index_bytes = 0.0
    for i, s in enumerate(xs[1:], start=1):
        name = names[i] if i < len(names) else ""
        eb = in_sizes[i] if i < len(in_sizes) else 0
        if name in ("", "UNDEFINED", "N/A", "NONE") or not eb:
            continue
        is_int = name.startswith("INT") or name.startswith("UINT")
        if is_int and len(s) >= 2 and s[0] == tlen:
            topk_sum += s[-1]
            index_bytes += numel(s) * eb
        elif (not is_int) and len(s) >= 3 and s[-1] == dim and numel(s) > numel(q):
            n_kv += 1
            kv_elem = eb
    if topk_sum <= 0 or heads <= 0 or dim <= 0 or tlen <= 0:
        return None, None, "SPARSE_ATTN_WINDOW_UNKNOWN", 1.0
    flops = 4 * tlen * heads * topk_sum * dim
    q_bytes = numel(q) * (in_sizes[0] if in_sizes and in_sizes[0] else 2)
    out_bytes = 0.0
    if os_ and numel(os_[0]) > 1:
        out_bytes = numel(os_[0]) * (out_sizes[0] if out_sizes and out_sizes[0] else 2)
    kv_bytes = tlen * topk_sum * dim * kv_elem
    note = "稀疏注意力(另有无索引KV,未计入)" if n_kv > 1 else "稀疏注意力"
    return float(flops), q_bytes + out_bytes + index_bytes + kv_bytes, note, 1.0


def model_op(
    rtype: str,
    in_shapes: Sequence[Sequence[int]],
    out_shapes: Sequence[Sequence[int]],
    in_dts: str,
    out_dts: str,
    int8_scale: float,
    tflops: Optional[float] = None,
    gbps: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], str, float]:
    """返回 (flops, bytes, note, rate)。note 以 |ceiling 结尾表示 FLOPs 为 buffer 上界。"""
    t = canon_type(rtype)
    if not in_shapes or not out_shapes:
        return None, None, "BAD_SHAPE", 1.0

    xs = list(in_shapes)
    os_ = list(out_shapes)
    o, x = os_[0], xs[0]
    in_sizes, out_sizes = resolve_itemsize(xs, in_dts, os_, out_dts)
    io = sum_bytes(xs, in_sizes) + sum_bytes(os_, out_sizes)
    rate = lowprec_rate(in_dts, int8_scale)

    if t in MATMUL_TYPES:
        f = matmul_flops(xs, os_)
        return (f, io, "MatMul", 1.0) if f is not None else (None, None, "BAD_SHAPE", 1.0)

    if t.startswith("QUANTBATCHMATMUL") or t.startswith("QUANTMATMUL"):
        if len(o) < 2 or len(x) < 1:
            return None, None, "BAD_SHAPE", 1.0
        m, n, k = prod(o[:-1]), o[-1], x[-1]
        if min(m, n, k) <= 0:
            return None, None, "BAD_SHAPE", 1.0
        return 2.0 * m * n * k, io, "W8A8-GEMM", rate

    if t.startswith("GROUPEDMATMULSWIGLUQUANT") or t.startswith("SWIGLUQUANT"):
        m = prod(o[:-1]) if len(o) >= 2 else None
        if m is None or len(x) < 1 or len(xs) < 2:
            return None, None, "BAD_SHAPE", 1.0
        k0, nout = x[-1], o[-1]
        n_gemm = _weight_n(xs[1], k0) or (2 * nout)
        if min(m, k0, n_gemm) <= 0:
            return None, None, "BAD_SHAPE", 1.0
        e, per, _ = _expert_weight(xs[1], in_sizes[1] if len(in_sizes) > 1 else 1)
        others = _io_except(xs, in_sizes, os_, out_sizes, 1)
        return _finalize_grouped(2.0 * m * k0 * n_gemm, others, per, e, m, rate, tflops, gbps, "MoE-SwiGlu")

    if t.startswith("GROUPEDMATMUL") or t.startswith("GROUPEDGEMM"):
        m = prod(o[:-1]) if len(o) >= 2 else None
        if m is None or len(x) < 1 or len(xs) < 2:
            return None, None, "BAD_SHAPE", 1.0
        k, n = x[-1], o[-1]
        if min(m, k, n) <= 0:
            return None, None, "BAD_SHAPE", 1.0
        e, per, _ = _expert_weight(xs[1], in_sizes[1] if len(in_sizes) > 1 else 1)
        others = _io_except(xs, in_sizes, os_, out_sizes, 1)
        return _finalize_grouped(2.0 * m * n * k, others, per, e, m, rate, tflops, gbps, "MoE-GEMM")

    if t.startswith("TRANSPOSEBATCHMATMUL") or t.startswith("BATCHMATMULT"):
        if len(o) < 2 or len(x) < 1:
            return None, None, "BAD_SHAPE", 1.0
        m, n, k = prod(o[:-1]), o[-1], x[-1]
        if min(m, n, k) <= 0:
            return None, None, "BAD_SHAPE", 1.0
        return 2.0 * m * n * k, io, "BatchedLinear", 1.0

    if t.startswith("SCATTERNDUPDATEV2") or t.startswith("SCATTERUPDATE"):
        if len(xs) < 3:
            return None, None, "BAD_SHAPE", 1.0
        eb = in_sizes[2] if len(in_sizes) > 2 and in_sizes[2] else 4
        upd = numel(xs[2]) * eb
        idx = numel(xs[1]) * (in_sizes[1] if len(in_sizes) > 1 and in_sizes[1] else 4)
        return 0.0, 2 * upd + idx, "KV写入", 1.0

    if "METADATA" in t:
        return None, None, "METADATA_SKIP", 1.0

    if t.startswith("SPARSEATTNSHAREDKV") or t.startswith("SPARSEATTN"):
        return _model_sparse_attn(xs, os_, in_sizes, out_sizes, in_dts)

    if t.startswith("COMPRESSOR"):
        cm = _model_compressor(xs, os_, in_sizes, out_sizes)
        return cm if cm is not None else (None, None, "BAD_SHAPE", 1.0)

    if t.startswith("VLLMQUANTLIGHTNINGINDEXER") or t.startswith("QUANTLIGHTNINGINDEXER"):
        return None, None, "INDEXER_SEQLEN_UNKNOWN", 1.0

    if t.startswith("HCPRE") and "INVRMS" not in t and "SINKHORN" not in t:
        if len(xs) >= 2 and len(xs[1]) >= 2 and x:
            m, n, k = x[0], xs[1][0], xs[1][-1]
            if min(m, n, k) > 0:
                return 2.0 * m * n * k, io, "HcPre", 1.0
        return 0.0, io, "HcPre", 1.0

    if t.startswith("HCPOST"):
        return 2.0 * numel(o), io, "HcPost", 1.0

    memcpy_prefix = (
        "DYNAMICQUANT", "RMSNORMDYNAMICQUANT", "DEQUANTSWIGLUQUANT", "TRITONRMSKERNEL",
        "INPLACEPARTIALROTARYMUL", "MOETOKENPERMUTE", "MOETOKENUNPERMUTE", "MOEGATINGTOPKHASH",
        "MOEINITROUTING", "INDEXCHECK", "SCATTERELEMENTSV2", "ONESHOT", "HCPRE",
    )
    if any(t.startswith(p) for p in memcpy_prefix):
        return 0.0, io, "搬运", 1.0

    return None, None, "UNMODELED_TYPE", 1.0


def theory_us(flops: float, nbytes: float, tflops: float, gbps: float, rate: float) -> Tuple[float, float, float]:
    t_c = flops / (tflops * 1e6 * rate) if flops else 0.0
    t_m = nbytes / (gbps * 1e3) if nbytes else 0.0
    return max(t_c, t_m), t_c, t_m


def find_kernel_details(input_path: str) -> Optional[str]:
    if not input_path:
        return None
    if os.path.isfile(input_path) and os.path.basename(input_path) == "kernel_details.csv":
        return input_path
    if not os.path.isdir(input_path):
        return None
    preferred = glob.glob(
        os.path.join(input_path, "**", "ASCEND_PROFILER_OUTPUT", "kernel_details.csv"),
        recursive=True,
    )
    files = preferred or glob.glob(os.path.join(input_path, "**", "kernel_details.csv"), recursive=True)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def fmt_shape(shape: Sequence[int]) -> str:
    return "x".join(str(d) for d in shape)


def judge_need_opt(rec: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    if rec["flops"] is None:
        return "无法判断", "未建模", ["算子类型暂无可靠模型，结合透视表 ratio 排查"]
    if rec["over"]:
        return "不据此优化", "模型高估(效率>150%)", [
            "理论明显高于实际，勿按浪费时长排序优化",
        ]
    eff = rec["eff"]
    bound = rec["bound"]
    if eff is not None and 0.8 <= eff <= 1.2:
        return "暂不优先优化kernel", "理论≈实际(±20%)", [
            "Cube 已接近峰值，单 kernel 微优化空间小",
            "总耗时仍高时：图级融合、合批、减调用、算法降算量",
        ]
    if eff is not None and eff >= 0.7:
        return "可选优化", "接近峰值", ["核对合批空间与精度路径是否走满 Cube"]
    tips = []
    if bound == "COMPUTE":
        tips = [
            "检查 M/N/K 是否过小导致 Cube 利用率不足",
            "尝试合批/增大 M、确认高性能 MatMul 库路径",
            "核对量化路径 INT8/FP8 峰值与 --flops/--int8-scale",
        ]
    elif bound == "MEMORY":
        tips = [
            "屋顶线偏访存：检查数据排布/NZ/对齐与重复搬运",
            "小 shape 时优先合批或融合，而非抠单 kernel",
        ]
    else:
        tips = ["结合透视表其它 ratio 继续定位"]
    if rec["dur"] < 10:
        tips.append("单次 <10us 时效率低常为 launch/调度开销，需结合调用次数综合判断")
    label = "需要优化" if (eff is not None and eff < 0.4) else "建议优化"
    level = "明显需优化" if (eff is not None and eff < 0.4) else ("偏低需关注" if eff is not None and eff < 0.7 else "-")
    return label, level, tips


def analyze_csv(
    csv_path: str,
    tflops: float,
    gbps: float,
    int8_scale: float,
    mac_threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = build_index(header)
        if norm_col("aic_mac_ratio") not in idx:
            raise SystemExit(
                "[ERROR] CSV 缺少 aic_mac_ratio 列，无法按算力瓶颈（aic_mac_ratio）过滤"
            )
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                dur = float(get_col(idx, ["Duration(us)", "Duration"], row).strip() or "0")
            except ValueError:
                dur = 0.0
            try:
                mac_v = float(get_col(idx, ["aic_mac_ratio"], row).strip())
            except ValueError:
                continue
            if mac_v < mac_threshold:
                continue

            name = get_col(idx, ["Name", "Op Name"], row)
            rtype = get_col(idx, ["Type"], row).strip()
            core = get_col(idx, ["Accelerator Core", "AcceleratorCore"], row).strip()
            if core.upper() == "AI_CPU":
                continue

            in_shapes = parse_shapes(get_col(idx, ["Input Shapes"], row))
            out_shapes = parse_shapes(get_col(idx, ["Output Shapes"], row))
            in_dts = get_col(idx, ["Input Data Types"], row)
            out_dts = get_col(idx, ["Output Data Types"], row)
            flops, nbytes, note, rate = model_op(
                rtype, in_shapes, out_shapes, in_dts, out_dts, int8_scale,
                tflops=tflops, gbps=gbps,
            )
            ceiling = isinstance(note, str) and note.endswith("|ceiling")
            if ceiling:
                note = note[: -len("|ceiling")]

            rec: Dict[str, Any] = {
                "name": name, "type": rtype, "core": core, "dur": dur,
                "flops": flops, "bytes": nbytes, "note": note,
                "in_s": in_shapes, "out_s": out_shapes,
                "in_dts": in_dts.strip().strip('"'), "out_dts": out_dts.strip().strip('"'),
            }
            if flops is not None and nbytes is not None:
                _, t_c, t_m = theory_us(flops, nbytes, tflops, gbps, rate)
                # FLOP_CEILING：容量 FLOPs 高于满张量搬运时，理论耗时取搬运，不抬高 max
                if ceiling and t_c > t_m:
                    t_th, bound = t_m, "MEMORY"
                else:
                    t_th = max(t_c, t_m)
                    bound = "COMPUTE" if t_c > t_m else ("MEMORY" if t_m > 0 else "OTHER")
                if ceiling:
                    note = note + "|ceiling"
                    rec["note"] = note
                rec.update({
                    "t_c": t_c, "t_m": t_m, "t_th": t_th, "bound": bound,
                    "eff": (t_th / dur if dur > 0 else None),
                    "over": False,
                })
                if rec["eff"] is not None and rec["eff"] > 1.5:
                    rec["over"] = True
            else:
                rec.update({
                    "t_c": 0.0, "t_m": 0.0, "t_th": 0.0,
                    "bound": "-", "eff": None, "over": False,
                })
            rows.append(rec)
    return rows


def print_report(rows: List[Dict[str, Any]], args: argparse.Namespace, csv_path: str, gbps: float) -> None:
    modeled = [r for r in rows if r["flops"] is not None]
    ok = [r for r in modeled if not r["over"]]
    total_dur = sum(r["dur"] for r in rows)
    modeled_dur = sum(r["dur"] for r in ok)
    modeled_th = sum(r["t_th"] for r in ok)

    print("=" * 100)
    print("算力瓶颈算子 Roofline 核对报告 (profiling-analysis-computing)")
    print("  输入: %s" % os.path.abspath(csv_path))
    print("  理论耗时 = max(计算量/算力, 搬运量/带宽)；效率 = 理论/实际")
    print("  FP16/BF16 算力 %.1f TFLOPS | INT8/FP8 Cube %.1f TOPS（×%.2f）" % (
        args.flops, args.flops * args.int8_scale, args.int8_scale))
    print("  内存带宽 %.3f TB/s（%.1f GB/s）" % (gbps / 1000.0, gbps))
    print("  模式: 仅分析算力瓶颈算子（aic_mac_ratio>=%.0f%%）" % (args.mac_threshold * 100))
    print("=" * 100)

    overall = modeled_th / modeled_dur * 100 if modeled_dur > 0 else 0
    print("\n[总体概况]")
    print("  算子总数: %d | 可建模: %d | 未建模: %d" % (
        len(rows), len(modeled), len(rows) - len(modeled)))
    print("  实际总耗时: %.1f us | 可建模实际: %.1f us | 理论合计: %.1f us" % (
        total_dur, modeled_dur, modeled_th))
    print("  整体效率(理论/实际): %.1f%%  →  %s" % (
        overall,
        "整体接近理想" if overall >= 70 else (
            "整体存在明显优化空间" if overall >= 40 else "整体效率低，存在系统性瓶颈")))

    close_n = sum(1 for r in modeled if r["eff"] is not None and 0.8 <= r["eff"] <= 1.2)
    slow_n = sum(1 for r in modeled if r["eff"] is not None and r["eff"] < 0.8 and not r["over"])
    print("  理论≈实际(±20%%): %d | 实际偏慢(效率<80%%): %d" % (close_n, slow_n))

    agg: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        a = agg.setdefault(r["type"], {"n": 0, "nmod": 0, "dur": 0.0, "th": 0.0, "waste": 0.0, "tc": 0.0, "tm": 0.0})
        a["n"] += 1
        a["dur"] += r["dur"]
        if r["flops"] is not None:
            a["nmod"] += 1
            a["th"] += r["t_th"]
            a["tc"] += r["t_c"]
            a["tm"] += r["t_m"]
            if not r["over"] and r["dur"] > r["t_th"]:
                a["waste"] += r["dur"] - r["t_th"]

    print("\n[按算子类型汇总] (按实际耗时降序)")
    print("%-24s %6s %6s %12s %12s %8s %10s  %s" % (
        "Type", "总数", "已建模", "实际(us)", "理论(us)", "效率%", "浪费(us)", "主导"))
    for t, a in sorted(agg.items(), key=lambda kv: kv[1]["dur"], reverse=True):
        eff = a["th"] / a["dur"] * 100 if a["dur"] and a["nmod"] else float("nan")
        bound = "COMPUTE" if a["tc"] > a["tm"] else "MEMORY"
        print("%-24s %6d %6d %12.1f %12.1f %7.1f%% %10.1f  %s" % (
            t[:24], a["n"], a["nmod"], a["dur"], a["th"], eff, a["waste"],
            bound if a["nmod"] else "-"))

    cands = [r for r in modeled if r["t_th"] > 0 and not r["over"]]
    cands.sort(key=lambda r: r["dur"] - r["t_th"], reverse=True)

    print("\n[Top %d 算力瓶颈算子核对] (按浪费时长降序)" % args.top)
    print("%-4s %-36s %-18s %-32s %9s %9s %9s %7s %-8s %s" % (
        "#", "Name", "Type", "Shape", "理论(us)", "实际(us)", "效率%", "主导", "是否需优化", "note"))
    for i, r in enumerate(cands[: args.top], 1):
        si = fmt_shape(r["in_s"][0]) if r["in_s"] else "-"
        so = fmt_shape(r["out_s"][0]) if r["out_s"] else "-"
        shape = (si + " -> " + so)[:32]
        name = r["name"][:36]
        opt, _, _ = judge_need_opt(r)
        print("%-4d %-36s %-18s %-32s %9.2f %9.2f %6.1f%% %-8s %s [%s]" % (
            i, name, r["type"][:18], shape, r["t_th"], r["dur"],
            (r["eff"] * 100 if r["eff"] is not None else float("nan")),
            r["bound"], opt, r["note"]))

    nbn = sum(1 for r in cands if r["eff"] is not None and r["eff"] < 0.4)
    bn_waste = sum(r["dur"] - r["t_th"] for r in cands if r["eff"] is not None and r["eff"] < 0.4)
    print("\n[是否需要优化 — 判断摘要]")
    print("  已建模: %d | 理论≈实际: %d | 偏慢: %d | 明显需优化: %d | 浪费合计: %.1f us (%.1f%%)" % (
        len(modeled), close_n, slow_n, nbn, bn_waste,
        bn_waste / total_dur * 100 if total_dur else 0))

    print("\n[初步优化/排查方向]")
    shown = 0
    for r in cands[:12]:
        opt, level, tips = judge_need_opt(r)
        if opt == "暂不优先优化kernel" and shown >= 2:
            continue
        print("  - %s | %s | %s | 效率=%s | %s" % (
            r["name"][:48], r["type"], opt,
            ("%.1f%%" % (r["eff"] * 100) if r["eff"] is not None else "-"), level))
        for tip in tips[:2]:
            print("      · %s" % tip)
        shown += 1
        if shown >= 8:
            break

    print("\n[结论]")
    if not modeled:
        print("  未找到可建模的算力瓶颈算子。")
    elif nbn == 0 and close_n >= max(1, len(modeled) // 2):
        print("  多数算力瓶颈算子理论≈实际：Cube 已打满，优先图级/算法级优化。")
    elif nbn == 0:
        print("  未发现效率<40%% 的算力瓶颈算子；对偏低条目按上方建议排查。")
    else:
        top_types = sorted(agg.items(), key=lambda kv: kv[1]["waste"], reverse=True)[:3]
        print("  存在明显需优化算子；优先级(按浪费): %s" % ", ".join(
            "%s(%.1fus)" % (t, a["waste"]) for t, a in top_types if a["waste"] > 0))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Name", "Type", "Core", "Duration(us)", "FLOPs", "Bytes",
            "t_compute(us)", "t_memory(us)", "t_theory(us)",
            "Efficiency(理论/实际)", "Bound", "是否需优化", "评估", "Flags",
            "Input Shapes", "Input Data Types", "Output Shapes", "Output Data Types",
        ])
        for r in rows:
            opt, level, _ = judge_need_opt(r)
            w.writerow([
                r["name"], r["type"], r["core"], "%.3f" % r["dur"],
                "" if r["flops"] is None else "%.0f" % r["flops"],
                "" if r["bytes"] is None else "%.0f" % r["bytes"],
                "%.3f" % r["t_c"], "%.3f" % r["t_m"], "%.3f" % r["t_th"],
                "" if r["eff"] is None else "%.4f" % r["eff"],
                r["bound"], opt, level, r.get("note", ""),
                ";".join(fmt_shape(s) for s in r["in_s"]),
                r.get("in_dts", ""),
                ";".join(fmt_shape(s) for s in r["out_s"]),
                r.get("out_dts", ""),
            ])
    print("\n明细已写入: %s" % os.path.abspath(path))


def self_test() -> None:
    tflops, gbps = 313.0, bandwidth_gbps(1.2, "TB/s")
    assert abs(gbps - 1200.0) < 1e-9
    cases = [
        ("MatMulV3", "4096,4096;4096,4096", "4096,4096", "DT_BF16;DT_BF16", "DT_BF16",
         2 * 4096 ** 3, "MatMul"),
        ("QuantBatchMatmulV3", "1024,4096;128,256,16,32;4096", "1024,4096", "INT8;INT8;DT_BF16", "DT_BF16",
         2 * 1024 * 4096 * 4096, "W8A8-GEMM"),
        ("HcPre", "1024,4,4096;24,16384;3;24", "1024,4096;1024,4;1024,4,4",
         "DT_BF16;FLOAT;FLOAT;FLOAT", "DT_BF16;FLOAT;FLOAT", 2 * 1024 * 24 * 16384, "HcPre"),
    ]
    for rtype, ins, outs, idt, odt, exp_f, note in cases:
        f, _, n, _ = model_op(rtype, parse_shapes(ins), parse_shapes(outs), idt, odt, 2.0)
        assert f == exp_f and n == note, (rtype, f, n)
    c = model_op(
        "Compressor",
        parse_shapes("8192,4096;256,4096;256,4096;4466,8,512;4,256;128;2049,64;2049,64;1,16640;2;;1"),
        parse_shapes("2049,128;4466,8,512"),
        "DT_BF16;DT_BF16;DT_BF16;FLOAT;FLOAT;DT_BF16;FLOAT;FLOAT;INT32;INT32;DT_UNDEFINED;INT32",
        "DT_BF16;FLOAT", 2.0)
    assert c[0] == 2 * 8192 * 4096 * 256 * 2 and c[1] < 4466 * 8 * 512 * 4 * 2
    s = model_op(
        "SparseAttnSharedkv",
        parse_shapes("1024,64,512;4466,128,1,512;4466,128,1,512;;1024,1,512;1,1040;1,260;2;2;;;1;64;1024"),
        parse_shapes("1024,64,512;"),
        "DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;INT32;INT32;INT32;INT32;INT32;DT_UNDEFINED;DT_UNDEFINED;INT32;FLOAT;INT32",
        "DT_BF16;FLOAT", 2.0)
    assert s[0] == 4 * 1024 * 64 * 512 * 512
    g = model_op(
        "GroupedMatmulSwigluQuant",
        parse_shapes("49152,4096;32,128,256,16,32;32,4096;49152;32"),
        parse_shapes("49152,2048;49152"),
        "INT8;INT8;FLOAT;FLOAT;INT64", "INT8;FLOAT", 2.0, tflops=tflops, gbps=gbps)
    assert str(g[2]).endswith("|ceiling")
    print("self-test ok")


def main() -> None:
    ap = argparse.ArgumentParser(description="kernel_details.csv 算力瓶颈 Roofline 核对")
    ap.add_argument("--csv", default=None, help="kernel_details.csv 路径")
    ap.add_argument("--input-path", default=None, help="profiling 根目录，自动查找 kernel_details.csv")
    ap.add_argument("--flops", type=float, default=None, help="FP16/BF16 Cube 峰值 TFLOPS")
    ap.add_argument("--bandwidth", type=float, default=None, help="内存带宽数值")
    ap.add_argument("--bw-unit", default="GB/s", help="GB/s 或 TB/s")
    ap.add_argument("--int8-scale", type=float, default=2.0, help="INT8/FP8 相对 --flops 倍率，默认 2")
    ap.add_argument("--mac-threshold", type=float, default=0.80, help="aic_mac_ratio 阈值，默认 0.80")
    ap.add_argument("--top", type=int, default=30, help="Top 明细条数")
    ap.add_argument("--out", default=None, help="明细 CSV 路径")
    ap.add_argument("--output-path", default=None, help="输出目录")
    ap.add_argument("--self-test", action="store_true", help="内置 shape 自检")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.csv is None and args.input_path:
        args.csv = find_kernel_details(args.input_path)
        if args.csv is None:
            raise SystemExit("[ERROR] 未找到 kernel_details.csv: %s" % args.input_path)
        print("[INFO] 使用 kernel_details.csv: %s" % os.path.abspath(args.csv))

    if args.csv is None or args.flops is None or args.bandwidth is None:
        raise SystemExit("[ERROR] 需要 (--csv 或 --input-path)、--flops、--bandwidth")
    if args.flops <= 0 or args.bandwidth <= 0 or args.int8_scale <= 0:
        raise SystemExit("[ERROR] --flops、--bandwidth、--int8-scale 必须 > 0")
    if not os.path.isfile(args.csv):
        raise SystemExit("[ERROR] 找不到文件: %s" % args.csv)

    if args.out is None and args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        args.out = os.path.join(args.output_path, "kernel_roofline_compute_check.csv")

    gbps = bandwidth_gbps(args.bandwidth, args.bw_unit)
    if gbps < 50:
        print("[WARN] 带宽按 %.3f GB/s 解释。若为 1.2 TB/s，请用 --bandwidth 1.2 --bw-unit TB/s"
              % gbps, file=sys.stderr)

    rows = analyze_csv(args.csv, args.flops, gbps, args.int8_scale, args.mac_threshold)
    print_report(rows, args, args.csv, gbps)
    if args.out:
        write_csv(args.out, rows)


if __name__ == "__main__":
    main()
