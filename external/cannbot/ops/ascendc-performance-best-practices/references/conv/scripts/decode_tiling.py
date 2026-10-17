#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Decode Conv2DV2 bin_tiling_data (int32 stream) into named fields.

Layout matches ops-nn Conv2DTilingData in:
  conv/conv2d_v2/op_host/op_tiling/conv2d_v2_tiling.h

uint64 fields are little-endian and appear as (lo, hi) int32 pairs in TTK dumps.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import struct
from pathlib import Path

# (name, ctype) in declaration order
FIELDS = [
    ("orgHi", "u64"),
    ("orgWi", "u64"),
    ("orgHo", "u64"),
    ("orgWo", "u64"),
    ("orgHixWi", "u64"),
    ("singleCoreBatch", "u64"),
    ("singleCoreHo", "u64"),
    ("singleCoreWo", "u64"),
    ("unionDataXt", "u64"),
    ("hin", "u64"),
    ("win", "u64"),
    ("hout", "u64"),
    ("wout", "u64"),
    ("orgCi", "u32"),
    ("orgCo", "u32"),
    ("singleCoreCi", "u32"),
    ("singleCoreCo", "u32"),
    ("hoL1", "u32"),
    ("woL1", "u32"),
    ("kAL1", "u32"),
    ("kBL1", "u32"),
    ("khL1", "u32"),
    ("kwL1", "u32"),
    ("nBL1", "u32"),
    ("hoL0", "u32"),
    ("woL0", "u32"),
    ("kL0", "u32"),
    ("nL0", "u32"),
    ("pBufferFlag", "u32"),
    ("groups", "u32"),
    ("enlarge", "u32"),
    ("singleCoreGroups", "u32"),
    ("singleCoreGroupOpt", "u32"),
    ("bUbNStep", "u32"),
    ("bUbKStep", "u32"),
    ("khUb", "u32"),
    ("kwUb", "u32"),
    ("kernelHxkernelW", "u32"),
    ("kernelHxkernelWxkernelD", "u32"),
    ("aL1SpaceSize", "u32"),
    ("multiNBL1", "u32"),
    ("cinAInCore", "u32"),
    ("cinATailInCore", "u32"),
    ("cinBInCore", "u32"),
    ("cinBTailInCore", "u32"),
    ("mStep", "u32"),
    ("kStep", "u32"),
    ("nStep", "u32"),
    ("fmapKStride", "u32"),
    ("weightKStride", "u32"),
    ("cinOffsetBlockInGM", "u32"),
    ("coutOffsetBlock", "u32"),
    ("nL1DivBlockSize", "u32"),
    ("kernelH", "u32"),
    ("kernelW", "u32"),
    ("strideH", "u32"),
    ("strideW", "u32"),
    ("dilationH", "u32"),
    ("dilationW", "u32"),
    ("padTop", "u32"),
    ("padBottom", "u32"),
    ("padLeft", "u32"),
    ("padRight", "u32"),
    ("innerBatch", "u32"),
    ("batch", "u32"),
    ("cin", "u32"),
    ("cout", "u32"),
    ("kh", "u32"),
    ("kw", "u32"),
    ("batchDim", "u32"),
    ("groupDim", "u32"),
    ("nDim", "u32"),
    ("hoDim", "u32"),
    ("woDim", "u32"),
    ("cinOpt", "u32"),
    ("coutOpt", "u32"),
    ("groupOpt", "u32"),
    ("iterateMNOrder", "u8"),
    ("biasFullLoadFlag", "u8"),
    ("fixpParamsFullLoadFlag", "u8"),
    ("hf32Enable", "u8"),
    ("hf32TransMode", "u8"),
    ("hasBias", "u8"),
    ("hasScale", "u8"),
    ("dualOutput", "u8"),
    ("quantMode0", "u8"),
    ("reluMode0", "u8"),
    ("clipMode0", "u8"),
    ("quantMode1", "u8"),
    ("reluMode1", "u8"),
    ("clipMode1", "u8"),
    ("fixedShiftValue", "u8"),
    ("offsetx", "i8"),
    ("roundMode", "i8"),
]

FIELD_HELP = {
    "orgHi/orgWi/orgHo/orgWo": "原始输入/输出 H/W",
    "orgHixWi": "Hin*Win",
    "orgCi/orgCo": "原始 Cin/Cout",
    "singleCoreBatch/Ho/Wo/Ci/Co/Groups": "单核切分量",
    "hoL1/woL1/kAL1/kBL1/nBL1": "L1 tile（feature/weight/K/N）",
    "hoL0/woL0/kL0/nL0": "L0 tile",
    "pBufferFlag": "L1/L0 ping-pong 开关位图",
    "aL1SpaceSize": "AL1 占用空间",
    "kernelH/W,strideH/W,dilationH/W,pad*": "卷积属性",
    "batchDim/groupDim/nDim/hoDim/woDim": "多核切分维数（block_dim 相关）",
    "hasBias/hf32Enable": "bias / HF32 使能",
    "tiling_key": "模板组合键（Fmap/Weight fullload、L1/L0 pingpong、HW/M order 等）",
}

# Conv3DBackpropInputArch35TilingData (conv3d_backprop_input_v2, arch35).
# Layout: TILING_DATA_FIELD_DEF packs fields tightly (data_size_ += sizeof), no padding.
CONV3DDX_FIELDS = [
    ("batchDim", "u32"),
    ("groupDim", "u32"),
    ("mDim", "u32"),
    ("kDim", "u32"),
    ("nDim", "u32"),
    ("dDim", "u32"),
    ("coreNum", "u64"),
    ("al0Pbuffer", "u8"),
    ("bl0Pbuffer", "u8"),
    ("cl0Pbuffer", "u8"),
    ("al1Pbuffer", "u8"),
    ("bl1Pbuffer", "u8"),
    ("iterateOrder", "u8"),
    ("c0", "u8"),
    ("c0BitsA", "u8"),
    ("c0BitsB", "u8"),
    ("enlarge", "u8"),
    ("hf32Flag", "u8"),
    ("initOutputFlag", "u8"),
    ("isBiasFullLoad", "u8"),
    ("enableVecTrans", "u8"),
    ("enableFullLoad", "u8"),
    ("quantMode", "u8"),
    ("batch", "u32"),
    ("cin", "u32"),
    ("cout", "u32"),
    ("cinG", "u32"),
    ("coutG", "u32"),
    ("cout1", "u32"),
    ("cin1", "u32"),
    ("cout1G", "u32"),
    ("cin1G", "u32"),
    ("dout", "u32"),
    ("ho", "u32"),
    ("wo", "u32"),
    ("di", "u32"),
    ("hi", "u32"),
    ("wi", "u32"),
    ("dk", "u32"),
    ("hk", "u32"),
    ("wk", "u32"),
    ("group", "u32"),
    ("oriGroup", "u32"),
    ("strideD", "u32"),
    ("strideH", "u32"),
    ("strideW", "u32"),
    ("padFront", "u32"),
    ("padBack", "u32"),
    ("padUp", "u32"),
    ("padDown", "u32"),
    ("padLeft", "u32"),
    ("padRight", "u32"),
    ("backpropPadTail", "i32"),
    ("backpropPadUp", "i32"),
    ("backpropPadDown", "i32"),
    ("backpropPadLeft", "i32"),
    ("backpropPadRight", "i32"),
    ("dilationD", "u32"),
    ("dilationH", "u32"),
    ("dilationW", "u32"),
    ("singleCoreGroup", "u32"),
    ("singleCoreCout", "u32"),
    ("singleCoreCin", "u32"),
    ("singleCoreDin", "u32"),
    ("baseM", "u32"),
    ("baseK", "u32"),
    ("baseN", "u32"),
    ("stepKa", "u32"),
    ("stepKb", "u32"),
    ("singleIterateDk", "u32"),
    ("singleCoreBatch", "u64"),
    ("singleCoreM", "u64"),
    ("enRelu", "u64"),
    ("kSegment", "u64"),
    ("kSegmentTail", "u64"),
    ("kValueSegment", "u64"),
    ("enableSplitK", "u8"),
    ("useUbAccumForSplitK", "u8"),
    ("offsetX", "i8"),
    ("kSCoutFullLoad", "u32"),
    ("kSUseWorkSpace", "u32"),
    ("khDilation", "u32"),
    ("kwDilation", "u32"),
    ("hoExpand", "u32"),
    ("woExpand", "u32"),
    ("dkHkWk", "u64"),
    ("hkWk", "u64"),
    ("fixedShiftVal", "u8"),
    ("reserved", "u8x8"),
]

# conv3d_backprop_input_v2 arch35 tiling_key: 5 template args, UINT LIST params are
# encoded by list index (index == value here), fields shifted by declared bit width.
CONV3DDX_TILING_KEY_FIELDS = [
    ("loadB2Condition", 8),  # 0=NO_TRANS_NO_REV,1=TRANS_AND_REV,2=REV_ONLY,3=TRANS_ONLY
    ("kernelSplitMode", 8),  # 0=NO_SPLIT,1=SPLIT_HW,2=SPLIT_H
    ("groupConvMode", 8),  # 0=ORIGIN,1=ENLARGE
    ("isBasicBlockTiling", 1),
    ("loadB1Condition", 8),  # 0=GM_TO_L1,1=VEC_TO_L1_C04,2=NO_HK,3=NO_HK_WK,4=SMALL_KERNEL
]


def decode_conv3ddx_tiling_key(key):
    key = int(key)
    out = {}
    off = 0
    for name, bw in CONV3DDX_TILING_KEY_FIELDS:
        out[name] = (key >> off) & ((1 << bw) - 1)
        off += bw
    return out


def _to_int32_list(raw: str):
    if not raw:
        return []
    nums = re.findall(r"np\.int32\((-?\d+)\)", raw)
    if nums:
        return [int(x) for x in nums]
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
    except Exception:
        pass
    return [int(x) for x in re.findall(r"-?\d+", raw)]


def decode_i32_stream(vals, fields=None):
    """Parse int32 stream into field dict using little-endian packing."""
    raw = b"".join(struct.pack("<i", v) for v in vals)
    off = 0
    out = {}
    for name, ctype in fields or FIELDS:
        if ctype == "u64":
            need = 8
            if off + need > len(raw):
                break
            out[name] = struct.unpack_from("<Q", raw, off)[0]
        elif ctype == "u32":
            need = 4
            if off + need > len(raw):
                break
            out[name] = struct.unpack_from("<I", raw, off)[0]
        elif ctype == "u8":
            need = 1
            if off + need > len(raw):
                break
            out[name] = struct.unpack_from("<B", raw, off)[0]
        elif ctype == "i32":
            need = 4
            if off + need > len(raw):
                break
            out[name] = struct.unpack_from("<i", raw, off)[0]
        elif ctype == "i8":
            need = 1
            if off + need > len(raw):
                break
            out[name] = struct.unpack_from("<b", raw, off)[0]
        elif ctype == "u8x8":
            need = 8
            if off + need > len(raw):
                break
            out[name] = list(struct.unpack_from("<8B", raw, off))
        else:
            raise ValueError(ctype)
        off += need
    out["_bytes_consumed"] = off
    out["_bytes_total"] = len(raw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--schema",
        default="conv2d",
        choices=["conv2d", "conv3ddx"],
        help="conv2d: Conv2DTilingData; conv3ddx: Conv3DBackpropInputArch35TilingData",
    )
    args = ap.parse_args()

    fields = CONV3DDX_FIELDS if args.schema == "conv3ddx" else FIELDS
    rows = list(csv.DictReader(open(args.results, encoding="utf-8")))
    decoded = []
    for r in rows:
        name = r.get("testcase_name", "")
        raw = r.get("bin_tiling_data") or r.get("dyn_tiling_data") or ""
        key = r.get("bin_tiling_key") or r.get("dyn_tiling_key") or ""
        fields_dec = decode_i32_stream(_to_int32_list(raw), fields)
        entry = {
            "testcase_name": name,
            "tiling_key": key,
            "perf_us": r.get("bin_perf_us") or r.get("dyn_perf_us"),
            "block_dim": r.get("bin_block_dim") or r.get("dyn_block_dim"),
            "tiling": fields_dec,
        }
        if args.schema == "conv3ddx" and key:
            try:
                entry["tiling_key_decoded"] = decode_conv3ddx_tiling_key(key)
            except Exception:
                pass
        decoded.append(entry)

    out = Path(args.out)
    out.write_text(json.dumps(decoded, indent=2, ensure_ascii=False), encoding="utf-8")

    # compact csv of key tiling knobs
    csv_path = out.with_suffix(".csv")
    keys = [
        "testcase_name",
        "tiling_key",
        "perf_us",
        "block_dim",
        "batch",
        "cin",
        "cout",
        "hin",
        "win",
        "hout",
        "wout",
        "kh",
        "kw",
        "hoL1",
        "woL1",
        "kAL1",
        "kBL1",
        "nBL1",
        "hoL0",
        "woL0",
        "kL0",
        "nL0",
        "pBufferFlag",
        "batchDim",
        "nDim",
        "hoDim",
        "woDim",
        "hasBias",
        "hf32Enable",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for item in decoded:
            t = item["tiling"]
            row = {k: item.get(k, t.get(k, "")) for k in keys}
            w.writerow(row)
    print(f"Wrote {out}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
