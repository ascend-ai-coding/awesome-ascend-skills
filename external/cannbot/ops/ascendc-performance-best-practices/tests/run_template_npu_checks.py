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

"""Materialize and validate Softmax/Scan templates on an arch35 device (logical ID 0).

Source the target CANN set_env.sh first. The work directory retains binaries and logs.
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work-dir", type=Path)
args = parser.parse_args()
sdk = os.environ.get("ASCEND_HOME_PATH")
if not sdk or not shutil.which("bisheng"):
    parser.error("source the target CANN set_env.sh before running this check")
skill = Path(__file__).resolve().parents[1]
work = args.work_dir or Path(tempfile.mkdtemp(prefix="perf-template-npu-"))
work.mkdir(parents=True, exist_ok=True)
root = work / "reduce"
root.mkdir(exist_ok=True)
scan_root = work / "scan"
scan_root.mkdir(exist_ok=True)
for group, output in (("reduce", root), ("scan", scan_root)):
    for source in (skill / "reference" / group / "templates").rglob("*.template"):
        if group == "reduce" and not (source.name.startswith("softmax") or source.stem == "kernel_utils"):
            continue
        (output / (source.stem + ".h")).write_bytes(source.read_bytes())
print(f"Validation artifacts: {work}", flush=True)
variants = [
    ("ar_small_r", "SoftmaxV2ArSmallR", "SoftmaxV2ArSmallRTilingData", "TilingArSmallR"),
    ("ar_full_load", "SoftmaxV2AR", "SoftmaxV2ARTilingData", "TilingArFullLoad"),
    ("ar_recompute", "SoftmaxV2ArRecompute", "SoftmaxV2ArRecomputeTilingData", "TilingArRecompute"),
    ("ara_full_load", "SoftmaxV2ARA", "SoftmaxV2ARATilingData", "TilingAraFullLoad"),
    ("ara_recompute", "SoftmaxV2ARARecompute", "SoftmaxV2ARARecomputeTilingData", "TilingAraRecompute"),
    ("ara_online", "SoftmaxV2ARAOnline", "SoftmaxV2ARAOnlineTilingData", "TilingAraOnline"),
]
BODY = r"""
#include "kernel_operator.h"
#include "softmax_v2_tiling_data.h"
#ifdef __NPU_DEVICE__
#include "softmax_v2_VARIANT.h"
#endif
__global__ __aicore__ __vector__ void CheckSoftmax(GM_ADDR x, GM_ADDR y, DATA data) {
#ifdef __NPU_DEVICE__
 AscendC::TPipe pipe;
 INIT
 op.Process();
#endif
}
#ifndef __NPU_DEVICE__
#include "softmax_v2_tiling.h"
#include <acl/acl.h>
#include <platform/platform_ascendc.h>
#include <vector>
#include <cmath>
#include <iostream>
#define CHECK(call) do { auto rc=(call); if(rc) {std::cerr<<#call<<" failed "<<rc<<'\n';return 1;} }while(0)
int main() {
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream));
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) return 1;
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    softmax_tiling::PlatformParam p{};
    p.ubSize = ub;
    p.numBlocks = platform->GetCoreNumAiv();
    if (!ub || !p.numBlocks) return 1;
    for (int test = 0; test < 3; ++test) {
        softmax_tiling::CaseShape s{3, RSIZES[test], A0, 0};
        DATA data{};
        auto blocks = softmax_tiling::TILER(s, p, data);
        if (blocks <= 0) {
            std::cerr << "tiling rejected\n";
            return 2;
        }
        size_t count = s.a1 * s.r * s.a0, bytes = count * sizeof(float);
        std::vector<float> x(count), y(count);
        for (size_t i = 0; i < count; ++i) x[i] = (int(i % 13) - 6) * 0.125f;
        void *dx = nullptr, *dy = nullptr;
        CHECK(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK(aclrtMemcpy(dx, bytes, x.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK(aclrtMemset(dy, bytes, 0xa5, bytes));
        CheckSoftmax<<<blocks, nullptr, stream>>>(static_cast<uint8_t*>(dx), static_cast<uint8_t*>(dy), data);
        CHECK(aclrtSynchronizeStreamWithTimeout(stream, 10000));
        CHECK(aclrtMemcpy(y.data(), bytes, dy, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        for (int64_t a1 = 0; a1 < s.a1; ++a1)
        for (int64_t a0 = 0; a0 < s.a0; ++a0) {
            double sum = 0;
            for (int64_t r = 0; r < s.r; ++r) sum += std::exp(double(x[(a1 * s.r + r) * s.a0 + a0]));
            for (int64_t r = 0; r < s.r; ++r) {
                size_t i = (a1 * s.r + r) * s.a0 + a0;
                double ref = std::exp(double(x[i])) / sum;
                if (!std::isfinite(y[i]) || std::abs(y[i] - ref) > 1e-5 + 1e-4 * std::abs(ref)) {
                    std::cerr << "FAIL test=" << test << " index=" << i << " got=" << y[i]
                              << " expected=" << ref << '\n';
                    return 3;
                }
            }
        }
        CHECK(aclrtFree(dx));
        CHECK(aclrtFree(dy));
    }
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    std::cout << "PASS VARIANT 3 FP32 CPU golden cases\n";
}
#endif
"""
for variant, cls, data, tiler in variants:
    init = (
        f"SoftmaxV2Ops::{cls}<float,float> op(&data);op.Init(x,y,&pipe);"
        if variant.startswith("ara")
        else f"SoftmaxV2Ops::{cls}<float,float> op(&pipe);op.Init(x,y,&data);"
    )
    s = BODY.replace("VARIANT", variant).replace("DATA", data).replace("INIT", init).replace("TILER", tiler)
    sizes = (
        "{4,8,16}" if variant == "ar_small_r" else ("{33,65,129}" if variant == "ara_full_load" else "{33,129,1025}")
    )
    s = s.replace("RSIZES[test]", f"std::vector<int64_t>{sizes}[test]").replace(
        "A0", "17" if variant.startswith("ara") else "1"
    )
    (root / (variant + ".asc")).write_text(s)


def compile_variant(v):
    name = v[0]
    r = subprocess.run(
        [
            "bisheng",
            "-xasc",
            "--npu-arch=dav-3510",
            "-std=c++17",
            "-O2",
            "-I" + os.environ["ASCEND_HOME_PATH"] + "/include",
            "-I" + os.environ["ASCEND_HOME_PATH"] + "/include/op_common",
            str(root / (name + ".asc")),
            "-o",
            str(root / name),
            "-L" + os.environ["ASCEND_HOME_PATH"] + "/lib64",
            "-ltiling_api",
            "-lplatform",
            "-lm",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
    )
    (root / (name + "-build.log")).write_bytes(r.stdout)
    return name, r.returncode


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    results = list(ex.map(compile_variant, variants))
failures = []
for name, rc in results:
    print(name, "build", rc, flush=True)
    if rc:
        failures.append(name)
        continue
    result = subprocess.run([str(root / name)], capture_output=True, timeout=90)
    output = result.stdout + result.stderr
    (root / (name + "-run.log")).write_bytes(output)
    print(output.decode(errors="replace"), flush=True)
    if result.returncode:
        failures.append(name)
scan_source = skill / "tests" / "scan_streaming_npu_check.asc"
scan_binary = scan_root / "streaming_check"
command = [
    "bisheng",
    "-xasc",
    "--npu-arch=dav-3510",
    "-std=c++17",
    "-O2",
    "-I" + str(scan_root),
    "-I" + sdk + "/include",
    "-I" + sdk + "/include/op_common",
    str(scan_source),
    "-L" + sdk + "/lib64",
    "-ltiling_api",
    "-lplatform",
    "-o",
    str(scan_binary),
]
result = subprocess.run(command, capture_output=True, timeout=240)
(scan_root / "build.log").write_bytes(result.stdout + result.stderr)
if result.returncode:
    failures.append("streaming_scan_build")
else:
    result = subprocess.run([str(scan_binary)], capture_output=True, timeout=90)
    output = result.stdout + result.stderr
    (scan_root / "run.log").write_bytes(output)
    print(output.decode(errors="replace"), flush=True)
    if result.returncode:
        failures.append("streaming_scan")
if failures:
    raise SystemExit("FAILED: " + ", ".join(failures))
print("PASS 18 Softmax and 4 Scan device cases")
