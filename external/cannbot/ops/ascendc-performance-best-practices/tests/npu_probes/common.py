# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import os
import subprocess
import time
from pathlib import Path

P = Path(__file__).resolve().parent
R = Path(os.environ.get("SKILLS_DEV_REPO", P.parents[3])).resolve()
S = R / "ops/ascendc-performance-best-practices"
if not os.environ.get("ASCEND_HOME_PATH"):
    raise SystemExit("Source the target CANN set_env.sh before running probes")
SDK = Path(os.environ["ASCEND_HOME_PATH"])
for directory in ("build", "logs", "src"):
    (P / directory).mkdir(exist_ok=True)
INC = [
    f"-I{SDK}/asc/include/tiling",
    f"-I{SDK}/asc/include",
    f"-I{SDK}/include",
    f"-I{SDK}/include/op_common",
]
LIB = [f"-L{SDK}/lib64", "-ltiling_api", "-lplatform"]
DRV = Path("/usr/local/Ascend/driver/lib64/driver")


def run(name, cmd, timeout=300):
    t = time.time()
    try:
        r = subprocess.run(
            [str(x) for x in cmd],
            cwd=P,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        out = r.stdout
        rc = r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") + b"\nTIMEOUT\n"
        rc = 124
    (P / "logs" / f"{name}.log").write_bytes(out)
    record = {
        "name": name,
        "command": [str(x) for x in cmd],
        "rc": rc,
        "seconds": round(time.time() - t, 3),
        "log": f"logs/{name}.log",
    }
    with (P / "results.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(name, "PASS" if rc == 0 else "FAIL", rc, flush=True)
    if rc:
        print(out.decode(errors="replace")[-5000:], flush=True)
        raise RuntimeError(name)
    return out.decode(errors="replace")


def build(name, source, extra=()):
    return run(
        name + "-build",
        [
            "bisheng",
            "-xasc",
            "--npu-arch=dav-3510",
            "-std=c++17",
            "-O2",
            *INC,
            str(source),
            *extra,
            "-o",
            P / "build" / name,
        ],
    )
