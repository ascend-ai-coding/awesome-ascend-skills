# Communication 算子 — 测试脚本模板

> **阅读时机**：渐进式生成步骤 5（README 之前，若 testcase-gen 未覆盖脚本）或补齐 `scripts/` 时读取本文件。
> 索引与约束见 [GUIDE.md](GUIDE.md)。

---

## scripts/gen_data.py → `<op_name>/scripts/gen_data.py`

```python
#!/usr/bin/env python3
# ============================================================
# SHMEM Communication 算子数据生成 + golden 计算模板
# 来源: shmem/examples/allgather/scripts/data_gen.py
# 使用: 复制到 <op_name>/scripts/gen_data.py，
#       按 design.md 的 correctness.oracle 实现 golden 逻辑
# ============================================================

import argparse
import json
import os
import numpy as np

np.random.seed(42)

DTYPE_MAP = {
    "int32":    np.int32,
    "float16":  np.float16,
    "float32":  np.float32,
    "bfloat16": "bfloat16",
}


def get_dtype(name: str):
    if name == "bfloat16":
        from ml_dtypes import bfloat16
        return bfloat16
    return DTYPE_MAP[name]


def gen_random_data(shape, dtype):
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 1000, size=shape, dtype=dtype)
    return np.random.uniform(0.0, 10.0, size=shape).astype(dtype)


# ==== 定制：按 design.md 的 correctness.oracle 实现 golden 逻辑 ====
def compute_golden(inputs, n_pes, elements, dtype):
    """示例: allgather golden — 拼接所有 PE 的输入"""
    golden = np.concatenate(inputs, axis=0)
    return golden


def main():
    parser = argparse.ArgumentParser(description="生成输入数据和 golden")
    parser.add_argument("--n_pes",    type=int, required=True)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--dtype",    type=str, default="int32")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    dtype = get_dtype(args.dtype)
    os.makedirs(args.data_dir, exist_ok=True)

    inputs = []
    for pe in range(args.n_pes):
        data = gen_random_data((args.elements,), dtype)
        data.tofile(os.path.join(args.data_dir, f"input_pe{pe}.bin"))
        inputs.append(data)

    golden = compute_golden(inputs, args.n_pes, args.elements, dtype)
    golden.tofile(os.path.join(args.data_dir, "golden.bin"))

    config = {
        "n_pes":    args.n_pes,
        "elements": args.elements,
        "dtype":    args.dtype,
        "rtol":     0,
        "atol":     0,
    }
    with open(os.path.join(args.data_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated data for {args.n_pes} PEs, {args.elements} elements, dtype={args.dtype}")


if __name__ == "__main__":
    main()
```

---

## scripts/check_result.py → `<op_name>/scripts/check_result.py`

```python
#!/usr/bin/env python3
# ============================================================
# SHMEM 算子精度验证模板
# 来源: shmem-ops-testcase-gen/references/correctness.md 标准 checker
# 使用: 复制到 <op_name>/scripts/check_result.py，
#       按 design.md 的 correctness.tolerance 调整阈值
# ============================================================

import argparse
import json
import sys
import numpy as np


def get_dtype(name: str):
    dtype_map = {
        "int32":    np.int32,
        "float16":  np.float16,
        "float32":  np.float32,
    }
    if name == "bfloat16":
        from ml_dtypes import bfloat16
        return bfloat16
    return dtype_map[name]


def check_precision(output, golden, rtol, atol, label=""):
    if output.shape != golden.shape:
        print(f"\033[31m✗ SHAPE MISMATCH {label}: output={output.shape} golden={golden.shape}\033[0m")
        return False

    if np.any(np.isnan(output)):
        nan_count = np.sum(np.isnan(output))
        print(f"\033[31m✗ NaN DETECTED {label}: {nan_count} NaN values\033[0m")
        return False

    if np.any(np.isinf(output)):
        inf_count = np.sum(np.isinf(output))
        print(f"\033[31m✗ Inf DETECTED {label}: {inf_count} Inf values\033[0m")
        return False

    diff = np.abs(output.astype(np.float64) - golden.astype(np.float64))
    golden_abs = np.abs(golden.astype(np.float64))
    rel_error = diff / (golden_abs + 1e-8)

    if rtol == 0 and atol == 0:
        mask = (output != golden)
    else:
        mask = (rel_error >= rtol) & (diff >= atol)

    error_indices = np.where(mask.flatten())[0]

    if len(error_indices) == 0:
        print(f"\033[32m✓ PASS {label}: all {output.size} elements within tolerance "
              f"(rtol={rtol}, atol={atol})\033[0m")
        return True

    print(f"\033[31m✗ FAIL {label}: {len(error_indices)}/{output.size} elements exceed tolerance\033[0m")
    print(f"  Max absolute error: {np.max(diff):.6e}")
    print(f"  Max relative error: {np.max(rel_error):.6e}")
    print(f"  Mean absolute error: {np.mean(diff):.6e}")

    flat_out = output.flatten()
    flat_gol = golden.flatten()
    for idx in error_indices[:10]:
        print(f"  Index {idx}: output={flat_out[idx]}, golden={flat_gol[idx]}, "
              f"diff={diff.flatten()[idx]:.6e}")

    return False


def main():
    parser = argparse.ArgumentParser(description="验证算子输出精度")
    parser.add_argument("--data_dir", type=str, required=True)
    # ==== 定制：如果每个 PE 的 golden 不同，调整 golden 读取逻辑 ====
    args = parser.parse_args()

    with open(f"{args.data_dir}/config.json") as f:
        config = json.load(f)

    dtype = get_dtype(config["dtype"])
    rtol = config.get("rtol", 0)
    atol = config.get("atol", 0)
    n_pes = config["n_pes"]

    golden = np.fromfile(f"{args.data_dir}/golden.bin", dtype=dtype)

    all_pass = True
    for pe in range(n_pes):
        output_path = f"{args.data_dir}/output_pe{pe}.bin"
        output = np.fromfile(output_path, dtype=dtype)
        if not check_precision(output, golden, rtol, atol, label=f"PE {pe}"):
            all_pass = False

    if all_pass:
        print(f"\033[32m✓ ALL {n_pes} PEs PASSED\033[0m")
        return 0
    else:
        print(f"\033[31m✗ SOME PEs FAILED\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## 算子 `scripts/run.sh` → `<op_name>/scripts/run.sh`

```bash
#!/bin/bash
# ============================================================
# SHMEM 算子多 PE 启动脚本模板
# 来源: shmem/examples/allgather/scripts/run.sh
# 使用: 复制到 <op_name>/scripts/run.sh
#       替换 <op_name> 占位符，按 compile/test contract 调整路径
# ============================================================

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
OP_DIR=$(dirname "$SCRIPT_DIR")
# custom-ops 独立工程：SHMEM 仓库根 = OP_DIR 的上两级
SHMEM_REPO=$(cd "${OP_DIR}/../.." && pwd)
EXEC_BIN="${OP_DIR}/build/bin/<op_name>"
DATA_DIR="${OP_DIR}/data"

# ---- 参数 ----
DEVICE_LIST="${1:-0,1}"
IFS=',' read -ra DEVICE_ID_LIST <<< "$DEVICE_LIST"
PE_SIZE=${#DEVICE_ID_LIST[@]}
ELEMENTS="${2:-1024}"
PERF_TIMES="${3:-0}"  # 0 = correctness only
DTYPE="${4:-int32}"
# IPPORT / SHMEM_UID_SESSION_ID 由 setup_shmem_runtime_env → setup_shmem_dynamic_endpoints 分配

echo "=== <op_name> === PE_SIZE=${PE_SIZE} ELEMENTS=${ELEMENTS} DTYPE=${DTYPE}"

# ---- 生成数据和 golden ----
mkdir -p "${DATA_DIR}"
python3 "${SCRIPT_DIR}/gen_data.py" \
    --n_pes "${PE_SIZE}" \
    --elements "${ELEMENTS}" \
    --dtype "${DTYPE}" \
    --data_dir "${DATA_DIR}"

# ---- SHMEM 环境（MUST 完整链路；函数体见 env-setup.snippet.md）----
setup_shmem_runtime_env "${SHMEM_REPO}" "${OP_DIR}" || exit 1

if [[ ! -x "${EXEC_BIN}" ]]; then
    echo "Build first: cmake --build ${OP_DIR}/build (see shmem-ops-compile-debug/references/custom-ops-entrypoints.md §1)" >&2
    exit 1
fi

# ---- 启动多 PE 进程 ----
PIDS=()
for (( idx = 0; idx < PE_SIZE; idx++ )); do
    ${EXEC_BIN} "${PE_SIZE}" "${idx}" "${IPPORT}" "${DATA_DIR}" \
        "${ELEMENTS}" "${PERF_TIMES}" &
    PIDS+=($!)
done

# ---- 等待所有进程 ----
EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || EXIT_CODE=1
done

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Some PE processes failed"
    exit 1
fi

# ---- 验证结果 ----
if [ "${PERF_TIMES}" -eq 0 ]; then
    python3 "${SCRIPT_DIR}/check_result.py" --data_dir "${DATA_DIR}"
    exit $?
fi
```
