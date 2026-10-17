# SHMEM 测试文件结构与代码模板

本文定义 SHMEM 算子测试脚本的标准文件结构和代码模板。规则和策略（测什么、精度容差、golden 策略）见 [correctness.md](correctness.md)。

## 1. 目录结构

```
<op_name>/
├── scripts/
│   ├── gen_data.py          # 输入数据生成 + golden 计算
│   ├── check_result.py      # 精度验证
│   ├── run.sh               # 端到端测试入口
│   └── cases.json           # 可选：case matrix 配置
├── data/                    # gen_data.py 生成的输入和 golden
├── output/                  # 算子输出
└── src/
    ├── main.cpp             # Host 入口
    ├── <op_name>_kernel.cpp
    └── <op_name>_kernel.h
```

---

## 2. gen_data.py 模板

```python
#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np

np.random.seed(42)

def gen_random_data(size, dtype):
    if dtype in (np.float16, np.float32, np.bfloat16):
        return np.random.uniform(low=0.0, high=10.0, size=size).astype(dtype)
    elif dtype in (np.int32, np.int8):
        return np.random.randint(0, 1000, size=size, dtype=dtype)
    return None

def golden_generate(args):
    """生成输入数据和 golden 输出"""
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = np.dtype(args.dtype)

    # 1. 为每个 PE 生成独立输入
    for pe_id in range(args.n_pes):
        input_data = gen_random_data((args.M, args.N), dtype=dtype)
        input_data.tofile(f"{args.out_dir}/input_pe{pe_id}.bin")

    # 2. 计算 golden 输出（按算子语义实现）
    # [替换为实际 golden 逻辑]
    golden = np.zeros((args.M, args.N), dtype=dtype)
    golden.tofile(f"{args.out_dir}/golden.bin")

    # 3. 保存配置
    config = {
        "n_pes": args.n_pes,
        "M": args.M,
        "N": args.N,
        "dtype": args.dtype,
        "numpy_dtype": str(dtype),
        "expected_elems": args.M * args.N,
    }
    with open(f"{args.out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pes', type=int, required=True)
    parser.add_argument('--M', type=int, required=True)
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--dtype', type=str, default='float16')
    parser.add_argument('--out_dir', type=str, default='./data')
    args = parser.parse_args()
    golden_generate(args)
```

**关键规范**：
- 固定 `np.random.seed(42)` 保证可复现
- dtype 从命令行参数接收，不硬编码
- 输入/golden 按 PE 分文件，文件名包含 pe_id
- 输出目录可配置
- 必须保存 config.json（供 check_result.py 读取）
- 浮点累加优先用 float32 中间结果

---

## 3. check_result.py 模板

### 3.1 per-PE 模式（推荐）

适用于每个 PE 有独立 golden 的集合通信/分布式算子。

```python
#!/usr/bin/env python3
import sys
import argparse
import json
import os
import numpy as np

def check_pe(actual, golden, pe, rtol, atol, dtype_str):
    abs_diff = np.abs(actual.astype(np.float64) - golden.astype(np.float64))
    golden_abs = np.abs(golden.astype(np.float64))
    eps = np.finfo(np.float64).tiny
    rel_diff = abs_diff / np.maximum(golden_abs, eps)

    max_ae = float(np.max(abs_diff))
    max_re = float(np.max(rel_diff))
    mean_re = float(np.mean(rel_diff))

    print(f"PE {pe}: dtype={dtype_str} MaxAE={max_ae:.6e} MaxRE={max_re:.6e} "
          f"MeanRE={mean_re:.6e} rtol={rtol} atol={atol}")

    if np.any(np.isnan(actual)) or np.any(np.isinf(actual)):
        print(f"PE {pe}: FAIL (nan/inf detected)")
        return False

    if rtol == 0 and atol == 0:
        ok = np.array_equal(actual, golden)
    else:
        ok = np.allclose(actual, golden, rtol=rtol, atol=atol)

    print(f"PE {pe}: {'PASS' if ok else 'FAIL'} ({len(actual)} elements)")
    if not ok:
        bad = np.logical_and(abs_diff > atol, rel_diff > rtol) if rtol > 0 else ~np.equal(actual, golden)
        idx = int(np.argmax(bad))
        print(f"  first mismatch idx={idx} actual={float(actual.astype(np.float64)[idx])} "
              f"expected={float(golden.astype(np.float64)[idx])} diff={float(abs_diff[idx])}")
    return ok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--atol", type=float, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "config.json")) as f:
        cfg = json.load(f)
    n_pes = int(cfg["n_pes"])
    expected_elems = int(cfg["expected_elems"])
    dtype = np.dtype(cfg.get("numpy_dtype", "float16"))

    failed = 0
    for pe in range(n_pes):
        actual = np.fromfile(os.path.join(args.output_dir, f"output_pe{pe}.bin"), dtype=dtype, count=expected_elems)
        golden = np.fromfile(os.path.join(args.data_dir, f"golden_pe{pe}.bin"), dtype=dtype, count=expected_elems)
        if not check_pe(actual, golden, pe, args.rtol, args.atol, str(dtype)):
            failed += 1

    if failed:
        print(f"FAIL: {failed}/{n_pes} PE outputs mismatched")
        sys.exit(1)
    print(f"PASS: all {n_pes} PE outputs match")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

### 3.2 单文件模式

适用于所有 PE 共享同一 golden 的场景。

```python
#!/usr/bin/env python3
import sys
import argparse
import numpy as np

def check(args):
    golden = np.fromfile(args.golden, dtype=args.dtype)
    output = np.fromfile(args.output, dtype=args.dtype)

    if golden.shape != output.shape:
        print(f"FAIL: shape mismatch golden={golden.shape} output={output.shape}")
        return 1

    if np.any(np.isnan(output)) or np.any(np.isinf(output)):
        print("FAIL: nan/inf detected in output")
        return 1

    abs_diff = np.abs(output.astype(np.float64) - golden.astype(np.float64))
    golden_abs = np.abs(golden.astype(np.float64))
    eps = np.finfo(np.float64).tiny

    max_ae = np.max(abs_diff)
    max_re = np.max(abs_diff / np.maximum(golden_abs, eps))
    mean_re = np.mean(abs_diff / np.maximum(golden_abs, eps))

    print(f"dtype: {args.dtype}")
    print(f"MaxAE: {max_ae:.6e}")
    print(f"MaxRE: {max_re:.6e}")
    print(f"MeanRE: {mean_re:.6e}")
    print(f"rtol: {args.rtol}, atol: {args.atol}")

    if args.rtol == 0 and args.atol == 0:
        pass_flag = np.array_equal(output, golden)
    else:
        pass_flag = np.allclose(output, golden, rtol=args.rtol, atol=args.atol)
    print("PASS" if pass_flag else "FAIL")
    return 0 if pass_flag else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dtype', type=str, default='float16')
    parser.add_argument('--rtol', type=float, required=True)
    parser.add_argument('--atol', type=float, required=True)
    args = parser.parse_args()
    sys.exit(check(args))
```

### 3.3 精度容差选择

check_result.py 的 `--rtol` 和 `--atol` 必须由 `scripts/run.sh` 显式传入，不依赖脚本默认值。容差值按 `op_kind`、`dtype` 和 `compute_times` 选取，权威规则见 [precision-standard.md](precision-standard.md)。

**check_result.py 必须实现**：
- 双统计判定：`precision_percent`（逐元素通过率 → 必须 100%）和 `eb`（平均偏置 → 必须 ≤ eb_threshold），见 [precision-standard.md §4](precision-standard.md)
- 可选 `--torch-output` 参数，启用 reference pass 兜底，见 [precision-standard.md §5](precision-standard.md)

**关键规范**：
- 退出码 0 = PASS，非 0 = FAIL
- 每个 PE 必须打印 MaxAE、MaxRE、MeanRE 和 tolerance 阈值
- transport/纯搬运算子使用 `np.array_equal()` 做精确比较
- 比较时转 float64 避免精度损失
- 检查 `nan`/`inf`
- `--rtol` 和 `--atol` 为 required 参数

---

## 4. `scripts/run.sh` 模板

> **custom-ops 独立工程（默认）**：算子内仍生成 `custom-ops/<op>/scripts/run.sh`，但 skill/README/交付 **MUST** 以 [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §2 为首选运行入口。build 失败提示 **MUST** 写同文件 §1 编译，禁止裸 cmake。

```bash
#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
OP_DIR=$(dirname "$SCRIPT_DIR")
# custom-ops：SHMEM 仓库根
SHMEM_REPO=$(cd "${OP_DIR}/../.." && pwd)
EXEC_BIN="${OP_DIR}/build/bin/<op_name>"

# ========== 参数 ==========
PE_SIZE="${1:-2}"
FIRST_NPU="${3:-0}"
TIMEOUT="${TIMEOUT:-120}"    # 超时上限 120 秒（2 分钟）
# IPPORT / SHMEM_UID_SESSION_ID 由 setup_shmem_dynamic_endpoints 分配；用户 export 时尊重其值

DATA_DIR=${OP_DIR}/data
OUTPUT_DIR=${OP_DIR}/output

# ========== 环境（MUST 完整链路，禁止只设 build/lib）==========
# 内联 setup_shmem_runtime_env（函数体见 env-setup.snippet.md）
setup_shmem_runtime_env "${SHMEM_REPO}" "${OP_DIR}" || exit 1

if [[ ! -x "${EXEC_BIN}" ]]; then
    echo "Build first: cmake --build ${OP_DIR}/build (see shmem-ops-compile-debug/references/custom-ops-entrypoints.md §1)" >&2
    exit 1
fi

# ========== 1. 生成测试数据 ==========
rm -rf ${DATA_DIR} ${OUTPUT_DIR}
mkdir -p ${DATA_DIR} ${OUTPUT_DIR}
${PYTHON_CMD:-python3} ${SCRIPT_DIR}/gen_data.py --n_pes ${PE_SIZE} --out_dir ${DATA_DIR}

# ========== 2. 启动多 PE 进程 ==========
pids=()
for (( idx=0; idx<${PE_SIZE}; idx++ )); do
    npu_id=$(( FIRST_NPU + idx ))
    ${EXEC_BIN} ${PE_SIZE} ${idx} ${IPPORT} ${npu_id} ${DATA_DIR} ${OUTPUT_DIR} &
    pids+=("$!")
done

# ========== 3. 超时等待（上限 ${TIMEOUT} 秒）==========
( sleep ${TIMEOUT} && echo "[TIMEOUT] exceeded ${TIMEOUT}s, killing processes" >&2 && kill "${pids[@]}" 2>/dev/null ) &
watchdog=$!

ret=0
for pid in "${pids[@]}"; do
    wait $pid || ret=1
done

kill $watchdog 2>/dev/null
wait $watchdog 2>/dev/null || true

if [[ $ret -ne 0 ]]; then
    echo "[FAIL] one or more PE processes failed or timed out"
    exit 1
fi

# ========== 4. 验证结果 ==========
${PYTHON_CMD:-python3} ${SCRIPT_DIR}/check_result.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --rtol <RTOL> \
    --atol <ATOL>
exit $?
```

**关键规范**：
- **MUST** `source ${SHMEM_REPO}/install/set_env.sh` 后再追加 `build/lib`；见 env-setup.snippet.md 的 `setup_shmem_runtime_env`
- **MUST** 调用 `setup_shmem_dynamic_endpoints`，禁止写死 `27010`/`8899`
- **超时上限 120 秒**（通过 `TIMEOUT` 环境变量可覆盖，但默认不超过 2 分钟）
- 支持参数化 PE 数、端口、首 NPU
- 先生成数据，再启动进程，最后验证
- 每个 PE 作为独立后台进程启动
- `--rtol` 和 `--atol` 必须显式填写，不用占位符
- 使用 `PYTHON_CMD` 环境变量指定 Python，不硬编码 `python3`

---

## 5. main.cpp 边界规则

算子 `main.cpp` 必须遵循以下边界：

| 允许 | 禁止 |
| --- | --- |
| 读取输入文件 | 生成 golden 数据 |
| 拷贝数据到设备 | 精度验证 / 误差打印 |
| 调用 Device kernel | 复杂 route/payload/tiling/packing 逻辑 |
| 拷贝输出并写入文件 | fork/spawn 多个 PE 子进程 |
| ACL/SHMEM 初始化和清理 | Host RMA 作为主通信路径 |
| 调用独立模块的 Host 计划函数 | 在 main.cpp 内实现 Host 计划逻辑 |

**拆分信号**：Host 文件超过 ~400 行，且大部分不是 lifecycle/资源管理/launch 编排 → 拆出独立 `.cpp/.h`。

**逻辑归属**：

| 逻辑类型 | 归属 |
| --- | --- |
| route table / payload 编码解码 / 数据预处理 | Python（测试数据）或 独立 C++ Host 模块（运行时） |
| layout / offset / shape 派生 / tiling 参数 | 独立 C++ helper |
| 文件读写 / 资源管理 | 可拆到 C++ helper |
| 多 PE 启动 | `scripts/run.sh` / 外部 launcher |

---

## 6. cases.json（可选）

```json
[
  {"case_id": "smoke_2pe_fp16", "n_pes": 2, "M": 128, "N": 64, "dtype": "float16", "rtol": 0, "atol": 0},
  {"case_id": "medium_8pe_fp16", "n_pes": 8, "M": 2048, "N": 1024, "dtype": "float16", "rtol": 0, "atol": 0}
]
```

用于批量测试多组 case 的场景，`scripts/run.sh` 可循环读取每条执行。
