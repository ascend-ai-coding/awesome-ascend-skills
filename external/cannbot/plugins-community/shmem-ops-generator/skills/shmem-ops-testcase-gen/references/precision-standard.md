# SHMEM 精度验证标准

本文定义 SHMEM 算子精度验证的容差体系、判定逻辑和算子映射规则。数据生成脚本 `gen_data.py` 和精度验证脚本 `check_result.py` 必须以此为标准。

---

## 1. 精度评估指标

| 指标 | 公式 | 说明 |
| --- | --- | --- |
| 最大绝对误差 (MaxAE) | max(\|output - golden\|) | 逐元素绝对误差的最大值 |
| 最大相对误差 (MaxRE) | max(\|output - golden\| / max(\|golden\|, eps)) | 逐元素相对误差的最大值 |
| 平均相对误差 (MeanRE) | mean(\|output - golden\| / max(\|golden\|, eps)) | 逐元素相对误差的均值 |
| 精确匹配 | output == golden (bitwise) | 整型或纯搬运算子要求 |

其中 `eps` 为 `np.finfo(np.float64).tiny`，用于防止分母为零。

---

## 2. 算子分类 (OpTypes)

每个 SHMEM 算子必须映射到以下分类之一，确定容差级别：

| 分类 | 说明 | 典型算子 |
| --- | --- | --- |
| MOVE | 纯搬运，数据无变化 | broadcast、scatter、allgather、put/get |
| COMPUTE_INTEGER | 整型计算/索引操作 | int32 reduce、dispatch 索引 |
| COMPUTE_QUANT | 量化计算 | 量化 matmul、量化 cast |
| COMPUTE_FLOAT | 通用浮点通信计算 | reduce、allreduce、reduce_scatter |
| COMPUTE_FLOAT_HIGH_PRECISION | 高精度浮点通信计算 | fp32 accumulate 的 matmul+reduce |

---

## 3. rtol / atol 取值

### 3.1 compute_times 计算

`compute_times` 是算子中浮点运算的累积次数，影响 rtol 取值：

```
compute_times = PE 数 × 每 PE 参与计算/累加的元素数
```

例如：8 PE 的 allreduce，每 PE 1024 元素 → `compute_times = 8 × 1024 = 8192`

### 3.2 各 dtype 各分类 rtol 值

**fp16**：

| 分类 | compute_times < 2048 | compute_times ≥ 2048 |
| --- | --- | --- |
| MOVE | 0 | 0 |
| COMPUTE_INTEGER | 0 | 0 |
| COMPUTE_QUANT | 2^-8 | 2^-8 |
| COMPUTE_FLOAT | 2^-8 | 2^-7 |
| COMPUTE_FLOAT_HIGH_PRECISION | 2^-11 | 2^-11 |

**bfloat16**：

| 分类 | compute_times < 2048 | compute_times ≥ 2048 |
| --- | --- | --- |
| MOVE | 0 | 0 |
| COMPUTE_INTEGER | 0 | 0 |
| COMPUTE_FLOAT | 2^-7 | 2^-6 |
| COMPUTE_FLOAT_HIGH_PRECISION | 2^-11 | 2^-11 |

**float32**：

| 分类 | compute_times < 2048 | compute_times ≥ 2048 |
| --- | --- | --- |
| MOVE | 0 | 0 |
| COMPUTE_INTEGER | 0 | 0 |
| COMPUTE_FLOAT | 2^-11 | 2^-10 |
| COMPUTE_FLOAT_HIGH_PRECISION | 2^-14 | 2^-14 |

**整型**（int32、int64、int8、uint8）：

| 分类 | rtol | atol |
| --- | --- | --- |
| MOVE | 0 | 0 |
| COMPUTE_INTEGER | 0 | 0 |

### 3.3 atol 取值规则

`atol` 统一等于 `rtol`。rtol=0 时 atol=0（bitwise exact）。

---

## 4. 双统计判定

### 4.1 precision_threshold（逐元素容差）

对每个元素：`|actual - golden| ≤ rtol × max(|golden|, 1)` 为通过。

`precision_percent = 通过元素数 / 总元素数 × 100`

**要求**：`precision_percent == 100`（所有元素必须通过逐元素容差）。

### 4.2 eb_threshold（平均偏置）

`eb = |mean(actual - golden) / max(|golden|)|`

即相对误差的均值绝对值，检测是否有系统性偏离（方向偏差）。

**要求**：`eb ≤ eb_threshold`。

`eb_threshold` 取值：

| dtype | eb_threshold |
| --- | --- |
| fp16 | 2^-10 |
| bf16 | 2^-7 |
| fp32 | 2^-14 |

### 4.3 综合判定

```
precision_pass = (precision_percent == 100)
eb_pass = (eb ≤ eb_threshold)
basic_pass = precision_pass AND eb_pass
```

---

## 5. 双级兜底判定

`check_result.py` **MUST** 支持两级判定：

| 级 | 名称 | 输入 | 判定 |
| --- | --- | --- | --- |
| **Basic** | 基础 pass | output vs golden | `basic_pass`（§4.3） |
| **Reference**（可选） | 参考 pass | output vs golden vs torch_output | output 与 torch_output 的 MARE/MERE/RMSE 比值均在门限内 |

### 5.1 Reference pass 判定

当同时提供 `--torch-output` 参数时启用。计算 SHMEM output 和 torch_output 各自对 golden 的相对误差指标：

```
MARE = MaxRE  ;  MERE = MeanRE  ;  RMSE = sqrt(mean((actual - golden)^2))
```

比值判定：

| 指标 | 比值门限 |
| --- | --- |
| MARE_ratio = MARE_shmem / MARE_torch | ≤ 10 |
| MERE_ratio = MERE_shmem / MERE_torch | ≤ 2 |
| RMSE_ratio = RMSE_shmem / RMSE_torch | ≤ 2 |

### 5.2 最终判定

```
overall_pass = basic_pass OR reference_pass
```

任一 pass 即视为精度通过。reference pass 的作用是兜底——当 SHMEM 的浮点累积误差在 torch 同等量级范围内时认为可接受。

---

## 5.1 `meta.op_kind` → OpTypes 映射

生成 `check_result.py` / 调用 `--rtol`/`--atol` 时，**先**按 `design.md` DSL 的 `meta.op_kind` 查下表，**再**按 §6 算子名表微调；二者冲突时以 §6 算子语义为准。

| `meta.op_kind` | 默认 OpTypes | rtol / atol | 参考 pass |
| --- | --- | --- | --- |
| `transport` | MOVE | 0 / 0 | 不需要 |
| `collective` | 浮点规约 → COMPUTE_FLOAT；整型搬运 → MOVE | 见 §3.2 或 0 | 浮点规约建议 |
| `compute` | COMPUTE_FLOAT 或 COMPUTE_INTEGER（按 dtype） | §3.2 或 0 | 浮点建议 |
| `fused_compute_comm` | COMPUTE_FLOAT（Matmul/GEMM 输出） | §3.2 | **必须** |

> 通算融合若输出为 bf16/fp16，仍用 COMPUTE_FLOAT 阈值；整型路由字段（如 dispatch index）用 COMPUTE_INTEGER。

---

## 6. SHMEM 算子映射表

| 算子 | OpTypes | rtol 来源 | 参考 pass |
| --- | --- | --- | --- |
| broadcast | MOVE | 0 | 不需要 |
| scatter | MOVE | 0 | 不需要 |
| allgather | MOVE | 0 | 不需要 |
| gather | MOVE | 0 | 不需要 |
| reduce（浮点） | COMPUTE_FLOAT | §3.2 | 建议 |
| allreduce（浮点） | COMPUTE_FLOAT | §3.2 | 建议 |
| reduce_scatter（浮点） | COMPUTE_FLOAT | §3.2 | 建议 |
| matmul_allreduce / matmul_reduce_scatter / allgather_matmul（通算融合） | COMPUTE_FLOAT | §3.2 | **必须** |
| dispatch（整型路由） | COMPUTE_INTEGER | 0 | 不需要 |

---

## 7. 比较规则

- 比较时 **MUST** 转 `np.float64` 避免精度损失
- **MUST** 检查 `nan` / `inf`
- `scripts/run.sh` 调用 `check_result.py` 时 **MUST** 显式传 `--rtol` 和 `--atol`，不依赖脚本默认值
- 退出码 0 = PASS，非 0 = FAIL
- 每 PE 打印 MaxAE、MaxRE、MeanRE 和 `rtol`/`atol`/`basic_pass`/`reference_pass` 结论
- `precision_threshold` 和 `eb_threshold` 的计算方法见 §4
