# AscendC 量化算子 (DynamicQuant 类) 知识库

> **本文档仅供参考，不作为绝对约束。** 内容基于一次具体硬件 (Atlas A2 / CANN 8.5.0) 与 CANN 版本下的实测与文档整理；不同 SoC、CANN 版本、kernel 设计场景下接口能力与精度行为可能不同。开发时以官方 API 文档与实测结果为准。

适用于将每行/每 token 维度的动态量化（per-token / per-row dynamic quant）落到 AscendC 的开发者。覆盖 Cast 接口能力、舍入语义、典型数据流约束、验证标准。

---

## 1. 算子语义

per-token dynamic quant 的核心步骤（每个 token / 每行独立）：

```
row_max_abs = max(|x|)
scale       = max(row_max_abs / 127.0, 1e-10)        # fp32
quantized   = round(x / scale).clamp(-128, 127)      # 输出 int8
```

输出两个张量：

| 输出 | dtype | 形状 | 备注 |
|---|---|---|---|
| `quantized` | int8 | 同输入 | 每行独立 round + clamp |
| `scale` | fp32 | 输入除最后一维 | 每行一个标量 |

`torch_npu.npu_dynamic_quant(x)` 是这一类算子的标杆参考，内部走 NPU 硬件 quant 单元，不通过标准 AscendC vector API 直接暴露。

---

## 2. Cast 接口能力（Atlas A2，CANN 8.5.0）

### 2.1 fp32 → 整数

| 目的类型 | 支持的 RoundMode |
|---|---|
| `int32_t` | RINT / FLOOR / CEIL / ROUND / TRUNC |
| `int64_t` | RINT / FLOOR / CEIL / ROUND / TRUNC |
| `int16_t` | RINT / FLOOR / CEIL / ROUND / TRUNC |
| `int8_t`  | **不支持直接 fp32 → int8 cast，必须走 fp16 中转** |

### 2.2 fp32 → 浮点

| 目的类型 | 支持的 RoundMode |
|---|---|
| `half` | NONE / ODD（NONE 在有损时 = RINT） |
| `bfloat16_t` | RINT / FLOOR / CEIL / ROUND / TRUNC |

### 2.3 half / bfloat16_t → 整数

| 源 → 目的 | 支持的 RoundMode |
|---|---|
| `half` → `int32_t` | RINT / FLOOR / CEIL / ROUND / TRUNC |
| `half` → `int16_t` | RINT |
| `half` → `int8_t`  | FLOOR / CEIL / ROUND / TRUNC / NONE |
| `bfloat16_t` → `int32_t` | RINT / FLOOR / CEIL / ROUND / TRUNC |

### 2.4 不存在的常用 Cast 重载

| Cast 形式 | 状态 |
|---|---|
| `Cast<int8_t, int32_t>` | 不存在（编译失败） |
| `Cast<int8_t, fp32>` | 不存在 |
| `Cast<int16_t, int32_t>` | 不存在 |

需要这些路径时，应组合可用 cast（fp16 中转、int16 中转 + host 端 `.to(int8)` 窄化）。

### 2.5 RoundMode 语义（`enum class RoundMode`）

| Mode | 含义 |
|---|---|
| `CAST_NONE = 0` | 有精度损失时 = `CAST_RINT`；无损失时不舍入 |
| `CAST_RINT` | 四舍六入五成双（IEEE round-half-to-even / banker's） |
| `CAST_FLOOR` | 向 -∞ 舍入 |
| `CAST_CEIL` | 向 +∞ 舍入 |
| `CAST_ROUND` | 四舍五入（远离零） |
| `CAST_TRUNC` | 向零舍入 |
| `CAST_ODD` | Von Neumann，最近邻奇数舍入 |

---

## 3. 量化路径选择

落地 fp32 → int8 时，两条等价但精度特征不同的路径：

### 路径 A：fp32 → fp16 → int8

```cpp
AscendC::Cast(yFp16, yFp32, AscendC::RoundMode::CAST_NONE, count);  // fp16 lossy → 自动 RINT
AscendC::Cast(yI8,   yFp16, AscendC::RoundMode::CAST_NONE, count);  // half→int8 → RINT
```

特点：
- fp16 的 ulp 在 `|x|≈100` 时约为 0.0625，会把接近整数 .5 的 fp32 值吸到可表示的 fp16 半步上，进而影响最终 int8 的舍入结果
- 与硬件融合 quant 单元的内部行为更接近

### 路径 B：fp32 → int16 → host narrow

```cpp
AscendC::Cast(yI16, yFp32, AscendC::RoundMode::CAST_RINT, count);  // 直接 banker's 入整
```
host 端：`y_int16.to(torch.int8)`（int16 中 [-128,127] 的低字节即 int8）。

特点：
- 中间精度全程保持 fp32，避免 fp16 ulp 干扰
- int16 GM 占用是 int8 的 2×，最大算子 (8192×16384) 约 256 MB 中间张量

---

## 4. 数值精度要点

### 4.1 `Div` 与 `Muls` 不等价

`x / scale`（向量 `Div(y, x, scaleVec)`）和 `x * (1.0f/scale)`（标量 `Muls(y, x, invScale)`）在 fp32 ulp 上有 1–2 位漂移，可能让正好处在整数 .5 边界的元素倒向另一侧。需要严格匹配 reference 的除法语义时，使用 `Div`（`scaleVec` 由 `Duplicate` 填充）。

### 4.2 标量写 GM 必须走 UB + DataCopyPad

```cpp
// 推荐
AscendC::LocalTensor<float> scaleLocal;
scaleOutQueue_.AllocTensor<float>(scaleLocal);
AscendC::Duplicate(scaleLocal, scale, 8);  // pad 到 32 B 满足 DataCopyPad 对齐
scaleOutQueue_.EnQue(scaleLocal);
scaleOutQueue_.DeQue<float>(scaleLocal);
StoreUbToGm(scaleGM_[rowIdx], scaleLocal, 1);
scaleOutQueue_.FreeTensor(scaleLocal);
```

`GlobalTensor::SetValue(idx, val)` 是直写指令，与 vector pipeline 不在同一通道，host 读回时不保证已落盘；用 UB + `DataCopyPad`（即 `StoreUbToGm`）走标准 EnQue/DeQue 才能正确同步。

### 4.3 输入 dtype 转 fp32 是无损的

`bf16 → fp32` 和 `fp16 → fp32` 均无损（fp32 的 mantissa/exponent 范围严格覆盖两者），用 `Cast<fp32, dataType, CAST_NONE>` 即可。

### 4.4 reduce 用 fp32 累加

`max(|x|)` 应在 fp32 域内用 `Abs` + `ReduceMax<float>` 完成；用输入 dtype 直接 reduce 会丢失 1 ulp 精度，导致 `scale = max/127` 与参考偏差。

---

## 5. TQue 与 Cast 接口的常见约束

### 5.1 `TQue<..., 0>`（depth=0）的 DeQue 形式

```cpp
AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQ;
AscendC::LocalTensor<int16_t> outLocal;

outQ.AllocTensor<int16_t>(outLocal);   // 引用形式：可
outQ.DeQue<int16_t>(outLocal);         // 引用形式：可

auto x = outQ.DeQue<int16_t>();        // 返回值形式：编译失败（depth=0 时 static_assert）
```

depth=0 + 返回值 DeQue 触发 `static_assert((depth != 0), "must use DeQue<LocalTensor&> api ...")`。两种解法：把 depth 改为 ≥1，或全部使用引用形式。

### 5.2 Cast 起始地址 32 B 对齐

`Cast` 的 `dst` / `src` 起始地址必须 32 B 对齐。`AllocTensor` 默认满足；`Get<T>()` 自 `TBuf` 取出的 LocalTensor 同样从 buffer 起始位置对齐。

### 5.3 `DataCopyPad` 写不足 32 B 的元素

写 1 个 fp32（4 B）回 GM 时仍需走 `DataCopyPad`，UB 一侧用 `Duplicate(buf, val, 8)` 填到 32 B 后传 `count = 1`：

```cpp
AscendC::DataCopyExtParams params{1, sizeof(float), 0, 0, 0};
AscendC::DataCopyPad(scaleGM_[rowIdx], scaleLocal, params);
```

---

## 6. 验证标准

`.claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py` 对比 reference 与 candidate 输出：

| 输出 dtype | 判定方式 |
|---|---|
| 浮点 (fp16 / bf16 / fp32) | MERE / MARE 阈值（threshold 取自 `PRECISION_THRESHOLDS`） |
| int8 / int16 | **元素级 max abs diff ≤ 1**（`INT_LSB_TOLERANCE`） |
| int32 / int64 / bool | `torch.equal` 严格按位 |

```python
INT_LSB_TOLERANCE = {
    torch.int8:  1,
    torch.int16: 1,
}
```

判定逻辑：

```python
diff = (rhs.to(torch.int32) - lhs.to(torch.int32)).abs()
max_abs = diff.max().item()
if max_abs <= tol:
    return True
```

容忍度是张量内的最大值，**不是平均值**：哪怕只有一个元素差 2，整张张量判 fail。

浮点阈值：

| dtype | MERE threshold | MARE threshold |
|---|---|---|
| float16 | 9.77e-4 | 9.77e-3 |
| bfloat16 | 7.81e-3 | 7.81e-2 |
| float32 | 1.22e-4 | 1.22e-3 |

scale 输出（fp32）走 MERE/MARE，量化输出（int8）走 ±1 LSB。

---

## 7. 输入与 Tiling

### 7.1 输入 shape 处理

per-token 量化对最后一维做 reduce + scale。任意维度输入可以 host 端 reshape 成 2D `[M, N]`：

```python
x_2d = x.reshape(-1, x.shape[-1]).contiguous()
y_2d, scale_1d = _ext.run_op(x_2d)
y     = y_2d.reshape(x.shape)
scale = scale_1d.reshape(x.shape[:-1])
```

### 7.2 N 较大时的 split-D

UB 容量下整行可能装不下 fp32 工作 buffer。N 拆为 `kBlockN`（典型 4096）后，每行内 reduce 与 quant 各做一遍 chunk 循环：reduce 阶段累计 chunk-max；quant 阶段用已确定的 scale 做整行划分。

### 7.3 Block / Core 划分

每核处理 `blockM` 行（典型 16）；AIV 子块 = `blockM / GetSubBlockNum()`；usedCoreNum = `min(20, ceil(M / blockM))`。

---

## 8. Pybind 与 Host 接口

### 8.1 Tiling struct

```cpp
struct DynamicQuantTiling {
    int32_t M, N;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t pad0;  // 32 B 对齐
};
```

### 8.2 Launch dispatch

```cpp
LaunchFn launch = nullptr;
if (x.scalar_type() == at::kHalf)         launch = dynamic_quant_do_fp16;
else if (x.scalar_type() == at::kBFloat16) launch = dynamic_quant_do_bf16;
else TORCH_CHECK(false, "unsupported dtype");
```

### 8.3 输出张量分配

| 路径 | y 输出 dtype |
|---|---|
| 路径 A（fp16 中转） | `at::kChar`（int8）直接出 |
| 路径 B（int16 + host narrow） | `at::kShort`（int16） |

scale 一律 `at::kFloat`。

---

## 9. KERNEL_TYPE 与 vec_num

| vec_num | KERNEL_TYPE 宏 | 每 block 组成 |
|---|---|---|
| 1 | `KERNEL_TYPE_MIX_AIC_1_1` | 1 AIC + 1 AIV |
| 2 | `KERNEL_TYPE_MIX_AIC_1_2` | 1 AIC + 2 AIV |

per-token quant 的 reduce + 单值 scale 写出场景，`KERNEL_TYPE_MIX_AIC_1_2` 即可（计算全在 AIV）。

---

## 10. 参考实现位置

| 算子 | 可复用模式 |
|---|---|
| `archive_tasks/rms_norm/` | per-row reduce + 标量写 scale 的 `vector_tile.h` / `_kernel.h` 模板 |
| `archive_tasks/reshape_matmul_rowwise_quant_int8/` | fp32 → fp16 → int8 cast 链（`quant_kernel.h`） |
| `archive_tasks/quant_matmul/` | int8 输出的 host pybind / tiling 模式 |

---

## 11. 涉及的 AscendC API 速查

| 用途 | API |
|---|---|
| 行内绝对值 | `AscendC::Abs(dst, src, count)` |
| 行内最大值 reduce | `AscendC::ReduceMax<float>(dst, src, work, count, false)` |
| 标量乘整向量 | `AscendC::Muls(dst, src, scalar, count)` |
| 向量除（标量需先 `Duplicate`） | `AscendC::Div(dst, src0, src1, count)` |
| 向量 clamp 上下界 | `AscendC::Mins`, `AscendC::Maxs` |
| 向量填充常量 | `AscendC::Duplicate(dst, scalar, count)` |
| 类型转换 | `AscendC::Cast(dst, src, RoundMode, count)` |
| GM↔UB 传输（任意 count） | `DataCopyPad`（封装为 `LoadGmToUb` / `StoreUbToGm`） |

完整定义见 `asc-devkit/docs/api/README.md`。
