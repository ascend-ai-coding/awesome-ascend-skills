# Triton API 在 Ascend 910_95 NPU 上的差异速查

> 触发条件：在迁移或编写 Triton kernel 时，需要确认 API 在 910_95 上的行为差异、替代方案或 NPU 特有扩展。

---

## 1. API 差异速查表

### 1.1 不支持的 API 及替代方案

| API | 不支持内容 | 替代方案 | 说明 |
|-----|-----------|---------|------|
| `tl.load` / `tl.store` | uint8/uint16/uint32/uint64/fp64 | int8/int16/int32/int64/fp32 | 硬件限制，需在 host 端转换类型 |
| `tl.load` / `tl.store` | `cache_modifier` / `eviction_policy` / `volatile` | 忽略即可 | NPU 无 GPU L1/L2 cache 层次，传入不报错但无效 |
| `tl.dot` | `input_precision="tf32"` | 使用 `"hf32"` | NPU 不支持 TF32，自动映射为 hf32 |
| `tl.dot` | `input_precision="tf32x3"` | 无替代 | NPU 不支持 tf32x3 |
| `tl.dot` | `out_dtype=bfloat16` | `out_dtype=float32` 后 `.to(tl.bfloat16)` | 需手动 cast |
| `tl.dot` | `max_num_imprecise_acc` | 忽略即可 | 传入被忽略，所有累加使用完整精度 |
| `tl.dot` | int16/int32 输入 | 仅支持 int8/fp16/bf16/fp32 | 矩阵乘输入类型受限 |
| `tl.sort` | 全类型 | 使用 `al.sort`（Ascend 扩展） | 毕昇编译器限制，标准 tl.sort 不可用 |
| `tl.round` | fp16/bf16 | 仅 fp32 | NPU 限制 |
| `tl.umulhi` | 负数输入 | 避免负数 | 仅支持 i32，不支持负数 |
| `tl.gather` | axis != n-1 | 仅支持最后一维 | NPU 限制 |
| `tl.permute` / `tl.trans` | 不相邻轴转置 | 仅支持相邻轴 | NPU 限制 |
| `tl.cat` | `can_reorder=False` | 必须使用 `can_reorder=True` | 当前实现限制，元素顺序可能重排 |
| `tl.flip` | 非 2 的幂维度 | 确保翻转维度和 numel 为 2 的幂 | 基于 bitonic merge 实现 |
| `tl.histogram` | 浮点输入 | 仅 1D 整数 | NPU 限制 |
| `tl.arange` | 非 int32 类型 | 仅 int32 | NPU 限制 |
| `// (floordiv)` | fp16/bf16/fp32 | 仅整数 | 浮点地板除不支持 |
| `tl.where` | int64 | 不支持 | API 支持矩阵中 i64 不支持 |
| `atomic_add` | int64/fp64 | int8/int16/int32/fp16/fp32/bf16 | 硬件限制 |
| `atomic_cas` | bf16 (910B) | 910_95 支持 bf16 | 910_95 支持更多类型 |
| 所有 atomic op | `sem` 非 `"acq_rel"` | 仅 `"acq_rel"` | NPU 唯一支持的内存序 |
| 所有 atomic op | `scope` 非 `"gpu"` | 仅 `"gpu"` | NPU 唯一支持的同步范围 |
| 部分 atomic op | loop 内使用 | 避免在循环中使用 atomic_and/or/xchg/xor/cas | NPU 限制 |

### 1.2 行为不同的 API

| API | GPU 行为 | 910_95 行为 | 影响及处理 |
|-----|---------|------------|-----------|
| `tl.load` (mask=False, other=None) | masked 位置值未定义 | `care_padding=True` 时填零；`care_padding=False` 时为随机值 | NPU 扩展参数，默认填零；设 `care_padding=False` 可提升性能 |
| `tl.store` (离散 mask) | 正常写入 | 拆解为 atomic {load, select, store} 组合 | 可能存在精度/正确性问题，建议使用连续 mask 或 block pointer |
| `tl.dot` 精度 | `"tf32"` 使用 19-bit TF32 | `"tf32"` 自动映射为 `"hf32"` | hf32 是 NPU 特有半精度 float32 格式 |
| `tl.dot` 输入精度 | `input_precision="hf32"` 不适用 | 非 fp32 输入时 hf32 被忽略，回退 `"ieee"` | 仅 fp32 输入可使用 hf32 |
| `tl.sum` / `tl.max` / `tl.min` | int8/int16 提升为 int32 归约 | int8/int16 **不提升**，直接归约 | 可能溢出，需手动 `.to(tl.int32)` |
| `tl.sum` / `tl.max` / `tl.min` | bf16 直接归约 | bf16 自动提升为 fp32 归约 | NPU 不支持 bf16 的 FMAX/FMIN/FCMP |
| `tl.maximum` / `tl.minimum` | bf16 直接比较 | bf16 提升为 fp32 比较 | 同上 |
| `/ (div)` | fp16/bf16 直接除 | fp16/bf16 自动提升为 fp32 执行 | NPU Vector 单元不原生支持低精度除法 |
| `tl.max` / `tl.min` | `propagate_nan` 参数 | 支持 `propagate_nan=True/False` | `True` 使用 `maximum`（传播 NaN），`False` 使用 `maxnumf`（不传播） |
| `atomic_max` / `atomic_min` | 浮点需 bitcast 为整数实现 | 硬件直接支持浮点 atomic_max/min | NPU 实现更简洁高效 |
| `tl.cast` | 标准 cast | 扩展 `overflow_mode` 参数（`"saturate"` 等） | NPU 特有扩展 |
| `tl.make_block_ptr` | 可通过调整 stride 顺序实现转置 | 不允许通过调整 stride 顺序实现转置 | 仅允许通过 `order` 参数表达转置语义 |

### 1.3 NPU 特有 API（Ascend 扩展）

以下 API 通过 `from triton.language.extra.cann.extension import *` 导入，习惯使用 `al` 作为别名：

```python
from triton.language.extra.cann import extension as al
```

| API | 类别 | 说明 | 典型用法 |
|-----|------|------|---------|
| `al.compile_hint` | 编译提示 | 指导编译器优化，如 `"dot_pad_only_k"` | `al.compile_hint(tensor, "dot_pad_only_k")` |
| `al.get_element` | 向量操作 | 获取张量中的单个元素 | `al.get_element(tensor, indices)` |
| `al.insert_slice` | 向量操作 | 将子张量插入到张量中 | `al.insert_slice(tensor, sub_tensor, offsets)` |
| `al.extract_slice` | 向量操作 | 从张量中提取子张量 | `al.extract_slice(tensor, offsets, sizes)` |
| `al.sort` | 向量操作 | 沿指定维度排序（替代 tl.sort） | `al.sort(tensor, dim, descending)` |
| `al.flip` | 向量操作 | 沿指定维度翻转 | `al.flip(tensor, dim)` |
| `al.cast` | 类型转换 | 扩展版 cast，支持 `overflow_mode` | `al.cast(tensor, dtype, overflow_mode="saturate")` |
| `al.fixpipe` | Cube 后处理 | L0C 到 UB 的数据搬运（仅 910_95） | `al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2ND)` |
| `al.sync_block_set` | 同步 | 生产者核心发送同步信号 | `al.sync_block_set("cube", "vector", 0)` |
| `al.sync_block_wait` | 同步 | 消费者核心等待同步信号 | `al.sync_block_wait("cube", "vector", 0)` |
| `al.sync_block_all` | 同步 | 全局屏障同步 | `al.sync_block_all("all", 0)` |
| `al.scope` | Cube-Vector 协同 | 指定代码块运行在 Cube 或 Vector 核心 | `with al.scope(core_mode="cube"):` |
| `al.multibuffer` | 存算并行 | 多缓冲设置（ping-pong 流水线） | `al.multibuffer(tensor, 2)` |
| `al.parallel` | 并行迭代 | 并行范围声明，支持 `bind_sub_block` | `for i in al.parallel(0, N, bind_sub_block=True):` |
| `al.index_put` | 内存操作 | 索引写入（UB 到 GM） | `al.index_put(dst, indices, values)` |
| `al.gather_out_to_ub` | 内存操作 | 索引读取（GM 到 UB） | `al.gather_out_to_ub(src, indices)` |
| `al.scatter_ub_to_out` | 内存操作 | 索引散列写入（UB 到 GM） | `al.scatter_ub_to_out(dst, indices, values)` |
| `al.index_select_simd` | 内存操作 | SIMD 并行索引选择 | `al.index_select_simd(src, indices)` |
| `al.int64` | 类型包装 | 用于自定义算子中指定 int64 参数 | `al.int64(value)` |
| `tl.PropagateNan` | 枚举 | NaN 传播控制：`NONE`/`ALL` | `tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)` |
| `al.Pipe` / `al.CORE` / `al.MODE` | 枚举 | 流水线/核心/执行模式枚举 | `al.CORE.CUBE_AND_VECTOR` |
| `al.FixpipeDMAMode` | 枚举 | fixpipe DMA 模式 | `al.FixpipeDMAMode.NZ2ND` |
| `al.FixpipePreQuantMode` | 枚举 | fixpipe 预量化模式 | `al.FixpipePreQuantMode.F322BF16` |
| `al.FixpipePreReluMode` | 枚举 | fixpipe 预 ReLU 模式 | `al.FixpipePreReluMode.NORMAL_RELU` |

---

## 2. 关键 API 详细说明

### 2.1 tl.load / tl.store 的 care_padding 参数

`care_padding` 是 Triton-Ascend 特有扩展参数，控制 `mask=False` 时的填充行为。默认 `care_padding=True` 时 MTE2 需等待 Vector 初始化，设为 `False` 可提升并行度。详细说明和代码示例见 [08-data-type-precision.md](08-data-type-precision.md#5-care_padding-参数详解)。

### 2.2 tl.dot 的 BLOCK_SIZE 对齐要求

910_95 上 `tl.dot` 的分块需满足以下对齐约束：

| 算子类型 | 对齐要求 | 说明 |
|---------|---------|------|
| VV 类（纯 Vector） | 尾轴 32B 对齐 | fp16: 尾轴元素数为 16 的倍数；fp32: 8 的倍数 |
| CV 类（Cube+Vector） | 尾轴 512B 对齐 | fp16: 尾轴元素数为 256 的倍数；fp32: 128 的倍数 |

**具体到 tl.dot 的 BLOCK_SIZE**：

```python
# CV 类算子（含 tl.dot）的 512B 对齐
# fp16/bf16 (2 bytes): BLOCK_K 须为 256 的倍数
# fp32 (4 bytes): BLOCK_K 须为 128 的倍数

# 推荐分块配置
BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 256  # fp16/bf16 时满足 512B 对齐
```

**compile_hint 优化**：当 mask 仅在 K 维度存在越界时，使用 `al.compile_hint` 可减少不必要的 padding：

```python
a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
al.compile_hint(a, "dot_pad_only_k")
b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
al.compile_hint(b, "dot_pad_only_k")
accumulator = tl.dot(a, b, accumulator)
```

### 2.3 归约操作（tl.sum / tl.max / tl.min）的精度要求

| 输入类型 | NPU 行为 | 风险 | 建议 |
|---------|---------|------|------|
| fp32 | 直接归约 | 无 | 正常使用 |
| fp16 | `tl.sum` 直接 fp16 归约；`tl.max`/`tl.min` 自动提升为 fp32 | `tl.sum` 需手动 `.to(tl.float32)` | 源码：sum 仅提升 bf16，max/min 对 bitwidth<32 的浮点自动提升 |
| bf16 | 自动提升为 fp32 归约 | 无 | 正常使用 |
| int32 | 直接归约 | 无 | 正常使用 |
| int8 / int16 | **不提升为 int32**，直接归约 | **可能溢出** | 手动 `.to(tl.int32)` 后归约 |

```python
# int8 归约可能溢出，需手动提升
x_int8 = tl.load(x_ptr + offsets, mask=mask, other=0)  # int8
s = tl.sum(x_int8.to(tl.int32), axis=0)  # 提升为 int32 后归约
```

### 2.4 tl.where 的标量退化问题

NPU 的 Vector CMP 不支持 int64 和 int32 类型，当 `tl.where` 的条件比较使用这些类型时，会退化为 scalar 计算，性能严重下降。

**典型场景**：使用 `tl.arange` 生成索引进行边界比较

```python
# 问题代码：cols 是 int32（tl.arange 返回 int32），cols < N 退化为 scalar CMP
cols = tl.arange(0, BLOCK_N)
xbar = tl.where(cols < N, x - mean, 0.0)

# 修复：转换为 fp32 使用 Vector CMP
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

**诊断方法**：使用 msProf 查看 `aiv_scalar_ratio`，若异常高则可能存在标量退化。

---

## 3. 910_95 特别注意

910_95 与 910B 的关键差异速查：

| 维度 | 910B | 910_95 |
|------|------|--------|
| UB 容量 | 192KB / 96KB(double buffer) | 256KB / 128KB(double buffer) |
| L0C 容量 | 128KB | 256KB |
| fixpipe | 不支持 | 支持 L0C -> UB 直通 |
| MultiBuffer | 默认开启 | 默认关闭 |
| FP8 | 不支持 | 支持 dot_scaled 和类型转换 |

> 完整硬件规格详见 [00-hardware-quick-ref.md](00-hardware-quick-ref.md)。

---

## 4. 数据类型速查

### 4.1 不支持的数据类型

以下类型在 910_95 上**普遍不支持**（硬件限制）：

| 类型 | 替代方案 |
|------|---------|
| uint8 | int8（注意符号位和范围差异） |
| uint16 | int16 |
| uint32 | int32 |
| uint64 | int64 |
| fp64 | fp32（精度降低，需评估影响） |

### 4.2 类型提升规则（NPU 特有）

| 场景 | NPU 行为 |
|------|---------|
| bf16 + 非bf16 类型运算 | 自动提升为 fp32 |
| bf16 除法 | 自动提升为 fp32 |
| fp16 除法 | 自动提升为 fp32 |
| bf16 比较 / maximum / minimum | 自动提升为 fp32 |
| bf16 归约 | 自动提升为 fp32 |
| int8/int16 归约 | **不提升**（与 GPU 不同） |

### 4.3 性能相关类型选择

| 操作 | 不推荐类型 | 推荐类型 | 原因 |
|------|-----------|---------|------|
| Vector ADD | int64 | int32 | int64 退化为 scalar |
| Vector CMP | int64/int32 | fp32 | int64/int32 退化为 scalar |
| 矩阵乘输入 | - | bf16 | bf16 范围与 fp32 一致，不易溢出 |
| 矩阵乘累加 | fp16 | fp32 | 硬件默认 fp32 累加 |

---

## 5. 常见迁移陷阱速查

| 陷阱 | 症状 | 解决方案 |
|------|------|---------|
| `tl.where` 标量退化 | aiv_scalar_ratio 异常高 | 将比较操作数转为 fp32 |
| int8/int16 归约溢出 | 归约结果错误 | 归约前手动 `.to(tl.int32)` |
| UB 溢出 | `ub overflow` 编译错误 | 减小 BLOCK_SIZE 或关闭 MultiBuffer |
| coreDim 超限 | `coreDim > UINT16_MAX` | 增大 BLOCK_SIZE 或 `TRITON_ALL_BLOCKS_PARALLEL=1` |
| 尾轴不对齐 | 性能严重下降 | VV: 32B 对齐，CV: 512B 对齐 |
| 离散 mask store | 正确性问题 | 使用连续 mask 或 block pointer |
| `tl.sort` 不可用 | 编译错误 | 使用 `al.sort`（Ascend 扩展） |
| `out_dtype=bfloat16` | ValueError | 使用 float32 后手动 `.to(tl.bfloat16)` |
| `cache_modifier` 无效 | 无效果 | NPU 无 GPU cache 层次，忽略即可 |
| atomic_add 多核累加 | 结果不正确 | NPU 不支持多核 add+保存中间结果 |

---

## 6. 相关文档链接

- [01-memory-ops.md](../docs_triton_ascend/02-Core-API/01-memory-ops.md) - tl.load/tl.store 详细说明
- [02-math-ops.md](../docs_triton_ascend/02-Core-API/02-math-ops.md) - 数学运算 API
- [03-reduction-ops.md](../docs_triton_ascend/02-Core-API/03-reduction-ops.md) - 归约操作 API
- [04-linear-algebra-ops.md](../docs_triton_ascend/02-Core-API/04-linear-algebra-ops.md) - tl.dot 详细说明
- [05-atomic-ops.md](../docs_triton_ascend/02-Core-API/05-atomic-ops.md) - 原子操作 API
- [06-shape-ops.md](../docs_triton_ascend/02-Core-API/06-shape-ops.md) - 形状操作 API
- [07-scan-sort-ops.md](../docs_triton_ascend/02-Core-API/07-scan-sort-ops.md) - 扫描与排序操作
- [08-comparison-logical-ops.md](../docs_triton_ascend/02-Core-API/08-comparison-logical-ops.md) - 比较与逻辑操作
- [01-api-support-matrix.md](../docs_triton_ascend/09-Reference/01-api-support-matrix.md) - API 支持矩阵
- [01-extension-overview.md](../docs_triton_ascend/03-Ascend-Extensions/01-extension-overview.md) - Ascend 扩展 API 总览
- [03-common-issues.md](../docs_triton_ascend/06-Migration-from-GPU/03-common-issues.md) - 迁移常见问题
- [04-runtime-errors.md](../docs_triton_ascend/07-Debugging/04-runtime-errors.md) - 运行时错误排查
- [02-tiling-strategy.md](../docs_triton_ascend/05-Performance-Optimization/02-tiling-strategy.md) - 分块策略
- [03-memory-model.md](../docs_triton_ascend/01-Programming-Model/03-memory-model.md) - NPU 内存层次
- [04-data-types.md](../docs_triton_ascend/01-Programming-Model/04-data-types.md) - 数据类型支持

### 源码参考

- [core.py - load/store 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1589-L1726)
- [core.py - dot 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1518-L1581)
- [core.py - PropagateNan 枚举](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L24)
- [core.py - maximum/minimum 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1954-L1975)
- [semantic.py - dot 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1559-L1626)
- [semantic.py - 类型提升规则](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L46-L109)
- [aux_ops.py - compile_hint 实现](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/aux_ops.py#L114-L151)
- [vec_ops.py - al.sort/al.cast 实现](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py)
- [compiler.py - min_dot_size](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L68-L69)
