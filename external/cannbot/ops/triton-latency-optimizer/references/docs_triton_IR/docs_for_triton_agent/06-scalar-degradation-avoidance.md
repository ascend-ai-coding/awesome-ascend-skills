# 标量降级规避指南

## 触发条件

当 agent 发现 Triton kernel 性能远低于预期（如 10x-100x 量级性能损失），应首先排查是否存在标量降级（Scalar Lowering）。典型症状包括：

- kernel 执行时间与纯 CPU 标量循环相当，远未达到 NPU 向量加速预期
- 编译产物 HIVM IR 中出现大量 `scf.for` 循环包裹 `arith` 标量操作
- 涉及 i64 类型、整数比较、或特定归约模式的计算路径

---

## 什么是标量降级

标量降级是指 HIVM 向量操作在编译过程中，因硬件向量计算单元不支持特定数据类型与操作组合，被编译器自动退化为逐元素的标量循环（`scf.for` + `arith` 标量操作）执行。

**性能影响**：向量指令一次处理多个数据元素，标量循环逐元素执行，性能损失通常在 **10x-100x** 量级。

**降级机制**：编译器通过 `ImplByScalarOpInterface` 接口判断 `shouldLowerToScalarLoops()` 返回值，为 true 时由 `HIVMLowerToLoopsPass` 执行降级。

> 标量降级是功能等价变换，不影响计算正确性，仅影响性能。不建议通过编译选项禁用降级，因为硬件不支持对应向量操作，强制禁用会导致后续编译阶段失败。

---

## 910_95 vs 910B 标量降级差异

910_95（Ascend910_95xx 系列，如 Ascend910_9589）属于 **Reg-based 架构**，与 910B（Mem-based 架构）在归约操作的降级条件上有显著差异。

| 架构分类 | 代表芯片 | 核间同步机制 | 判定函数 |
|---------|---------|------------|---------|
| **Reg-based** | Ascend910_95xx, Ascend310B, Ascend950 | 寄存器级指令（SetFlag/WaitFlag） | `isRegBasedArch()` |
| **Mem-based** | Ascend910B, Ascend910_93 | FFTS 内存机制 | `isMemBasedArch()` |

### 关键差异：归约操作降级条件

| reduceOp | 910_95（Reg-based） | 910B（Mem-based） |
|----------|-------------------|------------------|
| sum / prod / max / min / xori | **不降级** | 仅 i64 降级 |
| max_with_index / min_with_index | 仅内存对齐不合法时降级 | i16/i32/i64 降级；f16/f32/bf16 高维降级 |
| any / all / ori / andi / none | **不降级** | **不降级** |

**910_95 的核心优势**：基本归约（sum/prod/max/min/xori）在所有数据类型下均不触发标量降级，包括 i64。这意味着 910_95 上 i64 归约可以走向量路径，而 910B 上 i64 归约会退化为标量循环。

**910_95 的注意事项**：argmax/argmin（max_with_index/min_with_index）在内存访问对齐不合法时仍会降级。对齐合法性取决于操作数和结果的步幅布局是否满足对齐要求，缺少 `StridedLayoutAttr` 时默认判定为不合法。

> 源码参考：[ShouldLowerToScalarLoops.cpp:221-272](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L221-L272)

---

## 标量降级条件速查表

### 通用算术操作（VAdd/VSub/VMul/VMin/VMax/VAbs/VShL/VShR/VInterleave/VDeinterleave）

| 元素类型 | SIMD VF 模式 | SIMT VF 模式 |
|---------|-------------|-------------|
| f16/f32 | 不降级 | **降级** |
| i8/i16/i32 | 不降级 | **降级** |
| i64 | **降级** | **降级** |

降级条件：`hasPureBufferSemantics() AND (isSIMTVF() OR elemType == i64)`

### 比较操作（VCmp）

| 元素类型 | EQ | NE | LT/GT/LE/GE |
|---------|----|----|-------------|
| f16/f32 | 不降级 | 不降级 | 不降级 |
| i32 | 不降级 | 不降级 | **降级** |
| i8/i16/i64 | **降级** | **降级** | **降级** |

降级条件：`hasPureBufferSemantics() AND 整数类型 AND (非 i32 或 非 EQ/NE)`

### 扩展乘法（VMulExt）

| 元素类型 | 是否降级 |
|---------|---------|
| i32 | **降级**（始终） |
| i64 | **降级**（始终） |

### 累积操作（VCumsum/VCumprod）

| 元素类型 | 累积维度 = 最后维度 | 累积维度 != 最后维度 | 多个累积维度 |
|---------|-------------------|-------------------|------------|
| f16/f32/bf16/i8/i16/i32 | **降级** | 不降级 | 不降级 |
| i64 | **降级** | **降级** | 不降级 |

### 归约操作（VReduce）

#### 910_95（Reg-based）

| reduceOp | 降级条件 |
|----------|---------|
| sum/prod/max/min/xori | **不降级** |
| max_with_index/min_with_index | 仅内存对齐不合法时降级 |

#### 910B（Mem-based）

| reduceOp | 降级条件 |
|----------|---------|
| sum/prod/max/min/xori | i64 降级 |
| max_with_index/min_with_index | i16/i32/i64 降级；f16/f32/bf16 且 flatten rank > 2 降级 |

---

## 代码模式与规避方法

### 场景 1：int64 向量加法 -> int32 转换

**触发条件**：向量加法/减法/乘法等逐元素运算，操作数类型为 i64（包括 Python `int` 默认的 int64）

**降级原因**：VADD/VSUB/VMUL 等指令不支持 i64，退化为标量循环

**规避方法**：在 load 后立即转换为 int32，运算完成后再按需转回 int64

```python
# 退化代码：i64 加法触发标量降级
a = tl.load(ptr_a + offsets)   # 类型为 int64
b = tl.load(ptr_b + offsets)   # 类型为 int64
c = a + b                      # 退化为 scalar 计算

# 规避代码：转为 int32 走向量路径
a = tl.load(ptr_a + offsets).to(tl.int32)
b = tl.load(ptr_b + offsets).to(tl.int32)
c = a + b                      # 启用 VADD 向量指令
# 如需返回 int64，运算后再转回
c = c.to(tl.int64)
```

**性能收益**：数倍至数十倍（取决于数据规模）

**风险**：数据溢出。必须确认原始数据在 `[-2^31, 2^31-1]` 范围内


### 场景 2：整数比较 -> fp32 比较

**触发条件**：涉及整数类型的比较操作，特别是：
- `tl.where` 中的条件判断（如 `cols < N`）
- 条件选择操作
- 边界处理中的索引比较

**降级原因**：除 i32 的 EQ/NE 外，所有整数比较（LT/GT/LE/GE 及 i8/i16/i64 的全部比较）都触发标量降级

**规避方法**：将比较操作数转换为 fp32 后再比较

```python
# 退化代码：整数比较触发标量降级
cols = tl.arange(0, BLOCK_N)          # cols 类型为 i64
xbar = tl.where(cols < N, x - mean, 0.0)  # 退化为 scalar 计算

# 规避代码：转为 fp32 后比较
cols = tl.arange(0, BLOCK_N)
cols_cmp = cols.to(tl.float32)        # 转换为 fp32
xbar = tl.where(cols_cmp < N, x - mean, 0.0)  # 启用 vector 计算
```

**性能收益**：显著提升，尤其在处理不规则数据块时效果明显

**注意事项**：
- `tl.load`/`tl.store` 的 mask 参数中使用比较时，编译器通常会自动优化，无需手动转换
- fp32 可精确表示整数范围 `[-2^24, 2^24]`（约 16M），超出此范围可能丢失精度
- 对于 i32 的 EQ/NE 比较，无需转换，已走向量路径


### 场景 3：tl.where 中的整数条件退化

**触发条件**：`tl.where(condition, A, B)` 中 condition 涉及整数比较

**降级原因**：tl.where 内部生成 vcmp + vsel 模式，整数比较的 vcmp 降级使整个条件选择链路性能下降

**规避方法 A**：将条件中的整数转为 fp32（同场景 2）

```python
# 退化代码
result = tl.where(idx >= start, A, B)    # idx 为整数，退化为 scalar

# 规避代码
result = tl.where(idx.to(tl.float32) >= start, A, B)  # 走 vector 路径
```

**规避方法 B**：当 condition 仅在一个位置为 false 时，使用 get_element + insert_slice 替代

```python
# 退化代码：condition 仅在一个位置为 False
X = base + tl.arange(0, BLOCK_SIZE)
A = tl.load(A_ptr + X)
B = tl.load(B_ptr + X)
condition = (X != y)
A = tl.where(condition, A, B)  # 退化为 scalar 计算

# 规避代码：用 get_element + insert_slice 替代 where
try:
    import triton.language.extra.cann.extension as extension
except Exception:
    extension = tl

X = base + tl.arange(0, BLOCK_SIZE)
A = tl.load(A_ptr + X)
B = tl.load(B_ptr + X)
if base <= y < base + BLOCK_SIZE:
    _offs = y - base
    _val = extension.get_element(B, (_offs,))
    _tensor = tl.full((1,), _val, dtype=A.dtype)
    A = extension.insert_slice(A, _tensor, offsets=(_offs,), sizes=(1,), strides=(1,))
```

**性能收益**：消除 where 退化带来的 scalar 计算开销，避免离散访存


### 场景 4：归约操作的降级

**触发条件**：归约操作（tl.sum/tl.max/tl.min/tl.argmax/tl.argmin 等）在特定数据类型和架构下触发降级

#### 910_95 上的归约规避

910_95 对基本归约（sum/prod/max/min/xori）不降级，包括 i64 类型。**主要关注 argmax/argmin 的对齐问题**。

```python
# 910_95 上 argmax/argmin 可能因对齐问题降级
# 确保数据布局满足对齐要求，避免缺少 StridedLayoutAttr

# 如果 argmax/argmin 性能异常，检查输入 tensor 的 stride 信息
# 确保编译器能推断出合法的内存访问对齐
```

#### 910B 上的归约规避

```python
# 退化代码：i64 归约在 910B 上降级
x_i64 = tl.load(ptr + offsets)  # int64
result = tl.sum(x_i64)          # 退化为 scalar 计算

# 规避代码：转为 fp32 归约
x_fp32 = tl.load(ptr + offsets).to(tl.float32)
result = tl.sum(x_fp32)         # 走 vector 路径
# 如需整数结果再转回
result = result.to(tl.int64)
```

**910B 上 argmax/argmin 的规避**：

```python
# 退化代码：整数 argmax 在 910B 上始终降级
idx = tl.argmax(x_int, axis=0)   # i16/i32/i64 均降级

# 规避代码：转为 fp32 后 argmax
idx = tl.argmax(x_int.to(tl.float32), axis=0)  # 低维不降级
```

**性能收益**：从标量循环到向量归约，性能提升 10x-100x

### 场景 5：累积操作的降级

**触发条件**：cumsum/cumprod 操作中累积维度为 flatten 后的最后维度，或元素类型为 i64

**规避方法**：调整数据布局，使累积维度不在最后

```python
# 退化代码：累积维度为最后维度
# 假设 x shape 为 (M, N)，对 axis=-1 做 cumsum
result = tl.cumsum(x, axis=-1)  # 累积维度为最后维度，降级

# 规避代码：转置后累积，再转置回来
x_t = tl.trans(x)               # shape 变为 (N, M)
result_t = tl.cumsum(x_t, axis=-1)  # 累积维度不再是最后维度
result = tl.trans(result_t)      # 转回原始布局
```

**注意**：i64 类型的累积操作无论维度如何都会降级，应避免使用 i64 累积

### 场景 6：扩展乘法（VMulExt）

**触发条件**：使用高精度乘法（vmulext）

**降级原因**：vmulext 在 IR 层面仅支持 I32，而 I32 恰好触发标量降级，该操作实际上始终走标量路径

**规避方法**：使用 vmulextended（I16 输入）或手动拆分乘法

```python
# vmulext 始终降级，应避免使用
# 替代方案：使用 I16 输入的扩展乘法，或手动拆分
```

---

## 910_95 特别注意

### 1. Reg-based 架构的归约优势

910_95 属于 Reg-based 架构（通过 `isAscend950()` 判定），核间同步通过寄存器级指令实现，归约操作有更好的硬件向量支持：

- **基本归约不降级**：sum/prod/max/min/xori 在所有数据类型（包括 i64）下均走向量路径
- 这意味着 910_95 上 **i64 归约无需规避**，与 910B 的行为不同

### 2. argmax/argmin 对齐问题

910_95 上 argmax/argmin（max_with_index/min_with_index）仅在内存访问对齐不合法时降级。对齐合法性由 [isLegalAccessAlignment](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L139-L219) 判断：

- 检查操作数和结果的步幅布局是否满足 32 Byte 对齐要求
- **缺少 `StridedLayoutAttr` 时默认判定为不合法**，会触发降级
- 这是一个临时 workaround，后续编译器改进后将移除此限制

**规避建议**：如果 910_95 上 argmax/argmin 性能异常，检查输入 tensor 的 stride 信息是否完整

### 3. 910_95 设备型号列表

910_95 系列设备均属于 Reg-based 架构，包括但不限于：Ascend910_950z, Ascend910_9579, Ascend910_957b, Ascend910_957d, Ascend910_9581, Ascend910_9589, Ascend910_958a, Ascend910_958b, Ascend910_9599 等。

> 源码参考：[Utils.cpp:232-260](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HACC/Utils/Utils.cpp#L232-L260)

### 4. 910_95 降级条件差异汇总

| 操作类别 | 910_95 降级条件 | 910B 降级条件 | 差异说明 |
|---------|---------------|-------------|---------|
| 通用算术 | i64 或 SIMT VF | i64 或 SIMT VF | 无差异 |
| 比较操作 | 整数类型且(非 i32 或非 EQ/NE) | 同左 | 无差异 |
| 扩展乘法 | 始终降级 | 始终降级 | 无差异 |
| 累积操作 | i64 或累积维度为最后维度 | 同左 | 无差异 |
| **基本归约** | **不降级** | **i64 降级** | **核心差异** |
| **argmax/argmin** | **仅对齐不合法时降级** | **i16/i32/i64 降级；f16/f32/bf16 高维降级** | **核心差异** |

---

## 相关文档链接

### 源文档
- [HIVM 向量操作标量降级](../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md) — 标量降级机制完整文档

### 编译器源码
- [ShouldLowerToScalarLoops.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp) — 降级判断逻辑核心实现
- [LowerToLoops.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/LowerToLoops.cpp) — 降级实现（scf.for 循环生成）
- [ImplByScalarOpInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.td) — 接口定义
- [Utils.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HACC/Utils/Utils.cpp) — 架构判定函数（isRegBasedArch/isMemBasedArch/isAscend950）

### HIVM 操作详细文档
- [Binary Ops](../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md)（VAddOp, VSubOp, VMulOp, VMinOp, VMaxOp）
- [Compare Ops](../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/05-compare-ops.md)（VCmpOp）
- [Reduction Ops](../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/07-reduction-ops.md)（VReduceOp）
- [Cumulative Sort](../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/09-cumulative-sort.md)（VCumsumOp, VCumprodOp）
- [Special Ops](../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md)（VMulExtOp）
