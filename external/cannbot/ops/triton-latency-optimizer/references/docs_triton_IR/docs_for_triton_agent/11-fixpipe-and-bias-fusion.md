# FixPipe 随路操作与 Bias 融合优化

## 触发条件

当 Agent 处理包含以下模式的算子时，应考虑应用本文档中的优化：

1. **`tl.dot` + bias 累加**：矩阵乘法后对累加器加上 bias（`accumulator += b`）
2. **`tl.dot` + `tl.trans`**：矩阵乘法参数中同时出现 `tl.trans()` 和 `.to(dtype)`，且 `tl.trans()` 在 `.to(dtype)` 之前
3. **Cube 计算结果需后处理**：矩阵乘法结果需要量化、ReLU 激活、布局转换等后处理

---

## 一、Bias 融合优化

### 1.1 问题背景

NPU 采用 CV（Cube-Vector）分离架构。矩阵乘法在 Cube 单元执行，结果存储在 L0C 缓冲区。如果 bias 累加操作在 Vector 核上执行，需要将 Matmul 结果从 L0C 搬运到 UB 再由 Vector 计算，产生额外的数据搬运与 CV 间同步开销。

通过调整 bias 的加载时机和广播写法，使 AscendNPU-IR 识别并将 bias 累加融合到 Cube 流水线中，可以消除额外开销。

### 1.2 触发条件

代码中存在 `tl.dot` + bias 累加模式：先通过 `tl.dot` 计算矩阵乘法，然后对累加器 `accumulator` 加上 bias。

### 1.3 代码对比

**优化前（bias 在 matmul 后加载，隐式广播）：**

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K
    w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N
    x = tl.load(
        x_ptrs,
        mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    w = tl.load(
        w_ptrs,
        mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    accumulator = tl.dot(x, w, accumulator)

if HAS_BIAS:
    b_ptrs = b_ptr + offset_wn
    b = tl.load(b_ptrs, mask=offset_wn < N, other=0.0)
    accumulator += b[None, :]  # 隐式广播

y = accumulator.to(dtype)
```

**优化后（提前加载 bias，使用显式广播）：**

```python
if HAS_BIAS:
    offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    b_ptrs = b_ptr + offset_wn
    b = tl.load(b_ptrs, mask=offset_wn < N, other=0.0)

accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K
    w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N
    x = tl.load(
        x_ptrs,
        mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    w = tl.load(
        w_ptrs,
        mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    accumulator = tl.dot(x, w, accumulator)

if HAS_BIAS:
    accumulator += tl.broadcast_to(b[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))

y = accumulator.to(dtype)
```

### 1.4 关键改动点

| 改动 | 说明 |
|------|------|
| 提前加载 bias | 将 bias 的 `tl.load` 移到 matmul 循环之前，使 AscendNPU-IR 能在编译期识别出 Cube 累加模式 |
| 使用显式广播 | 将隐式的 `b[None, :]` 替换为 `tl.broadcast_to(b[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))`，使广播 shape 明确，帮助编译器正确生成 Cube 指令 |
| bias 加载不受 `for k` 影响 | bias 是沿 N 维的数据，与 K 维循环无关，提前加载在语义上完全等价 |

### 1.5 性能收益

| 方面 | 优化前 | 优化后 |
|------|--------|--------|
| bias 累加执行位置 | Vector 核执行，存在 Cube -> UB -> Vector 的数据搬运 | Cube 流水线内完成，消除跨核数据搬运 |
| CV 间同步 | Cube 与 Vector 间需要同步等待 | 无跨核同步，计算流水更紧凑 |

### 1.6 注意事项

- `tl.broadcast_to` 的目标 shape 必须与 `accumulator` 的 shape 一致，即 `(BLOCK_SIZE_M, BLOCK_SIZE_N)`
- 优化后 `offset_wn` 需在 `if HAS_BIAS` 内提前计算，确保指针计算与原逻辑一致
- 改动仅影响 IR 模式识别，不改变数值结果，可通过精度测试验证

---

## 二、Vec 到 Cube 随路转置优化

### 2.1 问题背景

在 Cube&Vector 融合算子中，`tl.trans()` 转置操作通常在 Vector 单元执行，产生额外开销。Cube 单元在从 Global Memory 加载数据时支持随路完成转置（不需要额外时间开销）。但如果在转置之前执行了 `.to(dtype)` 类型转换，该转换在 Vector 单元执行，导致 `tl.trans()` 也只能跟在 Vector 单元之后完成，无法利用 Cube 的随路转置能力。

### 2.2 触发条件

代码中同时出现以下特征：

1. **Cube&Vector 融合算子**：kernel 既包含 `tl.dot`（Cube 运算）又包含 Vector 运算
2. **`tl.trans()` 和 `.to(dtype)` 同时出现，且 `tl.trans()` 在 `.to(dtype)` 之前**，即以下代码模式之一：
   - `tl.dot(tl.trans(A).to(dtype), B)`
   - `tl.dot(A, tl.trans(B).to(dtype))`

### 2.3 代码对比

**优化前（`tl.trans()` 在前，`.to(dtype)` 在后）：**

```python
@triton.jit
def kernel(input1, input2, ...):
    # ...
    C = tl.dot(tl.trans(A).to(dtype), B)
    # ...
```

**优化后（`.to(dtype)` 在前，`tl.trans()` 在后）：**

```python
@triton.jit
def kernel(input1, input2, ...):
    # ...
    A = A.to(dtype)
    C = tl.dot(tl.trans(A), B)
    # ...
```

### 2.4 关键改动点

| 改动 | 说明 |
|------|------|
| 调整操作顺序 | 将 `.to(dtype)` 提到 `tl.trans()` 之前，仅调整顺序，不改变计算语义 |
| 仅对 `tl.dot` 参数有效 | 只有 Cube 单元加载数据时才有随路转置能力 |
| 左右参数均适用 | `tl.dot` 的第一个参数和第二个参数都可以应用此优化 |

### 2.5 性能收益

| 方面 | 优化前（`tl.trans().to()`） | 优化后（`.to() + tl.trans()`） |
|------|---------------------------|------------------------------|
| 类型转换位置 | Vector 单元执行 `.to(dtype)` | Vector 单元执行 `.to(dtype)` |
| 转置位置 | Vector 单元执行转置 | Cube 单元随路完成转置 |
| 端到端耗时 | 较长（额外搬运开销） | 较短（消除搬运开销） |

### 2.6 注意事项

- 如果 `tl.trans()` 和 `.to(dtype)` 只出现其中一个，不需要调整顺序
- 调整顺序后精度不会变化，先做类型转换再做转置与先做转置再做类型转换在数学上完全等价
- 纯 Cube 算子或纯 Vector 算子不需要此优化

---

## 三、FixPipe 随路操作汇总

FixPipe 是昇腾 NPU 上的关键硬件流水线，负责将 Cube 计算结果从 L0C 缓冲区搬运到其他存储层级，同时可选地执行格式转换、量化和 ReLU 激活等后处理操作。数据通路如下：

```
Cube 计算 (PIPE_M) -> L0C -> fixpipe (PIPE_FIX) -> UB/GM/L1 -> Vector 后处理 (PIPE_V)
```

### 3.1 随路操作一览

| 随路功能 | 枚举类型 | 可选值 | 说明 |
|----------|----------|--------|------|
| 布局转换 | `FixpipeDMAMode` | `NZ2ND` | NZ 格式转换为 ND 格式（最常用） |
| | | `NZ2DN` | NZ 格式转换为 DN 格式（仅 910_95） |
| | | `NZ2NZ` | 保持 NZ 格式直接搬运（默认） |
| 预量化 | `FixpipePreQuantMode` | `NO_QUANT` | 不做量化（默认） |
| | | `F322BF16` | float32 转 bfloat16 |
| | | `F322F16` | float32 转 float16 |
| | | `S322I8` | int32 转 int8 |
| 预激活 | `FixpipePreReluMode` | `NO_RELU` | 不做 ReLU（默认） |
| | | `NORMAL_RELU` | 标准 ReLU |
| | | `LEAKY_RELU` | Leaky ReLU |
| | | `P_RELU` | P-ReLU |
| 双目标输出 | `FixpipeDualDstMode` | `NO_DUAL` | 单目标输出（默认） |
| | | `ROW_SPLIT` | 按 M 维度拆分，M/2 x N 写入每个 UB |
| | | `COLUMN_SPLIT` | 按 N 维度拆分，M x N/2 写入每个 UB |

### 3.2 Python API

```python
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl

al.fixpipe(
    src,                    # tl.tensor，必须位于 L0C 内存区域
    dst,                    # bl.buffer，必须位于 UB 内存区域
    dma_mode=al.FixpipeDMAMode.NZ2ND,          # DMA 传输模式
    dual_dst_mode=al.FixpipeDualDstMode.NO_DUAL,  # 双目标输出模式
)
```

> 注意：当前 Python 层的 `fixpipe` 函数固定使用 `NO_QUANT` 和 `NO_RELU`，预量化和预 ReLU 通过内部语义层调用。未来版本可能会开放这些参数。

### 3.3 代码示例：基本 fixpipe 使用

```python
@triton.jit
def matmul_fixpipe_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
    al.fixpipe(acc, ub_buf, dma_mode=al.FixpipeDMAMode.NZ2ND)

    acc_tensor = bl.to_tensor(ub_buf)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc_tensor)
```

### 3.4 MLIR IR 示例

**NZ2ND 布局转换：**

```mlir
hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
                  ins(%l0c : memref<256x128xf16>)
                  outs(%gmCSubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
```

**随路量化（F322F16）：**

```mlir
%ret = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
                         ins(%l0c : tensor<256x128xf32>)
                         outs(%dst : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
```

**随路激活（Leaky ReLU）：**

```mlir
%ret = hivm.hir.fixpipe {pre_relu = #hivm.fixpipe_pre_relu_mode<LEAKY_RELU>}
                         ins(%l0c : tensor<256x128xf16>)
                         outs(%dst : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
```

### 3.5 对齐约束

| 数据位宽 | 条件 | 对齐要求 |
|----------|------|----------|
| 32 位（float32/int32） | 最后一维 | 对齐到 8 |
| 32 位（非 NZ2ND 模式） | 最后一维 | 对齐到 16 |
| 32 位（COLUMN_SPLIT 模式） | 最后一维 | 对齐到 32 |
| 32 位（NZ2DN 模式） | 第一维 | 对齐到 8 |
| 16 位（float16/int16/bfloat16） | 最后一维 | 对齐到 16 |
| 16 位（NZ2DN 模式） | 第一维 | 对齐到 16 |

---

## 四、910_95 特别注意

### 4.1 L0C -> UB 直通通路

910_95 系列上，FixPipe 支持将数据直接从 L0C 搬运到 UB（Unified Buffer），实现 Cube 计算到 Vector 后处理的零拷贝流水线。这是 910_95 的核心优势之一。

**对比其他系列：**

| 平台 | L0C -> GM | L0C -> L1 | L0C -> UB |
|------|-----------|-----------|-----------|
| A2/A3 | 支持 | 支持 | 不支持 |
| 910_95 | 支持 | 支持 | **支持** |

在 A2/A3 系列上，Cube 结果需先搬运到 GM，再由 Vector 从 GM 加载到 UB，存在额外搬运开销。910_95 的 L0C -> UB 直通通路消除了这一中间环节。

**MLIR IR 示例（L0C 到 UB 搬运）：**

```mlir
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_fixpipe_l0c_to_ub() {
    %ub_buf = memref.alloc() : memref<1024x2048xf16, #hivm.address_space<ub>>
    %subview = memref.subview %ub_buf[0, 0] [256, 128] [1, 1]
               : memref<1024x2048xf16, #hivm.address_space<ub>>
                 to memref<256x128xf16, strided<[2048, 1]>, #hivm.address_space<ub>>
    %l0c = memref.alloc() : memref<256x128xf16, #hivm.address_space<cc>>
    hivm.hir.fixpipe ins(%l0c : memref<256x128xf16, #hivm.address_space<cc>>)
                     outs(%subview : memref<256x128xf16, strided<[2048, 1]>, #hivm.address_space<ub>>)
    return
  }
}
```

### 4.2 dual_dst_mode（双目标输出模式）

910_95 独有的双目标输出模式允许 FixPipe 将一个 Cube 结果拆分为两个输出，分别写入不同的 UB 缓冲区，常用于 Mix 模式下的 Cube-Vector 协同计算。

| 模式 | 拆分方式 | 输出形状 | 约束 |
|------|----------|----------|------|
| `ROW_SPLIT` | 按 M 维度拆分 | M/2 x N 写入每个 UB | M 须为 2 的倍数 |
| `COLUMN_SPLIT` | 按 N 维度拆分 | M x N/2 写入每个 UB | N 须为 32 的倍数 |

**使用约束：**

- 仅在 `dma_mode` 为 `NZ2ND` 或 `NZ2NZ` 时可用
- 仅在数据从 L0C 搬运到 UB 时可用
- 仅在 910_95 系列上支持
- 32 位数据 `COLUMN_SPLIT` 模式下最后一维须对齐到 32

**MLIR IR 示例：**

```mlir
%l0c = memref.alloc() : memref<16x16xf16, #hivm.address_space<cc>>
%ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
hivm.hir.fixpipe ins(%l0c : memref<16x16xf16, #hivm.address_space<cc>>)
                 outs(%ub : memref<16x16xf16, #hivm.address_space<ub>>)
                 dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>
```

### 4.3 NZ2DN 模式

`NZ2DN` 布局转换模式仅在 910_95 上支持，可将 NZ 格式数据转换为 DN 格式后搬运。

```mlir
hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>}
                  ins(%l0c : memref<256x128xf16>)
                  outs(%dst : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
```

### 4.4 Cube-Vector 协同中的 FixPipe

在 910_95 的 Cube-Vector 协同模式中，FixPipe 是连接 Cube 和 Vector 的桥梁。Cube 完成计算后通过 FixPipe 将数据写入 UB，然后通过同步指令通知 Vector 核心数据已就绪。

```python
@triton.jit
def cube_vector_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    with al.scope(core_mode="cube"):
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + ...)
            b = tl.load(b_ptr + ...)
            acc = tl.dot(a, b, acc)

        ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
        al.fixpipe(acc, ub_buf)
        al.sync_block_set("cube", "vector", 0)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        acc_tensor = bl.to_tensor(ub_buf)
        result = tl.sigmoid(acc_tensor)
        tl.store(c_ptr + ..., result)
```

对应的 MLIR IR 中同步流程：

```mlir
hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot_result : tensor<32x32xf32>) outs(%ub_buf : memref<32x32xf32, #hivm.address_space<ub>>)
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
```

### 4.5 平台限制汇总

| 特性 | A2/A3 | 910_95 |
|------|-------|--------|
| L0C -> GM 搬运 | 支持 | 支持 |
| L0C -> L1 搬运 | 支持 | 支持 |
| L0C -> UB 直通 | 不支持 | **支持** |
| NZ2ND 布局转换 | 支持 | 支持 |
| NZ2DN 布局转换 | 不支持 | **支持** |
| dual_dst_mode | 不支持 | **支持** |
| 预量化 | 支持 | 支持 |
| 预 ReLU | 支持 | 支持 |

> **重要**：在非 910_95 平台上调用 `al.fixpipe()` 会抛出 `RuntimeError`。

---

## 五、优化决策速查表

| 优化 | 触发条件 | 核心改动 | 关键约束 |
|------|----------|----------|----------|
| Bias 融合 | `tl.dot` + bias 累加 | 提前加载 bias + `tl.broadcast_to` 替代隐式广播 | `broadcast_to` 目标 shape 须与 accumulator 一致 |
| 随路转置 | `tl.dot` 参数中 `tl.trans().to(dtype)` | `.to(dtype)` 提到 `tl.trans()` 之前 | 仅对 `tl.dot` 参数有效 |
| FixPipe NZ2ND | Cube 结果需 ND 格式 | 使用 `al.fixpipe` + `FixpipeDMAMode.NZ2ND` | 最后一维对齐要求 |
| FixPipe 预量化 | Cube 结果需类型转换 | FixPipe 随路量化（当前通过内部语义层） | 仅 910_95 支持 L0C->UB |
| FixPipe 预 ReLU | Cube 结果需 ReLU | FixPipe 随路激活（当前通过内部语义层） | 仅 910_95 支持 L0C->UB |
| FixPipe 双目标 | 需拆分 Cube 结果到两个 UB | `dual_dst_mode=ROW_SPLIT/COLUMN_SPLIT` | 仅 910_95，仅 L0C->UB |

---

## 相关文档链接

- [FixPipe Python API 文档](../docs_triton_ascend/03-Ascend-Extensions/03-fixpipe.md)
- [FixPipe IR 文档](../docs_ascendnpu_ir/01-HIVM-Dialect/01-DMA-Operations/06-fixpipe.md)
- [FixPipe 源码 - core.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L273-L333)
- [FixPipe 枚举定义 - core.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L247-L270)
- [FixPipe 语义层 - semantic.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/semantic.py#L132-L148)
- [FixPipe TableGen 定义 - HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L246-L326)
- [FixPipe 枚举定义 - HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L783-L859)
- [FixPipe 验证逻辑 - HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L848-L891)
