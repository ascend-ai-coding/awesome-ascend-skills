# Block Pointer 迁移注意事项

## 概述

`tl.make_block_ptr` 是 Triton 中用于描述结构化内存访问的高级抽象，在 GPU 和 NPU 上的行为存在显著差异。NPU 对 Block Pointer 的 stride、order 参数有严格限制，转置语义只能通过 order 参数表达，且与复杂循环/分支语句搭配时可能出现编译问题。本文详细说明 Block Pointer 在 NPU 上的行为差异、限制和替代方案。

## 关键概念

| 概念 | GPU 行为 | NPU 行为 | 差异说明 |
|------|---------|---------|---------|
| make_block_ptr 数据类型 | 支持所有类型 | 不支持 uint8/uint16/uint32/uint64/fp64 | 硬件限制 |
| stride 参数 | 可通过调整顺序实现转置 | 只能反映真实内存布局 | NPU 不支持 stride 交换转置 |
| order 参数 | 控制遍历顺序 | 控制遍历顺序 + 表达转置语义 | NPU 上 order 兼具转置功能 |
| advance 操作 | 无特殊限制 | 与复杂循环/分支搭配可能编译失败 | 泛化性问题 |
| 连续访存 | 灵活 | 强偏好连续访存 | 离散访存性能差 |
| 维度支持 | 1-5 维 | 1-5 维 | 无差异 |

## 详细内容

### 1. make_block_ptr 在 NPU 上的行为差异

#### 1.1 数据类型限制

NPU 的 Block Pointer 不支持以下数据类型：

| 数据类型 | GPU | NPU (A2/A3) | 替代方案 |
|---------|-----|-------------|---------|
| uint8 | 支持 | 不支持 | 使用 int8 |
| uint16 | 支持 | 不支持 | 使用 int16 |
| uint32 | 支持 | 不支持 | 使用 int32 |
| uint64 | 支持 | 不支持 | 使用 int64 |
| fp64 | 支持 | 不支持 | 使用 fp32 |
| int8 | 支持 | 支持 | - |
| int16 | 支持 | 支持 | - |
| int32 | 支持 | 支持 | - |
| int64 | 支持 | 支持 | - |
| fp16 | 支持 | 支持 | - |
| fp32 | 支持 | 支持 | - |
| bf16 | 支持 | 支持 | - |

#### 1.2 stride 和 order 参数的限制

**GPU**：可以通过调整 `stride` 参数的顺序实现转置语义。例如，一个 (M, N) 矩阵，正常 stride 为 (N, 1)，交换为 (1, N) 即可实现转置。

**NPU**：只允许通过调整 `order` 参数的顺序来表达转置语义，不能通过调整 `stride` 参数的顺序实现转置。`stride` 必须反映真实的内存布局。

```python
# GPU 写法：通过 stride 交换实现转置（NPU 不支持）
block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(M, N),
    strides=(1, M),      # stride 交换，GPU 上可实现转置
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(0, 1),
)

# NPU 写法：通过 order 参数表达转置
block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(M, N),
    strides=(N, 1),      # stride 反映真实内存布局
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),        # 通过 order 表达转置语义
)
```

**order 参数含义**：order 是一个元组，指定维度的遍历优先级。order 中靠前的维度是内层循环（变化最快），靠后的是外层循环。例如 `order=(1, 0)` 表示先遍历维度 1（列），再遍历维度 0（行），即按列优先访问。

#### 1.3 与分支、循环语句搭配的泛化性问题

当前 `tl.make_block_ptr` / `tl.make_tensor_ptr` 如果与较复杂的循环和分支语句搭配使用，可能会出现编译问题。这是已知的限制，正在通过大量泛化测试暴露问题并迭代解决。

```python
# 可能出现编译问题的写法
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # 复杂循环中使用 advance
    for i in range(10):
        if i % 2 == 0:  # 分支语句 + advance 可能导致编译问题
            block_ptr = tl.advance(block_ptr, (BLOCK_SIZE,))
        x = tl.load(block_ptr, boundary_check=[0])
```

### 2. advance 操作的限制

`tl.advance` 用于调整 Block Pointer 的偏移量。在 NPU 上使用时需要注意以下限制：

1. **与循环/分支搭配的泛化性问题**：在复杂控制流中使用 `tl.advance` 可能导致编译失败
2. **offsets 长度必须匹配**：`len(offsets)` 需要与 `len(base.offsets)` 相等
3. **替代方案**：当 `tl.advance` 不可用时，可以重新调用 `tl.make_block_ptr` 修改 offset 参数

```python
# 使用 tl.advance 调整偏移量
block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(XB, YB, ZB),
    strides=(YB * ZB, ZB, 1),
    offsets=(3, 1, 2),
    block_shape=(XB, YB, ZB),
    order=(2, 1, 0),
)
bbptr = tl.advance(block_ptr, (-3, -1, -2))

# 替代方案：重新创建 block_ptr
for block_idx in range(pid, NUM_BLOCKS, 20):
    task_hz_idx = block_idx // NUM_BLOCKS_M
    task_m_idx = block_idx % NUM_BLOCKS_M
    off_z = task_hz_idx // H
    off_h = task_hz_idx % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(task_m_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
```

### 3. 离散访存 vs 连续访存

NPU 对连续访存有强偏好，离散访存会导致严重的性能问题。

#### 3.1 问题场景

GPU 上常见的写法是将多维数据展平为一维，使用 stride 表示跨步：

```python
# GPU 风格：离散访存
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024,),       # 一维张量
    strides=(32,),       # 每次跳 32 个元素
    offsets=(i_t * 16,),
    block_shape=(BT,),
    order=(0,),
)
```

这种写法在 GPU 上利用线程绑定最低维度，但在 NPU 上会导致离散访存，性能极差。

#### 3.2 优化方案

将数据视为二维矩阵，确保最低维度连续：

```python
# NPU 优化：连续访存
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024, 32),    # 二维张量
    strides=(32, 1),     # 最低维度连续
    offsets=(i_t * BT, 0),
    block_shape=(BT, 32),
    order=(1, 0),        # 先行后列
)
```

**优化思路**：
- 把 (1024, 32) 看成二维矩阵，最低维度 32 是连续的
- stride 从 (32,) 改为 (32, 1)，每个线程块访问连续的 32 个元素
- 线程块绑定到行维度（1024），每个线程处理一整行的 32 个元素
- 访存变为连续，亲和性好

#### 3.3 非对齐访存的 UB 溢出问题

非对齐访存不仅影响性能，还可能导致 UB 溢出：

```text
问题：对 (64, 32) 二维数据搬运，stride (12832, 128) 是非对齐的
      对齐数据的 stride 应为 (32, 1)

解决：在最内轴新增大小为 1 的轴
      (64, 32) → (64, 32, 4)
      stride (12832, 128) → (12832, 128, 1)
      硬件要求 vector 算子场景 UB 内存 32 字节对齐
```

### 4. Block Pointer 的替代方案

当 Block Pointer 在 NPU 上遇到限制时，可以使用以下替代方案：

#### 4.1 手动指针算术

使用 `tl.arange` + `tl.load`/`tl.store` 替代 Block Pointer：

```python
# Block Pointer 写法
block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(M, N),
    strides=(N, 1),
    offsets=(pid * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),
)
x = tl.load(block_ptr, boundary_check=[0, 1])

# 替代方案：手动指针算术
offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)
offsets = offs_m[:, None] * N + offs_n[None, :]
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
x = tl.load(x_ptr + offsets, mask=mask)
```

**优点**：
- 更灵活，不受 Block Pointer 的泛化性限制
- 可以与任意控制流搭配
- 调试更直观

**缺点**：
- 代码更冗长
- 需要手动管理边界检查
- 编译器可能无法进行某些优化

#### 4.2 重新创建 Block Pointer 替代 advance

```python
# 不使用 advance，每次重新创建 block_ptr
for k in range(0, K, BLOCK_K):
    a_block_ptr = tl.make_block_ptr(
        base=A + pid_m * BLOCK_M * stride_am + k * stride_ak,
        shape=(M - pid_m * BLOCK_M, K - k),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    a = tl.load(a_block_ptr, boundary_check=[0, 1])
```

### 5. Block Pointer 转置示例

以下示例展示了在 NPU 上通过 order 参数实现转置的正确写法：

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(in0_ptr, out0_ptr,
                     in0_stride0, in0_stride1,
                     in0_stride_order0, in0_stride_order1,
                     out0_stride0, out0_stride1,
                     out0_stride_order0, out0_stride_order1,
                     s0, s1,
                     tile_size0, tile_size1):
    tile_id0 = tl.program_id(axis=0)
    tile_id1 = tl.program_id(axis=1)
    offset0 = (tile_id0 * tile_size0).to(tl.int32)
    offset1 = (tile_id1 * tile_size1).to(tl.int32)

    in0_bptr = tl.make_block_ptr(
        in0_ptr,
        (s0, s1),
        (in0_stride0, in0_stride1),
        (offset0, offset1),
        (tile_size0, tile_size1),
        order=(in0_stride_order0, in0_stride_order1)
    )
    in0 = tl.load(in0_bptr,
                  boundary_check=(in0_stride_order0, in0_stride_order1))

    out0_bptr = tl.make_block_ptr(
        out0_ptr,
        (s0, s1),
        (out0_stride0, out0_stride1),
        (offset0, offset1),
        (tile_size0, tile_size1),
        order=(out0_stride_order0, out0_stride_order1)
    )
    tl.store(out0_bptr, in0,
             boundary_check=(out0_stride_order0, out0_stride_order1))
```

**关键点**：
- `in0_stride_order` 通过 permute_order 计算：`[len(permute_order)-1-i for i in permute_order]`
- stride 始终反映真实的内存布局
- 转置语义完全由 order 参数表达

## NPU 适配要点

1. **stride 必须反映真实内存布局**：不能通过交换 stride 实现转置，只能通过 order 参数
2. **优先连续访存**：将离散的一维访存改为连续的二维访存
3. **避免复杂控制流中使用 advance**：使用重新创建 block_ptr 的方式替代
4. **注意数据类型限制**：Block Pointer 不支持 uint8/uint16/uint32/uint64/fp64
5. **注意泛化性问题**：与复杂循环/分支搭配时可能编译失败，必要时使用手动指针算术替代

## 常见问题（Q&A）

**Q1：为什么 GPU 上通过 stride 交换实现的转置在 NPU 上不工作？**

A：NPU 的 Cube Unit 和 Vector Unit 对内存布局有严格要求，stride 必须反映真实的物理内存布局。转置语义只能通过 order 参数表达。这是硬件层面的限制，无法通过软件绕过。

**Q2：tl.advance 在什么情况下会编译失败？**

A：当 `tl.advance` 与复杂的循环和分支语句搭配使用时，可能导致编译问题。建议在遇到编译失败时，改用重新创建 `tl.make_block_ptr` 的方式调整偏移量。

**Q3：如何判断访存是否连续？**

A：检查 stride 参数。对于二维张量 (M, N)，如果 stride 为 (N, 1)，则最低维度（维度 1）是连续的。如果 stride 为 (1, M)，则最低维度（维度 0）是连续的。NPU 偏好最低维度连续的访存模式。

**Q4：Block Pointer 和手动指针算术哪个性能更好？**

A：在 NPU 上，如果 Block Pointer 能正确编译，通常可以获得更好的性能，因为编译器可以进行更多优化。但如果 Block Pointer 导致离散访存，手动指针算术配合连续访存模式可能更优。

## 相关文档

- [01-架构差异](./01-architecture-differences.md)
- [02-代码迁移模式](./02-code-migration-patterns.md)
- [03-迁移常见问题](./03-common-issues.md)
- [tl.make_block_ptr.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Memory_Pointer_Ops/tl.make_block_ptr.md)
- [tl.advance.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Memory_Pointer_Ops/tl.advance.md)
