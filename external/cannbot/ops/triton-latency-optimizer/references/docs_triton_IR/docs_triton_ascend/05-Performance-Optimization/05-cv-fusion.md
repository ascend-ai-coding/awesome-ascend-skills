# Cube-Vector 融合算子优化

## 概述

Cube-Vector（CV）融合是昇腾 NPU 独有的优化技术，利用 Cube Core（矩阵乘法单元）和 Vector Core（向量计算单元）的异构并行能力，将矩阵乘法和后处理运算融合到同一个算子中执行。CV 融合按照 1:2 的 Cube:Vector 比例调度核心，通过 fixpipe 数据通路和 sync_block 同步机制实现核心间的高效协同，显著减少中间结果的 GM 写回和重新加载开销。

## 关键概念

| 概念 | 说明 | 关键参数 |
|------|------|----------|
| CV 融合 | Cube 和 Vector 核心协同工作，矩阵乘+后处理融合执行 | 1:2 Cube:Vector 比例 |
| Cube Core | 矩阵乘法单元，执行 tl.dot 运算 | 每个 AI Core 1 个 |
| Vector Core | 向量计算单元，执行 element-wise、activation 等运算 | 每个 AI Core 2 个 |
| fixpipe | L0C 到 UB 的数据通路，将 Cube 计算结果直接传递给 Vector | 仅 910_95/950 支持 |
| sync_block_set | 生产者核心发送同步信号 | sender, receiver, event_id |
| sync_block_wait | 消费者核心等待同步信号 | sender, receiver, event_id |
| sync_block_all | 全局屏障同步 | mode, event_id |
| sub_vec_id | 获取当前 Vector Core 在 AI Core 内的索引 | 返回 constexpr |
| sub_vec_num | 获取每个 AI Core 中 Vector Core 的数量 | 返回 constexpr（通常为 2） |
| al.Scope | 定义核心执行模式的作用域 | core_mode="cube" 或 "vector" |

## CV 融合的概念

### 1:2 Cube:Vector 比例

在昇腾 NPU 中，一个 AI Core 由 1 个 Cube Core 和 2 个 Vector Core 组成。CV 融合算子执行时，按照 1:2 的比例调度核心：

```
AI Core 0:  [Cube 0] + [Vector 0, Vector 1]
AI Core 1:  [Cube 1] + [Vector 2, Vector 3]
...
AI Core N:  [Cube N] + [Vector 2N, Vector 2N+1]
```

### 分核策略

对于 CV 融合算子，分核数应等于 **Cube 核数量**（即 aicore_num），而非 Vector 核数量：

```python
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

# CV 融合算子：分核数 = Cube 核数
NUM_CORE = aicore_num
grid = (NUM_CORE,)
```

### CV 融合 vs 分离执行

| 方式 | 数据流 | GM 访问次数 | 同步开销 | 适用场景 |
|------|--------|:----------:|:--------:|----------|
| 分离执行 | Cube -> GM -> Vector | 2 次（写+读） | 低 | 简单算子 |
| CV 融合 | Cube -> L0C -> UB -> Vector | 0 次 | 中 | 矩阵乘+后处理（仅 910_95） |

## fixpipe 在 CV 融合中的作用

### 概述

fixpipe 是 Cube Core 的 L0C 到 UB 的数据通路（仅 910_95 系列支持），允许 Cube 计算结果直接传递给 Vector Core 进行后处理，无需经过 GM 中转。A2/A3 系列不支持此通路，需要通过 GM 中转。

源码参考：[core.py - fixpipe](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L273-L333)

### API 说明

```python
from triton.language.extra.cann import extension as al

al.fixpipe(
    src,                    # 源张量，必须位于 L0C 内存区域
    dst,                    # 目标 buffer，必须位于 UB 内存区域
    dma_mode=al.FixpipeDMAMode.NZ2ND,      # DMA 传输模式
    dual_dst_mode=al.FixpipeDualDstMode.NO_DUAL,  # 双目标模式
)
```

### DMA 模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| NZ2ND | NZ 格式转 ND 格式 | 默认模式，通用场景 |
| NZ2DN | NZ 格式转 DN 格式 | 需要转置输出 |
| NZ2NZ | NZ 格式保持不变 | 无需格式转换 |

### 对齐要求

fixpipe 对目标 buffer 的对齐有严格要求：

| 数据类型 | 尾轴对齐 | 首轴对齐（NZ2DN 模式） |
|----------|----------|----------------------|
| float32 / int32 | 8 的倍数 | 8 的倍数 |
| float16 / bfloat16 / int16 | 16 的倍数 | 16 的倍数 |

> 注意：fixpipe 当前仅在 Ascend910_95/950 系列上支持。

## sync_block 在 CV 协同中的使用

### 同步机制概述

CV 融合中，Cube 和 Vector 核心需要协调执行顺序。Triton-Ascend 提供三种同步操作：

源码参考：[sync_block.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Extension_Ops/sync_block.md)

| 操作 | 说明 | 参数 |
|------|------|------|
| sync_block_set | 生产者发送同步信号 | sender, receiver, event_id |
| sync_block_wait | 消费者等待同步信号 | sender, receiver, event_id |
| sync_block_all | 全局屏障同步 | mode, event_id |

### 参数说明

| 参数 | 类型 | 取值范围 | 说明 |
|------|------|----------|------|
| sender | str | "cube" / "vector" | 发送方核心类型 |
| receiver | str | "cube" / "vector" | 接收方核心类型 |
| event_id | int | 0-15 | 事件 ID，区分不同同步点 |
| mode | str | "all_cube" / "all_vector" / "all" | 全局同步模式 |

### 同步流程

```
Cube Core:                           Vector Core:
计算 QK 矩阵乘 ─────┐
                     │
              sync_block_set         sync_block_wait
              ("cube","vector",0) ──> ("cube","vector",0)
                     │
                     │               执行 Softmax
                     │
                     │               sync_block_set
                     │               ("vector","cube",1)
                     │
              sync_block_wait <──── ("vector","cube",1)
              ("vector","cube",1)
                     │
              计算 PV 矩阵乘 ─────┐
                     │
              sync_block_set         sync_block_wait
              ("cube","vector",2) ──> ("cube","vector",2)
                     │
                     │               更新输出
```

## sub_vec_id / sub_vec_num 的使用

### sub_vec_id

获取当前 Vector Core 在 AI Core 内的索引（0 或 1）：

```python
from triton.language.extra.cann import extension as al

@triton.jit
def cv_kernel(...):
    # 在 Vector 作用域内获取当前 Vector Core 的索引
    with al.Scope(core_mode="vector"):
        vec_idx = al.sub_vec_id()  # 返回 0 或 1
```

### sub_vec_num

获取每个 AI Core 中 Vector Core 的数量（通常为 2）：

```python
from triton.language.extra.cann import extension as al

@triton.jit
def cv_kernel(...):
    # 获取 Vector Core 数量
    num_vec = al.sub_vec_num()  # 返回 2
```

### 使用场景

sub_vec_id 和 sub_vec_num 主要用于 CV 融合中 Vector Core 间的任务分配：

```python
@triton.jit
def cv_fusion_kernel(...):
    with al.Scope(core_mode="vector"):
        vec_idx = al.sub_vec_id()
        num_vec = al.sub_vec_num()
        # 根据 Vector Core 索引分配不同的计算任务
        # 例如：vec_idx=0 处理前半部分，vec_idx=1 处理后半部分
        chunk_size = N // num_vec
        start = vec_idx * chunk_size
        end = start + chunk_size
```

## CV 融合的适用场景

### 最佳适用场景：矩阵乘法 + 后处理

CV 融合最适合"矩阵乘法 + 后处理"的模式，其中 Cube Core 执行矩阵乘法，Vector Core 执行后处理（如 activation、normalization、type conversion 等）。

| 场景 | Cube 计算 | Vector 计算 | CV 融合收益 |
|------|-----------|-------------|-------------|
| Flash Attention | QK^T 矩阵乘、PV 矩阵乘 | Softmax、累加更新 | 高收益 |
| 矩阵乘 + Bias + GELU | A*B 矩阵乘 | Bias 加、GELU 激活 | 高收益 |
| 矩阵乘 + LayerNorm | A*B 矩阵乘 | 均值/方差计算、归一化 | 中收益 |
| 矩阵乘 + Quantize | A*B 矩阵乘 | 量化、类型转换 | 高收益 |

### 不适用场景

| 场景 | 原因 |
|------|------|
| 纯 Vector 算子 | 无 Cube 计算，无法形成 CV 协同 |
| 矩阵乘无后处理 | Cube 计算结果直接写回 GM，无需 Vector 参与 |
| 后处理依赖全局信息 | Vector 后处理需要跨核心同步，开销大 |

## 代码示例

### 示例1：基础 CV 融合模式

```python
import triton
import triton.language as tl
from triton.language.extra.cann import extension as al

@triton.jit
def cv_fusion_example(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Cube 核心执行矩阵乘法
    with al.Scope(core_mode="cube"):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak)
            b = tl.load(b_ptr + (offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn)
            acc += tl.dot(a, b)

        # 通知 Vector 核心矩阵乘完成
        al.sync_block_set("cube", "vector", 0)

    # Vector 核心执行后处理
    with al.Scope(core_mode="vector"):
        # 等待 Cube 核心完成矩阵乘
        al.sync_block_wait("cube", "vector", 0)

        # 执行后处理（如 activation）
        result = tl.sigmoid(acc.to(tl.float16))

        # 写回结果
        offs_m_full = tl.arange(0, BLOCK_M)
        offs_n_full = tl.arange(0, BLOCK_N)
        tl.store(c_ptr + offs_m_full[:, None] * stride_cm + offs_n_full[None, :] * stride_cn,
                 result)
```

### 示例2：Flash Attention CV 融合

```python
@triton.jit
def flash_attention_fwd(q_ptr, k_ptr, v_ptr, o_ptr, ...):
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Cube 核心执行 QK 和 PV 矩阵乘
    with al.Scope(core_mode="cube"):
        for start_n in range(0, N, BLOCK_N):
            qk = tl.dot(q, k)
            # 通知 Vector: QK 结果可用
            al.sync_block_set("cube", "vector", 0)
            # 等待 Vector 完成 Softmax
            al.sync_block_wait("vector", "cube", 1)
            pv = tl.dot(p, v)
            # 通知 Vector: PV 结果可用
            al.sync_block_set("cube", "vector", 2)

    # Vector 核心执行 Softmax 和累加更新
    with al.Scope(core_mode="vector"):
        for start_n in range(0, N, BLOCK_N):
            # 等待 Cube 完成 QK
            al.sync_block_wait("cube", "vector", 0)
            m_new, l_new, softmax_out = _softmax(qk, m_prev, l_prev)
            # 通知 Cube: Softmax 完成
            al.sync_block_set("vector", "cube", 1)
            # 等待 Cube 完成 PV
            al.sync_block_wait("cube", "vector", 2)
            acc = _update_output(pv, softmax_out, acc)

    # 全局同步
    with al.Scope(core_mode="cube"):
        al.sync_block_all("all", 0)

    tl.store(o_ptr + offsets, acc)
```

### 示例3：使用 sub_vec_id 分配任务

```python
@triton.jit
def cv_with_sub_vec(a_ptr, b_ptr, c_ptr, bias_ptr,
                    M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Cube 核心执行矩阵乘
    with al.Scope(core_mode="cube"):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + ...)
            b = tl.load(b_ptr + ...)
            acc += tl.dot(a, b)
        al.sync_block_set("cube", "vector", 0)

    # Vector 核心执行后处理，两个 Vector Core 分工
    with al.Scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)

        vec_idx = al.sub_vec_id()
        num_vec = al.sub_vec_num()

        # 每个 Vector Core 处理一半的行
        rows_per_vec = BLOCK_M // num_vec
        start_row = vec_idx * rows_per_vec
        end_row = start_row + rows_per_vec

        # 加载 bias
        bias = tl.load(bias_ptr + tl.arange(0, BLOCK_N))
        # 加 bias + ReLU
        local_acc = acc[start_row:end_row, :] + bias[None, :]
        result = tl.maximum(local_acc, 0.0)

        # 写回各自负责的行
        offs_m = start_row + tl.arange(0, rows_per_vec)
        offs_n = tl.arange(0, BLOCK_N)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                 result.to(tl.float16))
```

## NPU 适配要点

1. **分核数等于 Cube 核数**：CV 融合算子的 grid 应设为 `aicore_num`，而非 `vectorcore_num`
2. **512B 对齐要求**：CV 融合算子要求 Tensor 尾轴大小能被 512Bytes 整除
3. **sync_block 的 event_id 范围**：0-15，共 16 个独立事件，需避免冲突
4. **fixpipe 仅 910_95/950 支持**：其他型号需要通过 GM 中转
5. **al.Scope 必须正确使用**：Cube 计算必须在 `core_mode="cube"` 作用域内，Vector 计算在 `core_mode="vector"` 作用域内
6. **sync_block_set 和 sync_block_wait 必须配对**：sender 和 receiver 不能相同

## 常见问题 (Q&A)

**Q1: CV 融合一定比分离执行快吗？**

A: 不一定。CV 融合减少了 GM 访问次数，但增加了核心间同步开销。对于后处理很简单的场景（如仅加 bias），同步开销可能抵消收益。建议通过 msprof 对比两种方案的实际性能。

**Q2: 如何确定算子是否使用了 CV 融合？**

A: 如果算子中使用了 `tl.dot`，编译器会自动识别为 CV 算子。也可以通过 `al.Scope(core_mode="cube"/"vector")` 显式指定核心执行模式。

**Q3: sync_block 的 event_id 如何选择？**

A: event_id 用于区分不同的同步点，范围 0-15。在同一个循环中，不同的同步步骤应使用不同的 event_id，避免信号冲突。建议按顺序递增分配。

**Q4: 两个 Vector Core 之间如何同步？**

A: 使用 `sync_block_all("all_vector", event_id)` 进行同类型核心的全局同步，或使用 `debug_barrier` 进行 Vector Core 间的屏障同步。

## 相关文档

- [01-optimization-overview.md](./01-optimization-overview.md) - 优化策略总览
- [06-data-movement-optimization.md](./06-data-movement-optimization.md) - 数据搬运优化
- [07-profiling-guide.md](./07-profiling-guide.md) - 性能分析与瓶颈定位

### 源码参考

- [core.py - fixpipe/sync_block/sub_vec_id](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py)
- [sync_block.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Extension_Ops/sync_block.md) - 同步操作文档
- [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md) - CV 融合分核策略
