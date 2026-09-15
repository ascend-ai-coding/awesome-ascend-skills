# Multibuffer / 双缓冲优化（通信-计算重叠）

> 把**内存访问（MTE2/MTE3）**与**计算（VECTOR/CUBE）**在时间上重叠，是 Ascend NPU Triton kernel 最常见的通用加速手段之一。本文档集中说明 `multibuffer=True` 与手写双缓冲的用法、适用场景和回退策略。

---

## 1. 优化原理

Triton kernel 的典型执行流是：

```
MTE2 load  →  VECTOR/CUBE compute  →  MTE3 store
```

当 compute 无法完全隐藏 memory latency 时，MTE 管线成为瓶颈。`multibuffer=True` 让编译器在相邻迭代/不同 buffer 之间自动调度：在计算当前 tile 的同时，提前发起下一 tile 的 load，使 MTE2 与 VECTOR/CUBE 并行工作。

手动双缓冲是同一思想的显式实现：用户维护两份（或多份）buffer，当前迭代 compute 与下一迭代 load 重叠。

---

## 2. 何时使用

| 场景 | 建议 |
|------|------|
| kernel 内是 `load → compute → store` 的清晰流水线 | **优先尝试 `multibuffer=True`** |
| 性能瓶颈在 MTE2/MTE3，VECTOR/CUBE 利用率不高 | 尝试 multibuffer |
| 单次 tile 数据量较大（BLOCK 较大、UB 有余量） | multibuffer 更容易生效 |
| 已做 Tiling / Pass 合并 / 维度合并 / Block Size Scaling 后仍内存受限 | 叠加 multibuffer |
| 自动 multibuffer 导致 UB 溢出或编译失败 | 改用手动无条件 prefetch |

---

## 3. 自动 multibuffer：`multibuffer=True`

### 用法

```python
kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE, multibuffer=True)
```

### 适用算子类型

- **Elementwise / 激活函数**：`SwiGLU`、`GELU`、`SiluAndMul` 等 load→compute→store 清晰的 kernel。
- **Normalization apply kernel**：LayerNorm/BatchNorm/RMSNorm/Softmax 的 apply 阶段（scale/shift/store）。
- **归约算子**：对 stats kernel，可隐藏 load 到 vector sum 的等待。
- **Memory-bound copy / layout transform**：连续的 split/concat/transpose。
- **Matmul/Attention 的 latency-bound 阶段**：当 MTE2 operand load 与 CUBE dot 交替等待时。

### 何时无效或有害

- **UB 已接近上限**：再增加双 buffer 副本会溢出（verify 失败）。
- **kernel 内有复杂的跨迭代依赖**：编译器无法安全调度 load 提前。
- **瓶颈是 atomic/CUBE 利用率/标量循环**：multibuffer 不触及这些瓶颈。
- **条件 prefetch**：在某些偶数 shape 下会 miscompile。

---

## 4. 手动双缓冲 fallback

当 `multibuffer=True` 失败或不可用时，使用显式 prefetch。核心规则：

1. **无条件 clamped prefetch**：
   ```python
   next_idx = tl.minimum(i + 1, N - 1)
   next_tile = tl.load(ptr + next_idx * BLOCK)  # 提前发起
   ```
2. **不要加 if/条件分支**：条件 prefetch 在部分 shape 下会 miscompile。
3. **buffer 轮转**：当前迭代用已 prefetched 的数据，同时发起下一次 load。

### 示例

```python
@triton.jit
def kernel(input_ptr, output_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK

    # 预加载第一个 tile
    cur = tl.load(input_ptr + start + tl.arange(0, BLOCK))

    for i in range(1, (N + BLOCK - 1) // BLOCK):
        nxt_start = tl.minimum(start + i * BLOCK, N - BLOCK)
        nxt = tl.load(input_ptr + nxt_start + tl.arange(0, BLOCK))  # 下一 tile load

        out = compute(cur)                                           # 当前 tile compute
        tl.store(output_ptr + start + (i - 1) * BLOCK + tl.arange(0, BLOCK), out)

        cur = nxt  # 轮转

    # 最后一个 tile
    out = compute(cur)
    tl.store(output_ptr + ..., out)
```

---

## 5. 与各类优化点的关系

| 优化点 | 叠加 multibuffer 的作用 |
|--------|------------------------|
| **Tiling 优化** | 连续轴向量化后，隐藏 MTE2 load 与 VECTOR compute 之间的等待 |
| **Pass 合并** | 单次 load 多次使用，compute 链变长，multibuffer 隐藏下次 load |
| **维度合并** | 合并成大 load 后，MTE2 与 vector sum/逐元素 compute 重叠 |
| **Load 重排序** | load 重排解决单次迭代内依赖；multibuffer 解决跨迭代重叠 |
| **Block Size Scaling** | 大 BLOCK 给 multibuffer 更多余量，常组合出最佳性能 |
| **Scalar → Vector** | 向量化后计算密度提升，MTE2 容易成为瓶颈，multibuffer 补齐 |
| **维度合并与大 BLOCK 累加** | 归一化 stats/apply kernel 的 load→compute→store 流水线重叠 |
| **Latency-bound Tile 合并** | dot 的 operand load 与 CUBE compute 重叠，减少 issue/sync 等待 |
| **Cube/MTE3 解耦** | Stage A 纯 Cube 阶段，把下一 blk 的 load 与当前 dot 重叠 |
| **Workspace 物化解耦** | Pass A 的 workspace store 与下一 tile compute 重叠；Pass B 的 workspace load 与当前 compute 重叠 |

---

## 6. 调试与验证

### 启用后 verify 失败

1. **UB 溢出**：减小 BLOCK_SIZE 或关闭 multibuffer。
2. **编译失败**：尝试手动双缓冲替代。
3. **精度错误**：检查是否有未正确保护的越界 load（prefetch 索引必须 clamp）。

### 性能无提升

1. 用 profiling 确认瓶颈是否真的是 MTE2/MTE3。
2. 如果瓶颈是 CUBE/VECTOR 算力或 atomic，multibuffer 无效。
3. 尝试 Block Size Scaling + multibuffer 组合搜索。

---

## 7. 关键原则

1. **先自动，后手动**：优先 `multibuffer=True`，失败再手写双缓冲。
2. **无条件 clamp**：手动 prefetch 必须无条件 clamp 索引，禁止 if 条件。
3. **先定位瓶颈**：multibuffer 只对 memory/compute overlap 有效，非万能。
4. **注意 UB 余量**：双缓冲需要额外 buffer 空间，确保不溢出。
5. **与其他优化组合**：multibuffer 通常是 Tiling/Pass合并/Block Size Scaling 后的最后一层优化。

---

## 8. 快速决策树

```
kernel 有 load→compute→store 流水线？
├── 否 → multibuffer 不适用
└── 是
    ├── 自动 multibuffer=True 通过 verify？
    │   ├── 是 → benchmark；有提升则保留
    │   └── 否 → 手动无条件 clamped prefetch
    └── profiling 瓶颈在 MTE2/MTE3？
        ├── 是 → 叠加 multibuffer/prefetch
        └── 否 → 先解决真正的瓶颈
```
