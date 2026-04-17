# 昇腾 NPU 硬件约束详解

本文档详细说明昇腾 910B NPU 上 Triton Vector 类算子开发的硬件约束。在分析 UB 占用、设计分核策略时参考此文档。

---

## 1. 整体架构

昇腾 NPU 的计算核心包含三大引擎：
- **Scalar**：标量计算（地址计算、循环控制、条件判断）
- **MTE（Memory Transfer Engine）**：数据搬运（GM ↔ UB）
- **Vector**：向量计算（算术运算、规约、类型转换等）

三者可以流水并行执行，是性能优化的关键。当任何一个引擎的操作阻塞了其他引擎，流水并行就被破坏。

---

## 2. Vector 核

### 2.1 核数与并行

Vector 核的数量有限，决定了可并行执行的 program ID 数量。

### 2.2 分核原则

- **负载均衡**：将输入数据尽量均匀分配给各 Vector 核
- **逻辑一致**：每个 program ID 对应的 kernel 处理逻辑相同
- **避免过度细分**：合理控制 grid 大小
- **循环补偿**：当分核数超过可用核心数时，在 kernel 内部通过循环处理

### 2.3 典型分核模式

```python
core_num = get_vectorcore_num()
num_tokens = qkv.shape[0]

# front_core 处理多一个 Token，tail_core 处理少一个
front_core_num = core_num
if num_tokens % core_num != 0:
    front_core_num = num_tokens % core_num

num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num

tail_core_num = 0
if num_tokens > core_num:
    tail_core_num = core_num - front_core_num

num_tokens_each_tail_core = num_tokens // core_num
```

**关键**：front_core 和 tail_core 的工作量差异不应超过 1 个 Token，以保持负载均衡。

---

## 3. Unified Buffer（UB）

### 3.1 基本概念

UB 是 Vector 核的片上存储，所有计算都在 UB 中进行。数据流为：

```
Global Memory (GM) → [tl.load] → UB → [Vector 计算] → UB → [tl.store] → GM
```

### 3.2 容量

| 芯片型号 | UB 容量 | 可用上限 (Double Buffering) | 建议使用量 |
|---------|---------|------------------------|-----------|
| 910B | 192 KB | 96 KB | ~85 KB |

### 3.3 Double Buffering 机制

**原理**：将 UB 分为两个 Buffer（A 和 B）：
- 当 Vector 在 Buffer A 中计算时，MTE 同时将下一批数据搬入 Buffer B
- 当 Vector 切换到 Buffer B 计算时，MTE 将 Buffer A 的结果搬出并加载新数据
- 如此交替，实现 MTE 与 Vector 的流水并行

**硬性约束**：单次循环的 UB 占用必须 ≤ 总容量的一半，否则 Double Buffering 无法工作。

```
单次循环 UB 占用 ≤ 192 KB // 2 = 96 KB
预留临时变量 → 建议 ≤ 85 KB
```

### 3.4 UB 占用计算

统计 kernel 循环体内所有变量的 UB 同时占用的峰值：

**需要计入的变量**：
1. `tl.load` 加载的所有 tensor
2. 计算过程中产生的中间 tensor
3. `tl.store` 暂存的输出 tensor
4. 类型转换后的变量（bf16 → float32 大小翻倍）

**注意事项**：
- 不同数据类型占用不同：bf16 = 2 Bytes/元素，float32 = 4 Bytes/元素
- 对二维 tensor 使用一维索引和 mask 会额外占用大量 UB
- 二维 tensor 形状如 `(num_heads, head_size)` 需按完整形状计算

**计算公式**：
假设一个算子包含 load、compute 和 store 三个操作过程，`S_token_load`、`S_token_compute`、`S_token_store` 分别表示 load、 compute、 store 三个阶段中的 UB 峰值占用大小，S_static 表示静态 UB 占用（例如：kernel 在循环体外加载到 UB 的权重），则这个 kernel UB 同时占用的峰值和单次循环最大可处理 token 数为：
```
S_token = max(S_token_load, S_token_compute, S_token_store) + S_static
        =   Σ(load_tensor_i × bytes_per_element_i)
            + Σ(intermediate_tensor_j × bytes_per_element_j)
            + Σ(store_tensor_k × bytes_per_element_k)
            + S_static

N = 85 * 1024 // S_token  （使用整数除法）
```
S_static 较小时，可以利用预留的 UB 临时变量空间所消耗，不需要将其计入 S_token，当出现 UB 溢出时，可排查是否由 S_static 过大导致，如果是，则将其计入 S_token。

**计算公式示例**：
以 add_kernel 为例：
```python
@triton.jit
def add_kernel(x_ptr, 
               y_ptr, 
               out_ptr, 
               n,  # 元素总数量。
               BLOCK_SIZE: tl.constexpr,  # 分块元素数量。
               ): 
    pid = tl.program_id(0) 
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        # 分块大小为 BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        # 加载 x,y 数据到片上内存
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)
```
- 该 kernel 循环体外 NUM_CORE 和 NUM_BLOCKS 的 UB 占用较少，因此不在 S_token 中单独计算 S_static。
- 循环体内每次循环处理一个 BLOCK，处理过程可拆分为 load、add 和 store 三个阶段，
    - load 阶段：加载 `x` 和 `y` 用于`outout`的计算，load 阶段的 UB 占用峰值为 `x` 和 `y` 的占用之和：
        ```
        S_token_load = Σ(load_tensor_x × bytes_per_element_x + load_tensor_y × bytes_per_element_y)
                     = BLOCK_SIZE × x.dtype + BLOCK_SIZE × y.dtype
        ```
    - add 阶段：计算`output`，在这个过程中，`x` 和 `y` 与 `output` 同时存在， add 阶段的 UB 占用峰值为 `x`、`y` 和 `output` 的占用之和：
        ```
        S_token_add = Σ(load_tensor_x × bytes_per_element_x + load_tensor_y × bytes_per_element_y + intermediate_tensor_output ×        bytes_per_element_output)
                    = BLOCK_SIZE × x.dtype + BLOCK_SIZE × y.dtype + BLOCK_SIZE × output.dtype
        ```
    - store 阶段：循环体内仅`output`占用 UB ，`x`和`y`计算完`output`后不再参与计算，store 阶段的 UB 占用峰值为`output`大小：
        ```
        S_token_store = Σ(store_tensor_output × bytes_per_element_output)
                     = BLOCK_SIZE × output.dtype
        ```

- 则这个 kernel UB 同时占用的峰值为：
    ```
    S_token = max(S_token_load, S_token_add, S_token_store)
            = max(BLOCK_SIZE × x.dtype + BLOCK_SIZE × y.dtype,
                  BLOCK_SIZE × x.dtype + BLOCK_SIZE × y.dtype + BLOCK_SIZE × output.dtype,
                  BLOCK_SIZE × output.dtype)
    ```
- 根据计算出来的`S_token`，可以进一步优化该算子在循环内处理更多的 BLOCK，提升访存效率和性能，每个循环最多处理的 BLOCK 数为：
    ```
    N = 85 * 1024 // S_token 
    ```

---

## 4. MTE / Vector / Scalar 流水并行

### 4.1 理想流水

```
时间 →
Scalar: [addr1] [addr2] [addr3] ...
MTE:    [load1] [load2] [load3] ...
Vector:         [comp1] [comp2] ...
MTE:            [store1][store2]...
```

### 4.2 破坏流水的常见操作

| 操作 | 影响 | 替代方案 |
|------|------|---------|
| `tl.load` with mask | MTE 等待 Vector 生成 mask | mask 预计算 |
| `tl.load` with other | 内部调用 tl.where，阻止 load 并行 | 去掉 other，手动 tl.where |
| 大量 Scalar 计算 | Scalar 流水成为瓶颈 | 预计算、tl.arange 索引 |
| range() 循环 | 可能影响流水并行 | 确保循环体内 load/vector 可并行 |

### 4.3 检查流水是否正常

使用 msprof 工具采集 profiling 数据后，检查：
- MTE 和 Vector 是否有重叠执行区间
- Scalar 是否存在长时间独占执行
- 各引擎的利用率是否均衡

---

## 5. 数据对齐与连续性

### 5.1 内存连续性要求

传入 Triton kernel 的所有 tensor 必须内存连续。在 Python 侧调用前确保：
```python
tensor = tensor.contiguous()
```

### 5.2 连续加载

`tl.load` 应加载连续的多行数据。若数据分布离散（如经过 index_select），需逐行加载。

### 5.3 对齐

尽量保证加载的数据起始地址对齐（通常 32 Bytes 对齐），可提升 MTE 搬运效率。
