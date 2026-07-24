# 内存访问模式优化指南

## 触发条件

当 Triton Agent 在优化 kernel 内存访问模式时，遇到以下任一场景应参考本文档：

- kernel 中存在 `%` 取余运算用于边界处理
- 循环内存在多条 `tl.load` 且存在可重排空间
- 循环内使用 `+=` 更新指针地址
- kernel 入参中存在运行时不变的固定值参数

---

## 核心知识：4 个优化模式总览

| # | 优化模式 | 核心问题 | 优化手段 | 性能收益 |
|---|---------|---------|---------|---------|
| 1 | 变量取余 → Mask 替代 | `%` 导致标量化，破坏向量化访存 | 用 mask 显式处理边界，保持连续地址 | 2x - 10x |
| 2 | Load 指令重排序 | 有依赖的 load 阻塞无依赖的 load | 将无依赖的 load 提前发射，与上次 store 并行 | 1.05x - 1.20x |
| 3 | 避免循环内 `+=` 更新指针 | `+=` 产生 RAW 依赖，阻塞流水线 | 用基地址 + 偏移量独立计算地址 | 1.05x - 1.20x |
| 4 | 入参静态化 | 运行时参数阻止编译期常量折叠 | `tl.constexpr` 启用编译期优化 | 视场景而定 |

---

## 模式 1：变量取余 → Mask 替代

### 触发条件

- kernel 中使用 `%` 对地址索引取余（如 `(pid * BLOCK + offset) % N`）
- 使用取余处理矩阵尾块（padding 区域）的越界索引

### 原理

在昇腾 NPU 上，模运算 `%` 通过标量 ALU 执行，导致向量运算退化为逐元素标量计算。取余后的地址不连续，破坏向量化内存访问模式，向量宽度（如 128/256 bit）无法充分利用，性能可能下降 5-10 倍。

**核心思路**：移除 `%` 取余，保持连续地址计算，改用 `mask` 显式标记有效元素，`mask` 本身是向量化的比较操作，不破坏向量化访存。

| 特性 | 使用 `%` 取余 | 使用 Mask |
|------|--------------|-----------|
| 地址计算 | 标量化（逐个元素） | 向量化（SIMD） |
| 内存访问 | 离散、不连续 | 连续、对齐 |
| 边界处理 | 隐式（通过取余） | 显式（通过 mask） |
| 性能 | 低（可能慢 5-10 倍） | 高（充分利用向量宽度） |

### 代码对比

**优化前（使用 `%` 取余）**：

```python
offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offset_k = tl.arange(0, BLOCK_SIZE_K)

x_ptrs = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
w_ptrs = w_ptr + (offset_k[:, None] * N + offset_wn[None, :])

accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x = tl.load(
        x_ptrs,
        mask=offset_k[None, :] < K - k * BLOCK_SIZE_K,
        other=0.0,
        care_padding=False
    )
    w = tl.load(
        w_ptrs,
        mask=offset_k[:, None] < K - k * BLOCK_SIZE_K,
        other=0.0,
        care_padding=False
    )
    accumulator += tl.dot(x, w)
    x_ptrs += BLOCK_SIZE_K
    w_ptrs += BLOCK_SIZE_K * N
```

**优化后（使用 Mask 替代 `%`）**：

```python
offset_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offset_wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offset_k = tl.arange(0, BLOCK_SIZE_K)

a_ptrs_base = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
b_ptrs_base = w_ptr + (offset_k[:, None] * N + offset_wn[None, :])

msk_m = offset_xm < M
msk_n = offset_wn < N

accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K
    w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N

    x = tl.load(
        x_ptrs,
        mask=msk_m[:, None] & (offset_k[None, :] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    w = tl.load(
        w_ptrs,
        mask=msk_n[None, :] & (offset_k[:, None] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    accumulator += tl.dot(x, w)
```

### 关键点

1. **移除 `%` 操作**：`offset_xm` 和 `offset_wn` 不再取余，保持连续地址
2. **添加边界 mask**：`msk_m = offset_xm < M` 和 `msk_n = offset_wn < N`，在循环外计算一次
3. **组合 mask**：使用 `&`（位与）而非 `and`（逻辑与）组合多个 mask 条件；`and` 是 Python 逻辑操作符，返回单个布尔值，不适用于 tensor mask
4. **mask 计算位置**：边界 mask 在循环外计算，避免循环内重复计算
5. **不要遗漏维度**：确保所有需要边界处理的维度都有对应的 mask

---

## 模式 2：Load 指令重排序

### 触发条件

- 循环内存在多条 `tl.load` 指令
- 其中一条 load 与上一次循环的 store 存在数据依赖（如 load 同一地址），而另一条 load 无此依赖

### 原理

编译器不会修改用户 load 指令的顺序。当循环内存在多条 load 时，如果排在前的 load 因等待上一次循环的 store 而阻塞，排在后的无依赖 load 也无法提前发射，导致串行执行。

**核心思路**：将无数据依赖的 load 提前到有依赖的 load 之前，使其可与上一次循环的 store 并行执行，充分利用流水线并行能力。

**执行时序对比**：

```
优化前（load B 在前，阻塞 load A）:
  迭代 i:   load B → load A → calc → store O → store B
  迭代 i+1: load B(等store B) → load A(等load B) → ...

优化后（load A 在前，与 store B 并行）:
  迭代 i:   load A → load B → calc → store O → store B
  迭代 i+1: load A(与store B并行) → load B → ...
```

| 优化前 | 优化后 |
|--------|--------|
| load B 等待上一次 store B 完成后才能执行 | load A 无需等待，可以提前发射 |
| load A 必须等 load B 完成后才能执行 | load A 可以与上一次循环的 store B 并行执行 |
| 串行执行，并行度低 | 并行执行，并行度高 |

### 代码对比

**优化前（load B 在前，阻塞 load A）**：

```python
@triton.jit
def AB_load_kernel(
    A, B, B_index, O,
    B_DIM: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_range = tl.arange(0, B_DIM)

    for i in range(HEAD_NUM):
        p_A = A + i * HEAD_DIM + i_n * B_DIM + i_range
        p_O = O + i * HEAD_DIM + i_n * B_DIM + i_range
        p_B_index = B_index + i

        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        b_B = tl.load(p_B)

        b_A = tl.load(p_A)

        b_B += tl.sum(b_A)
        b_O = b_A * b_B

        tl.store(p_O, b_O)

        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        tl.store(p_B, b_B)
```

**优化后（load A 在前，与 store B 并行）**：

```python
@triton.jit
def AB_load_kernel(
    A, B, B_index, O,
    B_DIM: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_range = tl.arange(0, B_DIM)

    for i in range(HEAD_NUM):
        p_A = A + i * HEAD_DIM + i_n * B_DIM + i_range
        p_O = O + i * HEAD_DIM + i_n * B_DIM + i_range
        p_B_index = B_index + i

        b_A = tl.load(p_A)

        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        b_B = tl.load(p_B)

        b_B += tl.sum(b_A)
        b_O = b_A * b_B

        tl.store(p_O, b_O)

        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        tl.store(p_B, b_B)
```

### 关键点

1. **识别依赖关系**：分析循环内每条 load 与上一次循环 store 之间是否存在数据依赖
2. **无依赖 load 前置**：将与上次 store 无依赖的 load 提前到有依赖的 load 之前
3. **编译器不重排**：Triton 编译器不会自动调整 load 顺序，必须由开发者手动重排
4. **不影响语义**：重排仅改变 load 的发射时机，不改变计算逻辑

---

## 模式 3：避免循环内 `+=` 更新指针

### 触发条件

- 循环内使用 `ptr += offset` 更新指针地址
- 循环迭代次数较多（> 4）的场景

### 原理

`ptr += offset` 产生读后写依赖（RAW）：当前迭代的指针值依赖前一次迭代的写入结果。这导致编译器难以重排指令，阻塞流水线，无法充分利用昇腾 NPU 的多级流水架构。

**核心思路**：将地址的增量更新转换为绝对地址计算——`ptr = base + i * offset`，每次迭代独立计算地址，消除迭代间的 RAW 依赖。

| 方面 | 使用 `+=` | 使用基地址 + 偏移量 |
|------|-----------|---------------------|
| 数据依赖 | 存在 RAW 依赖，阻塞流水线 | 无依赖，各次迭代独立计算 |
| 指令调度 | 编译器难以重排指令 | 编译器可自由重排，提升并行度 |
| 流水优化 | 难以充分利用 NPU 多级流水 | 更好地匹配昇腾流水架构 |
| 向量化 | 可能阻碍向量化优化 | 有利于 SIMD 向量化 |

### 代码对比

**优化前（使用 `+=` 更新指针）**：

```python
offset_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offset_wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offset_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs_base = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
b_ptrs_base = w_ptr + (offset_k[:, None] * N + offset_wn[None, :])
msk_m = offset_xm < M
msk_n = offset_wn < N

x_ptrs = a_ptrs_base
w_ptrs = b_ptrs_base

accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x = tl.load(
        x_ptrs,
        mask=msk_m[:, None] & (offset_k[None, :] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    w = tl.load(
        w_ptrs,
        mask=msk_n[None, :] & (offset_k[:, None] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    accumulator += tl.dot(x, w)

    x_ptrs += BLOCK_SIZE_K
    w_ptrs += BLOCK_SIZE_K * N
```

**优化后（使用基地址 + 偏移量）**：

```python
offset_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offset_wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offset_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs_base = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
b_ptrs_base = w_ptr + (offset_k[:, None] * N + offset_wn[None, :])
msk_m = offset_xm < M
msk_n = offset_wn < N

accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K
    w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N

    x = tl.load(
        x_ptrs,
        mask=msk_m[:, None] & (offset_k[None, :] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    w = tl.load(
        w_ptrs,
        mask=msk_n[None, :] & (offset_k[:, None] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    accumulator += tl.dot(x, w)
```

### 关键点

1. **基地址在循环外计算**：`a_ptrs_base` 和 `b_ptrs_base` 在循环外计算一次，避免重复计算
2. **循环内独立计算地址**：`ptr = base + k * offset`，消除迭代间 RAW 依赖
3. **不要丢失基地址**：确保偏移量计算是 `base + offset`，而非仅 `offset`
4. **多维地址同理**：`w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N`，不同维度使用对应步长

---

## 模式 4：入参静态化（`tl.constexpr`）

### 触发条件

- kernel 入参中存在运行时不变的固定值参数
- 典型参数：BLOCK_SIZE（`BLOCK_M`、`BLOCK_N`、`BLOCK_K`）、STRIDE（`stride_m`、`stride_n`）等
- 如果不确定参数是否固定，应询问用户是否可设为 `tl.constexpr`

### 原理

将固定数值的入参声明为 `tl.constexpr`，编译器可在编译期进行常量折叠（constant folding）和常量传播（constant propagation），生成更优的指令序列。未声明为 `tl.constexpr` 的参数即使在运行时是固定值，编译器也无法利用其常量性质进行优化。

**优化效果**：
- 启用编译时常量折叠
- 帮助编译器进行更 aggressive 的常量传播
- 减少运行时分支判断开销
- 可能影响 tiling 策略和内存布局选择

### 代码对比

**优化前（stride 为普通入参）**：

```python
@triton.jit
def kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_an,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_an = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offset_am[:, None] * stride_am + offset_an[None, :] * stride_an
    # ...
```

**优化后（stride 声明为 `tl.constexpr`）**：

```python
@triton.jit
def kernel(
    A, B, C,
    M, N, K,
    stride_am: tl.constexpr,
    stride_an: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_an = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offset_am[:, None] * stride_am + offset_an[None, :] * stride_an
    # ...
```

### 关键点

1. **只有运行时不变的参数才适合**：`tl.constexpr` 要求参数在编译期已知，动态参数不可使用
2. **优先对性能敏感参数使用**：BLOCK_SIZE、stride 等直接影响内存访问模式和 tiling 策略的参数应优先考虑
3. **不确定时询问用户**：如果不确定某个参数是否在所有调用中都是固定值，应询问用户确认
4. **声明位置**：`tl.constexpr` 在函数签名中通过类型注解声明，格式为 `param_name: tl.constexpr`

---

## 910_95 特别注意

1. **向量宽度有限**：910_95 的 Vector Core 单次处理 256 bit（32 Bytes），标量化退化对性能影响尤为严重。模式 1（Mask 替代 `%`）在 910_95 上收益最大，可能达到 5-10x 加速。

2. **UB 容量仅 248KB**：910_95 UB 容量较小（248KB = 256KB - 8KB 预留），向量化访存对 UB 利用率至关重要。取余导致的离散访存会浪费宝贵的 UB 带宽。

3. **多级流水架构**：910_95 采用 Cube + 2 Vector 架构，流水线并行能力强但依赖正确调度。模式 2（Load 重排序）和模式 3（基地址 + 偏移量）能更好地匹配昇腾流水架构，减少流水线气泡。

4. **L0C 容量 256KB**：910_95 的 L0C 为 256KB（相比 910B 的 128KB 翻倍），矩阵乘法 accumulator 可充分利用。配合模式 4（`tl.constexpr`），编译器可更精确地规划 L0C tiling 策略。

5. **对齐要求**：910_95 L1 对齐 32B、L0C 对齐 512B、UB 对齐 32B。连续地址访问（模式 1 + 模式 3）有助于满足对齐约束，避免硬件异常。

6. **`tl.constexpr` 与 tiling**：910_95 的 UB/L1/L0C 容量有限，`tl.constexpr` 使编译器能在编译期确定 BLOCK_SIZE，从而精确计算 tiling 分块是否适配各级缓存。

---

## 相关文档链接

- [00-hardware-quick-ref.md](../docs_for_triton_agent/00-hardware-quick-ref.md) — Ascend910_95 硬件速查手册
