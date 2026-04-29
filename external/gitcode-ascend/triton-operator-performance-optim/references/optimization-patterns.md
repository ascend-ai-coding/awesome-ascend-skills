# Ascend Triton 性能优化模式与代码样例

本文档包含 Triton 算子在 Ascend NPU 上的具体优化代码模式，供优化工作流中按需参考。

## 基础调优四板斧

### 1. Block Size 与 Grid Size

```python
# BLOCK_SIZE 必须匹配 UB 容量（192KB）
# FP16 数据：BLOCK_SIZE 建议 1024-2048
# 矩阵乘法：BLOCK_M/N/K 必须为 16 的倍数（Cube 单元粒度）

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
```

**检查点**：
- [ ] BLOCK_SIZE 是否适配 UB 容量？
- [ ] 矩阵运算 BLOCK 是否为 16 的倍数？
- [ ] Grid 维度是否映射到 AI Core 物理布局（推荐 1D Grid，且不超过物理核数）？

### 2. 强制向量化内存访问

```python
# 连续内存访问
offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 好
# offsets = block_start + tl.arange(0, BLOCK_SIZE) * 2  # 差：非连续

# Mask 防止越界（Ascend 对越界访问零容错）
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
```

**检查点**：
- [ ] 内存访问是否连续？
- [ ] 是否添加了 Mask？
- [ ] 地址是否 32 字节对齐？

### 3. UB 缓存与数据复用（核内再分块）

```python
# 当 BLOCK_SIZE 过大时，在核内再分块以适配 UB 容量
for sub_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    offsets = start + sub_start + tl.arange(0, SUB_BLOCK_SIZE)
    x_chunk = tl.load(x_ptr + offsets, mask=mask)
    # ... 处理
    tl.store(y_ptr + offsets, y_chunk, mask=mask)
```

### 4. 编译时常量与循环展开

```python
# tl.constexpr 强制编译期确定
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    for i in tl.static_range(0, BLOCK_SIZE):  # 静态展开
        ...
```

## Ascend 硬件特化优化

### Cube 单元适配（矩阵乘法）

```python
# Cube 单元仅支持 16x16 基础粒度
BLOCK_M: tl.constexpr = 128  # 必须为 16 的倍数
BLOCK_N: tl.constexpr = 256
BLOCK_K: tl.constexpr = 64

# 精度策略：累加器用 FP32，写回 FP16
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + ...)  # FP16
    b = tl.load(b_ptr + ...)  # FP16
    acc += tl.dot(a, b)       # FP32 累加
tl.store(c_ptr + ..., acc.to(tl.float16))  # 写回 FP16
```

### 数值稳定性（归约操作）

```python
# 归约前必须升精度到 FP32
x_fp32 = x.to(tl.float32)
mean = tl.sum(x_fp32, axis=-1) / D
var = tl.sum((x_fp32 - mean[:, None])**2, axis=-1) / D
```

### UB 空间管理

```python
# 计算 UB 需求
UB_SIZE = 192 * 1024  # 192KB
input_buffer = D * 2      # FP16
upcast_buffer = D * 4     # FP32
output_buffer = D * 2     # FP16
total = input_buffer + upcast_buffer + output_buffer

# 单值缓冲区必须 32B 对齐
mean_buffer = 32  # 不是 4B
```

## 高级优化技术

### 算子融合

将 Memory-Bound 操作转化为 Compute-Bound 操作：

```python
# 融合前：多次 GM 访问
x = load(x_ptr)      # GM → UB
y = relu(x)          # UB 计算
store(y_ptr, y)      # UB → GM
z = load(y_ptr)      # GM → UB（冗余！）
w = softmax(z)       # UB 计算
store(w_ptr, w)      # UB → GM

# 融合后：减少 GM 访问
x = load(x_ptr)
y = relu(x)
w = softmax(y)       # 直接复用 UB 中的 y
store(w_ptr, w)
```

### Double Buffer

```python
# 乒乓加载，隐藏访存延迟
# Buffer A 加载时，Buffer B 计算
# Buffer B 加载时，Buffer A 计算
```

## 精度保护模式

任何优化都不能破坏数值精度。以下是必须遵守的精度保护模式：

### 归约操作升精度

```python
# 所有归约操作（sum/mean/max/var）必须在 FP32 下进行
x_fp32 = x.to(tl.float32)
result = tl.sum(x_fp32, axis=-1)  # FP32 归约
```

### 矩阵乘法混合精度

```python
# 存储 FP16，累加 FP32，写回 FP16
a = tl.load(a_ptr)           # FP16 加载
b = tl.load(b_ptr)           # FP16 加载
acc = tl.dot(a, b)           # FP32 累加
tl.store(c_ptr, acc.to(tl.float16))  # FP16 写回
```

### 精度验证方法

```python
# 优化后必须对比 PyTorch-NPU 原生实现
torch.testing.assert_close(triton_output, torch_output, rtol=1e-3, atol=1e-3)

# 测试多种输入规模，包括非对齐边界
for size in [127, 128, 255, 256, 1023, 1024, 4096]:
    verify_correctness(size)
```

---

## 实战踩坑记录：RoPE (npu_rotary_mul) 优化

### 踩坑 1：Per-row 循环的 MTE 粒度陷阱

**现象**：per-row 循环 kernel 的 aiv_time = 3450 us/block，MTE3_ratio = 95.6%，read_bw 仅 2.72 GB/s。

**根因**：每行 load/store 仅 128B（half_D=64 × 2B），产生 183K 条 MTE2 指令。torch_npu 仅用 1,271 条 MTE2 指令处理相同数据量（7.7KB/条 vs 128B/条）。

**教训**：per-row 循环在 Ascend NPU 上有严重的 MTE 粒度问题。每条 MTE 指令有固定开销（dispatch、地址计算），小事务的 overhead 占比极高。

**优化手法**：用 2D Tiling 批量化 load/store，增加单条 MTE 指令传输的数据量。

### 踩坑 2：2D Tiling 的标量开销

**现象**：改用 2D Tiling（ROWS_PER_TILE=64）后 MTE3 从 3298→153 us（21x 改善），但 scalar_ratio 从 40% 飙升到 85%。

**根因**：2D broadcasting `row_off[:, None] + col_off[None, :]` 创建 [64,64] 中间数组，加上 `global_rows // NS` 整数除法。

**教训**：
- 2D Tiling 能解决 MTE 粒度问题，但引入标量开销
- 标量开销 = O(total_elements)，与 ROWS_PER_TILE 大小无关（减小 RPT 不降低总标量开销）
- 需要权衡 MTE 效率 vs Scalar 效率

### 踩坑 3：循环内分支导致灾难性性能

**现象**：用增量指针追踪（`cur_s += 1; if cur_s == S_val: ...`）替代整数除法，性能从 3450 us/block 暴涨到 17783 us/call（约 5x 变慢）。

**根因**：Triton 将循环内 if 分支编译为 masked 操作。当分支修改多个变量时，编译器生成极其低效的代码。

**教训**：**永远不要在 Triton 循环内使用条件分支来修改变量**。整数除法虽然慢，但比分支好得多。

### 踩坑 4：预计算 offset 表导致编译器崩溃

**现象**：将 cos/sin 行偏移预计算为 int32 tensor，kernel 内 `tl.load` 读取后用于 2D broadcasting，触发编译器 assertion：
```
Assertion `addptrRes.hasOneUse() && "Invalid: tt.addptr has multiple users"' failed.
```

**根因**：Triton-Ascend 编译器的 BlockPtrAnalysis 要求每个 `tt.addptr` 只有一个 user。从 `tl.load` 读取的 offset 被用于两次 broadcasting 违反了此约束。

**教训**：Triton-Ascend 编译器有单用户限制。同一指针值不能用于多个 `tl.load` 的 offset 计算。

### 踩坑 5：UB 溢出的隐藏成本——2D offset 数组

**现象**：ROWS_PER_TILE=128 时编译报 UB overflow：requires 246 KB > 192 KB available。

**根因**：UB 预算计算时只考虑了数据 buffer（8 × 16KB = 128KB），忽略了 2D offset 数组：
- `off_first`, `off_second`: 各 32KB（128 × 64 × 4B）
- `half_mask`: 8KB
- 总计 ~200KB > 192KB

**教训**：**计算 UB 预算时必须包含 offset 数组和 mask 数组**，不只是数据 buffer：
```python
UB_total = data_buffers + offset_arrays + mask_arrays + index_arrays
# offset_arrays = 2 × ROWS × half_D × sizeof(int32)
```

**额外陷阱**：`triton.next_power_of_2()` 向上取整。当 max_rows_tile=48 时 `next_power_of_2(48)=64`，直接超过 UB 限制。必须用向下取整：`1 << (max_rows_tile.bit_length() - 1)`。另外编译器有 ~15% 额外开销，需乘以 0.65 安全系数。

### 踩坑 6：Tiling 策略决定了 MTE2 指令粒度

**现象**：torch_npu 和 Triton kernel 同样只用 Vector Core（aic_* 列均为 NA），但内存带宽差距大：
- torch_npu: read_bw = 24.5 GB/s, MTE2 指令 1,271 条（每条 ~7.7KB）
- Triton 2D Tiling: read_bw = 6.95 GB/s, MTE2 指令 106,191 条（每条 ~90B）

**根因**：**Tiling 策略决定了编译器能否生成大块 DMA 传输。**

torch_npu 用连续内存布局 + 大块 `DataCopy`，offset 是线性递增的，编译器可合并为单条 MTE2。而 2D broadcasting 的 `row_off[:, None] + col_off[None, :]` 对编译器来说是非连续访存模式，无法合并成大块 DMA。

**证据**：2D Tiling 理论上每次 `tl.load` 加载 [64,64]=8KB，205 tiles × 6 loads = 1230 次，应接近 torch_npu 的 1271 次。但实际 MTE2 是 106,191 条（预期值的 86 倍），说明编译器将每个 2D load 拆成了许多小操作。

**教训**：
- op_statistic 中 MIX_AIC 标签不代表 Cube 引擎被使用——检查 PipeUtilization.csv 的 aic_cube_ratio
- 差距不在硬件，而在 tiling 策略：**连续访存模式才能触发大块 DMA，2D broadcasting 的非连续模式不行**
- 优化方向：设计让内存访问连续的 tiling 方案，而非 2D broadcasting offset

**判断方法**：对比理论 MTE2 条数（total_data / ideal_block_size）与实际条数。如果差距 >10x，说明 tiling 策略导致编译器无法合并内存访问。

### 踩坑 7：Broadcast Stride vs Expand+Contiguous（连续访存是第一优先级）

**现象**：kernel 内用 broadcast stride 访问辅助张量（cos/sin），需整数除法+stride 乘法定位行，是非连续访存。改为 host 侧 `expand().contiguous()` 展平为 (total_rows, D) 后，所有张量统一 `row * D + col` 偏移。

**结果**：kernel 从 1362us → 752us（45% 提升），且比 per-(batch,seq) 调度+cos/sin 广播复用的 rope.py 实现（1360us）快 1.8x。

**核心结论**：**内存访问的连续性 > 数据复用次数**。rope.py 的 cos/sin 广播复用（加载一次、广播到所有 head）理论上省带宽，但 2D tile `(N, D/2)` 是非连续 stride 访问，编译器无法合并为大块 DMA。展开后虽然 cos/sin 被重复加载，但连续访存让 DMA 引擎满速运行。

**代码模式**：
```python
# Host 侧：expand + contiguous，消除 broadcast stride
cos_flat = cos.expand(x_shape).contiguous().reshape(total_rows, D)
sin_flat = sin.expand(x_shape).contiguous().reshape(total_rows, D)
# Kernel 内：x, cos, sin, out 全部用相同偏移
row_off = global_rows * D
off = row_off[:, None] + col_off[None, :]  # 四个张量通用
```

**适用条件**：辅助张量原始大小 << 主张量大小（expand 内存代价可接受）。

### 性能对比总结

| 版本 | Task Duration | 瓶颈 | 关键优化 |
|------|--------------|------|---------|
| per-row (div/mod) | 3605 us | MTE 粒度 (95.6%) | 基线 |
| 2D Tiling (RPT=64) | 1362 us | Scalar (85%) | 2D tile 批量 load/store |
| 增量指针追踪 | ~17800 us | Branch divergence | 反模式 |
| **连续访存 (expand)** | **752 us** | **Memory BW** | **expand+contiguous 消除 broadcast stride** |
| torch_npu (CANN) | 550 us | MTE2 (99.3%) | hand-tuned C++ kernel |

与 CANN 差距：kernel 30%（752 vs 550us），full call 45%（expand 开销）。
