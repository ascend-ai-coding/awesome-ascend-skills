# 优化策略总览

## 概述

Triton-Ascend 的性能优化需要深入理解昇腾 NPU 的硬件架构特征。与 GPU 的优化思路不同，NPU 采用 Cube+Vector 异构核心架构，拥有独立的片上内存层级（UB/L1/L0C），其优化核心在于最大化数据搬运与计算的并行度，以及合理利用有限的片上内存空间。本文档提供 NPU 性能优化的全局视图，帮助开发者系统性地理解和实施优化策略。

## 关键概念

| 概念 | 说明 | 典型值/范围 |
|------|------|-------------|
| AI Core | 昇腾 NPU 的计算核心，包含 1 个 Cube 核 + 2 个 Vector 核 | 芯片型号决定数量 |
| Cube Core | 矩阵乘法单元，执行 tl.dot 等矩阵运算 | 每个 AI Core 1 个 |
| Vector Core | 向量计算单元，执行 element-wise、reduction 等运算 | 每个 AI Core 2 个 |
| UB (Unified Buffer) | Vector Core 的片上存储，数据搬入/计算/搬出的工作区 | A2: 192KB, 910_95/950: 256KB |
| L1 Buffer | Cube Core 的片上存储，用于矩阵乘法的数据缓存 | 与芯片型号相关 |
| L0C | Cube Core 的累加器输出缓冲区 | 固定大小 |
| MTE2 | 数据从 GM 搬入 UB 的搬运流水线 | 带宽约 1.8TB/s |
| MTE3 | 数据从 UB 搬出至 GM 的搬运流水线 | 与 MTE2 共享带宽 |
| 存算并行 | 数据搬运与计算重叠执行，是 NPU 性能优化的核心手段 | 默认开启 multiBuffer |
| Tiling | 将大数据切分为小块，适配片上内存容量并实现存算并行 | BLOCK_SIZE / BLOCK_SIZE_SUB |
| CV 融合 | Cube 和 Vector 核心协同工作，1:2 比例调度 | 适用于矩阵乘+后处理场景 |

## 计算密集型 vs 访存密集型 Kernel 的优化策略

### 算子类型判定

| 算子类型 | 特征 | 典型算子 | 瓶颈所在 | 优化方向 |
|----------|------|----------|----------|----------|
| 计算密集型 | 计算量/访存量比值高 | 矩阵乘法 (tl.dot)、Conv | Cube/Vector 计算单元 | 增大分块提高计算密度、CV 融合 |
| 访存密集型 | 计算量/访存量比值低 | Element-wise Add、Copy、Mask | MTE2/MTE3 搬运流水线 | 存算并行、care_padding、对齐优化 |
| 混合型 | 既有矩阵乘又有向量运算 | LayerNorm、Softmax、Flash Attention | 取决于具体比例 | 综合优化、流水线调度 |

### 优化策略矩阵

| 优化手段 | 计算密集型 | 访存密集型 | 混合型 |
|----------|:----------:|:----------:|:------:|
| 增大 Tiling 分块 | 高优先 | 中优先 | 高优先 |
| 存算并行 (multiBuffer) | 高优先 | 高优先 | 高优先 |
| care_padding=False | 低优先 | 高优先 | 中优先 |
| CV 融合 | 高优先 | 不适用 | 高优先 |
| 对齐优化 | 中优先 | 高优先 | 高优先 |
| 数据类型优化 | 中优先 | 低优先 | 中优先 |
| Grid 分核优化 | 中优先 | 中优先 | 高优先 |

## 关键性能指标

### 1. AI Core 利用率

AI Core 利用率反映了计算核心的繁忙程度，是衡量算子性能的首要指标。

| 指标 | 含义 | 理想值 | 获取方式 |
|------|------|--------|----------|
| aiv_vec_ratio | Vector 流水利用率 | > 80% | msprof op_summary |
| aiv_cube_ratio | Cube 流水利用率 | > 80% | msprof op_summary |
| aiv_scalar_ratio | Scalar 流水利用率 | < 20% | msprof op_summary |

**诊断规则**：
- `aiv_scalar_ratio` 过高（>30%）说明存在标量退化，需检查数据类型是否被硬件支持
- `aiv_vec_ratio` 过低说明 Vector 流水断流，需检查 Tiling 和存算并行

### 2. UB 命中率

UB 命中率衡量数据在片上内存中的复用效率。

| 场景 | UB 命中率低的原因 | 优化手段 |
|------|-------------------|----------|
| Tiling 过小 | 频繁搬运小块数据，MTE2 开销大 | 增大 BLOCK_SIZE |
| 无存算并行 | 搬运和计算串行，UB 空闲等待 | 启用 multiBuffer，使用 for 循环 Tiling |
| 数据未复用 | 同一数据多次从 GM 搬入 | 调整计算顺序，先 load 再 select |

### 3. 流水线效率

流水线效率衡量 MTE2（搬入）、Compute（计算）、MTE3（搬出）三条流水线的重叠程度。

```
理想流水线（存算并行）：
MTE2:  |==搬入1==|==搬入2==|==搬入3==|
Compute:          |==计算1==|==计算2==|==计算3==|
MTE3:                      |==写出1==|==写出2==|

低效流水线（串行）：
MTE2:  |==搬入1==|          |==搬入2==|
Compute:          |==计算1==|          |==计算2==|
MTE3:                    |==写出1==|          |==写出2==|
```

## 优化检查清单

### 阶段一：基础适配

- [ ] **Grid 分核数对齐物理核数**：Vector 算子 = vectorcore_num，CV 算子 = aicore_num
- [ ] **核内循环处理**：使用 `range(pid, NUM_BLOCKS, NUM_CORE)` 模式分配任务
- [ ] **数据类型检查**：避免使用 int64 的 ADD/CMP 操作，改用 int32 或 fp32
- [ ] **设备指定**：`device='npu'` 替代 `device='cuda'`

### 阶段二：访存优化

- [ ] **尾轴对齐**：Vector 算子 32B 对齐，CV 算子 512B 对齐
- [ ] **care_padding**：对 padding 区域数据无依赖时设置 `care_padding=False`
- [ ] **连续访存**：确保 offsets 连续，避免离散访问
- [ ] **借轴转置**：短轴不对齐时，通过 reshape+trans 规避自动补齐

### 阶段三：Tiling 优化

- [ ] **BLOCK_SIZE 选择**：使用 autotune 自动搜索最优分块
- [ ] **二级分块**：大 BLOCK_SIZE + 小 BLOCK_SIZE_SUB 避免 UB 溢出
- [ ] **for 循环 Tiling**：在核内使用 for 循环实现存算并行
- [ ] **UB 空间检查**：确保单次分块数据量不超过 UB 容量

### 阶段四：高级优化

- [ ] **CV 融合**：矩阵乘+后处理场景使用 Cube+Vector 协同
- [ ] **multiBuffer**：启用双缓冲实现存算并行
- [ ] **L1 缓存利用**：矩阵乘法中利用 L1 缓存 A/B 矩阵分块
- [ ] **sync_block 同步**：CV 融合中使用同步指令协调核心执行

## 与 GPU 优化思路的差异

| 维度 | GPU 优化思路 | NPU 优化思路 | 关键差异 |
|------|-------------|-------------|----------|
| 核心数量 | SM 数量多（几十~几百），grid 可自由定义 | AI Core 数量少（几十），grid 应对齐物理核数 | NPU 需固定核数+核内循环 |
| 分核策略 | 一个 block 处理一块数据，block 数量可远大于 SM 数 | 一个核处理多块数据，核数等于物理核数 | NPU 采用"少核多任务"模式 |
| 片上内存 | Shared Memory 可编程管理，大小灵活 | UB/L1 有固定大小和对齐要求 | NPU 需严格 Tiling 和对齐 |
| 计算单元 | CUDA Core + Tensor Core，编译器自动调度 | Cube Core + Vector Core，需显式协调 | NPU 需要 CV 融合和 sync_block |
| 存算并行 | 通过 Shared Memory 手动实现双缓冲 | 编译器默认 multiBuffer，需 for 循环配合 | NPU 存算并行更依赖编译器 |
| 数据对齐 | 128B 对齐（warp 访存） | Vector 32B / Cube 512B 对齐 | NPU 对齐要求更严格且分场景 |
| 同步机制 | `__syncthreads()` 块内同步 | `sync_block_set/wait` 核间同步 | NPU 同步是跨核心的，不是线程级 |
| 标量退化 | 较少发生 | int64 ADD/CMP 会退化为标量计算 | NPU 需特别注意数据类型选择 |

### 迁移核心原则

1. **放弃 GPU "逻辑 grid 自由定义"**，转为昇腾"物理核组绑定"
2. **Vector 算子 32B 对齐，CV 算子 512B 对齐**，移除 GPU 专属同步 API
3. **Grid 优先用 1D**，2D 适配写法也会合并为 1D，实际 grid 值应对齐芯片物理核数
4. **核内做更细致的数据分块**，通过 for 循环实现多批次处理

## 优化工作流

### 迭代优化流程

性能优化是一个迭代过程，建议按以下流程进行：

```
步骤1：基础适配
  |
  +-- 将 device='cuda' 改为 device='npu'
  +-- 固定 Grid 分核数为物理核数
  +-- 核内使用 for 循环处理多块数据
  |
步骤2：正确性验证
  |
  +-- 确保算子计算结果正确
  +-- 处理 coreDim 超限和 UB 溢出问题
  |
步骤3：性能基线采集
  |
  +-- 使用 msprof op 采集上板 Profiling 数据
  +-- 记录 op_summary 中的关键指标
  |
步骤4：瓶颈分析
  |
  +-- 分析流水利用率（vec_ratio, mte2_ratio, scalar_ratio）
  +-- 对比理论参数，定位最大差距
  +-- 使用仿真流水图分析微观瓶颈
  |
步骤5：实施优化
  |
  +-- 根据瓶颈类型选择优化策略
  +-- 计算密集 → 增大分块、CV 融合
  +-- 访存密集 → 存算并行、care_padding、对齐
  +-- 标量退化 → 数据类型优化
  |
步骤6：验证与迭代
  |
  +-- 重新采集性能数据
  +-- 对比优化前后
  +-- 确认结果正确性
  +-- 如仍有瓶颈，返回步骤4
```

### 获取硬件信息

优化前需获取当前硬件的关键参数：

```python
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

print(f"Vector Core 数量: {vectorcore_num}")
print(f"AI Core (Cube) 数量: {aicore_num}")
# Vector Core 数量通常是 AI Core 数量的 2 倍
```

### 典型优化案例

#### 案例1：向量加法从 GPU 迁移

```python
# GPU 原始写法
def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# NPU 优化写法
def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    # 固定核数为物理核数
    NUM_CORE = vectorcore_num
    grid = (NUM_CORE,)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    # 核内循环处理多块数据
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
        y = tl.load(y_ptr + offsets, mask=mask, care_padding=False)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)
```

#### 案例2：coreDim 超限 + UB 溢出复合问题

```python
# 问题：N=1073741824, BLOCK_SIZE=4096
# coreDim = 262144 > 65535 (超限)
# 调大 BLOCK_SIZE=32768 后 coreDim=32768 (合规)
# 但出现 UB 溢出

# 解决：引入 BLOCK_SIZE_SUB 二级分块
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
                       BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)

# 调用
MAIN_BLOCK_SIZE = 32768  # 确保 coreDim 合规
SUB_BLOCK_SIZE = 1024    # 控制 UB 使用量
grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
masked_fill_kernel[grid](inp, mask, value, out, N, MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
```

## 常见问题 (Q&A)

**Q1: 为什么从 GPU 迁移的算子在 NPU 上性能很差？**

A: 最常见的原因是 grid 分核数过多。GPU 习惯用大量 block 并行，但 NPU 的 AI Core 数量有限，过多的 block 会导致多轮调度开销。解决方法是将 grid 固定为物理核数，在核内用 for 循环处理多块数据。

**Q2: 如何判断算子是计算密集型还是访存密集型？**

A: 通过 msprof 采集性能数据，查看 `aiv_mte2_ratio` 和 `aiv_vec_ratio`。如果 MTE2 占比高而 Vector 占比低，说明是访存密集型；反之则是计算密集型。也可以通过理论计算：搬运数据量 / 理论带宽 vs 计算数据量 / 理论算力。

**Q3: 优化后性能仍不理想，下一步怎么办？**

A: 按以下顺序排查：(1) 检查 scalar_ratio 是否过高（数据类型退化）；(2) 检查 MTE2 断流（Tiling 过小或无存算并行）；(3) 检查 Grid 分核是否合理；(4) 使用仿真流水图分析微观瓶颈。

**Q4: TRITON_ALL_BLOCKS_PARALLEL 环境变量的作用是什么？**

A: 当逻辑核数大于物理核数时，启用该环境变量（设为 1），编译器会自动调整逻辑核数量为物理核数，减少调度开销。但仅当逻辑核间可并行时方可启用。

## 相关文档

- [02-tiling-strategy.md](./02-tiling-strategy.md) - 分块策略详解
- [03-autotune-guide.md](./03-autotune-guide.md) - Autotune 使用与 AutoTilingTuner
- [04-care-padding.md](./04-care-padding.md) - care_padding 优化
- [05-cv-fusion.md](./05-cv-fusion.md) - Cube-Vector 融合算子优化
- [06-data-movement-optimization.md](./06-data-movement-optimization.md) - 数据搬运优化
- [07-profiling-guide.md](./07-profiling-guide.md) - 性能分析与瓶颈定位

### 源码参考

- [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md) - NPU 算子开发指南
- [performance_guidelines.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/performance_guidelines.md) - NPU 高性能编程指南
- [migrate_from_gpu.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/migrate_from_gpu.md) - GPU 迁移指南
