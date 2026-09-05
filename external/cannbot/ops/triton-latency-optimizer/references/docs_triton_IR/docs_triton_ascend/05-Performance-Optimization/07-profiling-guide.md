# 性能分析与瓶颈定位

## 概述

性能优化是一个迭代过程：采集数据 -> 分析瓶颈 -> 实施优化 -> 验证效果。Triton-Ascend 提供了 msProf 工具链进行上板 Profiling 和仿真流水图分析，结合理论参数计算和代码热点分析，可以系统性地定位性能瓶颈。本文档详细描述性能数据的采集方法、分析流程、瓶颈定位方法和常见性能反模式。

## 关键概念

| 概念 | 说明 | 获取方式 |
|------|------|----------|
| msProf | 昇腾性能数据采集和分析工具 | CANN 包自带 |
| 上板 Profiling | 在实际硬件上采集性能数据 | msprof op |
| 仿真流水图 | 在仿真器上采集指令级性能数据 | msprof op simulator |
| op_summary | 算子详细性能信息 CSV 文件 | Profiling 输出 |
| trace.json | 指令时间线文件 | 仿真输出 |
| visualize_data.bin | 指令关联看板数据 | 仿真输出 |
| PipeUtilization | 计算单元和搬运单元耗时占比 | op_summary |
| 理论参数 | 基于硬件规格计算的理想性能 | 手动计算 |
| 瓶颈定位 | 识别限制性能的最慢流水线 | 数据分析 |

## msProf 工具使用

### 上板 Profiling

msProf 工具用于采集和分析运行在昇腾 AI 处理器上算子的关键性能指标。

源码参考：[profiling.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/debug_guide/profiling.md)

#### 基本命令

```bash
# 采集指定 kernel 的性能数据
msprof op --kernel-name=target_kernel_name --output=$HOME/projects/output python3 $HOME/projects/test_op.py

# 采集所有算子的性能数据（不指定 --kernel-name）
msprof op --output=$HOME/projects/output python3 test_op.py
```

#### 示例：采集 LayerNorm 算子性能

```bash
msprof op --kernel-name=_layer_norm_fwd_fused python3 03-layer-norm.py
```

#### 输出文件

| 文件 | 说明 | 用途 |
|------|------|------|
| op_summary_*.csv | 算子详细信息 | 分析流水利用率 |
| PROF_* 文件夹 | 原始性能数据 | 深度分析 |

#### op_summary 关键指标

| 指标 | 含义 | 理想值 | 瓶颈信号 |
|------|------|--------|----------|
| aiv_vec_time(us) | Vector 流水执行时间 | - | - |
| aiv_vec_ratio | Vector 流水利用率 | > 80% | < 30% 说明 Vector 断流 |
| aiv_mte2_time(us) | MTE2 搬入时间 | - | - |
| aiv_mte2_ratio | MTE2 搬入占比 | < 50% | > 50% 说明搬运瓶颈 |
| aiv_mte3_time(us) | MTE3 搬出时间 | - | - |
| aiv_scalar_time(us) | Scalar 执行时间 | - | - |
| aiv_scalar_ratio | Scalar 流水利用率 | < 20% | > 30% 说明标量退化 |
| aic_cube_time(us) | Cube 流水执行时间 | - | - |
| aic_cube_ratio | Cube 流水利用率 | > 80% | < 30% 说明 Cube 断流 |

### 仿真流水图

仿真流水图提供指令级的性能数据，可以直观查看各条流水的执行情况和断流点。

#### 基本命令

```bash
# 设置仿真器路径
export LD_LIBRARY_PATH=/root/CANN/Install_CANN/Ascend/ascend_toolkit/latest/tools/simulator/{soc-version}/lib:$LD_LIBRARY_PATH

# 执行仿真流水图采集
msprof op simulator --kernel-name=_layer_norm_fwd_fused --soc-version=Ascend910B3 python3 03-layer-norm.py
```

#### 支持的 Soc-Version

| Soc-Version | | |
|:---:|:---:|:---:|
| Ascend910A | Ascend310 | Ascend310B1 |
| Ascend910B | Ascend310P1 | Ascend310B2 |
| Ascend910B1 | Ascend310P2 | Ascend310B3 |
| Ascend910B2 | Ascend310P3 | Ascend310B4 |
| Ascend910B2C | Ascend310P4 | - |
| Ascend910B3 | Ascend310P5 | - |
| Ascend910B4 | Ascend310P7 | - |

#### 输出文件

| 文件 | 可视化方式 | 说明 |
|------|-----------|------|
| trace.json | Chrome `chrome://tracing` 或 MindStudio Insight | 指令时间线 |
| visualize_data.bin | MindStudio Insight | 指令关联看板（含源码映射） |

#### trace.json 可视化

**Chrome 浏览器方式**：
1. 在 Chrome 中输入 `chrome://tracing`
2. 将 trace.json 拖入空白处
3. 使用快捷键：W（放大）、S（缩小）、A（左移）、D（右移）

**MindStudio Insight 方式**：
1. 打开 MindStudio Insight
2. 导入 trace.json
3. 查看指令详情、执行时间、调用栈和同步连线

#### visualize_data.bin 可视化

visualize_data.bin 提供与源代码对应的指令关联看板，可以精确定位性能瓶颈对应的代码行。

## 理论参数计算方法

### 搬运理论耗时

```python
# 搬运理论耗时 = 搬运数据量(Byte) / 理论带宽
# 示例：float 数据类型, 4096*4096 矩阵搬运
data_size = 4 * 4096 * 4096  # sizeof(float) * 4096 * 4096 = 64MB
bandwidth = 1.8e12  # 1.8 TB/s (GM 峰值带宽)
latency = data_size / bandwidth  # ≈ 37.28 us

# MTE2 + MTE3 同时搬运时
# 总耗时 = (MTE2搬运量 + MTE3搬运量) / GM带宽
```

### 计算理论耗时

```python
# 计算理论耗时 = 计算数据量(Element) / 理论算力
# 示例：float 数据类型, 32K Element Vector 计算
compute_size = 32 * 1024  # 32K elements
peak_flops = 11.06e12  # 11.06 TOPS (Vector 理论峰值)
latency = compute_size / peak_flops  # ≈ 0.003 us
```

### 理论 vs 实际对比

| 场景 | 理论值 | 实际值 | 差异原因 |
|------|--------|--------|----------|
| LayerNorm MTE2 | 0.1991 us | 远大于理论值 | Tiling 过小，冗余搬运指令 |
| Vector 计算 | 0.003 us | 远大于理论值 | 标量退化、流水断流 |
| Cube 矩阵乘 | 理论值 | 接近理论值 | Cube 利用率通常较高 |

## 瓶颈定位方法

### 方法一：通过上板 Profiling 分析流水情况

查看 op_summary_*.csv 文件，分析各流水线利用率：

```
诊断规则：
1. aiv_vec_ratio < 10%  → Vector 未充分发挥算力
2. aiv_scalar_ratio > 30% → 存在标量退化（数据类型不支持）
3. aiv_mte2_ratio > 50%  → 搬运瓶颈
4. aic_cube_ratio < 30%  → Cube 断流
```

**示例分析**：

| 指标 | 值 | 诊断 |
|------|-----|------|
| aiv_vec_ratio | 8% | Vector 利用率极低 |
| aiv_scalar_ratio | 60% | Scalar 是最长流水 |
| aiv_mte2_ratio | 25% | MTE2 占比适中 |

**结论**：Scalar 是瓶颈，需要分析是否存在标量退化。

### 方法二：通过上板 Profiling 分析 Tiling 情况

检查 Block Dim 是否超过物理核数：

```
诊断规则：
1. Block Dim > 物理Vector核数 → Host调度开销过大
2. Block Dim 远小于物理核数 → 核心利用率不足
3. 推荐 Block Dim = 物理核数
```

**示例**：某处理器有 48 个 Vector 核，但算子下发了 256 个 Block，导致多轮调度。

### 方法三：通过仿真流水图分析流水情况

查看仿真流水图中的断流现象：

```
诊断规则：
1. SCALAR/FLOWCTRL 指令饱和 → 标量计算过多
2. MTE2/VECTOR 有规律性断流 → 基本块切分过小
3. 大量空白段 → 流水线未充分利用
```

### 方法四：通过代码热点分析

使用 visualize_data.bin 在 MindStudio Insight 中查看指令与源码的映射：

```
诊断规则：
1. load 接口对应大量标量指令 → 离散访存或数据类型退化
2. 计算接口对应大量标量指令 → 数据类型不支持向量运算
3. 某行代码对应指令执行拍数异常高 → 优化该行代码
```

### 瓶颈分类与优化方向

| 瓶颈类型 | 特征 | 优化方向 |
|----------|------|----------|
| 计算密集 | aiv_vec_ratio 或 aic_cube_ratio 高 | 增大分块、CV 融合 |
| 访存密集 | aiv_mte2_ratio 高 | 存算并行、care_padding、对齐优化 |
| 标量退化 | aiv_scalar_ratio 高 | 数据类型优化（避免 int64 ADD/CMP） |
| 同步开销 | 流水图中有大量等待 | 减少同步、care_padding=False |
| Tiling 不合理 | 断流、冗余搬运 | 调整 BLOCK_SIZE、autotune |

## 常见性能反模式

### 反模式1：int64/i32 的 Compare 退化为标量

**问题**：i64/i32 的 cmp 在 NPU 上无法启用 Vector，退化为 scalar 计算。

**诊断**：aiv_scalar_ratio 异常高，仿真流水图中 SCALAR 指令饱和。

**解决**：将比较操作数转换为 fp32。

```python
# 优化前
cols = tl.arange(0, BLOCK_N)  # cols is int64
xbar = tl.where(cols < N, x - mean, 0.0)  # cols < N 退化为标量

# 优化后
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)  # 使用 vec_cmp
```

### 反模式2：Grid 分核数过多

**问题**：直接将 GPU 上的 Triton 算子拿到 NPU 上运行，grid 分核数远超物理核数。

**诊断**：op_summary 中 Block Dim 远大于物理核数，Host 调度开销大。

**解决**：固定核数为物理核数，核内循环处理。

```python
# 优化前
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
# 当 n_elements=1073741824, BLOCK_SIZE=2048 时
# coreDim = 524288 >> 物理核数

# 优化后
NUM_CORE = vectorcore_num
grid = (NUM_CORE,)
# 核内使用 for 循环处理
```

### 反模式3：Tiling 过小导致搬运冗余

**问题**：BLOCK_SIZE 过小，导致发射大量冗余搬运指令，MTE2 利用率低。

**诊断**：aiv_mte2_time 远大于理论搬运时间，MTE2 流水有大量断流。

**解决**：增大 BLOCK_SIZE 或使用 autotune 自动搜索。

### 反模式4：无 for 循环导致无法存算并行

**问题**：算子内无 Tiling 切分，单次执行完成，multiBuffer 无法使能。

**诊断**：仿真流水图中 MTE2 和 Vector 完全串行，无重叠。

**解决**：添加 for 循环实现 Tiling。

```python
# 优化前：单次处理
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
data = tl.load(input_ptr + offsets, mask=mask)
result = compute(data)
tl.store(output_ptr + offsets, result, mask=mask)

# 优化后：for 循环 Tiling
num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB
for sub_idx in range(num_sub_blocks):
    sub_offset = pid * BLOCK_SIZE + sub_idx * BLOCK_SIZE_SUB
    offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
    mask = offsets < n
    data = tl.load(input_ptr + offsets, mask=mask)
    result = compute(data)
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 反模式5：尾轴不对齐导致自动补齐

**问题**：Tensor 尾轴大小不满足 32B/512B 对齐要求，硬件自动补齐浪费空间和带宽。

**诊断**：UB 使用量异常大于预期，性能随 shape 变化波动大。

**解决**：使用借轴转置或 1D load 规避自动补齐。

```python
# 问题：shape=(2048, 3) bf16, 尾轴 3*2=6B 不满足 32B 对齐
# 解决：1D load + reshape
conv_state = tl.load(conv_state_ptr + ... + tl.arange(0, 2048 * 3))
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,)
```

### 反模式6：care_padding 导致 MTE2-Vector 同步

**问题**：care_padding=True（默认）时，MTE2 必须等待 Vector 初始化完成，降低并行度。

**诊断**：仿真流水图中 MTE2 和 Vector 存在明显的串行依赖。

**解决**：确认 padding 区域不影响结果后，设置 care_padding=False。

## 环境变量辅助分析

| 环境变量 | 说明 | 使用场景 |
|----------|------|----------|
| `TRITON_PRINT_AUTOTUNING=1` | 打印 autotune 调优信息 | 查看最优配置和调优过程 |
| `TRITON_ALL_BLOCKS_PARALLEL=1` | 自动根据物理核数优化逻辑核数 | Grid 分核数过多时 |
| `TRITON_DEBUG=1` | 保存中间编译产物 | 分析编译过程 |
| `TRITON_BENCH_METHOD=npu` | 使用 NPU 专用 benchmark 方法 | autotune 中的性能测量 |
| `TRITON_AUTOTUNE_PARALLEL_COMPILE=1` | 并行编译加速调优 | 减少调优耗时（默认开启） |

### 使用 TRITON_DEBUG 分析编译产物

```bash
# 设置环境变量保存中间产物
export TRITON_DEBUG=1

# 运行算子，保存 ~/.triton/cache/xxx.ttadapter

# 使用 bishengir-compile 分析 IR
bishengir-compile xxx.ttadapter \
    --target=Ascend910B3 \
    --enable-auto-multi-buffer=True \
    --enable-hfusion-compile=true \
    --enable-hivm-compile=true \
    --enable-triton-kernel-compile=true \
    --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync
```

通过对比 Triton 算子逻辑与 IR 内部的操作，观察是否有未映射成指令的操作，特别是是否存在纯 scalar 搬运或计算。

## 性能优化流程

```
1. 采集性能数据
   |
   +-- msprof op 上板 Profiling
   +-- msprof op simulator 仿真流水图
   |
2. 分析瓶颈
   |
   +-- 对比理论参数
   +-- 分析流水利用率
   +-- 分析 Tiling 情况
   +-- 代码热点分析
   |
3. 实施优化
   |
   +-- 计算密集 → 增大分块、CV 融合
   +-- 访存密集 → 存算并行、care_padding、对齐
   +-- 标量退化 → 数据类型优化
   +-- 同步开销 → 减少同步点
   +-- Tiling 不合理 → autotune
   |
4. 验证效果
   |
   +-- 重新采集性能数据
   +-- 对比优化前后
   +-- 确认结果正确性
```

## 常见问题 (Q&A)

**Q1: msprof 命令执行失败怎么办？**

A: 确认 CANN 包已正确安装，msopprof 可执行文件在 PATH 中。检查 `--soc-version` 参数是否与实际硬件匹配（可通过 `npu-smi info` 查看）。

**Q2: 仿真流水图和上板 Profiling 结果差异大吗？**

A: 仿真流水图提供指令级的精确数据，但不含实际硬件的动态调度影响。上板 Profiling 反映真实性能，但粒度较粗。建议先用上板 Profiling 定位大致瓶颈，再用仿真流水图分析微观原因。

**Q3: 如何快速判断算子性能是否已接近最优？**

A: 对比实际性能与理论性能的比值。如果实际性能达到理论性能的 70% 以上，通常认为已接近最优。理论性能 = max(搬运理论耗时, 计算理论耗时)。

**Q4: op_summary 中多个指标都很低，如何确定主要瓶颈？**

A: 找到耗时最长的流水线。例如，如果 scalar_time 远大于 vec_time 和 mte2_time，则 scalar 是主要瓶颈。如果 mte2_time 最长，则搬运是瓶颈。

**Q5: 如何分析 care_padding 优化的效果？**

A: 对比优化前后的 aiv_mte2_time 和 aiv_vec_ratio。优化后 MTE2 时间应减少，Vector 利用率应提升。也可以通过仿真流水图直观查看 MTE2 和 Vector 的重叠程度。

## 相关文档

- [01-optimization-overview.md](./01-optimization-overview.md) - 优化策略总览
- [02-tiling-strategy.md](./02-tiling-strategy.md) - 分块策略详解
- [04-care-padding.md](./04-care-padding.md) - care_padding 优化
- [06-data-movement-optimization.md](./06-data-movement-optimization.md) - 数据搬运优化

### 源码参考

- [profiling.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/debug_guide/profiling.md) - 性能分析方法
- [performance_guidelines.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/performance_guidelines.md) - NPU 高性能编程指南
- [migrate_from_gpu.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/migrate_from_gpu.md) - GPU 迁移指南
