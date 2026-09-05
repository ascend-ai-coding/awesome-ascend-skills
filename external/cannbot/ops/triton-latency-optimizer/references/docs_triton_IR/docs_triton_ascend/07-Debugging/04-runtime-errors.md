# 运行时错误排查

## 概述

Triton-Ascend 的运行时错误发生在 kernel 编译成功后、在 NPU 上执行时出现的问题。常见的运行时错误包括 UB 溢出（部分在编译期检测）、coreDim 超限、精度异常、数据类型不支持等。本文介绍运行时错误的排查方法、日志分析和常见错误的解决方案。

## 关键概念

| 错误类型 | 典型表现 | 排查工具 |
|---------|---------|---------|
| UB 溢出 | `ub overflow` 错误 | `ENABLE_PRINT_UB_BITS=1`，减小 BLOCK_SIZE |
| coreDim 超限 | `coreDim > UINT16_MAX` 错误 | `TRITON_ALL_BLOCKS_PARALLEL=1`，增大 BLOCK_SIZE |
| 精度异常 | 结果与参考值差异过大 | 解释器模式，`tl.device_print` |
| Scalar 退化 | Vector 流水利用率极低 | msProf，仿真流水图 |
| 离散访存 | MTE2 搬运时间过长 | IR 分析，调整 stride/order |
| 分批调度开销 | 算子整体执行时间长 | Grid 配置对齐物理核数 |
| 设备端错误 | NPU 执行中断或超时 | msProf 错误分析 |

## 详细内容

### 1. 常见运行时错误

#### 1.1 UB 溢出

UB 溢出通常在编译阶段被检测到，但也可能在运行时因动态数据规模导致。

**错误信息**：
```
ub overflow, requires 3072256 bits while 1572864 bits available!
(possible reason: large or block number is more than what user expect
due to multi-buffer feature is enabled and some ops need extra local buffer.)
```

**排查方法**：

1. 设置 `ENABLE_PRINT_UB_BITS=1` 获取当前 UB 占用量
2. 检查 BLOCK_SIZE 是否过大
3. 检查是否开启了 multibuffer（需要额外 UB 空间）
4. 检查 int8 类型（特殊处理会占用更大片上空间）

**解决方案**：

```python
# 方案 1：减小 BLOCK_SIZE
BLOCK_SIZE = 1024  # 从 32768 减小

# 方案 2：增加核内 Tiling
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_idx in range(num_sub_blocks):
        offsets = base_offset + sub_idx * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * 2.0, mask=mask)

# 方案 3：关闭 multibuffer
triton.Config({'XS': 128, 'multibuffer': False})
```

**UB 容量参考**：

| 硬件型号 | UB 容量 | 说明 |
|---------|---------|------|
| A2/A3 系列 | 192 KB (1,572,864 bits) | 最常见 |
| Ascend910_95/950 | 256 KB | 较新型号 |

**UB 使用量估算**：

- A2/A3 系列：开启 double buffer 时所有 tensor 总和 <= 96 KB，关闭 double buffer 时 <= 192 KB
- 910_95 系列：开启 double buffer 时所有 tensor 总和 <= 128 KB，关闭 double buffer 时 <= 256 KB
- int8 类型：占用空间约为其他类型的 2 倍

#### 1.2 coreDim 超限

**错误信息**：
```
coreDim=524288 can't be greater than UINT16_MAX
```

**排查方法**：

1. 计算当前 coreDim：`coreDim = ceil(N / BLOCK_SIZE)`
2. 检查是否超过 65535

**解决方案**：

```bash
# 方案 1：环境变量
export TRITON_ALL_BLOCKS_PARALLEL=1
```

```python
# 方案 2：动态计算 BLOCK_SIZE
N = x.numel()
min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
BLOCK_SIZE = max(32768, min_block_size)
```

**注意**：`TRITON_ALL_BLOCKS_PARALLEL` 要求 kernel 逻辑对执行顺序不敏感。

#### 1.3 精度异常

**典型表现**：NPU 运行结果与 PyTorch/CPU/GPU 参考结果差异过大（超出浮点误差范围）。

**排查方法**：

1. 使用解释器模式获取 CPU 基准：

```bash
export TRITON_INTERPRET=1
python your_program.py
```

2. 使用 `tl.device_print` 打印 NPU 上的中间结果：

```python
tl.device_print("intermediate: ", value)
```

3. 逐步对比 CPU 和 NPU 的中间结果

**常见精度问题原因**：

| 原因 | 表现 | 解决方案 |
|------|------|---------|
| 浮点计算顺序差异 | 微小差异（1e-6 量级） | 正常现象，使用 `torch.allclose` 验证 |
| 数据类型退化 | 较大差异 | 检查 int64/int32 是否导致 scalar 退化 |
| bf16/fp16 精度损失 | 中等差异 | 确保关键计算在 fp32 下进行 |
| mask 处理差异 | 特定位置差异 | 检查 boundary_check 和 mask 逻辑 |

#### 1.4 Scalar 退化

**典型表现**：Vector 流水利用率极低（<10%），Scalar 流水成为瓶颈。

**排查方法**：

1. 使用 msProf 采集性能数据，查看 `aiv_vec_ratio` 和 `aiv_scalar_ratio`
2. 使用仿真流水图分析 Scalar 指令占比
3. 使用代码热点分析定位 Scalar 退化的代码行

**常见 Scalar 退化原因**：

| 操作 | 不支持的数据类型 | 退化行为 |
|------|----------------|---------|
| Vector ADD | int64 | 退化为 Scalar 加法 |
| Vector CMP | int64, int32 | 退化为 Scalar 比较 |

**解决方案**：

```python
# Vector CMP 优化：将 int64/int32 转换为 fp32
cols = tl.arange(0, BLOCK_N)  # int64 类型
# 优化前：cols < N 导致 Scalar CMP
xbar = tl.where(cols < N, x - mean, 0.0)

# 优化后：转换为 fp32 使用 Vector CMP
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

#### 1.5 离散访存

**典型表现**：MTE2 搬运时间过长，与理论带宽差距大。

**排查方法**：

1. 使用 msProf 分析 MTE2 时间
2. 计算理论搬运时间：`搬运数据量 / 理论带宽`
3. 对比实际 MTE2 时间与理论时间

**解决方案**：

1. 调整数据布局，确保最低维度连续
2. 使用 `tl.make_block_ptr` 时，确保 stride 反映真实内存布局
3. 避免非对齐访存

### 2. 运行时日志分析

#### 2.1 msProf 上板 Profiling

```bash
msprof op --kernel-name=target_kernel_name --output=$HOME/projects/output python3 your_program.py
```

**关键输出文件**：
- `op_summary_*.csv`：算子详细信息，包含各流水线利用率
- `PipeUtilization.csv`：计算单元和搬运单元耗时占比

**关键指标**：

| 指标 | 含义 | 理想值 |
|------|------|--------|
| aiv_vec_ratio | Vector 流水利用率 | 接近 100% |
| aiv_scalar_ratio | Scalar 流水利用率 | 较低 |
| aiv_mte2_time | MTE2 搬运时间 | 接近理论值 |
| aiv_mte3_time | MTE3 搬运时间 | 接近理论值 |

#### 2.2 msProf 仿真流水图

```bash
export LD_LIBRARY_PATH=/root/CANN/Install_CANN/Ascend/ascend_toolkit/latest/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
msprof op simulator --kernel-name=target_kernel_name --soc-version=Ascend910B3 python3 your_program.py
```

**输出文件**：
- `trace.json`：可用 Chrome (`chrome://tracing`) 或 MindStudio Insight 查看
- `visualize_data.bin`：可用 MindStudio Insight 查看，支持指令关联看板

**分析方法**：

1. 查看 Vector 核的 SCALAR 和 FLOWCTRL 指令是否饱和
2. 查看是否有规律性的断流现象（大量无操作空白段）
3. 使用代码热点分析，查看 load/store 等接口对应的汇编指令类型

#### 2.3 性能瓶颈定位

**方法一：通过上板 Profiling 分析流水情况**

查看 `op_summary_*.csv` 中的流水利用率：
- `aiv_vec_ratio < 10%`：Vector 算力未充分发挥
- `aiv_scalar_ratio` 较高：Scalar 是最长流水，需要优化

**方法二：通过上板 Profiling 分析 Tiling 情况**

检查 Block Dim 是否超过物理核数：
- Block Dim > 物理核数 → Host 调度开销过大 → Tiling 优化

**方法三：通过仿真流水图分析流水情况**

- SCALAR/FLOWCTRL 饱和 → 标量计算优化
- 有规律性断流 → 基本块切分过小 → 流水优化 + Tiling 优化

**方法四：通过代码热点分析**

- load 接口对应标量指令占比高 → 不符合预期 → 标量计算优化

### 3. msProf 错误分析

当 msProf 采集过程中遇到错误时：

1. 检查 CANN 环境变量是否正确设置
2. 检查 `--soc-version` 参数是否与实际硬件匹配
3. 检查 `--kernel-name` 是否正确（使用 kernel 函数名）
4. 检查输出目录是否有写入权限

**支持的 soc-version**：

| Soc-Version | | |
|---|---|---|
| Ascend910A | Ascend310 | Ascend310B1 |
| Ascend910B | Ascend310P1 | Ascend310B2 |
| Ascend910B1 | Ascend310P2 | Ascend310B3 |
| Ascend910B2 | Ascend310P3 | Ascend310B4 |
| Ascend910B2C | Ascend310P4 | - |
| Ascend910B3 | Ascend310P5 | - |
| Ascend910B4 | Ascend310P7 | - |

可通过 `npu-smi info` 命令查看当前硬件版本。

### 4. 常见运行时错误及解决方案

#### 错误 1：UB 溢出（运行时触发）

部分场景下 UB 溢出在编译期无法完全检测，运行时才暴露。

**解决方案**：同编译期 UB 溢出，减小 BLOCK_SIZE 或增加 Tiling。

#### 错误 2：NPU 执行超时

**可能原因**：
1. Grid 分核数过多，分批调度开销大
2. 死锁（kernel 逻辑对执行顺序敏感）
3. 数据规模异常大

**解决方案**：
1. 将 Grid 对齐物理核数
2. 检查 kernel 是否存在跨核依赖
3. 减小数据规模进行测试

#### 错误 3：结果全零或全 NaN

**可能原因**：
1. 数据未正确加载（mask 错误）
2. 数据类型转换错误
3. 除零操作

**排查方法**：
1. 使用 `tl.device_print` 打印加载后的数据
2. 使用解释器模式验证
3. 检查 mask 和 boundary_check 逻辑

#### 错误 4：部分元素结果错误

**可能原因**：
1. 边界处理不正确
2. mask 逻辑有误
3. 数据对齐问题

**排查方法**：
1. 检查 `offsets < N` 等 mask 条件
2. 检查 `boundary_check` 参数
3. 使用解释器模式逐步验证

#### 错误 5：性能远低于预期

**可能原因**：
1. Scalar 退化（数据类型不支持 Vector 操作）
2. 离散访存
3. Grid 分核数过多
4. Tiling 不合理

**排查方法**：
1. msProf 上板 Profiling
2. 仿真流水图分析
3. 代码热点分析

### 5. 理论性能参考

**搬运相关流水理论耗时**：

```
理论耗时 = 搬运数据量 (Byte) / 理论带宽
```

例如：GM 峰值带宽约 1.8 TB/s，搬运 float 类型 4096 * 4096 矩阵：
```
理论耗时 = sizeof(float) * 4096 * 4096 / 1.8 TB/s ≈ 37.28 us
```

**计算相关流水理论耗时**：

```
理论耗时 = 计算数据量 (Element) / 理论算力
```

**注意事项**：
- MTE2/MTE3 同时搬运时共享带宽
- 小数据块搬运对带宽利用率低
- 实测性能通常低于理论值

## NPU 适配要点

1. **UB 溢出是最常见的运行时问题**：严格控制单次处理数据量
2. **msProf 是运行时调试的核心工具**：上板 Profiling + 仿真流水图
3. **Scalar 退化是性能问题的常见原因**：检查数据类型是否导致 Vector 操作退化
4. **Grid 对齐物理核数**：避免分批调度开销
5. **连续访存是性能基础**：确保 stride/order 配置正确

## 常见问题（Q&A）

**Q1：msProf 采集不到数据怎么办？**

A：检查以下几点：
1. `--kernel-name` 是否正确（使用 kernel 函数名，不是文件名）
2. CANN 环境变量是否正确设置
3. 是否有其他 msProf 进程在运行

**Q2：如何判断性能是否已经优化到位？**

A：对比实际性能与理论性能：
1. 计算搬运理论耗时：`数据量 / 带宽`
2. 计算计算理论耗时：`计算量 / 算力`
3. 如果实际耗时接近理论值，说明优化到位

**Q3：解释器模式结果正确但 NPU 上结果错误，一定是硬件 bug 吗？**

A：不一定。常见原因包括：
1. 浮点精度差异（解释器使用 fp64，NPU 使用 fp16/fp32）
2. mask/boundary_check 处理差异
3. 数据类型转换问题
4. NPU 特有的编译器优化导致行为差异

## 相关文档

- [01-调试方法总览](./01-debug-overview.md)
- [02-解释器模式调试](./02-interpreter-mode.md)
- [03-编译错误排查](./03-compile-errors.md)
- [05-调试相关环境变量](./05-environment-variables.md)
- [profiling.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/debug_guide/profiling.md)
- [performance_guidelines.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/performance_guidelines.md)
