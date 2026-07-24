# 高级优化技巧合集

> **触发条件**：Agent 需要进一步优化已跑通的 kernel 性能时，参考本文档进行系统性优化。

## 概述

当 Triton kernel 已在 Ascend NPU 上正确运行，但性能未达预期时，需要从多维度进行高级优化。本文档聚焦 **Profiling 方法**、**编译期优化** 和 **优化检查清单**，其他优化主题请参考专项文档：

| 优化主题 | 专项文档 |
|---------|---------|
| 算子融合 / FixPipe / L0C→UB 直通 | [05-cv-pipeline-optimization.md](05-cv-pipeline-optimization.md)、[11-fixpipe-and-bias-fusion.md](11-fixpipe-and-bias-fusion.md) |
| MultiBuffer / Double Buffer | [07-compile-params.md](07-compile-params.md) |
| 数据搬运 / care_padding | [04-memory-access-patterns.md](04-memory-access-patterns.md)、[08-data-type-precision.md](08-data-type-precision.md) |
| UB 缓存 / 核内再分块 | [03-tiling-and-grid.md](03-tiling-and-grid.md) |
| 910_95 硬件规格 / 对比表 | [00-hardware-quick-ref.md](00-hardware-quick-ref.md) |

---

## 1. Profiling 方法

### 1.1 torch_npu.profiler 用法

```python
import torch_npu

def profile_kernel(fn, *args, result_path="./result_profiling"):
    stream = torch.npu.current_stream()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=3, active=30, repeat=1, skip_first=10
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        experimental_config=experimental_config,
    ) as prof:
        stream.synchronize()
        for i in range(10 + (0 + 3 + 30) * 1):
            fn(*args)
            prof.step()
        stream.synchronize()
```

### 1.2 msprof op 命令行

```bash
msprof op --kernel-name=target_kernel_name --output=$HOME/projects/output python3 test_op.py

export LD_LIBRARY_PATH=/root/CANN/Install_CANN/Ascend/ascend_toolkit/latest/tools/simulator/{soc-version}/lib:$LD_LIBRARY_PATH
msprof op simulator --kernel-name=_layer_norm_fwd_fused --soc-version=Ascend910B3 python3 test_op.py
```

### 1.3 AiCMetrics.PipeUtilization 关键指标

| 指标 | 含义 | 理想值 | 瓶颈信号 |
|------|------|--------|----------|
| aiv_vec_ratio | Vector 流水利用率 | > 80% | < 30% 说明 Vector 断流 |
| aiv_mte2_ratio | MTE2 搬入占比 | < 50% | > 50% 说明搬运瓶颈 |
| aiv_scalar_ratio | Scalar 流水利用率 | < 20% | > 30% 说明标量退化 |
| aic_cube_ratio | Cube 流水利用率 | > 80% | < 30% 说明 Cube 断流 |

### 1.4 瓶颈分类与优化方向

| 瓶颈类型 | 特征 | 优化方向 |
|----------|------|----------|
| 计算密集 | vec_ratio 或 cube_ratio 高 | 增大分块、CV 融合 |
| 访存密集 | mte2_ratio 高 | 存算并行、care_padding、对齐优化 |
| 标量退化 | scalar_ratio 高 | 数据类型优化（避免 int64 ADD/CMP） |
| 同步开销 | 流水图中有大量等待 | 减少同步、care_padding=False |

### 1.5 仿真流水图分析

| 文件 | 可视化方式 | 说明 |
|------|-----------|------|
| trace.json | Chrome `chrome://tracing` 或 MindStudio Insight | 指令时间线 |
| visualize_data.bin | MindStudio Insight | 指令关联看板（含源码映射） |

仿真流水图诊断规则：

```
1. SCALAR/FLOWCTRL 指令饱和 → 标量计算过多
2. MTE2/VECTOR 有规律性断流 → 基本块切分过小
3. 大量空白段 → 流水线未充分利用
```

### 1.6 环境变量辅助分析

| 环境变量 | 说明 | 使用场景 |
|----------|------|----------|
| `TRITON_PRINT_AUTOTUNING=1` | 打印 autotune 调优信息 | 查看最优配置和调优过程 |
| `TRITON_ALL_BLOCKS_PARALLEL=1` | 自动根据物理核数优化逻辑核数 | Grid 分核数过多时 |
| `TRITON_DEBUG=1` | 保存中间编译产物 | 分析编译过程 |
| `TRITON_BENCH_METHOD=npu` | 使用 NPU 专用 benchmark 方法 | autotune 中的性能测量 |
| `TRITON_AUTOTUNE_PARALLEL_COMPILE=1` | 并行编译加速调优 | 减少调优耗时（默认开启） |

---

## 2. 编译期优化

### 2.1 tl.static_range 循环展开

`tl.static_range` 指导编译器在编译期将循环完全展开，消除循环控制开销：

```python
for i in tl.static_range(BLOCK_SIZE):
    x = tl.load(x_ptr + i)
    y = x * x
    tl.store(y_ptr + i, y)
```

**适用场景**：循环次数已知且较小（通常 < 10 次迭代）。`start`、`end`、`step` 必须为 `tl.constexpr`。循环次数过大会导致代码膨胀。

### 2.2 tl.constexpr 常量折叠

将固定数值的参数声明为 `tl.constexpr`，编译器可在编译期进行常量折叠和常量传播：

```python
@triton.jit
def kernel(A, B, C, M, N, K,
           stride_am: tl.constexpr,
           stride_an: tl.constexpr,
           BLOCK_SIZE_M: tl.constexpr,
           BLOCK_SIZE_K: tl.constexpr):
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_an = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offset_am[:, None] * stride_am + offset_an[None, :] * stride_an
```

**适用参数**：BLOCK_SIZE、STRIDE、shape 维度等在 kernel 生命周期内不会变化的固定值。

---

## 3. 性能优化检查清单

### 阶段一：基础适配

- [ ] Grid 分核数对齐物理核数：Vector 算子 = `vectorcore_num`，CV 算子 = `aicore_num`
- [ ] 核内循环处理：使用 `range(pid, NUM_BLOCKS, NUM_CORE)` 模式分配任务
- [ ] 数据类型检查：避免使用 int64 的 ADD/CMP 操作，改用 int32 或 fp32
- [ ] 设备指定：`device='npu'` 替代 `device='cuda'`

### 阶段二：访存优化

- [ ] 尾轴对齐：Vector 算子 32B 对齐，CV 算子 512B 对齐
- [ ] care_padding：对 padding 区域数据无依赖时设置 `care_padding=False`
- [ ] 连续访存：确保 offsets 连续，避免离散访问
- [ ] 先搬后选：大量数据整体搬运到 UB 再 select，优于多次小批量离散加载

### 阶段三：Tiling 优化

- [ ] BLOCK_SIZE 选择：使用 autotune 自动搜索最优分块
- [ ] 二级分块：大 BLOCK_SIZE + 小 BLOCK_SIZE_SUB 避免 UB 溢出
- [ ] for 循环 Tiling：在核内使用 for 循环实现存算并行
- [ ] UB 空间检查：确保单次分块数据量不超过 UB 容量

### 阶段四：存算并行

- [ ] MultiBuffer：启用双缓冲实现存算并行（910_95 需显式 `multibuffer=True`）
- [ ] for 循环配合：存算并行需要 for 循环 Tiling 作为前提
- [ ] UB 空间评估：开启 multiBuffer 后 UB 可用空间减半

### 阶段五：高级优化

- [ ] 算子融合：将 Memory-Bound 操作融合为 Compute-Bound 操作
- [ ] CV 融合：矩阵乘+后处理场景使用 Cube+Vector 协同
- [ ] 编译期优化：`tl.constexpr` 常量折叠、`tl.static_range` 循环展开
- [ ] 精度保护：归约操作升精度到 FP32，矩阵乘法混合精度

### 阶段六：910_95 专项

- [ ] multibuffer 需显式开启（910_95 默认 False）
- [ ] 利用 fixpipe L0C→UB 直通路径
- [ ] UB 256KB 允许更大分块
- [ ] Vector 核心数翻倍，grid 可使用更多核心

---

## 相关文档链接

- [性能分析与瓶颈定位](../docs_triton_ascend/05-Performance-Optimization/07-profiling-guide.md)
- [硬件速查手册](00-hardware-quick-ref.md)
- [编译参数指南](07-compile-params.md)
- [CV 流水线优化](05-cv-pipeline-optimization.md)
- [FixPipe 与 Bias 融合](11-fixpipe-and-bias-fusion.md)
- [内存访问模式](04-memory-access-patterns.md)
- [Tiling 与 Grid 策略](03-tiling-and-grid.md)
