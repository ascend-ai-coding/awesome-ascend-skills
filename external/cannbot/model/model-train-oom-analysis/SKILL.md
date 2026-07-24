---
name: external-cannbot-model-model-train-oom-analysis
description: 用于诊断 PyTorch on NPU 大模型训练中的 NPU OOM（Out of Memory）问题。当用户报告训练因内存不足崩溃、出现
  OOM 相关错误（OutOfMemoryError / NPU out of memory / workspace allocator / HCCL memory）、或需要进行训练内存优化时，触发本技能。按照日志分类
  → 静态估算 → snapshot 深度分析 → 优化建议的流程定位和解决问题。
original-name: model-train-oom-analysis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# model-train-oom-analysis 技能

用于 NPU 大模型训练 OOM 问题的系统化诊断与解决。

## 适用范围与约定

- 本技能的**方法论适用于 torchtitan-npu 等 PyTorch on NPU 训练框架**。
- 文中出现的 `torchtitan_npu/...` 路径与配置项均为 **torchtitan-npu 框架的具体示例**。torchtitan-npu 采用 **Python config registry**：配置是 `torchtitan_npu/models/<model>/config_registry.py` 中返回 `TrainerConfig(...)` 的函数（各项为 `TrainingConfig`/`ProfilingConfig`/`ActivationCheckpointConfig`/`OptimizerConfig`/`ParallelismConfig` 等 dataclass 字段），用 `--config <注册名>` 选择，并支持 `--<section>.<field>=<value>` 形式的 CLI override（不再使用 TOML 配置文件）。
- 使用其他框架时，请将上述路径、配置项与启动方式映射到对应框架的训练入口、配置项与并行接口。
- 内存估算公式、snapshot 分析维度、优化决策矩阵与框架无关，可直接复用。

## 参考资料

- 分析脚本：[scripts/analyze_snapshot.py](scripts/analyze_snapshot.py)
- 内存估算模板：[references/memory-estimation-templates.md](references/memory-estimation-templates.md)
- Mosaic 工具指南：[references/mosaic-analysis-guide.md](references/mosaic-analysis-guide.md)
- PyTorch Mosaic 教程：<https://docs.pytorch.org/tutorials/beginner/mosaic_memory_profiling_tutorial.html>

> 脚本路径均相对本技能目录。若以源码方式调用，请补全到 `model/model-train-oom-analysis/scripts/analyze_snapshot.py`。

## 适用场景

- 训练过程中出现 OOM 崩溃（`OutOfMemoryError`、`NPU out of memory` 等）。
- 需要评估当前模型/配置的内存需求是否合理。
- 训练前需要预估内存占用，选择合适的并行策略和 batch_size。

## 不适用场景

- 精度问题（loss 偏离、NaN/Inf）→ 使用 `model-train-accuracy-debug` 技能。
- 纯性能瓶颈（训练慢但不 OOM）。
- 非 NPU 设备的内存问题。

## 所需输入

- OOM 错误的完整训练日志（必需）。
- 训练配置（torchtitan-npu 为 config registry 注册名，经 `--config` 传入；其他框架为对应配置定义）（必需）。
- NPU 设备显存规格（如 Ascend 910B: 64GB）。

## 工作流

### Step 0：日志分析 — 确定 OOM 类型与时机（必做）

**这是所有后续步骤的前提，必须首先完成。**

分析训练日志中的 OOM 报错信息：

```bash
# 搜索 OOM 相关关键字
rg -n -i "out.of.memory|OOM|OutOfMemory|NPU.*memory|allocator|HCCL.*memory" <train_log>

# 确认 OOM 发生在第几步
rg -n -i "step|iteration" <train_log> | tail -5
```

根据报错关键字判定 OOM 来源，根据发生时机判定是配置问题还是内存泄漏：

| OOM 来源 | 特征关键字 | 后续步骤 |
|----------|-----------|----------|
| NPUWorkspaceAllocator | `workspace allocator`、`workspace memory` | → Step 1 → Step 2 |
| HCCL 通信 | `HCCL`、`hccl`、`communication` | → Step 1 → Step 2 |
| NPUCachingAllocator | `CachingAllocator`、`torch.OutOfMemoryError`、`NPU out of memory` | → Step 1 → Step 3 |
| 无法判断 | 不含上述明确关键字 | → Step 1 → Step 3 |

> [!IMPORTANT]
> OOM 类型判定直接决定后续方向。Workspace/HCCL OOM 需调水线（Step 2）；CachingAllocator OOM 需先估算配置是否超限（Step 1），再决定是否需要 snapshot 深度分析（Step 3）。

> [!WARNING]
> **渐进型 OOM 识别**：如果 OOM 发生在训练后期（如 step 10+，而非第 1-2 步），很可能是**内存泄漏**而非配置不足。
>
> - OOM 在 step 0-2 → 配置问题 → Step 1 → Step 4
> - OOM 在 step N（N 较大）→ 疑似泄漏 → Step 1 → Step 3（含 3.5 泄漏检测 + 3.6 代码审查）

### Step 1：静态内存估算 — 判断配置是否可行（必做）

**在采集 snapshot 之前，先做静态估算，快速判断当前配置下内存是否本身超限。**

#### 1.1 收集参数

**模型参数**：`dim`、`n_layers`、`n_heads`、`vocab_size`、`inter_dim`、MoE 参数（专家数、共享专家、激活专家数）等，从模型配置/定义获取。
> torchtitan-npu 示例：从 `torchtitan_npu/models/<model>/__init__.py`（flavors 函数，如 `_make_dsv3_model_config`）和 `torchtitan_npu/models/<model>/model.py`（Config，继承上游 torchtitan 的 ModelArgs）获取。

**训练参数**（从配置文件获取）：`local_batch_size`、`seq_len`、TP/PP/DP/EP/CP 各并行度、`activation_checkpoint` 策略、优化器类型。

#### 1.2 估算与判断

详细公式参见 [references/memory-estimation-templates.md](references/memory-estimation-templates.md)，核心：

```
训练峰值内存 ≈ 参数内存 + 梯度内存 + 优化器状态 + 激活内存 + 临时缓冲
估算总内存 ≤ NPU 总显存 × 0.95  (预留 5% 安全余量)
```

#### 1.3 决策路由

| 估算结果 | 判断 | 后续步骤 |
|---------|------|----------|
| 峰值估算 > NPU 显存 × 0.95 | **配置本身超限**，无需 snapshot | → 直接 Step 4 调整配置 |
| Workspace/HCCL OOM 且估算合理 | 临时内存不足 | → Step 2 调水线 |
| CachingAllocator OOM 且估算接近上限 | 需确认实际分布 | → Step 3 snapshot 分析 |
| 估算远低于上限但仍 OOM | 可能碎片化或泄漏 | → Step 3 snapshot 分析 |

> [!TIP]
> **配置超限是最常见的 OOM 根因**——参数量过大、batch_size/seq_len 过大、未启用激活重计算、优化器状态未分片/卸载等都会导致显存需求超出单卡上限。此时无需采集 snapshot，直接根据 Step 4 优化矩阵调整配置即可。

### Step 2：临时内存 OOM — 配置 PTA 内存水线

当 OOM 来自 `NPUWorkspaceAllocator` 或 HCCL 通信，且 Step 1 估算表明模型本身可以放下时，问题根因是 PTA 缓存分配器占用了过多显存，导致临时工作空间或通信缓冲区无法分配。

通过限制 PTA 显存占用上限，为临时内存留出空间。其底层基于 `torch.npu.set_per_process_memory_fraction()`，各框架暴露的配置项不同。

> torchtitan-npu 示例：在配置函数的 `training=TrainingConfig(...)` 中设置，或启动时 CLI override：
> ```python
> training=TrainingConfig(torch_npu_memory_ratio=0.9),  # 限制 PTA 最多使用 90% 显存，留 10% 给 workspace/HCCL
> ```
> ```bash
> --config <注册名> --training.torch_npu_memory_ratio=0.9
> ```
> 该功能在 `torchtitan_npu/train.py` 的 `_patch_for_train_npu_memory()` 中实现。其他框架可直接在训练入口调用 `torch.npu.set_per_process_memory_fraction(ratio)`。

- 首次尝试可设 `0.9`，若仍 OOM 降低到 `0.85`。
- 若过低导致 CachingAllocator 本身 OOM，说明模型内存需求过高，需回到 Step 1 重新估算并进入 Step 4 优化配置。

### Step 3：CachingAllocator OOM — 采集并分析 Memory Snapshot

当 Step 1 估算无法确认问题根因（估算接近上限、或估算合理但仍 OOM）时，采集 memory snapshot 做详细分析。

#### 3.1 启用 Memory Snapshot 采集

通过 `torch.npu.memory._record_memory_history()` + `_dump_snapshot()` 采集，多数训练框架已封装该能力。

> torchtitan-npu 示例：在 `torchtitan_npu/models/<model>/config_registry.py` 对应配置函数的 `profiling=ProfilingConfig(...)` 中开启：
> ```python
> profiling=ProfilingConfig(
>     enable_memory_snapshot=True,
>     save_memory_snapshot_folder="memory_snapshot",
> ),
> ```
> 或启动时用 CLI override（无需改注册文件）：
> ```bash
> --config <注册名> --profiling.enable_memory_snapshot --profiling.save_memory_snapshot_folder=memory_snapshot
> ```
> torchtitan 在 OOM 发生时也会自动转储快照。

#### 3.2 脚本化分析

使用 [scripts/analyze_snapshot.py](scripts/analyze_snapshot.py) 对 `.pickle` 文件进行多维度分析：

> [!NOTE]
> snapshot 基于 pickle 反序列化加载，请仅分析**可信来源**（自己训练产出）的 `.pickle` 文件，不要加载来历不明的快照。

```bash
# 单快照分析
python scripts/analyze_snapshot.py <snapshot.pickle>

# JSON 输出（便于程序化处理）
python scripts/analyze_snapshot.py <snapshot.pickle> --json

# 自定义 Top-N
python scripts/analyze_snapshot.py <snapshot.pickle> --top-n 20
```

脚本输出包含 **6 个维度**的分析：

**① 内存分配概览**：Reserved / Allocated / Active / 碎片化率 / 设备利用率

**② 语义类别分类**：按 Mosaic 分类体系将内存细分为激活、梯度、优化器状态、参数、通信缓冲、注意力层、MoE 层等类别，每项给出大小和占比

**③ 峰值内存分析**：定位峰值时刻，展示峰值各类别占比和 Top 调用栈

**④ 时间线分析**：追踪 alloc/free 事件序列，识别训练阶段（增长/释放/稳定），检测潜在内存泄漏

**⑤ 碎片化深度分析**：空闲块分布、最大空闲块、小块占比，并给出碎片化等级评估

**⑥ Top-N 大块分配**：最大内存块的大小、类别和完整调用栈

**分析判断矩阵**：

| 指标 | 阈值 | 对应优化措施 |
|------|------|-------------|
| 激活内存占比 | > 40% | 启用 activation checkpoint (`--activation_checkpoint.mode=full` 或 `selective`) |
| 优化器状态占比 | > 40% | 启用优化器状态卸载/分片（如 swap optimizer、FSDP 分片） |
| 梯度内存占比 | > 30% | 增大 TP/PP 并行度 |
| 通信缓冲占比 | > 20% | 调整并行策略、检查 HCCL 配置 |
| 碎片化率 | > 20% | 调用 `torch.npu.empty_cache()`、调整分配策略 |
| 其他/未知占比 | > 20% | 使用 Mosaic 深度分析定位来源 |
| 潜在泄漏标记 | 触发 | 对比多 step snapshot 确认 |

#### 3.3 深度分析 — Mosaic 工具（可选）

当基础分析无法充分定位问题时（尤其是 "其他/未知" 类别占比高），使用 [Mosaic](https://github.com/facebookresearch/mosaic) 做更精确的分析。详见 [references/mosaic-analysis-guide.md](references/mosaic-analysis-guide.md)。

```bash
# 峰值栈追踪
mosaic_get_memory_usage_peak --snapshot <snapshot.pickle>

# 分类可视化 Profile
mosaic_get_memory_profile --snapshot <snapshot.pickle> \
    --out-path profile.html \
    --profile categories \
    --preserve-allocation-order

# 自定义模式匹配（追踪特定操作的内存占用）
mosaic_get_memory_profile --snapshot <snapshot.pickle> \
    --out-path custom_profile.html \
    --profile custom \
    --custom-profile '{"hccl": "hccl", "expert": "expert", "moe": "moe"}'
```

#### 3.4 对比分析 — 双快照差异对比

对比两个场景（如：开启 AC 前后、不同 batch_size、不同并行策略）的内存差异：

```bash
python scripts/analyze_snapshot.py \
    <baseline.pickle> <modified.pickle> \
    --label-a "无AC" --label-b "有AC"
```

输出包含：各指标的并排对比表格、各类别的内存变化（增/减量和百分比）、关键洞察。

#### 3.5 泄漏检测 — 多步骤 Snapshot 对比

**当 Step 0 判断为渐进型 OOM（OOM 发生在训练后期）时执行此步。**

```bash
# 传入 3+ 个不同 step 的 snapshot，自动分析增长趋势
python scripts/analyze_snapshot.py \
    step5.pickle step10.pickle step20.pickle
```

| 现象 | 结论 | 下一步 |
|------|------|--------|
| 内存单调增长，某类别持续增长 | 确认泄漏，定位到具体类别 | → Step 3.6 代码审查 |
| 内存增长但后期稳定 | warmup 阶段正常增长 | → 排除泄漏，回到 Step 4 |
| 内存未增长但仍 OOM | 非泄漏，内存本身不足 | → Step 1 估算 + Step 4 优化 |

#### 3.6 代码审查 — 泄漏根因定位

**当 Step 3.5 确认存在泄漏时执行此步。**

审查 Checklist：

- [ ] **异步通信 + 重计算交互**：当 `activation_checkpoint` + TP 同时启用时，`redistribute(async_op=True)` 产生的异步 tensor 在重计算过程中可能未被正确 `wait`，导致中间张量无法释放。检查 `RowwiseParallel`/`ColwiseParallel` 的输出处理函数是否显式调用了 `wait_tensor()`
- [ ] **Tensor 引用持有**：检查是否有调试代码、全局变量或日志记录意外持有了 tensor 引用
- [ ] **自定义 autograd Function**：检查 `save_for_backward` 保存的 tensor 是否在 `backward` 中被正确释放
- [ ] **Hook 注册**：检查是否有未清理的 forward/backward hook 持有 tensor

```bash
# 检查并行化代码中的异步通信（路径以实际框架的并行化模块为准）
rg -n "async_op|redistribute|_prepare_output_fn" <parallelize_module>

# 检查是否有 wait_tensor 配对
rg -n "wait_tensor|async_op=True" <source_root>
```

> [!TIP]
> 通信类泄漏的典型修复方式是在 `redistribute` 后显式调用 `torch.ops._c10d_functional.wait_tensor()` 确保异步操作完成。

### Step 4：优化建议矩阵

根据 Step 0~3 的分析结果，按照以下矩阵给出针对性优化建议（配置项以 torchtitan-npu 为例，其他框架对应调整）：

> 配置项以 torchtitan-npu 为例，给出 CLI override 写法（等价于在 config registry 配置函数对应 dataclass 字段中修改）。

| 问题诊断 | 优化措施 | 配置修改（torchtitan CLI override 示例） | 预期效果 |
|---------|---------|-------------|---------|
| 临时内存不足（Workspace/HCCL OOM） | 配置内存水线 | `--training.torch_npu_memory_ratio=0.85~0.95` | 为临时内存预留空间 |
| 激活内存过大 | 启用全量激活重计算 | `--activation_checkpoint.mode=full` | 激活内存降至最低 |
| 激活内存过大（折中） | 选择性激活重计算 | `--activation_checkpoint.mode=selective --activation_checkpoint.selective_ac_option=op` | 激活内存适度降低 |
| 激活内存过大 | 减小 batch_size | `--training.local_batch_size` 减半 | 激活内存近似线性降低 |
| 激活内存过大 | 减小 seq_len | `--training.seq_len` 减半 | 激活内存近似线性降低 |
| 优化器状态过大 | 启用 swap optimizer | `--optimizer.swap_optimizer --optimizer.swap_optimizer_times=16` | 优化器状态卸载到 CPU |
| 参数+梯度内存过大 | 增大 TP 并行度 | `--parallelism.tensor_parallel_degree` ↑ | 参数/梯度按 TP 切分 |
| 参数+梯度内存过大 | 增大 PP 并行度 | `--parallelism.pipeline_parallel_degree` ↑ | 参数按层切分到不同卡 |
| MoE Expert 内存过大 | 增大 EP 并行度 | `--parallelism.expert_parallel_degree` ↑ | Expert 在更多卡间分布 |
| 内存碎片化严重 | 清理缓存 | 在代码中适时调用 `torch.npu.empty_cache()` | 释放碎片化内存 |
| 长序列内存过大 | 启用 Context Parallel | `--parallelism.context_parallel_degree` ↑ | 序列维度切分 |
| 内存泄漏（通信类） | 异步通信加 wait | 修改并行化代码，在 `redistribute` 后调用 `wait_tensor()` | 消除泄漏 |
| 内存泄漏（其他类） | 审查引用持有 | 检查并修复意外的 tensor 引用持有 | 消除泄漏 |

决策优先级（按推荐顺序）：

1. **首先**：确认 OOM 类型（一次性 vs 渐进型），若为渐进型先做泄漏检测
2. **若为泄漏**：通过多步骤 snapshot 定位泄漏类别，审查对应代码
3. **若为临时内存 OOM**：先调水线
4. **然后**：启用 activation checkpoint（对训练效果无影响，仅牺牲少量性能）
5. **接着**：启用 swap_optimizer（减少 NPU 内存压力，增加少量 CPU 通信开销）
6. **再考虑**：调整并行策略（可能需要更多卡或修改训练逻辑）
7. **最后**：减小 batch_size / seq_len（影响训练效率和收敛行为）

### Step 5：补充工具 — msmemscope（可选）

当上述步骤无法充分定位问题时，可使用 msmemscope（MindStudio MemScope）进行更深层次的内存分析。

> 项目地址：<https://gitcode.com/Ascend/msmemscope>

msmemscope 提供以下高级分析能力：

- **内存泄漏检测**：识别训练过程中随 step 增长的异常内存分配
- **内存拆解分析**：按组件细粒度分解显存使用（比 memory snapshot 更精细）
- **内存对比监测**：对比不同训练阶段（warmup vs 稳定训练）的内存差异

使用方式参考 msmemscope 官方文档中的 PyTorch 采集示例和分析指南。

## 输出要求

最终诊断报告必须包含：

- OOM 类型判定（来源及日志证据）
- 静态内存估算结果（参数/梯度/优化器/激活各占多少）
- memory snapshot 分析结果（若已采集：各类别占比、碎片化率、Top 内存分配）
- 具体优化建议（含配置修改示例）
- 优化后的预期内存占用
- 是否需要进一步分析的建议

## 关键路径（torchtitan-npu 示例）

> 以下路径为 torchtitan-npu 框架的定位锚点示例；使用其他框架时映射到对应的训练入口、模型定义与并行模块。

| 类别 | 路径 |
| --- | --- |
| 训练入口 | `torchtitan_npu/entry.py` |
| 训练 patch（含内存水线） | `torchtitan_npu/train.py` |
| 自定义配置 | `torchtitan_npu/config/configs.py` |
| 模型定义 | `torchtitan_npu/models/<model>/model.py` |
| 模型参数 flavors | `torchtitan_npu/models/<model>/__init__.py` |
| 训练配置 | `torchtitan_npu/models/<model>/config_registry.py`（`--config` 传注册名，非文件路径） |
| 并行逻辑 | `torchtitan_npu/models/<model>/parallelize.py` |
