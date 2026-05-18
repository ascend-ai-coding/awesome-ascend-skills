# VERL 7 项加速特性参考

## 特性一览

| 位 | 名称 | Hydra 参数 | 优化效果 | 类型 |
|----|------|-----------|----------|------|
| 0 | Remove Padding | `actor_rollout_ref.model.use_remove_padding=True` | 减少无效计算 | Type A |
| 1 | Dynamic Batch Size | `actor_rollout_ref.actor.use_dynamic_bsz=True` + ref/rollout | 内存优化+负载均衡 | Type A |
| 2 | Offload | `param_offload`/`optimizer_offload`/`grad_offload` | 显存省 30-50% | Type A |
| 3 | Prefix Cache | `actor_rollout_ref.rollout.enable_prefix_caching=True` | 加速相同前缀生成 | Type B |
| 4 | Recompute | `recompute_granularity/method/num_layers` (3行) | 显存省 30-40% | Type B |
| 5 | Swap Optimizer | `swap_optimizer=True` | 内存优化 | Type B |
| 6 | VPP | `virtual_pipeline_model_parallel_size=2` | 吞吐提升 | Type B |

## 各特性详细说明

### 0. Remove Padding

减少无效 padding token 的计算，提升计算效率。

**适用场景**：样本长度差异大、padding 占比高。

在序列长度分布较均匀的情况下，额外开销可能抵消优化收益。

单独开启吞吐约 46.03。

### 1. Dynamic Batch Size

按 token 实际长度切分 micro batch，提升计算密度。

单独开启易导致显存峰值波动过大触发 OOM，建议搭配 offload/swap_optimizer 使用。

### 2. Offload（Optimizer/Grad Offload）

将模型参数、优化器状态卸载至 CPU 内存，降低 NPU 显存峰值。

- `param_offload`：参数卸载
- `optimizer_offload`：优化器状态卸载（**注意：与 Swap Optimizer 互斥**）
- `grad_offload`：梯度卸载

单独开启运行时间会延长（CPU-NPU 搬运开销），属于"以时间换空间"方案。

### 3. Prefix Cache

复用 KV 缓存中的前缀部分，减少重复计算。

前缀重复率低时优化效果有限，峰值显存可达 98%。

### 4. Recompute（重计算）

重新计算中间激活值替代缓存，典型"时间换空间"优化。

配置参数：
- `recompute_granularity=full`
- `recompute_method=block`
- `recompute_num_layers=8`（可调）

显存占用稳定在 96% 左右，吞吐约 44.17。

### 5. Swap Optimizer

在权重更新阶段将优化器状态换入换出显存，轻量级显存优化。

**互斥警告**：不能与 `optimizer_offload=True` 同时开启！

已知存在保存检查点时 NoneType 报错问题，需手动修复 Megatron 源码。

### 6. VPP（虚拟流水线并行）

通过虚拟流水线切分提升并行效率。

在单机小规模并行下性能反而下降；验证阶段易爆显存，建议 `test_freq=-1` 关闭验证。

## 推荐组合

| 场景 | 掩码 | 说明 |
|------|------|------|
| 全关闭（基准） | `0000000` | baseline 对比 |
| 通用优化 | `1100000` | RmvPad + DynBSZ |
| 显存紧张 | `0010110` | Offload + Recompute + SwapOpt |
| 高吞吐 | `1100011` | RmvPad + DynBSZ + SwapOpt + VPP |

## 环境变量优化

| 变量 | 值 | 效果 |
|------|---|------|
| `TASK_QUEUE_ENABLE` | 2 | 吞吐 56.28，最优单变量 |
| `INF_NAN_MODE_ENABLE` | 1 | 保留原始 Inf/NaN，提升稳定性 |
| `MULTI_STREAM_MEMORY_REUSE` | 1 | 多流内存复用，默认开启 |
