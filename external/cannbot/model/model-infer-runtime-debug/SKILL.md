---
name: external-cannbot-model-model-infer-runtime-debug
description: 基于 PyTorch 框架的昇腾 NPU 模型推理运行时错误诊断与修复技能。系统化排查模型加载、初始化、推理执行全链路的运行时错误，包括
  aicore timeout、HCCL 通信错误、OOM、算子约束违反、推理卡住等。触发场景：NPU 运行时错误（RuntimeError、aicore timeout
  507014、HCCL timeout、device synchronize 失败、kernel crash、EZ9999/EE9999 错误码）、推理过程卡住不返回、权重加载阶段
  crash、模型加载成功但 forward 失败、分布式推理某些 rank 挂死等。
original-name: model-infer-runtime-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# NPU 推理运行时错误诊断与修复

核心理念：**先定位再修复**。不覆盖精度问题，精度调优见 model-infer-precision-debug。 —— 通过二分法逐步缩小故障范围，避免在错误方向浪费时间。

---

## 诊断路径总览

```
问题发生
  │
  ├── 有明确错误信息 → 通用诊断流程
  │     ├── aicore timeout (507014) → 二分法定位
  │     │     ├── 定位到阶段 → 层 → 模块 → 算子
  │     │     └── 查表「常见算子约束」→ 修复策略 → 验证
  │     ├── OOM → 检查参数量 / batch_size / seq_len / 中间 buffer
  │     ├── Shape 不匹配 → 检查 TP/EP 切分维度 / Prefill vs Decode 分支
  │     └── HCCL timeout → 检查各 rank 代码路径一致性 / 通信组创建
  │
  └── 无明确错误 → npu-smi 状态检查
       ├── 部分 rank 缺失 / HBM 不均 → 多卡部署诊断
       │     ├── 权重加载 crash → TP 切分越界 / vocab pad
       │     ├── Config 字段缺失 → Config 兼容性
       │     └── 静默失败 → 逐 rank 检查日志
       ├── 有进程 + HBM 不变 + CPU 100% → 推理卡住诊断 - 模型构造阶段
       ├── 有进程 + HBM 不变 + CPU 0% → 推理卡住诊断
       │     ├── 检查残留进程
       │     ├── eager 也卡 → sync+print 定位（通用流程第二步）
       │     └── 仅 graph 卡 → 比较 eager/graph 代码路径差异
       └── 0 进程 → 查 rank 0 日志 → 通用诊断流程
```

### npu-smi 状态检查

当没有明确错误信息时，通过设备状态判断问题类型：

```bash
npu-smi info
```

| npu-smi 表现 | 含义 | 诊断路径 |
|-------------|------|---------|
| 前 N 进程存在，后面缺失，HBM 不均 | 高位 rank 在权重加载时 crash | → 多卡部署诊断 |
| 8 进程，HBM 均匀，AICore 全 0% | 所有 rank 卡在通信等待 | → 推理卡住诊断 |
| 有进程，HBM 不变 ~3GB，CPU 100%+ | CPU 密集操作（模型构造/权重初始化） | → 推理卡住诊断 - 模型构造阶段 |
| 有进程，HBM 不变 ~3GB，CPU 0% | 死锁/通信等待 | → 推理卡住诊断 |
| 8 进程，HBM 持续增长 | 权重加载中 | 等待 |
| 0 进程 | 全部 crash | 查 rank 0 日志 → 通用诊断流程 |

---

# 通用诊断流程

适用于有明确错误信息的场景（aicore timeout、HCCL error、OOM、shape mismatch 等）。按分类 → 定位 → 查表 → 修复 → 验证线性推进。

## 第一步：错误分类

拿到错误信息后，先判断属于哪一类。不同类别的根因和排查路径完全不同，分类错误会浪费大量时间。

### A. aicore timeout（错误码 507014）

**特征**：`AclrtSynchronizeDeviceWithTimeout`、`error code is 507014`、`aicore timeout`、`fftsplus aivector error`、`Kernel task happen error, retCode=0x25`

**含义**：某个 NPU 算子在设备上执行时超时未返回。这是最常见也最难排查的错误，因为报错的位置（synchronize 调用处）通常不是出错的位置（某个具体算子）。

**常见根因**：
- 算子入参违反硬件约束（如 MC2 要求 ep_world_size >= 16，见「常见算子约束」）
- 算子 shape 超出硬件限制（如单次 matmul 的 M/N/K 维度超限）
- 死锁：部分 rank 走了不同的通信路径，导致集合通信永远等不齐
- 内存越界：slot_mapping/block_table 索引超出 KV cache 分配范围

### B. HCCL 通信错误

**特征**：`HCCL_CONNECT_TIMEOUT`、`HCCL error`、`AllReduce/AllToAll timeout`、`HCCL_EXEC_TIMEOUT`

**含义**：分布式通信操作超时或失败。

**常见根因**：
- 各 rank 进入通信操作的顺序/次数不一致（代码分支导致部分 rank 跳过某次通信）
- 通信组创建时参数错误（group_stride、group_num 不匹配 world_size）
- 网络问题（跨节点时 HCCL_IF_IP 配置错误）
- 某些 rank 已经 crash 但其他 rank 还在等它参与通信

### C. OOM（显存不足）

**特征**：`NPU out of memory`、`Tried to allocate X GiB`、`ENOMEM`

**含义**：NPU HBM 不够。

**常见根因**：
- 模型参数/KV cache/activation 总和超过单卡显存
- 中间 tensor 未及时释放（常见于 MoE EP all-to-all 中间 buffer）
- batch_size 或 seq_len 超出预期

### D. Shape 不匹配

**特征**：`RuntimeError: shape mismatch`、`expected size X but got Y`、`mat1 and mat2 shapes cannot be multiplied`

**含义**：tensor 维度不对。

**常见根因**：
- TP 切分后维度未正确除以 tp_size
- EP 切分后每卡专家数计算错误
- Prefill/Decode 分支传入了错误 shape 的 tensor

### E. 算子约束违反

**特征**：不直接报约束错误，通常表现为 aicore timeout 或 SIGABRT。需要通过二分法定位后，查阅算子文档确认。

**含义**：NPU 自定义算子对入参有隐含约束（数据类型、维度范围、硬件平台限制），违反时行为未定义。

---

## 第二步：二分法定位（sync+barrier 检查点法）

核心方法。当错误发生在 `torch.npu.synchronize()` 时，错误信息只告诉你"设备上有东西出错了"，不告诉你具体哪个算子。通过插入检查点逐步缩小范围。

### 2.1 定位到阶段（Prefill vs Decode）

在 runner 的 `model_inference` 方法中，prefill 和 decode 调用之间插入同步检查：

```python
# 在 prefill 之后、decode 之前
torch.npu.synchronize()
logging.info("prefill passed")
```

**关键陷阱**：NPU 操作是异步的。`model.prefill()` 返回不代表 prefill 完成 —— 只有 `torch.npu.synchronize()` 返回才代表所有已提交的操作完成。所以报错位置（synchronize 调用处）不一定是出错位置（某个具体算子）。确保检查点覆盖所有可能的异步操作。

### 2.2 定位到层

确认出错阶段后（如 Decode），在每个 decoder layer 之间插入检查点：

```python
for i, layer in enumerate(self.layers):
    hidden_states = layer(hidden_states, ...)
    torch.npu.synchronize()
    logging.info(f"layer {i} passed")
```

如果 Layer 0 就超时 —— 问题在第一层内部。如果 Layer 2 超时 —— 问题在第 2 层或其子模块。

### 2.3 定位到模块

进入出错层，在各子模块之间插入检查点：

```python
# 在 DecoderLayer.forward 中
hidden_states = self.input_layernorm(hidden_states)
torch.npu.synchronize(); logging.info("  norm passed")

hidden_states = self.self_attn(hidden_states, ...)
torch.npu.synchronize(); logging.info("  attention passed")

hidden_states = self.mlp(hidden_states)  # MoE
torch.npu.synchronize(); logging.info("  MoE passed")
```

### 2.4 定位到算子

进入出错模块，在每个 NPU 算子调用之间插入检查点。此时通常可以定位到具体算子及其入参，结合下文「常见算子约束」判断根因。

### 2.5 注意事项

- **先用 `torch.npu.synchronize()` 做单卡定位**，确认是哪个算子出错。不要一开始就加 `dist.barrier()`——barrier 会在某 rank 已崩溃时导致其他 rank 永久挂起
- **怀疑多 rank 不同步时**，用 `dist.barrier()` 或 `torch.distributed.monitored_barrier()` 确认各 rank 是否走到同一位置。`monitored_barrier` 在超时后会报告哪个 rank 未到达
- **查看所有 rank 的日志**，确认各 rank 到达的检查点是否一致（不一致说明有条件分支差异 → 通信死锁）
- **不要一次插入太多检查点** —— 每次 sync 有开销，且大量 sync 可能改变时序。先粗粒度，再细粒度
- **保留日志**：把每轮定位的检查点输出保存，供后续分析

---

## 第三步：常见 NPU 算子约束速查

以下约束来自本仓实际调试经验，定位到具体算子后应回到当前调用的 API 签名和官方文档核对，不要跨算子套用规则。违反约束的典型表现是 aicore timeout。

### MoE 相关算子

| 算子 | 约束 | 违反表现 | 解决方案 |
|------|------|---------|---------|
| `npu_moe_distribute_dispatch_v2` (MC2) | Atlas A2: `ep_world_size` 须为 {16,32,64,128,256}（fullmesh）或 {16,32,64}（hierarchy） | aicore timeout，Decode 阶段挂死 | EP<16 时回退到 double_routing 路径（npu_moe_init_routing_v2 + manual all_to_all） |
| `npu_moe_distribute_combine_v2` (MC2) | 同上 | 同上 | 同上 |
| `npu_moe_init_routing_v2` | 无 EP 最小限制，但 `moe_chunk_max_len` 为 0 可能导致空 tensor | SIGABRT 或 shape error | 设置合理的 moe_chunk_max_len（如 1024） |
| `npu_moe_gating_top_k` | `k` 不能超过专家总数 | 静默错误输出或 crash | 检查 reduced model 的 moe_topk 是否已调整 |

### FA (Flash Attention) 算子

| 约束 | 说明 |
|------|------|
| `head_num` 须匹配实际 Q heads | MLA 模型中 head_num = num_attention_heads，不是 kv_heads |
| Prefill `sparse_mode=3` 需配合 `atten_mask` | mask shape 须为 `[1, 1, max_s, max_s]`，不能省略 |
| `input_layout` 按 API 和阶段区分 | `npu_fusion_attention` 用 BSH/SBH/BSND/BNSD/TND；`npu_fused_infer_attention_score` 用复合 layout 如 TND_NTD、BSND_NBSD，以实际 API 签名为准 |
| Decode PA 模式下 `block_table` | shape `[batch, max_blocks]`，值不能超出 cache 分配的 block 数 |
| NZ 格式 cache | 需要 `torch_npu.npu_format_cast` 转换，且 hidden_dim 须为 16 的整数倍 |

> `actual_seq_lengths` 相关参数以"当前调用算子的签名与官方 API 文档"为准，不跨算子套用规则：
> - `torch_npu.npu_fused_infer_attention_score` / `torch_npu.npu_fusion_attention`：`actual_seq_lengths*` 为 `List[int]`（文档为 int64 语义）
> - `torch_npu.npu_kv_quant_sparse_flash_attention`：`actual_seq_lengths_query/kv` 为 `Tensor`（文档为 int32）

### 通信算子

| 约束 | 说明 |
|------|------|
| `all_to_all_single` 的 split sizes | input_splits 和 output_splits 之和须分别等于 input 和 output 的第 0 维 |
| 通信组内所有 rank 须同时到达 | 任何条件分支差异都可能导致部分 rank 跳过通信 → 死锁 |
| HCCL group 创建顺序 | 所有 rank 须以相同顺序创建相同的通信组 |

---

## 第四步：修复策略模式

定位根因后，选择合适的修复策略。

### 模式 1：算子回退（Operator Fallback）

当某个高性能算子在当前硬件/配置下不可用时，回退到功能等价但限制更少的替代算子。

```python
# 示例：MC2 MoE 算子 EP<16 时回退到 double_routing
def forward(self, hidden_states, is_prefill):
    topk_indices, topk_weights, _ = self.router(hidden_states)
    if is_prefill:
        return self.moe_infer_double_routing(hidden_states, topk_indices, topk_weights)
    else:
        if self.moe_ep_size < 16:
            # MC2 requires ep_world_size >= 16 on A2; fall back to double_routing
            return self.moe_infer_double_routing(hidden_states, topk_indices, topk_weights)
        return self.moe_infer_dispatch_combine(hidden_states, topk_indices, topk_weights)
```

**适用场景**：硬件约束、EP/TP size 不满足算子要求
**注意**：回退通常有性能损失，需记录在优化报告中

### 模式 2：参数修正（Parameter Fix）

算子入参计算错误（如 FA `head_num` 用了 KV heads 而非 Q heads、TP 切分后维度未除以 tp_size）。定位到具体算子后，逐个核对入参与算子文档/参考模型的差异。

**适用场景**：shape 不匹配、dtype 错误、维度计算错误

### 模式 3：路径统一（Path Unification）

部分 rank 走不同代码路径导致通信死锁。

```python
# 错误：不同 rank 可能在不同 step 进入 prefill/decode
if is_prefill:
    self.all_reduce(...)  # rank 0 执行
# rank 1 已进入 decode，不执行 all_reduce → 死锁

# 修复：确保所有 rank 同步状态后再分支
dist.barrier()
is_prefill_tensor = torch.tensor([int(is_prefill)], device="npu")
dist.broadcast(is_prefill_tensor, src=0)
is_prefill = bool(is_prefill_tensor.item())
```

**适用场景**：多 rank 死锁、HCCL timeout

### 模式 4：配置降级（Configuration Downgrade）

配置组合不兼容时，调整 YAML 参数（如降低 `moe_chunk_max_len`、减小 `batch_size`）。

**适用场景**：OOM、性能异常

---

## 第五步：验证清单

修复后，按以下清单验证：

- [ ] **单步验证**：在出错位置前后加 sync+barrier，确认不再超时
- [ ] **全流程验证**：移除所有调试检查点，运行完整 warmup + inference
- [ ] **全 rank 验证**：检查所有 rank 的日志，确认全部成功（`grep "model run success" log_*.log`）
- [ ] **输出一致性**：各 rank 的最终输出应一致（对 DP 模式，同 DP group 内的 rank 输出应一致）
- [ ] **性能记录**：记录修复后的 Prefill/Decode 耗时，与修复前对比
- [ ] **回退文档**：如果使用了算子回退，记录性能影响和未来可恢复条件

---

# 特定场景诊断

以下场景有独立的诊断路径，不走通用流程。通过入口 npu-smi 表判断后直接跳转。

## 多卡部署诊断

多卡部署引入了一类单卡不存在的问题：权重加载切分、vocab 对齐、部分 rank 崩溃、config 兼容性。共同特点：**部分 rank 成功、部分 rank 失败**，错误出现在权重加载阶段而非 forward 阶段。

### 权重加载 TP 切分越界

**典型错误**：
```
RuntimeError: start (105984) + length (35328) exceeds dimension size (131125)
```
或：
```
ValueError: 131125 is not divisible by 8
```

**根因**：ColumnParallelLinear / VocabParallelEmbedding 将权重沿 output 维度切分为 tp_size 份。如果 checkpoint 的实际维度不等于模型参数的维度，或者不能被 tp_size 整除，高位 rank 的 `narrow()` 操作就会越界。

**常见场景**：

| 场景 | 说明 | 修复 |
|------|------|------|
| **多模态模型文本推理** | `embed_tokens` 用 full vocab (282624) 但 `lm_head` 用 text+special (131125)，两者维度不同 | lm_head 单独使用 `text_vocab_plus_multimodal_special_token_size` |
| **vocab 不整除 tp_size** | 131125 % 8 ≠ 0 | 向上 pad 到最近的 tp_size 倍数：`padded = ((raw + tp - 1) // tp) * tp` |
| **padded 参数 vs 原始 checkpoint** | 模型参数 131128（padded）但 checkpoint 权重只有 131125 | 加载时先 pad checkpoint weight 再传给 weight_loader |

**修复模板（vocab pad + weight pad）**：

```python
# 1. 创建 lm_head 时 pad vocab size
lm_head_raw = getattr(config, "text_vocab_plus_multimodal_special_token_size", config.vocab_size)
self.lm_head_vocab_size = ((lm_head_raw + tp_size - 1) // tp_size) * tp_size

# 2. 加载权重时 pad checkpoint tensor
if "lm_head" in name and loaded_weight.shape[0] < self.lm_head_vocab_size:
    padded = torch.zeros(self.lm_head_vocab_size, *loaded_weight.shape[1:],
                         dtype=loaded_weight.dtype, device=loaded_weight.device)
    padded[:loaded_weight.shape[0]] = loaded_weight
    loaded_weight = padded
```

**排查技巧**：如果只有高位 rank (rank 3-7) crash 而低位 rank (0-2) 正常，几乎一定是切分越界 —— 因为低位 rank 的 `start_idx` 还在合法范围内。

### Config 兼容性问题

从 HuggingFace 原始模型加载时，config.json 可能缺少代码期望的字段，或字段格式不同。

**常见问题**：

| 问题 | 表现 | 修复 |
|------|------|------|
| 缺少 `rope_parameters` | `TypeError: 'NoneType' object is not subscriptable` at RotaryEmbedding init | Config `__init__` 中当 `rope_parameters=None` 时构造 default：`{"rope_type": "default", "rope_theta": self.rope_theta}` |
| `vocab_size` 含多模态 token | Embedding 过大 / lm_head 维度不匹配 | 区分 `vocab_size`（embedding 用）和 `text_vocab_plus_multimodal_special_token_size`（lm_head 用）|
| `num_hidden_layers` 歧义 | Dual-sublayer 架构下 HF 可能存 28（attention 层数）但物理层只有 14 | 用 property 覆盖：`@property num_hidden_layers -> return self.num_layers` |
| `auto_map` 指向原始模型代码 | 加载时尝试 import 原始 modeling 文件 | 传入自定义 config class 到 `from_pretrained` |

**最佳实践**：在 config class 的 `__init__` 末尾为缺失字段补 defensive defaults。对有模型差异的参数（如 `rope_theta`），从 config.json 读取而非硬编码默认值，缺失时报错。

### 部分 Rank 静默失败

有时 rank crash 了但错误被吞掉，存活的 rank 卡在下一个 `dist.barrier()` 或集合通信上，看起来像"推理卡住"。

**诊断流程**：

```bash
# 1. npu-smi 确认进程数
npu-smi info | grep "Process id"

# 2. 逐 rank 检查日志末尾
for i in $(seq 0 7); do
    echo "=== Rank $i ===" 
    tail -5 logs/log_${i}.log
done

# 3. 检查是否所有 rank 都到达了同一个里程碑
grep "model run success" logs/log_*.log
# 或
grep "Loading weights took" logs/log_*.log
```

### 多卡权重加载检查清单

在发起多卡推理前，逐项确认：

- [ ] **embed_tokens 维度**：checkpoint 里 embed_tokens.weight.shape[0] 和 config.vocab_size 一致
- [ ] **lm_head 维度**：checkpoint 里 lm_head.weight.shape[0] 可能小于 embed_tokens（多模态模型常见）
- [ ] **vocab 整除 tp_size**：lm_head 和 embed_tokens 的 output_size 都能被各自的 tp_size 整除（不能则 pad）
- [ ] **ignore 规则覆盖**：`_keys_to_ignore_on_load_unexpected` 包含所有不需要的权重前缀（ngram、visual、audio、mtp 等）
- [ ] **config 字段完整**：`rope_parameters`、`num_layers`、`zero_expert_num` 等代码必需字段在 config.json 中有或有 default
- [ ] **模型路径绝对路径**：executor 框架中相对路径可能在不同 rank 的 cwd 下解析不同

---

## 推理卡住诊断

推理卡住与 crash 不同——进程仍在运行但不产出结果。需要区分真卡住和假卡住。先通过入口 npu-smi 表判断状态，再按以下路径排查。

### eager 模式复现

```
卡住 → 换 eager YAML 跑同样模型
  ├── eager 也卡 → bug 在模型代码 → 用 sync+print 定位
  ├── eager 正常 → bug 在图模式适配
  │     ├── 看报错：ERR03007 / graph break → 找不兼容的算子
  │     └── 无报错但慢/卡 → 比较 eager 和 graph 模式代码路径差异
  └── 两者都正常但上次卡了 → 检查是否有残留进程占 NPU
```

### 模型构造阶段卡住（CPU 密集）

**症状**：日志停在 HCCL init 之后，HBM 不增长，但 CPU 100%+

**常见根因**：

| 原因 | 机制 | 修复 |
|------|------|------|
| **`nn.Embedding` 大 vocab 初始化** | 大 vocab 的 `nn.Embedding` 用 `normal_()` 填充随机值，多个 embedding 累计耗时可达数分钟 | 用 `no_init_weights()` 上下文包裹，跳过将被 checkpoint 覆盖的参数初始化 |
| **`VocabParallelEmbedding` 瞬态内存** | 继承 `nn.Embedding`，`super().__init__(full_vocab, dim)` 先创建全 vocab tensor，再用 per-partition tensor 替换，瞬态 CPU 内存远超最终占用 | 用 `nn.Module` + `torch.empty(per_partition_size)` 直接创建 per-partition 权重，避免瞬态全 vocab 分配 |
| **`post_init()` 遍历所有参数** | HuggingFace `PreTrainedModel.post_init()` 调用 `_init_weights` 对每个参数做初始化，大模型参数量多时很慢 | 对从 checkpoint 加载的大参数模块，用 `no_init_weights()` 跳过 |

**诊断技巧**：

```bash
# 区分真卡和 CPU 密集慢
ps aux | grep "infer.py" | awk '{print $2, $3"%CPU", int($6/1024)"MB"}'
# CPU 100%+ = 还在算（慢但没死）
# CPU 0%    = 真卡住（等通信/IO）
```

**修复模板（大 embedding 模块）**：

```python
from transformers.modeling_utils import no_init_weights

# 用 no_init_weights 包裹，避免对即将被 checkpoint 覆盖的参数做随机初始化
with no_init_weights():
    self.large_embeddings = LargeEmbeddingModule(config, ...)
```

### 残留进程导致 HCCL 卡住

**症状**：新启动的推理卡在 HCCL init，不报错不超时

**根因**：上次推理的进程未正确退出，仍占用 NPU 设备或 HCCL 端口

**修复**：

```bash
# 1. 杀死残留进程
ps aux | grep "infer.py" | grep -v grep | awk '{print $2}' | xargs kill -9

# 2. 等待 NPU 释放
sleep 3

# 3. 确认 NPU 完全空闲
npu-smi info | grep "No running"
# 期望输出 8 行 "No running processes found"

# 4. 重新启动推理
```

**预防**：每次启动推理前，先检查并清理残留进程。

---

# 附录

## 运行监控与提效

以下命令示例基于当前仓默认的日志路径和启动脚本约定，executor 或日志组织变更后需相应调整。

### 核心原则

**判断完成状态看进程，不看日志流**。多卡推理通过 `infer.sh` 后台启动 N 个进程再 `wait`。rank 0 的 stdout 通过 `tee` 到终端，可能缓冲截断；其他 rank 只写日志文件。

### 完成检测方法

| 方法 | 命令 | 说明 |
|------|------|------|
| **查 NPU 进程** | `npu-smi info \| grep "running process"` | 无进程 = 已结束 |
| **查非 rank0 日志** | `grep "model run success" logs/*/log_1.log` | Rank 1-7 日志不经 tee，完整写入。**不要只查 rank 0** |
| **查所有 rank** | `for i in $(seq 0 7); do grep -l "model run success" logs/*/log_${i}.log; done \| wc -l` | 期望输出 = world_size |

### 日志关键字判断阶段

```bash
grep "Loading safetensors" logs/*/log_1.log     # 正在加载权重
grep "Loading weights took" logs/*/log_1.log    # 加载完成
grep "inference time cost of prefill" logs/*/log_1.log  # 推理已开始
grep "model run success" logs/*/log_1.log       # 推理已结束
```

### 一条命令获取全局状态

```bash
echo "=== NPU ===" && npu-smi info | grep -E "Process|HBM" | head -5 && \
echo "=== Log ===" && tail -3 logs/*/log_1.log && \
echo "=== Done? ===" && grep -c "model run success" logs/*/log_*.log 2>/dev/null
```

### 常见误判场景

| 现象 | 原因 | 正确做法 |
|------|------|---------|
| 终端无新输出，以为在运行 | Rank 0 的 tee 管道已关闭或缓冲未刷 | 查 `npu-smi` 或查 rank 1-7 日志 |
| Rank 0 日志没有 "model run success" | tee 截断 | **查 rank 1 的日志** |
| `sleep 120` 后还没完成 | 模型大/首次编译慢 | 先 `npu-smi` 判断存活，再看 HBM 变化判断阶段 |
| grep 日志找不到关键字 | 进程 crash 了没写 success | 查 `npu-smi` 确认进程不在 → 查 `tail -20 log_*.log` 看错误 |

### Agent 工作模式

1. **启动推理**（`bash infer.sh`，前台或 `run_in_background`）
2. **不要盲目 sleep** —— 用 `run_in_background` 等通知，或前台直接看输出
3. **10 分钟无进展 → 主动排查**：推理启动超过 10 分钟仍无 `model run success` 输出，不要继续等待，按「推理卡住诊断」流程排查（`npu-smi` → 判断状态 → 分流）
4. **超时后第一步查 `npu-smi`**，不要先查日志
5. **检查完成时查 rank 1（不是 rank 0）**—— rank 1 日志完整无截断
6. **一次检查多个信号**：`npu-smi` + `tail log_1.log` + `grep "model run success" log_*.log` 并行执行

