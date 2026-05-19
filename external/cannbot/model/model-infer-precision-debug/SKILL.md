---
name: external-cannbot-model-model-infer-precision-debug
description: 基于 PyTorch 框架的昇腾 NPU 模型推理精度问题诊断技能。当前主要覆盖 KVCache / FlashAttention 相关精度问题，包括
  Prefill/Decode 对齐、cache 更新错误、slot/block mapping 错误、attention 路径切换后的精度异常等。触发场景：优化改造后精度验证未通过、模型输出与基线存在显著偏差、Prefill
  和 Decode 精度表现不一致、出现 NaN/Inf、量化模式下精度异常放大等。
original-name: model-infer-precision-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# 推理精度问题诊断技能

> 当前主要覆盖 KVCache / FA 相关精度问题，其他推理精度问题后续补充。
> 不覆盖运行时错误（crash、hang、OOM、算子约束违反等见 model-infer-runtime-debug）。

按"症状分类 → 快速验证 → 分模块定位 → 逐层对比 → 陷阱修复"的分层策略排查精度问题。完整校验工具见 `references/fa_debug_utils.md`。

## 重要原则

- **对比驱动**：所有精度判定必须基于与基线（eager 模式 / 优化前）的数值对比，不可凭直觉
- **逐步缩小范围**：先定位阶段（Prefill/Decode）→ 再定位模块（KVCache 写入/FA 计算/后处理）→ 最后定位参数
- **最小改动验证**：每次只修改一个变量，验证后再进入下一个
- **保留现场**：调试过程中的中间 tensor 和日志必须保留，用于后续分析
- **实际执行测试**：所有调试步骤由 Agent 实际执行，不将测试工作委托给用户

---

## 排查工作流

接到精度调试任务后，按以下步骤执行：

**第一步：收集信息** — 获取精度症状、优化阶段、执行/量化模式、基线输出

**第二步：快速诊断** — 根据症状分类表判断方向 → 执行快速验证清单（含 FA 入参检查）→ 发现问题直接修复

**第三步：分模块定位** — 二分法确定 Prefill/Decode → 针对性插桩验证 → PA 场景执行专项排查

**第四步：精细定位（按需）** — 使用逐层对比框架，dump 基线 → compare 优化后 → 定位首次偏差层

**第五步：修复与验证** — 参照常见陷阱表修复 → 重新运行完整精度验证

**第六步：输出调试报告** — 问题概要、根因分析、修复措施、修复验证、遗留风险

---

## 第二步：快速诊断

### 2.1 症状分类

| 症状 | 可能原因 | 优先排查方向 |
|------|---------|-------------|
| 输出全为重复 token 或乱码 | KVCache 写入位置错误 | slot_mapping / scatter_update_ 参数 |
| 首个 token 正确，后续偏移 | Decode 阶段 kv_len 更新错误 | kv_len 生命周期 |
| Prefill 正确，Decode 偏差大 | Decode FA 入参错误 | sparse_mode / actual_seq_lengths |
| 输出含 NaN/Inf | 数值溢出 | scale 参数、dtype 不匹配 |
| 精度逐层累积偏差 | FA 算子数值精度 | input_layout / NZ 格式 / 量化参数 |
| 多 batch 部分样本错误 | batch 间缓存交叉污染 | block_table 构造 / kv_len_offset |
| 量化模式下偏差显著放大 | 量化参数不匹配 | dequant_scale / antiquant_mode |

### 2.2 快速验证清单

- [ ] 1. dtype 一致性：KVCache tensor 与模型计算 dtype 一致（BF16 / FP16）
- [ ] 2. device 一致性：所有 tensor 均在 NPU 上，无意外的 CPU tensor
- [ ] 3. block_table dtype：必须为 torch.int32
- [ ] 4. sparse_mode 正确：Prefill=3（因果），Decode=0（dense）；sparse_mode=2 不推荐
- [ ] 5. Prefill 和 Decode 的 FA 调用参数需分别配置（sparse_mode、atten_mask、actual_seq_lengths 等）
- [ ] 6. scale 参数：1.0 / sqrt(head_dim)，确认 head_dim 值正确
- [ ] 7. num_heads / num_kv_heads：与模型配置一致，多卡时已按 rank 切分
- [ ] 8. kv_len 初始值：Prefill 后 kv_len 等于输入序列长度，非 0
- [ ] 9. input_layout 与 tensor shape 匹配
- [ ] 10. actual_seq_lengths 构造方式与 layout 匹配（TND 用 cumsum，BSH 用原值）
- [ ] 11. atten_mask 与 sparse_mode 匹配（详见附录 F.1）；Decode 单 token 查询通常不需要 mask
- [ ] 12. FA v1 vs v2 参数名：未混用（详见附录 F.2）
- [ ] 13. Decode 阶段未使用 sparse_mode=3
- [ ] 14. PA 场景下 mask 最后一维 >= block_table 第二维 × block_size
- [ ] 15. MLA rope 参数：query_rope/key_rope 同时传或同时不传；rope D=64，query D 仅支持 512/128
- [ ] 16. inner_precise 行无效：Prefill + 自定义 mask 有全遮蔽行时需设为 2/3
- [ ] 17. kv_len 层内只读：kv_len 在 attention/cache 内部未被修改（禁止层内递增，否则各层写入位置错位导致精度彻底损坏）

---

## 第三步：分模块定位

### 3.1 排查策略：二分法定位

```
模型输出与基线不一致
    │
    ├─ 1. 仅运行 Prefill，对比输出
    │   ├─ Prefill 输出一致 → 问题在 Decode 阶段（跳到 3.3）
    │   └─ Prefill 输出不一致 → 问题在 Prefill 阶段（跳到 3.2）
    │
    ├─ 2. Decode 第 1 步 vs 第 N 步
    │   ├─ 第 1 步就不一致 → FA 入参或 KVCache 初始状态问题
    │   └─ 第 N 步开始不一致 → kv_len 累计更新或缓存溢出问题
    │
    └─ 3. 逐层对比（跳到第四步）
```

### 3.2 Prefill 阶段精度排查

**KVCache 写入验证**：在 `scatter_update_` 调用前后插桩，检查：
1. 写入前缓存是否为零（首次 Prefill）
2. 写入后缓存内容与 key_states 是否一致（阈值 1e-6）
3. scatter_update_ 的 axis 参数与 layout 是否匹配（BSH → axis=1, TND → axis=0）

**FA 入参验证**：打印并校验 Q/K/V shape、actual_seq_qlen/kvlen、sparse_mode、是否含 NaN/Inf

### 3.3 Decode 阶段精度排查

**KVCache 写入验证**：
1. 验证 slot_mapping 计算（连续缓存：`expected_slot = kv_len.view(-1,1) + kv_len_offset`；PA：通过 block_table 索引）
2. 验证写入位置未越界
3. 验证 kv_len 逐步递增正确

**FA 入参验证**：
1. sparse_mode 应为 0，actual_seq_qlen 每个值应为 1（MTP 除外）
2. block_table dtype 必须为 int32
3. KV cache 已写入区域无 NaN/Inf

### 3.4 分页注意力（PA）专项排查

1. **block_table 静态校验**：`numel == batch_size × (max_seq_len // block_size)`
2. **block_table 值域校验**：应为 `[0, total_blocks)` 的排列，无重复
3. **slot_mapping 与 block_table 交叉验证**

---

## 第四步：逐层精细对比

### 4.1 逐层对比框架

在 Transformer Block 的 8 个关键位置插入 checkpoint，与基线对比：

| 序号 | 位置 | 重要程度 |
|:----:|------|:--------:|
| 1 | Block 输入 | 普通 |
| 2 | Attention Norm 后 | 普通 |
| 3 | QKV 投影后 | 普通 |
| 4 | RoPE 后 | 普通 |
| 5 | KVCache 写入后 | **关键** |
| 6 | FA 输出 | **关键** |
| 7 | Attention 残差后 | 普通 |
| 8 | Block 输出 | 普通 |

**对比指标**（均转为 float32 计算）：max_abs_diff / max_rel_diff / cosine_similarity

**判定阈值**：BF16 模式 max_rel < 1e-3，量化模式 max_rel < 1e-2

**使用流程**：dump 基线 → compare 优化后 → 定位首次出现偏差的层和位置

---

## 第五步：常见精度陷阱与修复

### 5.1 KVCache 写入相关

| 问题 | 根因 | 修复方案 |
|------|------|---------|
| 缓存写入位置偏移 | scatter_update_ 的 axis 与 layout 不匹配 | BSH → axis=1, TND → axis=0 |
| 多 batch 缓存交叉写入 | kv_len_offset 未按 max_seq_len 对齐 | `kv_len_offset = arange(0, batch*max_seq_len, max_seq_len)` |
| Prefill 后缓存部分为零 | kv_len 计算错误 | 检查 kv_len 是否等于实际有效序列长度（具体值视框架约定） |
| NZ 格式下缓存异常 | cache_mode 设置错误 | PA 模式使用 `cache_mode="PA_NZ"` |
| 融合写入算子结果不一致 | npu_kv_rmsnorm_rope_cache 参数不匹配 | 对齐 epsilon 和 RoPE 频率 |

### 5.2 FA 算子相关

| 问题 | 根因 | 修复方案 |
|------|------|---------|
| Prefill 正常，Decode 偏差大 | Decode 使用了 sparse_mode=3 | Decode 设 `sparse_mode=0` |
| actual_seq_lengths 错误 | TND 应用 cumsum，BSH 用原始值 | 参考 `model-infer-kvcache` skill 的数据流章节 |
| Decode 首步输出为零 | actual_seq_qlen 为 0 | Decode 时应为 1 或 cumsum([1,...]) |
| MLA absorb 模式 key≠value | cache_nope 读写不一致 | FA 中 key 和 value 都传 cache_nope |
| scale 参数错误 | FA v1 用 `scale`，v2 用 `softmax_scale` | 确认参数名与版本匹配 |

### 5.3 量化相关

| 问题 | 根因 | 修复方案 |
|------|------|---------|
| W8A8 精度从某层开始放大 | dequant_scale 与权重不匹配 | 检查 per-channel/per-tensor scale |
| W8A8C8 KVCache 精度差 | KV 缓存量化精度损失 | 检查 KV 量化 scale；考虑 BF16 KV |
| antiquant_mode 不匹配 | FA v1/v2 量化接口不同 | v1 用 antiquant_*，v2 用 dequant_* |

### 5.4 Prefill/Decode 一致性

| 问题 | 根因 | 修复方案 |
|------|------|---------|
| Prefill 不走 KVCache 写入 | 条件分支逻辑错误 | 确保 Prefill 也执行 scatter_update_ |
| Decode 使用了 Prefill 的 FA 参数配置 | 未分别配置 | Prefill 和 Decode 分别配置 sparse_mode、atten_mask 等参数 |
| 切换阶段时 kv_len 未更新 | Prefill 后未执行 kv_len += seq_len | 在 Prefill→Decode 切换时确保更新 |
| Decode 传入了 atten_mask | sparse_mode=0 不需要 | Decode 设 `atten_mask=None` |

---

## 附录 F：FA 入参配置参考

> 表格供快速验证清单引用，完整校验代码见 `references/fa_debug_utils.md`，可在 FA 调用前插入 `debug_fa_all_params()` 一键检查。

### F.1 atten_mask 与 sparse_mode 联合校验

| sparse_mode | 含义 | atten_mask 要求 | 适用阶段 | 常见错误 |
|:-----------:|------|----------------|---------|---------|
| 0 | defaultMask | 可选，通常 **None** | **Decode** | Decode 误传 mask |
| 1 | allMask | **必传** `(Q_S, KV_S)` | 特殊 | 忘记传 mask |
| 2 | leftUpCausal | **不推荐** | — | — |
| 3 | rightDownCausal | **必传** `(2048, 2048)` 下三角 | **Prefill** | mask 不是下三角；Decode 误用 |
| 4 | band（滑窗） | **必传** `(2048, 2048)` | Prefill / Decode | pre_tokens/next_tokens 错误 |

### F.2 FA v1 vs v2 关键参数名映射

| 功能 | FA v1 | FA v2 | 易错点 |
|------|-------|-------|--------|
| 缩放系数 | `scale` | `softmax_scale` | 默认 1.0，传错名精度崩溃 |
| Q head 数 | `num_heads` | `num_query_heads` | 传错名默认 1 |
| Q 长度 | `actual_seq_lengths` | `actual_seq_qlen` | 名称完全不同 |
| KV 长度 | `actual_seq_lengths_kv` | `actual_seq_kvlen` | 名称完全不同 |
| KV 量化 | `antiquant_scale/offset` | `dequant_scale_key/value` | v2 分离 key/value |

### F.3 input_layout 与 tensor shape 匹配

| layout | Q 维度 | K/V 维度 | N 轴含义 |
|--------|--------|---------|---------|
| BSH | 3D `[B, S, H*D]` | 3D `[B, S, H*D]` | 无独立 N 轴 |
| BNSD | 4D `[B, N, S, D]` | 4D `[B, N, S, D]` | N = num_heads |
| TND | 3D `[T, N, D]` | 3D `[T, N, D]` | N = num_heads，T = packed tokens |

PA 模式下 KV 按 block_table 索引，shape 与非 PA 不同。

### F.4 actual_seq_lengths 构造

| layout | actual_seq_qlen | actual_seq_kvlen |
|--------|----------------|-----------------|
| TND | 累积和 `cumsum([s1, s2, ...])` | 累积和 |
| BSH / BNSD | 原始值 `[s1, s2, ...]` | 原始值 |

Decode 时 actual_seq_qlen 每个值通常为 1（MTP 场景除外）。

### F.5 inner_precise 行无效

| 值 | 精度模式 | 行无效修正 | 适用场景 |
|:-:|------|:---:|---------|
| 0 | 高精度 | 否 | 默认 |
| 1 | 高性能 | 否 | 精度要求不高 |
| 2 | 高精度 | **是** | Prefill + 自定义 mask 有全遮蔽行 |
| 3 | 高性能 | **是** | 同上但允许精度损失 |

### F.6 MLA rope 约束

- `query_rope` 和 `key_rope` 必须同时传或同时不传
- rope D 必须为 64，query D 仅支持 512 或 128
- D=512 时仅支持 `sparse_mode` 为 0、3、4

---

## 精度调试报告模板

```markdown
## 精度调试报告

### 问题概要
- 症状：{描述}
- 定位阶段：{Prefill / Decode / 两者}
- 定位模块：{KVCache 写入 / FA 计算 / 量化参数 / ...}

### 根因分析
- 原因：{具体原因}
- 影响范围：{受影响的层/模块}

### 修复措施
| 修改文件 | 修改内容 | 修改原因 |
|---------|---------|---------|
| {file} | {change} | {reason} |

### 修复验证
- 精度对比：{通过/未通过，max_rel_diff, cosine_similarity}
- 输出一致性：{token 序列对比结果}

### 遗留风险
- {如有，列出可能的后续风险}
```

---

## 其他已知精度问题

> 以下问题来自本仓已适配模型的调试经验，不属于 KVCache/FA 范畴但会导致精度异常。

| 问题 | 症状 | 根因 | 修复 |
|------|------|------|------|
| **MoE 专家权重未加载** | MoE 输出全零或接近零，cosine~0.4-0.8 | Checkpoint 用 packed tensor 但 load_weights 用 per-expert mapping，权重未正确拷贝 | 添加 packed expert weight loader，按 EP rank 切片后 copy_ 到参数 |
| **Zero expert 处理错误** | MoE 输出偏差或切换路由算子后精度骤降至 ~40-50% | 零专家 identity 贡献公式错（应用 original_input 而非 MoE 输出），或路由算子 v1 无 `active_expert_range` 导致零专家 ID 污染 token 排列 | 路由前用 mask 分离零专家，路由后加回 `original_input * weight` 的 identity 贡献 |
| **DP+EP token 不同步 → NaN** | Decode 若干步后出现 NaN，之后全部 NaN | EP all_to_all 有浮点非确定性，DP 各 rank token 选择分叉 → KV cache 累积偏差 | 每步 Decode 后 `dist.broadcast(next_token, src=0)` 强制所有 DP rank 使用相同 token |
| **跳过训练组件的 scale 缺失** | 输出语义正确但严重重复/退化 | 模型训练时有额外组件，跳过后 embedding 幅度不匹配 | 保留训练时的 normalization factor |
