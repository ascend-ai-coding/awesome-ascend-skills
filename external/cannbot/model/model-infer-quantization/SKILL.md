---
name: external-cannbot-model-model-infer-quantization
description: infer 仓模型量化适配改造技能。分析并接入既有 compressed-tensors 量化方案和权重，完成量化产物契约检查、结构参考匹配、量化
  runtime 映射、权重加载、post-load 处理、融合算子量化冲突回退、真实生效验证和收益评估。触发：模型优化流程中的量化初评估、量化改造任务、compressed-tensors
  量化产物接入时使用；不重新设计上游量化算法，不实现 compressed-tensors 之外的量化路线。
original-name: model-infer-quantization
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# 模型量化适配改造

分析 infer 模型代码与既有 `compressed-tensors` 量化产物，按模型结构匹配仓库参考经验，完成量化接入、验证和收益评估。若当前任务只要求初评估，则只输出量化适配方案，不修改代码；若任务要求量化改造，则在契约和用户决策满足后实施代码改造。

---

## 适用范围

- `compressed-tensors` 量化方案和权重接入
- `quantization_config` / `compression_config` 解析和透传
- `Linear` / `ReplicatedLinear` / `RowParallelLinear` / `MoEGMM` / `KVCache` 等 infer 运行对象映射
- 量化权重加载、scale dtype 处理、转置、NZ/base format、smooth scale、专家权重处理
- Dense 模型的 `gate/up`、`q/k/v` post-load 运行时融合恢复
- MoE / MLA / Indexer / KVCache 量化契约适配
- 融合算子不支持量化时的非融合路径回退和需求记录
- 量化真实生效验证、显存/Prefill/Decode/部署卡数收益评估

**不做**：

- 不重新跑上游量化实验
- 不修改量化方案来绕过 infer 侧报错
- 不实现 GPTQ / AWQ / 外部 patch / service 部署等非主线路线
- 不替代模型迁移、并行化、KVCache/FA、融合算子或图模式 Skill

---

## 工作流程

```
第一步：确认输入和执行模式
    ↓
第二步：检查量化产物契约
    ↓
第三步：拆解模型结构与量化对象
    ↓
第四步：匹配仓库参考与评估接入分级
    ↓
第五步：方案审查
    ↓
第六步：实施量化接入（仅改造任务执行）
    ↓
第七步：验证真实生效与收益
    ↓
第八步：写回 progress.md 与经验沉淀
```

**禁止**：跳过契约检查直接改代码；跳过结构拆解按模型名套参考；量化报错后修改 `targets` / `ignore` / 张量语义再宣称成功。

---

## 参考文档使用矩阵

| 步骤 | 主要读取 | 用途 |
| --- | --- | --- |
| 第二步 契约检查 | `quantization-contract.md` §1-§4 | 产物字段 / 张量语义 / 运行对象映射 |
| 第三步 结构拆解 | `quantization-contract.md` §4-§6 + `quantization-structure-cards.md` 选定结构卡 | 映射量化对象到 runtime；识别结构指纹 |
| 第四步 匹配参考与分级 | `quantization-structure-cards.md` 结构卡 A/B/C + `quantization-fusion-and-benefit.md` §A、§B.3 | 命中改造要点和模型经验；融合兼容性；主线路径 A1-A4 |
| 第六步 实施改造 | `quantization-structure-cards.md` 命中卡的「改造要点 / 已确认模型经验 / post-load 例外」 | 模型 forward / post-load / scale 适配 |
| 第七步 验证收益 | `quantization-fusion-and-benefit.md` §B.1 五维口径 | 收益判定证据和分级 |
| 第八步 经验沉淀 | `quantization-structure-cards.md` 经验沉淀模板 | 按结构卡追加新经验 |

> 回退规则、验证要求、后验决策三段为本 SKILL 唯一真相源（第六步 6.5 / 第七步 / 第八步 8.2），ref 不复述。

---

## 第一步：确认输入和执行模式

### 1.1 输入

| 输入 | 适用模式 | 说明 |
| --- | --- | --- |
| `model_dir` | 初评估 / 改造 / 验证 | infer 模型工作目录 |
| `quant_export_dir` | 初评估 / 改造 / 验证 | 量化权重和量化方案目录 |
| `mode` | 初评估 / 改造 / 验证 | 执行模式 |
| `baseline` | 改造 / 验证 | 非量化部署基线；改造任务还应参考量化前最新基线 |
| `user_decision` | 改造 | 用户明确继续量化的决策 |

### 1.2 模式分流

| 模式 | 允许动作 | 禁止动作 |
| --- | --- | --- |
| `初评估` | 分析量化产物、结构、显存、接入分级，写入量化初评估报告 | 修改模型代码 |
| `改造` | 按已确认量化方案接入 runtime、加载权重、处理 post-load、跑量化基线 | 修改量化方案或权重语义 |
| `验证` | 验证量化真实生效、输出可用性、性能和显存收益 | 自行调试或改代码 |

### 1.3 输入缺失处理

- 缺少 `model_dir`：转 `model-infer-migrator` 或要求补充模型目录。
- 缺少 `quant_export_dir`：停止量化路径，要求补齐量化产物目录。
- 改造或验证缺少非量化基线：先补部署基线，不进入收益验证。
- 改造模式缺少用户决策：只做分析，不改代码。

### 完成标志

- [ ] 执行模式已明确
- [ ] 量化产物目录已确认
- [ ] 若为改造或验证任务，至少有一份可复现基线
- [ ] 改造任务已有用户继续量化决策

---

## 第二步：检查量化产物契约

**读取**：

- `{quant_export_dir}/config.json`、`model.safetensors.index.json`、`deploy_quantization.md`
- `references/quantization-contract.md` §1-§4（最小交付物 / config 字段 / 张量语义 / 运行对象映射）

### 2.1 契约检查项

- `config.json` 包含 `quantization_config` 或等价 `compression_config`
- `quant_method` 为 `compressed-tensors`
- `config_groups` 能描述权重量化和激活量化
- `targets` / `ignore` 能区分量化模块和浮点回退模块
- `model.safetensors.index.json` 可索引量化张量和保留张量
- 量化张量语义明确，例如 `qweight`、`weight_scale`、`weight_bias`、smooth scale、zero point
- 量化张量名可映射到 infer 模型参数前缀
- `deploy_quantization.md` 写清量化对象、量化模式、回退策略和下游加载要求

### 2.2 契约不满足时

任一关键项缺失时，不要猜测，不要硬编码适配，不要改量化方案。输出补充诉求：

| 字段 | 内容 |
| --- | --- |
| 缺失项 | 缺少的配置、张量或说明 |
| 影响模块 | 受影响的模型模块和 runtime object |
| 期望格式 | infer 侧需要的字段、命名、shape、dtype、scale 规则 |
| 阻塞原因 | 为什么无法安全接入 |

### 完成标志

- [ ] 契约满足 / 不满足 / 需补充的结论已明确
- [ ] 量化对象、浮点回退对象和张量语义已确认
- [ ] 不满足时已输出补充诉求，不进入代码改造

---

## 第三步：拆解模型结构与量化对象

**读取**：

- `{model_dir}/agentic/progress.md`、`config/*.yaml`、`runner_*.py`、`models/modeling_*.py`
- `references/quantization-contract.md` §4-§6（运行对象映射 / 9 步运行机制 / 四层落点）作为映射依据

### 3.1 识别结构指纹

按真实结构判断，不按模型名判断：

| 结构指纹 | 判断依据 |
| --- | --- |
| `Dense Decoder` | decoder-only，无 routed experts，有 attention projection 和 dense MLP |
| `MoE Decoder` | 有 router/topk、routed experts、expert gate/up/down |
| `MLA + MoE / KVCache` | 有 latent KV、MLA prolog、KV cache scale 或 MLA absorb |
| `Indexer / 长序列 + MoE` | 有 Indexer、LI cache、Sparse FA、Hadamard 或长序列专属契约 |
| 特殊 packed expert | 专家权重或 scale 有特殊 pack/shard 规则 |

### 3.2 建立量化对象映射

| 量化目标 | infer runtime object |
| --- | --- |
| Dense Linear | `Linear` / `ReplicatedLinear` / `RowParallelLinear` |
| Dense MLP gate/up/down | 可量化 Linear；必要时 post-load 恢复 `gate/up` 融合 |
| Attention q/k/v/o | 可量化 Linear；必要时 post-load 恢复 `q/k/v` 融合 |
| MoE experts | `MoEGMM` / `FusedMoEGMM`，不满足时回退 per-expert Linear |
| KVCache | `kv_cache_scheme` / cache scale / cache runtime |
| `ignore` 模块 | 浮点回退，不宣称已量化 |

### 3.3 记录运行场景特征

- Prefill / Decode 分支差异
- 部署卡数、TP/EP、是否 online split weight
- 量化模式：如 `W8A8` / `W8A8C8` / `W4A8C8`
- 当前量化前模型是否已有融合算子或其它前序优化
- 可能影响量化的 dtype / layout / scale / cache 组织

### 完成标志

- [ ] 结构指纹已明确
- [ ] 量化对象到 infer runtime object 的映射已形成
- [ ] `ignore` 和浮点回退模块已列出
- [ ] Prefill / Decode 差异和部署形态已纳入判断

---

## 第四步：匹配仓库参考与评估接入分级

**读取**：

- `references/quantization-structure-cards.md`：按结构卡 A/B/C 选定方案、改造要点、已确认模型经验
- `references/quantization-fusion-and-benefit.md` §A：融合算子兼容性分级和路线级判断
- `references/quantization-fusion-and-benefit.md` §B.3：主线参考路径 A1-A4

### 4.1 按结构匹配参考卡

优先匹配现有结构卡：

- `Dense Decoder`：重点检查 `gate/up` 和 `q/k/v` 是否被量化产物拆散，是否需要 post-load 融合恢复。
- `MoE Decoder`：重点检查 `Linear` 与 `MoEGMM` 的量化模式、expert scale、smooth scale、TP/EP shard。
- `MLA + MoE / KVCache`：重点检查 MLA prolog、KVCache C8、cache scale、MoE expert compute。
- `Indexer / 长序列 + MoE`：重点检查 IndexerProlog、LI cache、Sparse FA、Hadamard 和输出 dtype 契约。

**结构卡命中 ≠ 经验全套用**：即便结构指纹命中已有卡，「已确认模型经验」中的 forward 替身、scale 路径、post-load 例外等，必须基于当前模型实际算子链独立核对，不可「卡命中→经验全套用」。

只有当前结构无法落入已有卡，或出现新的 runtime object、张量语义、post-load 规则、融合回退模式时，才新增参考卡。

### 4.2 融合算子兼容性判断

按 `references/quantization-fusion-and-benefit.md` §A.1 兼容性分级（A/B/C/D）和 §A.2 算子映射判断量化与融合算子的交互：

- A 级（量化主线）：保留融合算子，接入量化输入和 scale。
- B 级（共存）：输入 dtype/layout 契约满足时保留。
- C 级（需替身）：先做 post-load 结构融合或换量化友好算子。
- D 级（专属契约）：必须先看输出 dtype、cache layout、必需 side tensor。

冲突处理由主流程在改造前向用户确认。回退规则见第六步 6.5；本步不擅自修改量化方案。

### 4.3 接入分级（按模块/能力逐项打级，不压成单维）

按 §3.2 模块映射逐项打级（Linear / MoEGMM / KVCache / Indexer 等独立给级），评估卡里逐项写入；整体动作按最坏级触发，但补充诉求和记录保留逐项分级，避免丢失「哪些模块已就绪、哪些卡住」的信息。

| 级别 | 判定 | 动作 |
| --- | --- | --- |
| L0 | 只需配置或 YAML 对齐 | 补配置，直接验证 |
| L1 | 需要模型映射、runner 或 post-load | 改模型/runner/加载逻辑 |
| L2 | 缺 infer 量化 runtime object / 算子能力（compressed-tensors 无对应 method/kernel） | 停止当前适配，标出 runtime gap，向主流程/用户确认是否另起 runtime 补齐任务；补 runtime 是独立工作量，不在本次适配内默默展开 |
| L3 | 量化契约缺失，无法安全接入 | 停止落代码，输出补充诉求 |

### 完成标志

- [ ] 已命中结构参考卡或说明新卡必要性
- [ ] 融合算子量化兼容性已初步判断
- [ ] 接入分级和理由已明确
- [ ] 初评估模式下已形成量化方案初评估报告

---

## 第五步：方案审查

进入代码改造前，审查以下项目。未完成则回到对应步骤补齐。

### 审查项

- [ ] 契约 Gate 通过，或已停止并输出补充诉求
- [ ] 结构指纹不是按模型名推断
- [ ] `quant_target -> infer runtime object` 映射完整
- [ ] `targets` / `ignore` 与模型参数前缀已核对
- [ ] 量化张量名、shape、dtype、scale 语义已核对
- [ ] post-load 处理项已列出
- [ ] 融合算子冲突和回退策略已列出
- [ ] 非量化基线和量化前最新基线已确认
- [ ] 改造模式下已有用户继续量化决策

若当前任务仅要求初评估，本步结束后输出初评估报告并写回 `progress.md`，不进入实施。

### 初评估输出格式

```markdown
## 量化方案初评估

> 字段与 model-infer-optimize 的 progress_template「阶段 0.5」对齐，主流程可 drop-in 填入。

### 基本结论
- 模型：
- 结构指纹：
- compressed-tensors 契约结论：
- 结构参考卡：
- 当前评估目标：

### 量化产物信息
- 量化产物目录：
- 目标量化模式：
- 期望消费方式：
- 当前核心配置格式：

### 契约核对
- `config.json`：
- `model.safetensors.index.json`：
- `deploy_quantization.md`：
- 当前 compressed-tensors 判定结果：

### 运行对象映射
- `{quant_target} -> {runtime_object}`（逐项）
- ignore / 显式浮点回退模块：
- 推荐首版无缝接入范围：

### infer 侧重点核查项
- `quantization_config` 是否可直接复用：
- `process_weights_after_loading` 后处理是否写清：
- `kv_cache_scheme` 或等价缓存契约是否写清：
- 融合算子兼容性风险（移交阶段 4 复核）：

### 显存与部署形态初评估
- 当前基线部署形态：
- 量化后部署判断：
- **覆盖率与收益上限提示**（advisory，非契约强校验）：
    - 已量化 / 未量化算力切片估计占比
    - 收益上限：「仅显存」/「显存 + 部分性能」/「显存 + 完整性能」
    - 工况收益应可追溯到实测对照点；外推须注明外推起点与假设
    - 量化模式与工况匹配性记入补充诉求（如 batch=1 latency-critical 下 W8A8 vs W4A16 预期差异）

### 接入分级结论
- 建议分级（L0/L1/L2/L3）：
- 判定理由：
- 升级到 L2/L3 的条件：

### 对后续阶段的影响
- 对阶段 1（并行化）：
- 对阶段 4（量化改造）：
- 若阶段 4 复核发现融合与量化冲突：

### 当前建议
- 建议结论：进入量化改造 / 暂不量化 / 先补契约
- 前提条件：
- 若前提不满足：
- 需补充的量化算法或产物契约：
```

---

## 第六步：实施量化接入

**读取**：`references/quantization-structure-cards.md` 命中结构卡的「改造要点 / 已确认模型经验 / post-load 例外」。

仅在改造任务中执行。每次改造围绕“一个模块映射或一个 post-load 问题”推进，验证通过后再继续下一个风险点。

### 6.1 接入 `quant_config`

> 改造前先确认目标模型的 model loading 入口（不同模型可能走不同 loader），在该入口处接入量化方案解析。

1. 从 `config.json` 读取 `quantization_config` 或 `compression_config`。
2. 在实际 model loading 入口把量化配置解析成运行时 `CompressedTensorsConfig`（实现上可能是 `get_quant_config(...)`、`CompressedTensorsConfig.from_config(...)`，或 model worker / loader 内部 helper）。
3. 将 `quant_config` 挂到模型 config 或 runner，并继续透传到各可量化模块。
4. 保持模块 prefix 稳定，使 `targets` / `ignore` 可正确命中。
5. 输出头模块（如 `lm_head`）即使匹配 `targets:["Linear"]` 又不在 `ignore`，模型层构造时不传 `quant_config` 即可。

### 6.2 替换或映射 runtime object

- Dense Linear：接入 `Linear` / `ReplicatedLinear` / `RowParallelLinear` 的量化方法。
- MoE experts：优先映射到 `MoEGMM` / `FusedMoEGMM`；统一 `W8A8` 可检查 `gmm_quant_mode` 是否继承 `mm_quant_mode`。
- 混合位宽 MoE：必须有独立 `targets=["MoEGMM"]` config group。
- KVCache：按 `kv_cache_scheme` 和 cache scale 接入。
- `ignore` 模块：显式浮点回退，不参与量化收益统计。

### 6.3 处理权重加载和 post-load

必须核对：

- 量化张量名与参数前缀
- TP/EP shard 和 online split 规则
- weight 转置
- NZ/base format
- scale dtype
- smooth scale 完整性
- MoE expert pack/unpack
- cache scale 来源

### 6.4 恢复量化后运行时融合

Dense 模型优先检查：

- 量化产物是否拆散原 fused `gate/up`
- 量化产物是否拆散原 fused `q/k/v`
- 是否可在 post-load 后恢复更大的量化 matmul

原则：恢复运行时融合是为了减少小粒度量化 matmul 和动态量化开销；不能改变量化方案语义。

### 6.5 处理融合算子冲突（唯一回退规则真相源）

> 本节是融合算子量化冲突的唯一回退规则真相源；其它 reference 不再复述本规则。

如果融合算子拒绝量化 dtype/layout/scale：

1. 不修改量化方案、`targets`、`ignore` 或张量语义。
2. 检查主流程是否已传入用户确认的融合冲突处理原则；若没有，停止改造并输出待决策项。
3. 若用户已接受回退，将该模块回退到原有非融合算子路径。
4. 继续按原量化产物加载和验证。
5. 记录融合算子名称、失败输入契约、错误信息、回退点、后续融合量化需求。

### 6.6 ignore 不完备的兼容处理

> 触发条件见 `references/quantization-contract.md` §3「张量语义要求」的 *ignore 完备性启发式*：产物里既不在 `ignore` 也无 `weight_scale` 的 Linear 候选模块。

#### 判定流程

**前置 0 步：模型层跳过等价**

模型代码若已通过常规机制（如 transformers `_keys_to_ignore_on_load_unexpected` / `load_state_dict(strict=False)` / 模型层不传 `quant_config`）让缺 scale 模块事实上不参与量化加载，等价合规。此时不必继续触发本节其它步骤，但仍需记录补充诉求，让产物方下版补 `ignore`。

1. 读 `deploy_quantization.md` 量化范围声明（量化目标 / bit 配置 / 明确量化或非量化模块）。
2. **量化范围明确不覆盖该模块**（如“量化目标: moe, mlp”不含 mtp）：视为产物方 implicit BF16 + ignore 漏列；进入隐式回退。
3. **量化范围模糊或与 ignore 不一致**：停止改造，输出补充诉求，要求产物方明确说明 `{module list}` 是 implicit BF16 还是 scale 丢失，并补 `ignore` 或 scale。
4. **量化范围明确应该量化但 scale 缺失**：停止改造，输出“产物 scale 丢失，回压重出”的补充诉求。

#### 隐式回退动作（仅步骤 2 命中时）

- 在 infer 侧 `quant_config` 解析后，把这些前缀注入 effective ignore set（扩展 `CompressedTensorsConfig.ignore`），让 `get_quant_method` 走 `UnquantizedLinearMethod` 浮点路径。
- **不修改产物 `config.json`**（产物语义只读）。
- 在 `progress.md` 记录：implicit BF16 模块前缀列表 / 量化范围依据（来自 `deploy_quantization.md` 哪一段） / 补充诉求模板（产物方下版应将 X 显式补入 `ignore`）。

#### 禁止动作

- 不要造假 scale 强行量化。
- 不要 int8 forward without scale。
- 不要修改产物 `config.json`。
- 不要替产物方决策“漏 ignore 还是漏 scale”。

### 完成标志

- [ ] 量化配置已透传
- [ ] 模块映射已完成
- [ ] 权重加载和 post-load 已处理
- [ ] 融合冲突已处理或记录
- [ ] ignore 不完备处已按 §6.6 完成隐式回退或停止回压
- [ ] 没有擅自修改量化方案语义

---

## 第七步：验证真实生效与收益

> 本步是量化验证要求的唯一真相源；其它 reference 不再复述本规则。

**读取**：`references/quantization-fusion-and-benefit.md` §B.1 五维口径。

量化改造不能只看代码 diff，必须证明真实运行。

### 7.1 功能验证

- 量化模型可加载
- Prefill / Decode 至少跑通一次
- 输出可读、不重复、非全零、非提前 EOS
- 日志或状态文件能证明走到量化 runtime
- 验证命令记录了模型路径、量化产物路径、部署卡数和 YAML

**等价性自检（接线正确性，非精度评测）**

量化算法的绝对精度由产物方在 `deploy_quantization.md` 报告，本 skill 不评测精度；这里只验证「接线对不对」——量化路径输出是否与同模型 infer BF16 路径一致：

- 固定一组 prompt（覆盖分布即可，无需标注），量化基线与 infer BF16 基线各跑一次 greedy。
- 文本 diff：记首个分歧 token 位置 + 是否语义等价。W8A8 允许细微 token 差异，重点是不应出现乱码 / 早停 / 与 BF16 显著走偏。
- 这能抓功能验证漏掉的失败模式：scale/layout 接错时「能跑但输出垃圾」。
- 若连 BF16 参照都跑不了：标「等价性未核，转产物方精度报告 / 评测侧」，不静默判通过。

### 7.2 生效验证（probe 优先，贴输出，不空打勾）

- **probe-B 对象级（主，零额外开销）**：加载后抽查量化目标层，`type(layer.quant_method).__name__` 不应为 `UnquantizedLinearMethod`（应是 `CompressedTensorsW8A8Int8LinearMethod` / MoEGMM method 等）。只查 load 后构造结果，不跑 decode、不 profiling。
- **probe-C 权重级（廉）**：从加载日志统计命中的 `qweight`/`weight_scale` 等量化张量数，与期望量化模块数对账。
- **probe-A 算子级（复用 §7.3 性能那次 profiling，不单独跑）**：若已 profiling，在 `op_statistic` grep `QuantBatchMatmul`/`GroupedMatmul`/`DequantSwigluQuant`/`DynamicQuant` 计数 >0；未跑性能则不强制。

并记录：已量化模块 / `ignore` 浮点回退模块 / 融合算子回退模块。

### 7.3 性能和显存验证

对比口径：

| 指标 | 对比对象 |
| --- | --- |
| 显存 | 非量化基线、量化前最新基线、量化基线 |
| Prefill | 非量化基线、量化前最新基线、量化基线 |
| Decode | 非量化基线、量化前最新基线、量化基线 |
| 部署卡数 | 原部署形态、量化后部署形态 |

> 性能数据需多轮取中位数并充分 warmup，避免 single-sample 推断。

若初评估判断量化后单卡可能满足，优先给出单卡验证结果。

### 7.4 失败处理

- 映射、加载、post-load、scale dtype 或 runtime object 错误：优先修复 infer 侧。
- 融合算子量化冲突：回退非融合路径继续验证，并记录需求。
- 契约缺失或权重不完整：停止改造，输出补充诉求。
- 性能未提升但功能生效：保留证据，分析动态量化开销、算子粒度、部署配置和 profile。性能反常时先做归因分离再下结论，避免在未做分离前直接采用单一根因解释。

### 完成标志

- [ ] 功能验证通过
- [ ] 量化输出与 BF16 等价性自检通过（或已标记未核原因）
- [ ] 量化真实生效证据完整（probe-B 至少一层 quant_method 断言通过）
- [ ] 显存、Prefill、Decode、部署卡数有对比口径
- [ ] 失败项或回退项已记录

---

## 第八步：写回 progress.md 与经验沉淀

**读取**：`references/quantization-structure-cards.md` 末尾「经验沉淀模板」与「新结构补卡规则」。

### 8.1 写回 progress.md

每次执行后写入 `{model_dir}/agentic/progress.md` 工作区：

- 当前模式：初评估 / 改造 / 验证
- 量化权重目录、量化模式、覆盖范围
- `compressed-tensors` 契约结论
- 结构指纹、命中参考卡、接入分级
- `quant_target -> infer runtime object` 映射
- `ignore` / 浮点回退模块
- 改造文件、关键实现点、post-load 处理
- 融合算子回退清单、原始错误、回退点、后续融合量化需求
- 验证命令、日志证据、量化是否真实生效
- 显存、Prefill、Decode、部署卡数对比
- 阻塞点、补充诉求、用户待决策项

### 8.2 后验决策依据（唯一后验决策真相源）

> 本节是量化基线产出后用户决策的唯一真相源；其它 reference 不再复述本规则。

量化基线产出后，本 Skill 只输出证据和建议，不替用户做最终取舍；最终确认由主流程负责呈现给用户。需要给主流程提供三种选择的判断依据：

- 采用部分融合算子回退的量化方案：接受当前量化收益，并把回退项沉淀为融合算子量化需求。
- 保留融合算子、跳过量化：量化收益不足或回退代价过高时，继续非量化优化链路。
- 修正量化方案后迭代验证：量化方案需要补契约、补 target、补 ignore、补张量语义或补上游量化能力。

不要在量化基线产出前给最终建议；不要替用户确认是否接受量化方案。

### 8.3 经验沉淀

任务完成后，若产生可复用经验，追加到对应结构卡或参考卡。

沉淀规则：

- 只按结构沉淀，不按模型名堆卡。
- 现有结构卡可覆盖时，只补差异点和实践数据。
- 只有出现新结构、新 runtime object、新张量语义、新 post-load 规则或新融合回退模式时，才新增参考卡。

样例至少包含：

- 适用结构指纹和量化模式
- 量化方案来源、不可修改项、实际消费方式
- 改造方案：配置、runner、模型映射、post-load、回退策略
- 改造要点：可复用代码位置、关键 runtime object、验证命令
- 问题及解决方案：错误现象、根因、处理方式
- 融合算子回退清单和融合量化需求
- 量化基线收益和用户最终决策

---

## 输出格式

结束时按顺序输出：

1. 契约结论：满足 / 不满足 / 需补充。
2. 当前模式：初评估 / 改造 / 验证。
3. 量化产物目录、量化模式、覆盖范围。
4. 结构参考卡、接入分级、关键映射。
5. 已量化模块、浮点回退模块、融合算子回退模块。
6. 验证证据和收益结论。
7. 后续决策选项和建议。
8. 阻塞点或补充诉求。
9. 已沉淀或待沉淀的参考卡位置。

---

## 参考文档索引

| 主题 | 路径 | 覆盖章节 |
| --- | --- | --- |
| 产物契约 / 张量语义 / 运行对象映射 / 本仓 9 步机制 / 关键代码入口 | `references/quantization-contract.md` | §1-§7 |
| 典型结构卡 + 模型经验 + post-load 例外 + 新结构补卡规则 | `references/quantization-structure-cards.md` | 结构卡 A/B/C + post-load 汇总 + 经验沉淀模板 |
| 融合算子兼容性 + 收益判断口径 + 主线路径 + 文件职责 | `references/quantization-fusion-and-benefit.md` | §A 兼容性 / §B 收益 / §C 文件职责 |
