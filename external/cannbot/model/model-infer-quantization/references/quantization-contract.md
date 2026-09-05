# 量化产物接入契约与运行机制

本卡描述 infer 仓 `compressed-tensors` 主线接入侧的事实：产物最小交付物、配置字段、张量语义、运行对象映射，以及本仓量化运行机制。其它量化路线不在本 skill 中实现。

> 回退规则、验证要求、后验决策见 `SKILL.md` 第六步 6.5 / 第七步 / 第八步 8.2，本卡不复述。

## 1. 最小交付物

| 文件 | 要求 |
|------|------|
| `config.json` | 包含 `quantization_config` 或等价 `compression_config` |
| `model.safetensors.index.json` | 能索引量化张量和保留张量 |
| `*.safetensors` | 包含量化权重、scale、bias 等运行所需张量 |
| `deploy_quantization.md` | 写清量化对象、张量语义、回退策略和下游加载要求 |

缺任一关键交付物：不进入改造，输出量化算法/产物契约补充诉求。

## 2. `config.json` 字段要求

必须能读出：

- `quant_method: compressed-tensors`
- `config_groups`
- `targets`
- `ignore`
- 权重量化格式、激活量化格式、dtype
- 可选 `kv_cache_scheme`

禁止 infer 侧因报错擅自修改 `targets`、`ignore`、量化 dtype 或张量语义后宣称成功。

## 3. 张量语义要求

量化权重索引必须能说明：

- 权重张量，例如 `qweight`
- scale 张量，例如 `weight_scale`
- 可选 bias / zero point / smooth scale
- 张量名到 infer 模型参数名前缀的映射关系
- 未量化模块的保留权重

张量语义不清时，不能猜测落代码；输出补充诉求。

**ignore 完备性启发式**：若产物中存在既不在 `ignore`、又没有对应 `weight_scale` 的 `Linear` 候选模块，说明 `ignore` 可能漏列 implicit BF16 模块，或 scale 丢失。命中此启发式时按 SKILL 第六步 §6.6 判定处理，不擅自决策。

## 4. 运行对象映射

优先把量化目标映射到 infer 已有运行对象：

| 量化目标 | infer 运行对象 |
|----------|----------------|
| Dense Linear | `Linear` / `ReplicatedLinear` / `RowParallelLinear` |
| Dense MLP gate/up/down | 可量化 Linear；必要时 post-load 恢复 `gate/up` 融合 |
| Attention q/k/v/o | 可量化 Linear；必要时 post-load 恢复 `q/k/v` 融合 |
| MoE Expert gate/up/down | `MoEGMM` / `FusedMoEGMM`，不满足时回退 per-expert Linear |
| KV Cache | `kv_cache_scheme` / cache scale / cache runtime |
| `ignore` 模块 | `UnquantizedLinearMethod` 浮点回退，不参与量化收益统计 |

无法映射时，先判断是 infer runtime 能力缺失还是量化产物契约缺失。契约缺失时停止（对应 L3）；runtime 缺失时也停止当前适配，标出 runtime gap 并确认是否另起 runtime 补齐任务（对应 L2，见 SKILL.md 第四步 §4.3），不在本次适配内默默补 kernel。

## 5. 本仓量化运行机制（9 步）

本仓当前量化主线是 `compressed-tensors`，按以下顺序生效：

1. runner / config 读取 `config.json::quantization_config` 或 `compression_config`。
2. 在真实 model loading 入口把量化配置解析成运行时 `CompressedTensorsConfig`；实现可能在 runner / model worker / loader 内部 helper，不假定唯一入口函数名。
3. 解析阶段通常调用 `CompressedTensorsConfig.from_config()`，处理 `config_groups`、`targets`、`ignore`、`kv_cache_scheme`，生成 `target_scheme_map`、`mm_quant_mode`、`gmm_quant_mode`。
4. 将 `quant_config` 挂到模型 config / runner，并在模型构造 `LinearBase` 或 `FusedMoEGMM` 时继续透传。
5. `quant_config.get_quant_method(layer, prefix)` 按 `ignore`、layer prefix、module class 和 target 匹配运行方法。
6. `create_weights()` 注册 int8/int32 权重、scale、smooth scale 等参数。
7. weight loader 从 safetensors 加载量化张量。
8. `process_weights_after_loading()` 做转置、NZ/base format、scale dtype、MoE pack/unpack 等后处理。
9. forward 走 NPU 量化 kernel，例如动态 activation quant + `npu_quant_matmul`，或 MoE `npu_grouped_matmul`。

能否接入的关键不是模型名，而是：结构里的量化对象能否映射到 §4 的 runtime object，产物契约（§1-§3）是否写清，post-load 是否能还原 runtime 需要的 layout。

## 6. 四层落点

业界主流推理量化通常拆成四层，本仓按这四层落地：

| 层次 | 主流做法 | 本仓落点 |
|------|----------|----------|
| 产物契约 | 配置声明量化对象、bit、strategy、ignore/fallback | `quantization_config` / `deploy_quantization.md` |
| 模块替换 | Linear、MoE、KVCache 等运行对象消费量化权重 | `LinearBase` / `FusedMoEGMM` / KVCache |
| 权重后处理 | 转置、pack/unpack、device format、scale dtype | `process_weights_after_loading()` |
| 验证闭环 | 加载证据、输出可用性、性能/显存收益、回退记录 | 量化基线 |

## 7. 关键代码入口

| 主题 | 路径 |
| --- | --- |
| 量化 runtime 入口 | `cann-recipes-infer/module/quantization/__init__.py` |
| 量化配置接入入口 | 实际模型加载入口（例如 `cann-recipes-infer/executor/core/model_worker/model_worker.py`） |
| compressed-tensors 配置解析 | `cann-recipes-infer/module/quantization/compressed_tensors/compressed_tensors.py` |
| Linear W8A8 runtime | `cann-recipes-infer/module/quantization/compressed_tensors/compressed_tensors_w8a8_int8.py` |
| MoE GMM runtime | `cann-recipes-infer/module/quantization/compressed_tensors/compressed_tensors_moe_gmm.py` |
