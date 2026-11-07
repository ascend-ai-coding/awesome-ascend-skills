# YAML 配置模板（两模式共用）

cann-recipes-infer 框架部署模式和独立部署模式 yaml schema 完全一致，均为 4 段式：`model_config` / `data_config` / `parallel_config` / `scheduler_config`。差异仅在解析方式（框架部署走 `InferenceConfig.from_dict`，独立部署 Runner 用 `yaml.safe_load` 直接读）。

`InferenceConfig` 共 5 个子配置：上述 4 段 + `DisaggConfig`。`DisaggConfig.disaggregation_mode` 默认 `NONE`（offline），无需在 YAML 配置；migrator 默认产物只覆盖前 4 段。online（PD 分离）需要的 disagg 字段见 `cann-recipes-infer/docs/design/online_inference_design.md`。

---

## 1. yaml schema（完整字段，单卡默认值）

```yaml
model_name: "{model-key}"             # 必须与 support_models.py 中 key 一致
world_size: {N}                       # 1=单卡 / >1=多卡（多卡值由 model-infer-parallel-impl skill 推导）

model_config:
  model_name: "{model-key}"
  model_path: "{absolute_or_relative_weights_path}"
  exe_mode: "eager"                   # ["eager", "ge_graph", "npugraph_ex"]
  enable_profiler: False
  with_ckpt: True
  enable_weight_nz: True              # 可选，部分模型量化场景需要
  custom_params:                      # 模型特有开关（如 enable_multi_streams / moe_chunk_max_len / perfect_eplb）
    {key}: {value}

data_config:
  dataset: "default"                  # framework 模式从 cann-recipes-infer/dataset/default_prompt.json 读
  input_truncated_len: 4096
  prompts:                            # 独立部署专属（framework 模式忽略此字段）
    - "..."

parallel_config:
  world_size: {N}                     # 与顶层 world_size 一致
  attn_tp_size: {value}               # 单卡=1；多卡按 parallel-impl 推导
  dense_tp_size: {value}              # 多卡 + Dense FFN 切分时配置
  moe_tp_size: {value}                # MoE 模型需要
  embed_tp_size: {value}
  lmhead_tp_size: {value}
  o_proj_tp_size: {value}             # MLA 模型需要
  # cp_size / kvp_size：长序列场景；parallel-impl skill 暂不直接支持

scheduler_config:
  batch_size: {全局 batch}             # 多卡按 attn_dp_size 推导每 rank batch
  max_new_tokens: 32
  max_prefill_tokens: 4096
  block_size: 128                     # Paged 模式才需要，由 kvcache skill 改造时引入
```

> migrator 阶段产物默认值：`world_size=1`、各 `*_tp_size=1`。多卡场景由 model-infer-parallel-impl skill 按 model-infer-parallel-analysis 推导结果填。

---

## 2. 命名规范

命名维度：参考仓内已注册模型命名，按"模型_rank_N_拓扑_后端_场景"或"场景_模型_rank_N_拓扑_量化_特性"组合维度。禁止用非结构性差异的临时描述符（如 `_4k1k`、`_b8`、`_test`）做差异化命名。

粒度判定：按差异类型决定

- 结构性差异（建 yaml）：并行拓扑、量化模式、prefill/decode 场景、特性开关组合（如 mtp / sp / kvp / eplb / cache_compile / superkernel）整套切换。
- 运行时参数差异（改 yaml 字段即可，不建 yaml）：batch_size、input_truncated_len、max_new_tokens、temperature 等。
- exe_mode 切换：通常改 yaml 字段即可（不建 yaml）；仅当后端切换伴随多个特性开关组合差异时才单独建。

保留策略：仅保留能成功运行的 yaml；单卡等早期临时配置在验证完成后及时清理。

布尔值统一用 `True` / `False`，不用 `true` / `false`（与仓内主流风格一致）。

---

## 3. 多卡填值（由 model-infer-parallel-impl skill 完成）

根据 model-infer-parallel-analysis 决策结果填入 `parallel_config` 段：

- `world_size`：顶层 + `parallel_config` 双写一致
- `attn_tp_size` / `dense_tp_size` / `moe_tp_size` / `embed_tp_size` / `lmhead_tp_size` / `o_proj_tp_size`：按 parallel-analysis 推导值填
- `moe_ep_size` / `attn_dp_size`：框架自动推导（`world_size // *_tp_size`），不需手填
- `custom_params`：多卡特性开关（`enable_multi_streams` / `moe_chunk_max_len` / `perfect_eplb` 等）

---

## 4. 参考仓内已注册模型实例

两种命名模板各取一个代表：

```
config/
├── {model_name}_rank_64_64ep_w8a8c8_decode_aclgraph_benchmark.yaml  # 模型_rank_N_拓扑_量化_场景_后端 形态
└── decode_{model_name}_rank_128_128ep_a8w8c8_mtp.yaml               # 场景_模型_rank_N_拓扑_量化_特性 形态
```

---

## 5. 完成标志

- [ ] 每种部署场景有独立的 YAML 文件
- [ ] 配置文件命名符合规范（命名维度 + 粒度判定）
- [ ] 单卡早期临时配置已清理（保留策略）
