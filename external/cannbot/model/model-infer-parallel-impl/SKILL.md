---
name: external-cannbot-model-model-infer-parallel-impl
description: 基于 PyTorch 框架的昇腾 NPU 模型推理并行切分实施技能。根据已确认的 parallel_config，实施模型代码的并行化改造，包括并行线性层替换、MoE
  并行模式适配、通信组创建、Embedding/LMHead 并行、YAML 配置生成和权重转换。支持 infer 仓框架部署与独立部署两种模式。触发场景：model-infer-parallel-analysis
  完成后需要实施改造、现有模型需要支持新的并行配置。
original-name: model-infer-parallel-impl
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# 模型并行切分实施

按已确认的 parallel_config 实施模型并行化改造，覆盖并行层替换、通信组接入、YAML 配置生成、权重处理与验证。

---

## 部署模式判定

**框架部署模式**：模型类构造接受 `infer_config + comm_manager`，在 `__init__` 调 `init_parallel_comm_group()` 按需 `comm_manager.register_group()` 注册通信组（框架自动判断 HCCL 物理组复用），启动通过 `cann-recipes-infer/executor/scripts/function.sh::launch`。

**独立部署模式**：Runner 自管推理循环（含进程组初始化、通信组建立、ForwardMetaData 构造、prefill / decode 调度），通过 `torchrun` 启动。

### 判定优先级

| 优先级 | 判定来源 | 说明 |
|---|---|---|
| 1 | dispatch prompt 显式指定 | prompt 含 `部署模式: 框架部署` 或 `部署模式: 独立部署` → 直接采用 |
| 2 | 用户对话指定 | 仅独立调用（用户直接 invoke skill）时适用 |
| 3 | migrator 阶段产物自检 | 模型注册到 `cann-recipes-infer/executor/core/support_models.py` → **框架部署**；模型目录有 `runner_*.py` 不继承 framework 基类、自管推理循环 → **独立部署**；不确定时默认框架部署 |

判定后跳转：
- 框架部署模式 → 「## 框架部署模式」
- 独立部署模式 → 「## 独立部署模式」

> 两模式共通流程（配置参数速查 / YAML 配置生成 / 权重处理 / 验证）见后续公共章节。

---

## 配置参数 → 实施步骤速查

根据 parallel_config 中各参数值，确定需要执行哪些步骤：

| 参数条件 | 需要的通信组 | 需要的代码改造 |
|---------|------------|--------------|
| `attn_tp_size > 1` | `attn_tp_group` | Attention QKV/O 替换为 ParallelLinear |
| `attn_tp_size = 1` 且 `world_size > 1` | 无 Attention TP 组（走 DP） | Attention 不需要 TP 替换 |
| `dense_tp_size > 1` | `dense_tp_group` | Dense FFN Gate/Up/Down 替换 |
| `moe_tp_size > 1` | `moe_tp_group` | MoE 专家 FFN 做 TP 切分 |
| `moe_tp_size = 1` 且有 MoE | `moe_ep_group` + `moe_ep_group_mc2`（均需 HCCL group name；MC2 组须 `allow_physical_reuse=False` 区分独立物理组） | MoE EP dispatch/combine 实现 |
| `o_proj_tp_size ≠ attn_tp_size` | `oproj_tp_group`（独立组） | O_proj 使用独立通信组 |
| `embed_tp_size > 1` | `embed_tp_group` | Embedding 替换为 VocabParallelEmbedding |
| `lmhead_tp_size > 1` | `lmhead_tp_group` | LMHead 替换为 ColumnParallelLinear |
| 相邻模块 TP 度不同 | — | 模块间插入 AllGather/ReduceScatter 重排 |

> `tp_size = 1` 的模块不需要 TP 替换和对应通信组。先扫一遍 parallel_config，标注哪些模块需要改，再逐步执行。

---

## 框架部署模式

接入 `cann-recipes-infer/executor/core/` 框架，模型类构造接受 `comm_manager`，启动走 `cann-recipes-infer/executor/scripts/function.sh::launch`。

### 实施流程

```
第一步：确认输入 + 选择参考模型
    ↓
第二步：通信组创建
    ↓
第三步：逐模块并行层替换
    ↓
第四步：Embedding / LMHead 并行
    ↓
后续 YAML / 权重 / 验证 走公共章节
```

**禁止**：跳过第二步直接替换层。

---

### 第一步：确认输入 + 选择参考模型

#### 1.1 确认 parallel_config

从编排层（主 agent prompt 或用户输入）获取已确认的配置：

```yaml
parallel_config:
  attn_tp_size: {value}       # attn_dp_size = world_size // attn_tp_size
  dense_tp_size: {value}
  moe_tp_size: {value}        # moe_ep_size = world_size // moe_tp_size
  embed_tp_size: {value}
  lmhead_tp_size: {value}
  o_proj_tp_size: {value}     # MLA 模型需要，非 MLA 可省略
```

#### 1.2 选择参考模型

根据目标代码实现模式，选择仓库中最接近的已适配模型阅读其 modeling 代码：

| 实现模式 | 参考模型 | 关注点 |
|---------|---------|--------|
| 标准 GQA + 纯 TP | GPT-OSS | QKVParallelLinear 基础用法 |
| 标准 GQA + MoE EP | Qwen3-MoE | MoE routing + npu_swiglu 融合 |
| MLA + 模块差异化 TP + EP | DeepSeek-R1 / V3.2 | MLA 投影切分、oproj 独立 group、EP dispatch/combine |

**必须**：读取参考模型的 `modeling_*.py`，了解并行层替换和通信组使用方式。框架的通信组由 `CommManager` 在 ModelWorker 初始化时统一创建，模型类内通过 `comm_manager.get_group("attn_tp_group")` 等接口获取。

#### 完成标志

- [ ] parallel_config 已确认
- [ ] 参考模型已选定并阅读关键代码

---

### 第二步：通信组创建

并行层替换前，必须先在模型初始化中创建通信组。

#### 2.1 通信组创建与获取

CommManager 提供 `register_group()` API，模型在 `__init__` 调用 `init_parallel_comm_group()` **按需注册**自己用到的所有通信组，框架按 (`group_num`, `group_size`, `group_stride`) 形状自动判断 HCCL 物理通信组复用，需要独立物理组（MC2 fullmesh_v2、multi-stream 副本组）时显式 `allow_physical_reuse=False`。

```python
from executor.core.config import InferenceConfig, CommManager

class XxxForCausalLM(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig,
                 comm_manager: CommManager = None, prefix: str = ""):
        self.comm_manager = comm_manager
        self.world_size = infer_config.parallel_config.world_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        # ... 其他 *_size
        self.init_parallel_comm_group()  # 在子模块构造前完成注册
```

`init_parallel_comm_group` 内部调 `self.comm_manager.register_group(name, group_num, group_size, ...)` 注册三类组：

- **基础 TP 组**（按本模型启用的切分注册）：`attn_tp_group` / `embed_tp_group` / `lmhead_tp_group`
- **条件组**（`*_size > 1` 时启用）：`moe_tp_group` / `moe_ep_group` / `dense_tp_group` / `oproj_tp_group`；其中 `moe_ep_group` 需传 `group_stride` 和 `return_name=True`
- **特殊独立物理组**（必须 `allow_physical_reuse=False`）：`moe_ep_group_mc2`（MC2 fullmesh_v2 通信）、`*_stream1` 副本组（multi-stream micro-batch）等；MC2 组的 `hccl_buffer_size` 通过 `calc_moe_hccl_buffer_size` 算出

模板代码、参数表与上游 `qwen3_moe` / `deepseek_r1` 参考实现见 `references/framework_moe_parallel.md` 的"通信组注意事项"段。

**取通信组**：模型内部用 `comm_manager.get_group(name)` 取 ProcessGroup，`comm_manager.get_group_name(name)` 取 HCCL name（NPU dispatch / combine 算子要求），`comm_manager.get_rank(name)` 取组内 rank。

#### 2.2 模块级差异化并行的通信组

当 attn_tp ≠ dense_tp ≠ moe_tp 时，各模块从 `comm_manager` 取不同的通信组（`attn_tp_group` / `dense_tp_group` / `moe_ep_group` 等）。模型在 `init_parallel_comm_group()` 里只需按 YAML 配置 `*_size > 1` 的模块注册对应组即可，框架会自动判断物理组复用。

#### 2.3 DP 大小自动计算

```
attn_dp_size  = world_size // attn_tp_size
moe_dp_size   = world_size // moe_tp_size
moe_ep_size   = moe_dp_size       # EP size = DP size
embed_dp_size = world_size // embed_tp_size
```

#### 完成标志

- [ ] YAML `parallel_config` 各 `*_tp_size` 字段已按目标策略配置
- [ ] 模型代码通过 `comm_manager.get_group(...)` 正确取用每个模块的通信组
- [ ] DP / EP size 由框架自动推导（world_size // tp_size）

---

### 第三步：逐模块并行层替换

> 并行层替换同时支持了权重在线切分：ParallelLinear 内置 `weight_loader()`，按 `tp_rank` 自动加载切片。权重处理详见公共章节「## 权重处理」。

**通用规则**：所有线性 / Embedding 层（Attention QKV/O、Dense MLP、Embedding、LMHead）一律用对应 ParallelLinear 包装，不论 tp_size 取值（含 `=1`）。tp_size=1 时不引入通信开销，保留统一 weight_loader、quant_method dispatch 与 comm_manager 联动。

#### 3.1 Attention 层

QKV → `QKVParallelLinear`，O → `RowParallelLinear`，均使用 `attn_tp_group`。MLA 模型 O_proj 可能使用独立的 `oproj_tp_group`。

> **forward 签名约定**：设计文档 `executor_design.md` §6.1 要求模型类 forward 接收 `slot_mapping` / `block_table`；参考模型统一通过 `**kwargs` 透传，不显式列参数（见 `cann-recipes-infer/models/qwen/models/modeling_qwen.py::forward`），照此写法即可。

> **paged 模式联动**：若模型已通过 kvcache skill 改造为 paged 模式（实现 `get_cache_info()` + 构造 `cache_entries`），attn_tp_size 改动后必须同步更新 `cache_entries` 的 `num_head` 字段为 `num_key_value_heads_per_rank = max(num_kv_heads // attn_tp_size, 1)`，否则 KV 按错误的 head 数预分配。

> **TP+DP 通信模式（`attn_tp_size > 1` 且 `attn_dp_size > 1`）**：注册模型 attention 层用 `all_gather_into_tensor` + `reduce_scatter_tensor` 配对 `attn_tp_group`（不用 `all_reduce`）：layer 间 hidden 维持 `[T/tp, H]` 切分以省 tp_size 倍激活内存，attention 入口 AllGather 还原全 T，出口 ReduceScatter 切回。通信原语要求各 rank 等长，因此 model 顶层 forward 必须做：
>
> 1. packed T pad 到 `attn_tp_size` 倍数
> 2. 按 rank 切 input_ids `[rank * T/tp : (rank+1) * T/tp]`（建议 position_ids / cos/sin 不切，每 rank 保留 T_real，简化 attention 内部对齐）
> 3. attention 内部 QKV 前 AllGather，O 投影后 ReduceScatter
>
>    备注：framework metadata（`slot_mapping` / `actual_seq_lengths_kv`）按 T_real 构造，cache write 和 FA 必须 T_real。AllGather 出 `[T_padded, H]` 后用 `prompt_tokens` slice 回 T_real 再算，O 投影后 cat-zeros 补回 `[T_padded, H]` 给 ReduceScatter。
> 4. Prefill 末尾选末 token：AllGather hidden → `index_select(cu_seq_q - 1)` → lm_head（原 BSND 路径是本地取末位 → lm_head → gather logits，TND 顺序倒置）
>
> 参考 `cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py::Qwen3MoeModel.forward` 和 `Qwen3MoeAttention.forward`。

#### 3.2 Dense FFN 层

Gate/Up → `MergedColumnParallelLinear`，Down → `RowParallelLinear`，激活用 `torch_npu.npu_swiglu`，均使用 `dense_tp_group`。

#### 3.3 MoE 层（有 MoE 时必须处理）

- `moe_tp_size > 1`：专家 FFN 做 TP 切分 + AllReduce
- `moe_tp_size = 1`：EP 模式，Prefill 用 re_routing + AllToAll，Decode 用 dispatch/combine + AllToAll

> MoE 并行与融合算子紧密耦合，详见 `references/framework_moe_parallel.md`（含完整代码和算子说明）。

> MoE 模型必填属性：config 暴露 `num_experts_per_tok` / `n_routed_experts`（hccl_utils 计算 MC2 buffer 用），ForCausalLM 实例暴露 `self.num_experts` / `self.num_experts_per_tok`（EPLB hook 校验）。HF 上游若用其他命名（如 `moe_topk`）加 `@property` alias。

#### 3.4 模块间数据重排（当相邻模块 TP 度不同时）

边界处需要 AllGather/ReduceScatter 做数据重排。

> 上述 1-4 项的代码示例（Attention / Dense FFN / Embed / LMHead / 数据重排）见 `references/framework_code_examples.md`。

#### 完成标志

- [ ] Attention 层 QKV/O 已替换为并行版本
- [ ] Dense FFN 层 Gate/Up/Down 已替换（如有差异化 tp）
- [ ] MoE 层已按选定模式实现（TP 或 EP Prefill/Decode）
- [ ] 模块间数据重排已正确插入（不同 TP 度边界的 AllGather/ReduceScatter）
- [ ] 各模块使用了正确的通信组

---

### 第四步：Embedding / LMHead 并行

- Embedding → `VocabParallelEmbedding`，按词表维度切，`tp_rank=comm_manager.get_rank("embed_tp_group")`
- LMHead → `ColumnParallelLinear`，`tp_rank=comm_manager.get_rank("lmhead_tp_group")`

完整代码示例见 `references/framework_code_examples.md`「Embedding / LMHead」节。

#### 完成标志

- [ ] Embedding 已按 embed_tp_size 并行化
- [ ] LMHead 已按 lmhead_tp_size 并行化
- [ ] 约束检查通过（见公共章节「## 验证」配置校验）

---

## 独立部署模式

Runner 自管 `dist.init_process_group` + ParallelContext 构造，启动通过 `torchrun`。modeling 代码 import 路径与框架部署一致，仅构造签名一处差异（`comm_manager` → `parallel_ctx`）。

完整 ParallelContext 骨架、Runner 改造、modeling 改造、启动模板见 `references/standalone_parallel.md`。

### 复用与自管边界

按功能边界划分：

- **自管**：进程组初始化、通信组配置数据结构（`ParallelContext`，对外接口对齐 CommManager）、通信组建立工厂、推理循环（`Runner.model_generate`，命名与签名与仓库 framework 模式 `ModelRunner.model_generate` 对齐，独立部署 inline 实现，不继承基类）、ForwardMetaData 构造、YAML 解析、启动脚本（torchrun）、入口（infer.py）。所有自管逻辑 inline 在 `runner_{model_name}.py` 内。
- **复用**：仓内通用算子层与工具层 —— `cann-recipes-infer/module/linear.py`（ParallelLinear）、`cann-recipes-infer/module/fuse_moe_gmm.py`（FusedMoEGMM）、`cann-recipes-infer/module/quantization/*`、`cann-recipes-infer/executor/utils/hccl_utils.py`（init_comm_group / get_group_name）、`cann-recipes-infer/executor/utils/forward_metadata.py`、`cann-recipes-infer/executor/model_loader/*`。这些与 `torch_npu` / `numpy` 同性质，import 不引入控制流约束。

modeling 代码与框架部署的差异仅在构造签名（`comm_manager` → `parallel_ctx`，接口对齐 `get_group(name)` / `get_group_name(name)` / `get_rank(name)`），forward 计算逻辑、ParallelLinear 用法、MoE EP 算子链、所有 `cann-recipes-infer/module/*` import 路径完全一致。运行时配置（如 `scheduler_config.batch_size`）通过 `parallel_ctx.runtime_cfg["..."]` 访问。

### 实施流程

```
第一步：Runner 改造（dist 初始化 + ParallelContext inline + 注入 modeling）
    ↓
第二步：modeling 代码改造（构造签名 + 并行层替换；并行层替换引用框架部署第三/四步）
    ↓
第三步：启动脚本（torchrun + infer.sh + infer.py）
    ↓
后续 YAML / 权重 / 验证 走公共章节
```

---

### 第一步：Runner 改造

`runner_{model_name}.py` 顶部 inline `ParallelContext` + `build_parallel_context`（约 80 行），`__init__` 内做 dist 初始化、构造 ParallelContext、把 `parallel_ctx` 注入 modeling。

```python
class {ModelName}Runner:
    def __init__(self, yaml_file_path):
        with open(yaml_file_path) as f:
            cfg = yaml.safe_load(f)

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

        self.parallel_ctx = build_parallel_context(cfg)

        self.model = {ModelName}ForCausalLM(
            config=..., parallel_ctx=self.parallel_ctx, prefix="").to(f"npu:{local_rank}")
        self._load_weights(...)
        self._init_kvcache()
```

完整 `ParallelContext` dataclass 与 `build_parallel_context` 工厂见 `references/standalone_parallel.md` §1。

#### 完成标志

- [ ] runner 顶部 inline `ParallelContext` + `build_parallel_context`
- [ ] runner `__init__` 内完成 dist 初始化（读 torchrun 注入的 RANK / WORLD_SIZE / LOCAL_RANK）
- [ ] runner 把 `parallel_ctx` 注入 modeling 类构造

---

### 第二步：modeling 代码改造

构造签名从 `(config, prefix)` 改为 `(config, parallel_ctx, prefix)`，属性访问通过 `parallel_ctx.get_group(name)` / `get_group_name(name)` / `get_rank(name)`。

ParallelLinear 替换、Embedding/LMHead 并行、模块间数据重排、MoE EP 算子链与框架部署完全一致，引用框架部署「第三步」「第四步」与 `references/framework_code_examples.md` / `references/framework_moe_parallel.md`。差异仅在 group 取用从 `comm_manager.xxx` 改为 `parallel_ctx.xxx`，`group_ep` 参数从 `parallel_ctx.get_group_name("moe_ep_group")` 取。

#### 完成标志

- [ ] modeling 类构造签名为 `(config, parallel_ctx, prefix)`
- [ ] ParallelLinear 实例化用 `parallel_ctx.get_rank(name)` 取 tp_rank
- [ ] MoE 模型 dispatch/combine 用 `parallel_ctx.get_group_name("moe_ep_group")` 取 group name

---

### 第三步：启动脚本

`infer.sh` 用 `torchrun` 拉起，从 yaml 读 `world_size` 决定 `--nproc_per_node`，HCCL 环境变量在脚本内 export。完整 `infer.sh` / `infer.py` 模板见 `references/standalone_parallel.md` §4。

#### 完成标志

- [ ] `infer.sh` 用 `torchrun` 拉起，`--nproc_per_node` 从 yaml.world_size 推导
- [ ] `infer.py` argparse 接 `--yaml_file_path`，实例化 Runner，先 `model_generate(prompts, warm_up=True)` 预热再 `warm_up=False` 正式跑
- [ ] HCCL 环境变量（`HCCL_BUFFSIZE` / `HCCL_OP_EXPANSION_MODE` / `HCCL_CONNECT_TIMEOUT`）已在 infer.sh export

---

## YAML 配置生成

YAML schema 两模式 4 段式完全一致：`model_config` / `data_config` / `parallel_config` / `scheduler_config`。差异仅在解析方式（框架部署走 `InferenceConfig.from_dict`，独立部署 Runner 用 `yaml.safe_load` 直接读）。

> 顶层 `world_size` 与 `parallel_config.world_size` 双写：前者供启动脚本（框架部署 `function.sh::launch` / 独立部署 `infer.sh`）读取，后者供业务代码（`InferenceConfig` / `build_parallel_context`）读取，需保持一致。

### 配置模板

```yaml
model_name: "{model_name}"
world_size: {W}

model_config:
  model_name: "{model_name}"
  model_path: "/path/to/weights"
  exe_mode: "eager"                  # 初始用 eager，后续可切 ge_graph
  with_ckpt: True
  enable_weight_nz: True
  enable_profiler: False
  custom_params:                     # 模型特有开关放这里
    enable_multi_streams: False
    moe_chunk_max_len: 65536         # MoE 专用，Decode 用 1024，Prefill 用 65536
    perfect_eplb: False

data_config:
  dataset: "default"                  # framework 模式从 cann-recipes-infer/dataset/default_prompt.json 读
  input_truncated_len: {根据场景}
  prompts:                            # 独立部署专属：yaml 内嵌 prompts（framework 模式忽略此字段）
    - "What is the capital of France?"

parallel_config:
  world_size: {W}
  attn_tp_size: {value}
  dense_tp_size: {value}
  moe_tp_size: {value}        # MoE 模型需要
  embed_tp_size: {value}
  lmhead_tp_size: {value}
  o_proj_tp_size: {value}     # MLA 模型需要

scheduler_config:
  batch_size: {根据显存估算}   # 全局 batch；按 attn_dp_size 推导每 rank batch（框架部署：ExecutionEngine 自动推导；独立部署：Runner 自管推导）
  max_new_tokens: {根据场景}
  block_size: 128             # Paged 块粒度
```

### 命名规范

```
config/
├── {model_name}_rank_{W}_{W}ep_decode.yaml          # Decode 纯 EP
├── {model_name}_rank_{W}_{tp}tp_prefill.yaml        # Prefill 纯 TP
├── {model_name}_rank_{W}_densetp{n}_ep{m}.yaml      # 混合模式
└── ci/
    └── {model_name}_ci.yaml                          # CI 测试用
```

### 完成标志

- [ ] 每种部署场景有独立的 YAML 文件
- [ ] 配置文件命名符合规范

---

## 权重处理

并行层替换后，需要确保权重能正确加载到各卡。两模式都直接 import 仓内 `cann-recipes-infer/module/linear.py`、`cann-recipes-infer/module/fuse_moe_gmm.py` 等，weight_loader 行为完全一致。

### 在线权重切分（推荐）

框架/Runner 在运行时通过各模块的 `weight_loader()` 自动按 rank 加载对应切片（新框架默认行为，无需 YAML 开关）。所有 rank 读同一份完整 checkpoint，各自只保留本卡需要的部分。

**TP 权重加载**：`ColumnParallelLinear` / `RowParallelLinear` 的 `weight_loader` 按 `tp_rank` 取对应列/行切片。

**EP 权重加载**（MoE 模型）：`FusedMoEGMM` 的 `weight_loader` 按 `ep_rank` 过滤专家——只保留 `[ep_rank * experts_per_rank, (ep_rank+1) * experts_per_rank)` 范围内的专家权重，丢弃其他。`load_weights()` 中需通过 `make_expert_params_mapping(num_experts=...)` 生成全局专家映射，逐个传入 `weight_loader(expert_id=...)`。

适配要点：
- 模型类须实现 `load_weights()` 方法，遍历权重文件并匹配到各模块的 `weight_loader()`
- MoE 模型必须调用 `make_expert_params_mapping` 生成专家权重映射
- `MergedColumnParallelLinear`（如 gate+up 合并）需要特殊的 weight_loader 处理 slice 顺序
- 模型类须实现 `process_weights_after_loading()`，加载完成后遍历子模块触发权重后处理（量化 scale 处理、NZ 格式转换等）：

```python
def process_weights_after_loading(self):
    for _, module in self.named_modules():
        qm = getattr(module, "quant_method", None)
        if qm is not None and hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(module)
```

参考实现：`cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py::process_weights_after_loading`、`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py::process_weights_after_loading`。

> 上述示例适用于 gpt_oss / qwen3_moe 等标准模型；MLA + 量化模型（如 deepseek_r1）通常需要扩展（KV-B split、scales dtype cast、跨 EP smooth_scale all_gather 等），照搬 `modeling_deepseek.py::process_weights_after_loading` 而非示例。

### 离线权重转换（备选）

若模型未实现 online split，或需要预切权重用于离线部署：

```bash
bash utils/weight_convert.sh \
    --input_path /path/to/origin \
    --output_path /path/to/output \
    --world_size {W} \
    --quant_mode {w8a8/w8a8c8/...}
```

输出为 `rank_0/` ~ `rank_N/` 目录结构，每个 rank 只包含该卡需要的权重切片。

> 注意：offline 预切的权重与 parallel_config 绑定。改了配置必须重新转换。

参考实现：`cann-recipes-infer/models/deepseek_r1/utils/convert_model.py`、`cann-recipes-infer/models/deepseek_r1/utils/weight_convert.sh`。独立部署可复制此对到模型目录改造。

### 完成标志

- [ ] 权重加载方式已确定（online split / offline convert）
- [ ] 若 online split：`load_weights()` 和 `weight_loader()` 已实现
- [ ] 若 offline convert：转换脚本已编写或复用，输出目录结构正确

---

## 验证

### 配置校验

框架的 `InferenceConfig._validate()` 在 `from_dict` 时自动校验 `world_size` 与各 `*_tp_size` 的整除性。下面是常见的额外语义校验，可放在模型自身的初始化或独立校验函数里（独立部署模式 Runner 不走 InferenceConfig，需自行调用此校验）：

```python
assert world_size % attn_tp_size == 0
assert world_size % moe_tp_size == 0
assert world_size % embed_tp_size == 0
assert world_size % lmhead_tp_size == 0
assert num_attention_heads % attn_tp_size == 0
assert num_key_value_heads % attn_tp_size == 0  # GQA
assert num_experts % ep_size == 0               # MoE
assert embed_tp_size >= attn_tp_size
assert embed_tp_size % attn_tp_size == 0
```

### 功能验证

1. 确认推理实际加载的是修改后的代码（检查模型注册表、import 路径、日志中的模块路径等，确保运行时代码路径与修改路径一致）
2. `infer.sh` 中 `YAML` 变量指向目标配置（两模式都用 `YAML`，框架部署的 `function.sh::launch` 与独立部署的 torchrun 启动都从此变量读路径）
3. 执行 `bash infer.sh`
4. 检查 Prefill + Decode 推理成功（无 crash）
5. 检查各 Rank 输出形状一致
6. 如加载权重：检查输出文本合理性

每种配置独立验证，通过后再验下一个。

### 权重加载验证

加载权重验证时，在 YAML 中设置 `with_ckpt: True` + `model_path`，确认各 rank 权重加载无报错。

### 完成标志

- [ ] 配置校验脚本通过
- [ ] 至少一种配置的 Prefill + Decode 验证通过
- [ ] 使用 `enable_profiler: True` 运行一次，生成 profiler 数据供后续策略校准
- [ ] 验证结果已输出

---

## 常见错误

| 错误模式 | 根因 | 预防 |
|---------|------|------|
| 跳过通信组创建直接替换层 | 运行时找不到 group | 框架部署第二步必须先于第三步完成；独立部署 ParallelContext 必须先 build 再注入 modeling |
| 所有模块用同一个 tp_group | 未区分 attn_tp / dense_tp / moe_tp | 「配置参数 → 实施步骤速查」表逐项检查 group 来源 |
| EP 模式下 Prefill/Decode 用同一套代码 | 两阶段的 routing 算子不同 | 参考 `framework_moe_parallel.md` 中 Prefill/Decode 分支 |
| 模块间 TP 度不同但缺少数据重排 | 相邻模块 tensor shape 不匹配 | 速查表"相邻模块 TP 度不同"行提示了此步骤 |
| embed_tp_size < attn_tp_size | 框架约束：embed 输出需能被 attn 消费 | 「## 验证」配置校验脚本检查 |
| 权重加载 shape 不匹配 | 改了 parallel_config 但未重新处理权重 | 「## 权重处理」确认权重处理方式 |

---

## 仓库参考实现索引

| 实现模式 | 参考文件 | 搜索关键词 |
|---------|---------|-----------|
| TP 线性层替换 | `cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py` | `QKVParallelLinear` |
| 通信组取用 | `cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py`、`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` | `comm_manager.get_group` |
| 通信组创建实现 | `cann-recipes-infer/executor/core/config/comm_manager.py` | `CommManager.initialize` |
| MoE EP Prefill | `cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` | `npu_moe_re_routing` |
| MoE EP Decode | `cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` | `npu_moe_distribute_dispatch` |
| Embed/LMHead 并行 | `cann-recipes-infer/models/kimi-k2-thinking/models/modeling_deepseek.py` | `VocabParallelEmbedding` |
| oproj_tp 独立配置 | `cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py` | `oproj_tp` |
| 权重转换脚本 | `cann-recipes-infer/models/deepseek_r1/utils/` | `weight_convert` |
| YAML 多场景配置 | `cann-recipes-infer/models/deepseek_r1/config/` | decode / prefill 分离 |
