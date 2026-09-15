# 独立部署模式：自管 ParallelContext 与启动模板

本文件提供独立部署模式的产物模板：
1. **ParallelContext + build_parallel_context**：替代仓内 CommManager 的轻量数据结构，inline 在 runner 内
2. **Runner 多卡改造点**：dist 初始化、ParallelContext 构造、modeling 注入
3. **modeling 代码改造**：构造签名与属性访问方式
4. **启动脚本与入口**：torchrun + infer.sh + infer.py

替换 `{model_name}`（小写下划线）和 `{ModelName}`（驼峰）后使用。

---

## §0 复用与自管边界（按功能维度）

### 自管（runner 内 inline）

| 功能 | 实现 |
|---|---|
| 进程组初始化 | `dist.init_process_group("hccl")`，读 torchrun 注入的 RANK / WORLD_SIZE / LOCAL_RANK |
| 通信组配置数据结构 | `ParallelContext` dataclass，对外接口对齐 CommManager（`get_group(name)` / `get_group_name(name)` / `get_rank(name)`），含 attn_tp / moe_ep / moe_tp / embed_tp / lmhead_tp / dense_tp / o_proj_tp 各 group + group_name + rank 字段，以及 `runtime_cfg` dict（携带 yaml 的 scheduler / data / custom_params 段供 modeling 读取） |
| 通信组建立工厂 | `build_parallel_context(yaml_cfg)`，内部调 `executor.utils.hccl_utils.init_comm_group` |
| 推理循环 | `Runner.model_generate(prompts, warm_up=False)`：tokenize → prefill → decode loop → 仅 rank 0 输出（migrator 阶段已建立单卡版本，本阶段加 rank 0 过滤） |
| ForwardMetaData 构造 | runner 内 `_build_forward_metadata(...)`：填 is_prefill / kv_len / actual_seq_lengths / attention_mask / slot_mapping / block_table |
| YAML 解析 | `yaml.safe_load`（不走 `InferenceConfig.from_dict`） |
| 启动脚本 | `infer.sh` 用 `torchrun --nproc_per_node=$WORLD_SIZE` 拉起 |
| 入口 | `infer.py`：argparse + 实例化 Runner |

### 复用（直接 `import` 仓内）

| 功能 | import 路径 | 适用场景 |
|---|---|---|
| 通信组建立底层工具 | `from executor.utils.hccl_utils import init_comm_group, get_group_name` | 全部多卡场景 |
| ForwardMetaData 数据结构 | `from executor.utils.forward_metadata import ForwardMetaData` | 全部多卡场景（也可参考字段在 runner 内 inline 一个轻量 dataclass） |
| ParallelLinear（QKV/O/Gate/Up/Down/Embed/LMHead） | `from module.linear import ColumnParallelLinear, RowParallelLinear, QKVParallelLinear, VocabParallelEmbedding` | 全部多卡场景 |
| FusedMoEGMM | `from module.fuse_moe_gmm import FusedMoEGMM` | MoE 模型 |
| 量化 | `from module.quantization import ...` | 量化模型 |
| 权重加载工具 | `from executor.model_loader.weight_utils import default_weight_loader` | 按需 |

torch_npu 原生算子（`npu_moe_distribute_dispatch_v2` / `npu_grouped_matmul` / `npu_fused_infer_attention_score` 等）与部署模式无关，modeling 内直接调用。

### 与框架部署对照

| 项 | 框架部署 | 独立部署 |
|---|---|---|
| 进程组初始化 | ModelWorker 自动 | Runner 内 `dist.init_process_group("hccl")` |
| 通信组管理 | `executor.core.config.comm_manager.CommManager` | `ParallelContext`（runner 内 inline） |
| 通信组接口 | `mgr.get_group(name)` / `get_group_name(name)` / `get_rank(name)` | **同左（接口对齐）** |
| 通信组建立底层 | `executor.utils.hccl_utils.init_comm_group` | **同左** |
| 推理循环 | ExecutionEngine + Scheduler | `Runner.model_generate(prompts, warm_up)`（命名与签名对齐 framework `ModelRunner.model_generate`，独立部署 inline 实现） |
| YAML 解析 | `InferenceConfig.from_dict` | `yaml.safe_load` 直接读 |
| 启动 | `bash infer.sh → launch → 多卡 fork python` | `torchrun --nproc_per_node=N infer.py` |
| ParallelLinear / FusedMoEGMM / 量化 | `from module.* import ...` | **同左** |
| modeling 构造签名 | `(config, infer_config, comm_manager, prefix)` | `(config, parallel_ctx, prefix)` |
| modeling 读取运行时配置 | `infer_config.scheduler_config.batch_size` 点访问 | `parallel_ctx.runtime_cfg["batch_size"]` dict 访问 |
| modeling forward 计算 | 与独立模式一致 | 与框架模式一致 |
| YAML schema | 4 段式 | **同左** |

---

## §1 ParallelContext + build_parallel_context

inline 在 `runner_{model_name}.py` 顶部，约 80 行：

```python
from dataclasses import dataclass, field
from typing import Optional
import torch.distributed as dist

from executor.utils.hccl_utils import init_comm_group


@dataclass
class ParallelContext:
    """轻量并行上下文，对外接口与 CommManager 对齐：
    - 属性访问：ctx.attn_tp_group / ctx.attn_tp_rank
    - 方法访问：ctx.get_group(name) / ctx.get_group_name(name) / ctx.get_rank(name)
    """
    world_size: int
    global_rank: int

    # 各 *_size 字段供 modeling 直接读取（避免误用 world_size 当 attn_tp_size）
    attn_tp_size: int = 1
    moe_ep_size: int = 1
    moe_tp_size: int = 1
    embed_tp_size: int = 1
    lmhead_tp_size: int = 1
    dense_tp_size: int = 1
    o_proj_tp_size: int = 1

    attn_tp_group: Optional[dist.ProcessGroup] = None
    attn_tp_rank: int = 0
    moe_ep_group: Optional[dist.ProcessGroup] = None
    moe_ep_group_name: Optional[str] = None
    moe_ep_rank: int = 0
    moe_tp_group: Optional[dist.ProcessGroup] = None
    moe_tp_rank: int = 0
    embed_tp_group: Optional[dist.ProcessGroup] = None
    embed_tp_rank: int = 0
    lmhead_tp_group: Optional[dist.ProcessGroup] = None
    lmhead_tp_rank: int = 0
    dense_tp_group: Optional[dist.ProcessGroup] = None
    dense_tp_rank: int = 0
    o_proj_tp_group: Optional[dist.ProcessGroup] = None
    o_proj_tp_rank: int = 0

    # modeling 内若读 infer_config.scheduler_config / infer_config.model_config.custom_params 等运行时配置，
    # 这里携带原始 yaml 的对应段，让 modeling 改读 runtime_cfg["batch_size"] 等即可。
    runtime_cfg: dict = field(default_factory=dict)

    def get_group(self, name: str) -> Optional[dist.ProcessGroup]:
        return getattr(self, name, None)

    def get_group_name(self, name: str) -> Optional[str]:
        return getattr(self, f"{name}_name", None)

    def get_rank(self, name: str) -> int:
        return getattr(self, name.replace("_group", "_rank"), 0)


def build_parallel_context(yaml_cfg: dict) -> ParallelContext:
    parallel_cfg = yaml_cfg["parallel_config"]
    ws = parallel_cfg["world_size"]
    gr = dist.get_rank()
    attn_tp = parallel_cfg.get("attn_tp_size", 1)
    moe_tp = parallel_cfg.get("moe_tp_size", 1)
    moe_ep = parallel_cfg.get("moe_ep_size", ws // moe_tp if moe_tp > 0 else 1)
    embed_tp = parallel_cfg.get("embed_tp_size", attn_tp)
    lmhead_tp = parallel_cfg.get("lmhead_tp_size", embed_tp)
    dense_tp = parallel_cfg.get("dense_tp_size", attn_tp)
    o_proj_tp = parallel_cfg.get("o_proj_tp_size", attn_tp)

    runtime_cfg = {
        **yaml_cfg.get("scheduler_config", {}),
        **yaml_cfg.get("data_config", {}),
        **(yaml_cfg.get("model_config", {}).get("custom_params") or {}),
        "exe_mode": yaml_cfg.get("model_config", {}).get("exe_mode", "eager"),
    }

    ctx = ParallelContext(world_size=ws, global_rank=gr,
                          attn_tp_size=attn_tp, moe_ep_size=moe_ep, moe_tp_size=moe_tp,
                          embed_tp_size=embed_tp, lmhead_tp_size=lmhead_tp,
                          dense_tp_size=dense_tp, o_proj_tp_size=o_proj_tp,
                          runtime_cfg=runtime_cfg)

    ctx.attn_tp_group = init_comm_group(gr, ws // attn_tp, ws, 1, "attn_tp_group")
    ctx.attn_tp_rank = dist.get_rank(ctx.attn_tp_group) if ctx.attn_tp_group else 0

    if moe_ep > 1:
        ctx.moe_ep_group, ctx.moe_ep_group_name = init_comm_group(
            gr, ws // moe_ep, ws, ws // moe_ep, "moe_ep_group", return_name=True)
        ctx.moe_ep_rank = dist.get_rank(ctx.moe_ep_group)

    if moe_tp > 1 and moe_tp != attn_tp:
        ctx.moe_tp_group = init_comm_group(gr, ws // moe_tp, ws, 1, "moe_tp_group")
        ctx.moe_tp_rank = dist.get_rank(ctx.moe_tp_group)
    elif moe_tp == attn_tp:
        ctx.moe_tp_group, ctx.moe_tp_rank = ctx.attn_tp_group, ctx.attn_tp_rank

    # rank list 一致时复用 attn_tp_group（与仓内 CommManager 的 canonicalization 一致）
    if embed_tp == attn_tp:
        ctx.embed_tp_group, ctx.embed_tp_rank = ctx.attn_tp_group, ctx.attn_tp_rank
    else:
        ctx.embed_tp_group = init_comm_group(gr, ws // embed_tp, ws, 1, "embed_tp_group")
        ctx.embed_tp_rank = dist.get_rank(ctx.embed_tp_group) if ctx.embed_tp_group else 0

    if lmhead_tp == embed_tp:
        ctx.lmhead_tp_group, ctx.lmhead_tp_rank = ctx.embed_tp_group, ctx.embed_tp_rank
    else:
        ctx.lmhead_tp_group = init_comm_group(gr, ws // lmhead_tp, ws, 1, "lmhead_tp_group")
        ctx.lmhead_tp_rank = dist.get_rank(ctx.lmhead_tp_group) if ctx.lmhead_tp_group else 0

    if dense_tp != attn_tp:
        ctx.dense_tp_group = init_comm_group(gr, ws // dense_tp, ws, 1, "dense_tp_group")
        ctx.dense_tp_rank = dist.get_rank(ctx.dense_tp_group) if ctx.dense_tp_group else 0
    else:
        ctx.dense_tp_group, ctx.dense_tp_rank = ctx.attn_tp_group, ctx.attn_tp_rank

    if o_proj_tp != attn_tp:
        ctx.o_proj_tp_group = init_comm_group(gr, ws // o_proj_tp, ws, 1, "o_proj_tp_group")
        ctx.o_proj_tp_rank = dist.get_rank(ctx.o_proj_tp_group) if ctx.o_proj_tp_group else 0
    else:
        ctx.o_proj_tp_group, ctx.o_proj_tp_rank = ctx.attn_tp_group, ctx.attn_tp_rank

    return ctx
```

> 注意 `dataclass` 默认值含 `field(default_factory=dict)`，记得 `from dataclasses import dataclass, field`。

---

## §2 Runner 多卡改造点

migrator 阶段产出的 Runner 是 `world_size=1` 单卡的。本阶段加 dist 初始化 + ParallelContext 构造 + modeling 注入：

```python
import os
import yaml
import torch
import torch_npu
import torch.distributed as dist
from transformers import AutoTokenizer

from models.modeling_{model_name} import {ModelName}ForCausalLM
from models.configuration_{model_name} import {ModelName}Config


class {ModelName}Runner:
    def __init__(self, yaml_file_path):
        with open(yaml_file_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.model_cfg = cfg["model_config"]

        # torchrun 注入：RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

        self.parallel_ctx = build_parallel_context(cfg)

        self.config = {ModelName}Config.from_pretrained(self.model_cfg["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg["model_path"], trust_remote_code=True)

        device = f"npu:{local_rank}"
        self.model = {ModelName}ForCausalLM(
            config=self.config,
            parallel_ctx=self.parallel_ctx,
            prefix="",
        ).to(device, dtype=torch.bfloat16)
        self.model.eval()

        self._load_weights(self.model_cfg["model_path"])
        self._init_kvcache()

    def _load_weights(self, model_path):
        # 复用 ParallelLinear.weight_loader / FusedMoEGMM.weight_loader 自动按 rank 切
        # 实现可参考仓内 cann-recipes-infer/executor/model_loader/weight_utils.py
        ...

    def _build_forward_metadata(self, batch_input_ids, kv_len, is_prefill):
        # 框架部署模式由 ExecutionEngine 在每步推理前构造 ForwardMetaData。
        # 独立部署 Runner 必须自管，至少填好以下字段：
        #   is_prefill / kv_len / actual_seq_lengths_q / actual_seq_lengths_kv
        #   attention_mask（prefill 用 causal mask；decode 必须为 None）
        #   slot_mapping / block_table（paged 模式才需要，由 kvcache skill 改造引入）
        # 参考字段定义：cann-recipes-infer/executor/utils/forward_metadata.py::ForwardMetaData
        # 简单做法：在 runner 内 inline 一个轻量 dataclass 覆盖必填字段，避免引入 executor.core 依赖
        ...

    @torch.no_grad()
    def model_generate(self, prompts, warm_up=False):
        # tokenize batch → prefill（计时 log）→ decode loop（每步计时 log）→ detokenize（仅 rank 0 输出）
        # warmup 短路计时与文本输出；日志格式 `{model_name} inference time cost of {stage} is X ms` 与仓库对齐
        # migrator 阶段已建立单卡逻辑，本阶段在 detokenize 处加 `if int(os.environ.get("RANK", 0)) == 0:` 过滤
        ...
```

> `ForwardMetaData` 在仓内位于 `cann-recipes-infer/executor/utils/forward_metadata.py`（不属 `cann-recipes-infer/executor/core/`），独立部署可直接 `from executor.utils.forward_metadata import ForwardMetaData` 复用，或参考其字段定义在 runner 内 inline 一个轻量 dataclass。两种做法都不破坏边界。

---

## §3 modeling 代码改造

构造签名从 migrator 阶段的 `(config, prefix)` 改为 `(config, parallel_ctx, prefix)`：

```python
from module.linear import QKVParallelLinear, RowParallelLinear, ColumnParallelLinear


class {ModelName}Attention(nn.Module):
    def __init__(self, config, parallel_ctx, layer_idx, prefix=""):
        super().__init__()
        self.attn_tp_size = parallel_ctx.attn_tp_size
        attn_tp_rank = parallel_ctx.get_rank("attn_tp_group")

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=config.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            tp_size=self.attn_tp_size,
            tp_rank=attn_tp_rank,
        )
        self.o_proj = RowParallelLinear(
            config.hidden_size, config.hidden_size,
            tp_size=self.attn_tp_size, tp_rank=attn_tp_rank,
        )
```

ParallelLinear 替换示例、Embedding / LMHead 并行、模块间数据重排参考 `framework_code_examples.md`。

### MoE EP 接入

算子链（`npu_moe_init_routing_v2` / `npu_moe_re_routing` / `npu_moe_distribute_dispatch_v2` / `npu_moe_distribute_combine_v2` / `npu_grouped_matmul` / `npu_moe_finalize_routing`）与框架部署 100% 一致，参考 `framework_moe_parallel.md`。差异仅在通信组与 rank 取用方式：

```python
# MoE 模块 __init__ 内（独立部署版 set_mc2_kwargs 的对照）
self.moe_ep_size = parallel_ctx.moe_ep_size
self.moe_tp_size = parallel_ctx.moe_tp_size
self.moe_ep_group = parallel_ctx.get_group("moe_ep_group")
moe_ep_group_name = parallel_ctx.get_group_name("moe_ep_group")  # NPU dispatch/combine 算子要求

global_rank = parallel_ctx.global_rank
self.dispatch_kwargs = {
    "group_ep": moe_ep_group_name,
    "ep_world_size": self.moe_ep_size,
    "ep_rank_id": global_rank // self.moe_tp_size,
    "group_tp": moe_ep_group_name,
    "tp_world_size": self.moe_tp_size,
    "tp_rank_id": global_rank % self.moe_tp_size,
    "moe_expert_num": config.n_routed_experts,
    "shared_expert_rank_num": getattr(config, "shared_expert_rank_num", 0),
    "global_bs": 0,
    "expert_shard_type": 0,
    "x_active_mask": None,
    "scales": None,         # BF16；W8A8 时填 smooth_scale
    "quant_mode": 0,        # BF16；W8A8 时填 2
}
self.combine_kwargs = {k: v for k, v in self.dispatch_kwargs.items()
                       if k not in ("scales", "quant_mode")}
```

MoE 专家计算复用 `from module.fuse_moe_gmm import FusedMoEGMM`，`tp_size` / `tp_rank` / `ep_size` / `ep_rank` 全部从 `parallel_ctx` 取。HCCL buffer 大 batch 时通过 `infer.sh` export `HCCL_BUFFSIZE` 调高（如 512），`init_comm_group` 默认读环境变量。

---

## §4 启动脚本与入口

`infer.sh`：

```bash
#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(realpath "${SCRIPT_DIR}/../..")
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}:${PYTHONPATH}"

# 用户按本机情况 source CANN 环境
# source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

YAML="${SCRIPT_DIR}/config/{model_name}.yaml"
WORLD_SIZE=$(python3 -c "import yaml; print(yaml.safe_load(open('${YAML}'))['parallel_config']['world_size'])")

export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

torchrun --nproc_per_node=${WORLD_SIZE} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-29500} \
    "${SCRIPT_DIR}/infer.py" --yaml_file_path="${YAML}"
```

> 多机：`torchrun --nnodes=N --node_rank=$NODE_RANK --master_addr=<主节点 IP> ...`，由用户在外层环境注入。HCCL_BUFFSIZE 大 batch / 大 hidden_size 可调高（如 512）。
>
> `MASTER_PORT=29500` 是 PyTorch torchrun 默认 rendezvous 端口，与框架部署的 `MASTER_PORT=6038/6138` / `HCCL_IF_BASE_PORT=23456` 端口体系独立；混部场景仅需保证不与同机其他 rendezvous 服务冲突。

`infer.py`：

```python
import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner_{model_name} import {ModelName}Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None,
                        help="可选：单 prompt 覆盖 yaml data_config.prompts（调试用）")
    args = parser.parse_args()

    with open(args.yaml_file_path) as f:
        cfg = yaml.safe_load(f)

    prompts = [args.prompt] if args.prompt else cfg["data_config"]["prompts"]

    runner = {ModelName}Runner(args.yaml_file_path)
    runner.model_generate(prompts, warm_up=True)
    runner.model_generate(prompts, warm_up=False)


if __name__ == "__main__":
    main()
```

---

## §5 覆盖能力

| 模型架构 | 是否支持 | 备注 |
|---|---|---|
| 标准 LLM（GQA / MHA）+ TP | ✅ | 复用 `QKVParallelLinear` |
| MLA + 模块差异化 TP | ✅ | 复用 `ColumnParallelLinear` / `RowParallelLinear` 组合实现 MLA 投影；micro_batch / SP+TP 等高级特性需复用 modeling 内部 helper（如 deepseek_r1 的 `gather_prefill_outputs`、`forward_lm_head`），这些 helper 不依赖 `cann-recipes-infer/executor/core/`，可正常迁移 |
| MoE EP / TP（含 Shared Expert） | ✅ | 复用 `FusedMoEGMM` + torch_npu dispatch/combine 算子 |
| 量化（W8A8 / W4A16 / W8A8C8） | ✅ | 复用 `module.quantization` |
