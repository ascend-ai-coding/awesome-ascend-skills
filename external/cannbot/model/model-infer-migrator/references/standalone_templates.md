# 独立部署模式模板和速查

本文件提供独立部署模式（不接入 cann-recipes-infer 框架）的产物模板：
1. **Runner 类骨架**：自定义 Runner（不继承框架 ModelRunner）
2. **modeling forward 接口**：自定义参数（`past_key_values` / `kv_len` / `attention_mask`）
3. **infer.py 入口**：实例化 Runner（传 yaml 路径），先 warmup 再正式调 `model_generate`
4. **YAML / infer.sh 模板**

替换 `{model_name}`（小写下划线）和 `{ModelName}`（驼峰）后使用。

> **命名与契约对齐**：方法名 `model_generate(prompts, warm_up=False)`、yaml 单一来源、`{model_name} inference time cost of {stage} is X ms` 日志格式均与仓库 framework 部署模式对齐，便于 `scripts/collect_baseline.py` 解析与后期 promote 到 framework 模式时无缝衔接；与 framework 模式的唯一差异是不继承 `executor.model_runner.ModelRunner`，forward loop 在 runner 内自管。

---

## 1. Runner 类骨架（runner_{model_name}.py）

```python
import os
import time
import logging
from pathlib import Path
import yaml
import torch
import torch_npu
from transformers import AutoTokenizer

from models.modeling_{model_name} import {ModelName}ForCausalLM
from models.configuration_{model_name} import {ModelName}Config

logger = logging.getLogger(__name__)


def get_init_attn_mask(mask_length, device):
    """预创建 prefill 用 causal mask（kvcache 阶段改造为 FA 算子时需符合算子约束：dtype bool/int8/uint8, shape 2048x2048）。"""
    return ~torch.tril(
        torch.ones((mask_length, mask_length), dtype=torch.bool, device=device)
    )


class {ModelName}Runner:
    """{ModelName} 独立部署 Runner（不依赖 cann-recipes-infer/executor/core/）"""

    def __init__(self, yaml_file_path):
        with open(yaml_file_path) as f:
            self.runner_settings = yaml.safe_load(f)

        self.model_name = self.runner_settings["model_config"]["model_name"]
        self.model_path = self.runner_settings["model_config"]["model_path"]
        self.max_new_tokens = self.runner_settings["scheduler_config"]["max_new_tokens"]
        self.input_truncated_len = self.runner_settings["data_config"].get("input_truncated_len", 2048)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = f"npu:{local_rank}"
        torch.npu.set_device(local_rank)

        self.config = {ModelName}Config.from_pretrained(self.model_path)

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载 model
        torch.npu.reset_peak_memory_stats()
        torch.npu.synchronize()
        self.model = {ModelName}ForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # 预创建 prefill causal mask
        self.attn_mask = get_init_attn_mask(2048, self.device)

        # 预计算 RoPE cos/sin（根据模型实现）
        # self.model.model.rotary_emb._precompute_cos_sin(...)

        # 可选：torch.compile 编译 decode（graph mode）
        # self._compile_decode()

    def init_static_kvcache(self, batch_size, max_seq_len=2048):
        """静态预分配 KVCache：每层 (k, v) tensor，BSH layout。"""
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        kv_dim = num_kv_heads * head_dim

        cache = []
        for _ in range(num_layers):
            k = torch.zeros(batch_size, max_seq_len, kv_dim,
                            dtype=torch.bfloat16, device=self.device)
            v = torch.zeros(batch_size, max_seq_len, kv_dim,
                            dtype=torch.bfloat16, device=self.device)
            cache.append((k, v))
        return cache

    @torch.no_grad()
    def model_generate(self, prompts, warm_up=False):
        """批量推理：tokenize → prefill → decode loop → detokenize。

        - warm_up=True：仅跑一遍预热（编译图、cache 预热），不输出计时与文本
        - warm_up=False：正式计时，日志格式 `{model_name} inference time cost of {stage} is X ms`
          与仓库 framework 模式对齐，便于 `scripts/collect_baseline.py` 解析
        """
        # tokenize batch
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.input_truncated_len)
        input_ids = inputs.input_ids.to(self.device)
        batch_size, input_seq_len = input_ids.shape

        # 初始化 KVCache
        past_key_values = self.init_static_kvcache(batch_size)
        generated_ids = input_ids.clone()

        # === Prefill 阶段 ===
        kv_len = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)
        position_ids = torch.arange(input_seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        torch.npu.synchronize()
        t0 = time.time()
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            kv_len=kv_len,
            attention_mask=self.attn_mask[:input_seq_len, :input_seq_len],
            is_prefill=True,
        )
        torch.npu.synchronize()
        prefill_ms = (time.time() - t0) * 1000
        if not warm_up:
            logger.info(f"{self.model_name} inference time cost of prefill is {prefill_ms:.2f} ms")

        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # === Decode 阶段 ===
        for step in range(self.max_new_tokens - 1):
            kv_len = torch.tensor([input_seq_len + step] * batch_size, dtype=torch.int32, device=self.device)
            position_ids = kv_len.unsqueeze(-1)

            torch.npu.synchronize()
            t0 = time.time()
            outputs = self.model(
                input_ids=next_token,
                position_ids=position_ids,
                past_key_values=past_key_values,
                kv_len=kv_len,
                attention_mask=None,  # decode 不传 attention_mask
                is_prefill=False,
            )
            torch.npu.synchronize()
            decode_ms = (time.time() - t0) * 1000
            if not warm_up:
                logger.info(f"{self.model_name} inference time cost of decode is {decode_ms:.2f} ms")

            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        # detokenize（仅正式跑输出；多卡场景由 parallel-impl 阶段加 rank 0 过滤）
        if not warm_up:
            for i in range(batch_size):
                output_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                logger.info(f"Request {i}: outputs: {output_text}")
        return generated_ids
```

---

## 2. modeling forward 接口（独立模式）

独立模式下 modeling 类的 forward 接受显式参数（不用 ForwardMetaData）：

```python
class {ModelName}Attention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        kv_len: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ):
        # ... QKV 投影、RoPE 等 ...

        # 写入 KV cache（标准 indexing 即可，scatter_update_ 改造由 kvcache skill 处理）
        if past_key_value is not None and kv_len is not None:
            past_key, past_value = past_key_value
            # 简化示例：用标准 PyTorch indexing 写入
            past_key[:, kv_len, :] = key_states.flatten(-2, -1)
            past_value[:, kv_len, :] = value_states.flatten(-2, -1)
            key_states = past_key[:, :kv_len + 1, :].view(...)
            value_states = past_value[:, :kv_len + 1, :].view(...)

        # attention 计算保留 HF 原版 SDPA / 标准 PyTorch
        # （NPU FA 算子 npu_fused_infer_attention_score 等改造由 kvcache skill 处理）
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask if is_prefill else None,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        return attn_output


class {ModelName}ForCausalLM(nn.Module):
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        kv_len: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ):
        # 主体 forward → lm_head → logits
        ...
```

---

## 3. infer.py 入口模板

```python
# infer.py
import os
import sys
import argparse
import logging
import yaml
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner_{model_name} import {ModelName}Runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="{ModelName} NPU Inference")
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

## 4. infer.sh 模板（独立部署模式）

```bash
#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# CANN 环境：用户按本机情况 source（独立部署不依赖仓库脚本，自包含）
# source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

YAML="${SCRIPT_DIR}/config/{model_name}.yaml"

python3 ${SCRIPT_DIR}/infer.py --yaml_file_path "${YAML}"
```

> migrator 阶段是 `world_size=1` 单卡，不引入 vendoring 的 `cann-recipes-infer/executor/scripts/`；parallel-impl skill 改造为多卡时会引入。所有运行时参数（prompts / max_new_tokens 等）通过 yaml 单一来源传入，CLI 仅保留 `--yaml_file_path` 与可选 `--prompt`（调试用）。

---

## 5. YAML 模板（与框架部署模式同 4 段式）

```yaml
model_name: "{model-key}"
world_size: 1

model_config:
  model_name: "{model-key}"
  model_path: "{absolute_or_relative_weights_path}"
  exe_mode: "eager"
  with_ckpt: True
  enable_profiler: False

data_config:
  dataset: "default"
  input_truncated_len: 4096
  prompts:
    - "What is the capital of France?"

parallel_config:
  world_size: 1
  attn_tp_size: 1

scheduler_config:
  batch_size: 1
  max_new_tokens: 100
  max_prefill_tokens: 4096
```

> 字段语义与框架部署模式相同；独立模式 Runner 自己解析（不依赖 `InferenceConfig.from_dict`）。`data_config.prompts` 是独立部署专属字段（framework 模式走 `dataset` 读外部 fixture，独立模式直接在 yaml 内嵌一行示例 prompt 跑通即可）。
>
> 布尔值统一用 `True` / `False`，不用 `true` / `false`（与仓内主流风格一致）。
>
> 命名维度：参考仓内已注册模型，按"模型_rank_N_拓扑_后端_场景"组合，禁止用非结构性差异的临时描述符（如 `_4k1k`、`_b8`）；新建 yaml 只针对结构性差异（拓扑 / 量化 / prefill/decode / 特性开关组合），运行时参数或 `exe_mode` 切换改字段即可。

---

## 6. README 模板

README 模板见共享文件 `references/readme_template.md`（框架部署与独立部署共用）。独立部署模式：删除 `<!-- 仅框架部署 -->` 标注的"注册"段；按 `<!-- 仅独立部署 -->` 标注在"当前状态"末段追加 "独立 Runner，不接入 cann-recipes-infer/executor/core/ 框架。"。
