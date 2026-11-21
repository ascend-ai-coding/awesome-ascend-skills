# 常见问题

按通用性分两组：
- **通用问题**：所有模型迁移都可能遇到
- **条件问题**：仅在特定架构（MoE / 多 DP）下触发

---

## 通用问题

### 问题 1: `cann-recipes-infer/models/<name>/models/__init__.py` 遮蔽顶层 `models` namespace

- **现象**：`bash infer.sh` 启动报 `ModuleNotFoundError: No module named 'models.gpt_oss'`（或其他同级模型）
- **根因**：从 `cann-recipes-infer/models/<your_model>/` 启动时，Python `sys.path[0]` 指向该目录。其下的 `cann-recipes-infer/models/__init__.py`（哪怕空文件）会把这个子目录变成 regular package，遮蔽仓库根的 `cann-recipes-infer/models/` namespace package
- **修法**：删除 `cann-recipes-infer/models/<your_model>/models/__init__.py`。已注册参考模型（qwen3_moe / gpt_oss / deepseek_r1 等）下都没有这个文件
- **检查**：`ls cann-recipes-infer/models/<your_model>/models/__init__.py` 应该报 file not found

### 问题 2: Decode 分支没把 `attention_mask` 置 None

- **现象**：decode 第一步 `aclnnFusedInferAttentionScoreV3` 报 `batch size mismatch`，类似 `batchSize[2048] of atten_mask != batchSize[1] of query`
- **根因**：`forward_metadata.attention_mask` 是 prefill 时构造的 `[max_seq, max_seq]` causal mask；decode 步 query 长度为 1，这个 mask shape 完全对不上
- **修法**：Attention.forward 里按 `is_prefill` 分支：
  ```python
  if forward_metadata.is_prefill:
      atten_mask = forward_metadata.attention_mask
  else:
      atten_mask = None  # decode 只靠 actual_seq_lengths_kv + sparse_mode
  ```

### 问题 3: `infer.sh` 没 export `PYTHONPATH`

- **现象**：8 卡启动后 rank 0 的 log 里 `ModuleNotFoundError: No module named 'models.gpt_oss'`
- **根因**：从模型目录启动时 `sys.path[0]` 是 `cann-recipes-infer/models/<your>/`，不是仓库根，导致 `cann-recipes-infer/executor/core/support_models.py` import 兄弟模型失败
- **修法**：`infer.sh` 在 source set_env / function 之后加：
  ```bash
  export PYTHONPATH="${SCRIPT_PATH}/../..:${PYTHONPATH}"
  ```
- **判断方式**：和问题 1 现象一样，但问题 1 是 `__init__.py` 遮蔽（删 init 修），问题 3 是 PYTHONPATH 缺失（加 export 修）。两条都查

---

## 条件问题（仅适用部分架构）

### 问题 4（仅 MoE / 量化模型）：缺 `process_weights_after_loading`

- **现象**：prefill 时 `aclnnGroupedMatmul` 报 `Dim 1 mismatch`（如 2816 vs 1408），或量化模型推理报 scale 维度错
- **根因**：MoE 的 `gate_up_proj` / `down_proj` 权重加载后需要做 transpose + NZ 打包；量化模型权重需要按 rank 切分 scale。框架的 `ModelWorker._process_weights_after_loading` 会调 `model.process_weights_after_loading()`，模型类没实现就跳过了
- **修法**：ForCausalLM 顶层实现：
  ```python
  def process_weights_after_loading(self):
      for _, module in self.named_modules():
          qm = getattr(module, "quant_method", None)
          if qm is not None and hasattr(qm, "process_weights_after_loading"):
              qm.process_weights_after_loading(module)
  ```
- **不适用**：纯密集模型（Qwen 标准 GQA）通常用不上这个钩子，但建议都加上以防量化场景

### 问题 5（仅多 DP MoE）：用全局 `batch_size` 而非 `batch_size_per_dp_rank`

- **现象**：ge_graph 模式编译期报 `aclnnMoeInitRouting: row_idx shape mismatch`；eager 能跑但 ge_graph 挂
- **根因**：MoE 模块预分配 `row_idx_decode` / `expert_count` 等 per-rank buffer 时用了 `infer_config.scheduler_config.batch_size`（全局 batch=8），但实际每 rank 只处理 `batch_size // attn_dp_size` 条
- **修法**：用框架已经做好除法的 `infer_config.scheduler_config.batch_size_per_dp_rank`：
  ```python
  # 错：bs = infer_config.scheduler_config.batch_size
  bs = infer_config.scheduler_config.batch_size_per_dp_rank
  self.row_idx_decode = torch.zeros(bs, ..., device=...)
  ```
- **判断方式**：eager 跑通但切 ge_graph 挂 → 多半是 per-rank buffer sizing 错；尤其 MoE local-expert 路径的 `npu_moe_init_routing` / `npu_moe_compute_expert_tokens` 报 shape 错时优先怀疑

---

未列出的运行时错误（HCCL timeout / aicore timeout / OOM 等）转 model-infer-runtime-debug skill。
