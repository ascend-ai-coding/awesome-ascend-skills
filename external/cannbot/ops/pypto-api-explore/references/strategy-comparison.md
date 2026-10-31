# 多组合策略对比与推荐

## 1. GELU

| 方案 | 实现 | 精度 | 推荐 |
|------|------|------|------|
| **A: tanh 近似** | `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))` | PASS | ✅ 通用 |
| B: erf 精确 | `x * 0.5 * (1 + erf(x / sqrt(2)))` | 精确 | erf 硬件加速状态不确定；BF16/FP16 推理下 tanh 近似误差远小于量化误差 |

## 2. RoPE

| 方案 | 实现 | 推荐 |
|------|------|------|
| **A: 标准组合** | `cos/sin→mul→neg→concat→mul→add`，PASS | ✅ 通用 |
| B: fused RMSNorm+RoPE | RMSNorm 与 RoPE 融合为单 kernel | 两算子紧邻时（如 Qwen3 1.7B），融合减少 GM 搬运 |

## 3. Attention

| 方案 | 实现 | 精度 | 推荐 |
|------|------|------|------|
| A: 标准组合 | `matmul→scale→softmax→matmul` 整行整块 | **FAIL** | ❌ |
| **B: online softmax + 分块** | `amax→sub→exp→sum→div` + `is_loop_begin/end` 状态管理 | PASS | ✅ decode：O(1) 内存、数值稳定，参考 gym gemma4 `gqa_decode_attn` |
| **C: flash attention kernel** | `sparse_flash_attention` 融合实现 | 未验证 | ✅ 生产（已有成熟 kernel 时） |

## 4. MoE Routing

| 方案 | 实现 | 推荐 |
|------|------|------|
| **A: sigmoid gating** | `exp→add→div`(sigmoid 展开) `→ topk → gather → sum → div` | ✅ gym 覆盖的 MoE 模型（GLM、LLaDA2、MiniMax）主流选择 |
| B: softmax gating | `amax→sub→exp→sum→div` `→ topk → gather` | 模型架构要求 softmax 归一化时 |

## 5. Linear (with bias)

| 方案 | 实现 | 推荐 |
|------|------|------|
| A: matmul + add | `pypto.add(pypto.matmul(x, w, dtype, b_trans=True), bias)` | 简单场景 |
| **B: extend_params 融合 bias** | `pypto.matmul(x, w, dtype, extend_params={"bias_tensor": bias})` | ✅ 少一次 GM 读写，1 次 kernel |

## 6. Embedding

| 方案 | 实现 | 推荐 |
|------|------|------|
| **A: gather** | `pypto.gather(weight, dim=0, indices=input_ids)` | ✅ 支持多维 indices，[B, S] input_ids 无需 flatten |
| B: index_select | `pypto.index_select(weight, dim=0, index=input_ids)` | 1D index 场景 |

## 7. repeat_kv

| 方案 | 实现 | 推荐 |
|------|------|------|
| **A: unsqueeze + expand_clone + reshape** | `unsqueeze(kv, 2) → expand_clone([B,N_kv,G,S,D]) → reshape([B,N_q,S,D])` | ✅ 默认；decode（S=1）内存开销可忽略 |
| B: head loop 复用 | attention head loop 中每 G 个 Q head 共用同一 KV head | 长序列 prefill：零额外内存 |

## 8. mean

`sum(dim) / N`：一步 reduce + 一步 elementwise，无需备选方案。BF16 输入在 `sum` 前 `cast` 到 FP32（`pypto.sum` 的 FP32 硬约束）。
