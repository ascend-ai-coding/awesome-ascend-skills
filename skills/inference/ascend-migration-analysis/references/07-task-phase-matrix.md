# Domain 7: Task-Phase Dependency Matrix

## Overview

将前 6 个域的分析结果按任务执行阶段拆分，识别最小迁移集，估算分阶段工作量。

## Method

### Step 1: Identify Task Execution Phases

阅读项目的入口脚本（如 `run_demo_xxx.py`、`generate.py`），识别推理/训练流程中的不同阶段。

常见模式：
- **多阶段推理**: 基础生成 → 蒸馏加速 → 超分精炼（如视频生成的 480p → 720p）
- **多任务共享模型**: Text-to-Video / Image-to-Video / Video-Continuation 共享 DiT
- **可选优化**: LoRA / 蒸馏 / 量化 / 稀疏注意力 是可选功能
- **训练 vs 推理**: 训练中的 backward kernel 在推理时不需要

### Step 2: Trace Dependency per Phase

对每个阶段，追踪完整的代码执行路径：

```
入口脚本
  → Pipeline.generate_xxx()
    → Model.forward()
      → Block.forward() × N
        → Attention (哪个后端？FA2/FA3/xformers/BSA/SDPA?)
        → Norm (RMSNorm/LayerNorm, 是否有 .float()?)
        → FFN
      → FinalLayer
    → VAE.decode()
    → Scheduler.step()
```

对每个阶段记录：
- 使用了哪个 Attention 后端？
- 是否触发 Triton/CUDA kernel？
- 是否使用 torch.compile？
- 是否使用分布式通信？

### Step 3: Build Dependency Matrix

```markdown
| Dependency | Phase A (基础) | Phase B (优化) | Phase C (精炼) |
|------------|:-:|:-:|:-:|
| torch.cuda.* API | **需要** | **需要** | **需要** |
| NCCL → HCCL | **需要** | **需要** | **需要** |
| autocast('cuda') | **需要** | **需要** | **需要** |
| Flash Attention | **需要** | **需要** | **需要** |
| BSA/Triton kernel | 不需要 | 不需要 | **需要** |
| torch.compile | 可选 | 可选 | 可选 |
| LoRA | 不需要 | **需要** | **需要** |
```

### Step 4: Determine Minimal Migration Set

找出完成最小可用功能（如仅 480p 输出）所需的依赖子集。

```
Minimal Set (Phase A only):
  - Device layer adaptation (Domain 2)
  - Flash Attention replacement (Domain 3)
  - Basic operator replacement (Domain 6)
  - No BSA, no Triton, no compile

Incremental Set (Phase B):
  - LoRA loading (usually pure PyTorch, compatible)

Full Set (Phase C):
  - BSA → RainFusion (Domain 4)
  - All custom kernels replaced
```

## Effort Estimation Template

```markdown
### Migration Effort Estimation

#### Minimal Migration (Phase A only)

| Task | Domain | Effort |
|------|--------|--------|
| Device layer replacement | 2 | X days |
| Flash Attention replacement | 3 | X days |
| Operator precision fixes | 6 | X days |
| **Subtotal** | | **X days** |

#### Full Migration (All phases)

| Task | Domain | Effort |
|------|--------|--------|
| Minimal migration above | | X days |
| Custom kernel replacement | 4 | X days |
| Distributed verification | 5 | X days |
| End-to-end verification | - | X days |
| **Total** | | **X weeks** |
```

## Common Phase Patterns

### Video Generation (DiT-based)

| Phase | Description | Typical Dependencies |
|-------|-------------|---------------------|
| Stage 1 | Base resolution generation | FA2/FA3, device APIs, autocast |
| Stage 2 | Distilled/accelerated generation | Same as Stage 1 + LoRA |
| Stage 3 | Super-resolution refinement | Same + BSA/Sparse Attention |

### LLM Inference

| Phase | Description | Typical Dependencies |
|-------|-------------|---------------------|
| Prefill | Prompt encoding + attention | FA2 varlen, RoPE |
| Decode | Token-by-token generation | FA2 with KV cache |
| Quantization | INT8/INT4 inference | npu_fused_infer_attention_score |

### Training

| Phase | Description | Typical Dependencies |
|-------|-------------|---------------------|
| Forward | Standard inference path | All attention backends |
| Backward | Gradient computation | Custom backward kernels (if any) |
| Optimizer | Weight update | AMP, gradient scaling |

## Decision Framework

```
仅需要基础功能？
  → Yes → 仅迁移 Minimal Set
  → No ↓

需要高性能/精炼？
  → Yes → 迁移 Full Set (含 custom kernel)
  → No → Minimal + selective

需要训练？
  → Yes → Full Set + backward kernel 迁移
  → No → Minimal or Full (取决于推理需求)

需要多卡？
  → Yes → 增加分布式通信适配
  → No → 单卡设备层替换即可
```
