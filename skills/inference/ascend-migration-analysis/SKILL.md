---
name: ascend-migration-analysis
description: 通用 PyTorch 项目 Ascend NPU 迁移可行性分析。系统化扫描代码库中的 CUDA/GPU 依赖，按 7 大域分类评估（设备层、注意力机制、自定义算子、分布式通信、精度策略、第三方依赖、编译加速），并基于 Wan2.2 实际迁移经验提供逐项替代方案。适用于评估任何 DiT/Transformer 类模型（视频生成、LLM、多模态等）在昇腾 NPU 上的运行可行性与迁移工作量估算。
keywords:
  - ascend
  - npu
  - migration
  - cuda
  - analysis
  - compatibility
  - assessment
  - 迁移分析
  - 兼容性评估
  - 昇腾
---

# Ascend NPU Migration Analysis Skill

## Purpose

对任意 PyTorch 项目进行系统化的 Ascend NPU 迁移可行性分析。扫描代码库中的 CUDA/NVIDIA 依赖，按域分类，逐项给出替代方案，并估算迁移工作量。

## When to Use

- 用户询问某个项目/模型能否在 Ascend NPU 上运行
- 用户需要评估 PyTorch 项目迁移到昇腾 NPU 的工作量
- 用户需要对 CUDA→Ascend 迁移做全面的技术尽职调查
- 用户遇到 NPU 兼容性问题，需要排查哪些 CUDA 依赖未替换

## Analysis Workflow

对目标项目执行以下 7 步分析，每步对应一个迁移域。详细方法和替代方案见各域的 reference 文档。

### Step 1: Third-Party Dependency Audit (Domain 1)

Read `references/01-dependency-audit.md`.

扫描内容：
- `requirements.txt` / `setup.py` / `pyproject.toml` 中的 NVIDIA 专属依赖
- `flash-attn`, `triton`, `xformers`, `nvidia-ml-py`, `cuda-python`, `cupy` 等
- 间接依赖中的 CUDA 硬编码

产出：**依赖兼容性矩阵**（每个依赖标注：兼容 / 需替换 / 不支持 / 不涉及）

### Step 2: Device Layer Scan (Domain 2)

Read `references/02-device-layer.md`.

扫描内容（使用 grep/rg）：
- `torch.cuda.*` 全系列 API
- 硬编码 `"cuda"` 设备字符串
- `backend="nccl"` 分布式后端
- `amp.autocast(device_type='cuda')` 混合精度
- `torch.compile` / `@torch.compile` 使用
- `torch.cuda.Stream` / `torch.cuda.Event`
- `init_device_mesh("cuda")`

产出：**设备层替换清单**（文件:行号 → 替代方案）

### Step 3: Attention Mechanism Analysis (Domain 3)

Read `references/03-attention-mechanism.md`.

扫描内容：
- `flash_attn` (FA2) 导入与调用
- `flash_attn_interface` (FA3) 导入与调用
- `xformers` 注意力调用
- `F.scaled_dot_product_attention` 调用
- 手动实现的 attention (Q×K^T softmax 等)
- 注意力的 tensor layout (BSHD / BNSD 等)

产出：**注意力后端替换方案**（每种 attention 类型 → 对应 NPU 替代）

### Step 4: Custom Operator / Kernel Analysis (Domain 4)

Read `references/04-custom-operators.md`.

扫描内容：
- `.cu` / `.cuh` / `.cpp` 自定义 CUDA 内核文件
- `@triton.jit` 装饰的 Triton kernel
- `torch.utils.cpp_extension.CUDAExtension` 构建
- `torch.autograd.Function` 子类中的自定义前向/反向
- `csrc/` / `ops/` / `kernels/` 目录

产出：**自定义算子迁移方案**（每个 kernel 的功能描述 + 替代路径）

### Step 5: Distributed Communication Analysis (Domain 5)

Read `references/05-distributed.md`.

扫描内容：
- `dist.init_process_group` 后端选择
- `dist.all_to_all` / `dist.all_gather` / `dist.broadcast` 等
- `init_device_mesh` / DeviceMesh 使用
- Ulysses / Ring / Sequence Parallel 模式
- `torch.distributed.P2POp` / `batch_isend_irecv`

产出：**分布式通信适配方案**

### Step 6: Precision & Numerics Analysis (Domain 6)

Read `references/06-precision-strategy.md`.

扫描内容：
- `.float()` / `.double()` 类型转换
- `torch.bfloat16` / `torch.float16` dtype 使用
- `torch.complex128` / `torch.float64` 高精度使用
- `autocast` 上下文中的 dtype 设置
- 数值敏感操作中的 float32 保护 (RMSNorm, LayerNorm, AdaLN)
- `dtype == torch.float32` assert 语句

产出：**精度策略调整清单**

### Step 7: Task-Phase Dependency Matrix (Domain 7)

Read `references/07-task-phase-matrix.md`.

将前 6 步的发现按**任务执行阶段**拆分，识别：
- 哪些依赖在每个阶段都需要（P0）
- 哪些依赖仅在特定阶段/模式需要（P1）
- 哪些依赖在当前任务中不涉及（可跳过）

产出：**按阶段的依赖矩阵** + **最小迁移集** + **完整迁移工作量估算**

## Output Format

分析完成后，输出结构化报告：

```markdown
# {Project} Ascend NPU Migration Assessment

## Executive Summary
- Overall feasibility: [可行 / 需适配 / 困难 / 不可行]
- Estimated effort: [X 周]
- Blockers: [list]

## Dependency Matrix (per task phase)
| Dependency | Phase A | Phase B | Replacement | Effort |
|------------|---------|---------|-------------|--------|

## Detailed Findings (per domain)
### Domain 1: Third-Party Dependencies
### Domain 2: Device Layer
### Domain 3: Attention Mechanism
### Domain 4: Custom Operators
### Domain 5: Distributed Communication
### Domain 6: Precision Strategy

## Migration Roadmap
## Risk Matrix
```

## Quick Reference: Common Replacements

| CUDA Pattern | Ascend Replacement | Confidence |
|-------------|-------------------|------------|
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` or `transfer_to_npu` | High |
| `backend="nccl"` | `backend="hccl"` | High |
| `autocast('cuda')` | `autocast('npu')` | High |
| `device="cuda"` | `device="npu"` | High |
| `flash_attn.flash_attn_func` | `mindiesd.attention_forward(op_type="ascend_laser_attention")` | High |
| `flash_attn.flash_attn_varlen_func` | `mindiesd.attention_forward(op_type="fused_attn_score")` | Medium |
| `xformers.ops.memory_efficient_attention` | `mindiesd.attention_forward(op_type="fused_attn_score")` | High |
| `F.scaled_dot_product_attention` | torch_npu 原生支持 | High |
| `@triton.jit` kernels | 需逐个分析，可能用 mindiesd/RainFusion/NPU原生算子替代 | Low-Medium |
| `torch.compile` | 禁用或使用 torch_npu backend | Medium |
| RMSNorm `.float()` | `torch_npu.npu_rms_norm()` | High |
| LayerNorm `.float()` | 移除 `.float()`，NPU 原生 BF16 | High |
| RoPE complex128 | 降为 complex64 或用 `mindiesd.rotary_position_embedding()` | High |
| `init_device_mesh("cuda")` | `"npu"` | Medium |
| `.cu` / CUDAExtension | 需重写为 Ascend 算子 | Low |

## Notes

- 本 skill 的替代方案知识库来自 Wan2.2 CUDA→Ascend 实际迁移经验
- DiT/Transformer 类架构的迁移模式高度可复用
- 分析结果需结合实际 CANN/torch_npu 版本验证
- 部分替代方案依赖 mindiesd 库，需确认环境中是否可用
