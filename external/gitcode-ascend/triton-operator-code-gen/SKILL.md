---
name: external-gitcode-ascend-triton-operator-code-gen
description: 根据 Ascend NPU 算子设计文档（或直接需求）生成 Triton kernel 代码。当用户需要实现 Triton 算子、将设计文档转为可执行代码时使用。核心产出：kernel
  代码 + 基本正确性测试。关键词：Triton kernel、算子实现、代码生成、code generation。
original-name: triton-operator-code-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# Triton 算子代码生成

## 核心原则

**计算逻辑 → Tiling 策略 → 代码实现**。顺序不可颠倒。

## Triton API 路径优先级

1. `tl.tanh` / `tl.erf` / `tl.sqrt` — 推荐，性能最优
2. `tl.math.tanh` / `tl.math.erf` — 备选
3. `triton.language.extra.ascend.libdevice` / `triton.language.extra.libdevice` — 以上找不到时

## 自由度：中高（计算逻辑不可错，实现方式灵活）

## 参考资源加载

| 阶段 | 必须加载 | 不要加载 |
|------|----------|----------|
| 设计 Tiling | [`hardware-architecture.md`](references/hardware-architecture.md) | templates.md |
| 生成 Kernel | [`templates.md`](references/templates.md) | hardware-architecture.md |
| 查找 API 签名 | [`triton-api-reference.md`](../triton-operator-shared/references/triton-api-reference.md) | — |

**MANDATORY**：对应阶段前完整阅读上述文件，不设行数限制。
**加载触发**：需要确认某个 tl/libdevice API 的签名或参数时，完整阅读 `triton-api-reference.md`。

## 工作流

### 1. 理解需求 → 确认计算逻辑

提取数学公式、输入输出规格、约束条件。用伪代码描述计算过程，**必须与用户确认**。

### 2. 实现 Tiling 策略

**输入依赖**：若已有设计文档（design skill 产出），按其 Tiling 策略实现；若无，则在此设计。

**核间切分两原则**：
1. `grid = 物理核数`（`get_npu_aicore_num()` 或 `get_npu_vectorcore_num()`）
2. 核内循环处理多任务，每个核自己计算要处理的数据（负载均衡）
3. 超过物理核数的grid，超出部分循环串行下发，只会增加开销，无法增加并行度

```python
core_num = get_npu_aicore_num()
grid = (core_num,)
# kernel 内：
pid = tl.program_id(0)
num_core = tl.num_programs(0)
blocks_per_core = tl.cdiv(total_blocks, num_core)
for block_idx in range(pid * blocks_per_core, min(...)):
    ...
```

**UB 空间**：安全 BLOCK_SIZE = `(196608 - 32) / (缓冲区数 × dtype大小) × 0.8`

### 3. 生成 Kernel 代码

按算子类型选择模板（详见 templates.md）：

| 算子类型 | 核心类型 | 模板 |
|----------|----------|------|
| 归约类 | vector core | 模板 1 |
| GEMM/注意力 | AI core | 模板 2,6 |
| 激活/损失/索引/MoE/后处理 | vector core | 模板 3-5,7-8 |
| 卷积 | AI core | 模板 9 |

### 4. 生成基本正确性测试

生成一个**基本正确性测试**（单 shape × 单 dtype 的 smoke test），确保 kernel 可编译运行且结果正确。**全面的精度验证**由 precision-eval skill 负责。

## 反模式清单（NEVER）

- ❌ 不确认计算逻辑就写代码
- ❌ 忽略 UB 大小（192KB）
- ❌ 归约不升精度 FP32 / 使用 int64
- ❌ grid > 65535 / grid ≠ 物理核数
- ❌ kernel 中用第三方库 / 逐元素计算
- ❌ **矩阵乘法退化为 Vector Core 逐元素乘加**——Cube Core 矩阵吞吐远高于 Vector Core（数十倍）。即使 GEMV（N=1）也必须用 `tl.dot` + pad B 到 BLOCK_N=16，禁止用 Vector Core 逐行算内积
- ❌ 在 NPU 使用 GPU 专用参数（num_stages/num_warps/num_ctas）
- ❌ 用 PyTorch 而非 Triton 实现算子
- ❌ 不测试算子正确性 / 不在 NPU 上测试
- ❌ **归约类算子用双 pass 模式**（多次 load 同一数据分别算统计量和归一化）——应单 pass：一次 load 后 `tl.sum(x, 1)` 在 UB 内完成统计和归一化，见模板 1
- ❌ **矩阵乘法大矩阵不对角线调度**——大矩阵（BLOCK_THRESHOLD ≥ 4）用顺序调度会导致 L2 缓存颠簸，应使用对角线调度，见模板 2
- ❌ **`for/while` 循环内 `return` / `break`** — Triton IR 结构化控制流不支持 early exit，编译必报错。替代：`while` + 布尔标志，或 `tl.where` mask
- ❌ **`tensor[i]` 索引操作**（读取/赋值/切片）— Triton tensor 不支持 Python 风格下标。替代：`tl.where` 赋值、`tl.gather` 读取、`tl.extract_slice` 切片

## 常见陷阱

| 陷阱 | 症状 | 解决 |
|------|------|------|
| 计算逻辑错 | 输出不符预期 | 伪代码描述并确认 |
| UB 溢出 | "ub overflow" | 减小 BLOCK_SIZE |
| coreDim 超限 | "coreDim > UINT16_MAX" | 增大 BLOCK_SIZE 或设 `TRITON_ALL_BLOCKS_PARALLEL=1` |
| 精度损失 | FP16 不准确 | 归约前升 FP32 |
| 索引不够 | D-cache 报错 | 超大 shape 用 int64 |
| 归约类算子慢 5x+ | aiv_scalar > 80% | 单 pass：一次 load + `tl.sum(x, 1)` UB 内全计算 |
| 大矩阵 L2 缓存颠簸 | GEMM 吞吐低于预期 | 对角线调度（`task_m_idx = block_idx % NUM_BLOCKS_M`），见模板 2 |

## 严禁 PyTorch 替代
必须使用 Triton kernel 完成核心计算，严禁以下行为:
- 禁止用 torch.matmul / torch.mm / torch.bmm / torch.einsum 替代 matmul kernel
- 禁止用 F.conv1d / F.conv2d / F.conv3d / F.conv_transpose* 替代 conv kernel
- 禁止用 nn.Conv1d / nn.Conv2d / nn.Conv3d / nn.LayerNorm / nn.BatchNorm* / nn.GroupNorm / nn.InstanceNorm 等高阶接口
- 禁止用 F.layer_norm / F.batch_norm / F.group_norm / F.instance_norm 替代 norm kernel
- 禁止用 torch.cumsum / torch.cumprod / torch.flip 等直接替代 scan kernel
- 禁止写了 @triton.jit kernel 但 forward 不调用它
唯一允许的 torch 调用: tensor 分配(empty/zeros/ones)、形状操作(view/reshape/permute/contiguous)、数据搬运(to/npu)