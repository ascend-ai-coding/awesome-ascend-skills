---
name: external-gitcode-ascend-triton-operator-code-gen
description: 根据 Ascend NPU 算子设计文档（或直接需求）生成 Triton kernel 代码。当用户需要实现 Triton 算子、将设计文档转为可执行代码时使用。核心产出：kernel
  代码 + 基本正确性测试。关键词：Triton kernel、算子实现、代码生成、code generation。
original-name: triton-operator-code-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
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

**MANDATORY**：对应阶段前完整阅读上述文件，不设行数限制。

## 工作流

### 1. 理解需求 → 确认计算逻辑

提取数学公式、输入输出规格、约束条件。用伪代码描述计算过程，**必须与用户确认**。

### 2. 实现 Tiling 策略

**输入依赖**：若已有设计文档（design skill 产出），按其 Tiling 策略实现；若无，则在此设计。

**核间切分两原则**：
1. `grid = 物理核数`（`get_npu_aicore_num()` 或 `get_npu_vectorcore_num()`）
2. 核内循环处理多任务，每个核自己计算要处理的数据（负载均衡）

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
- ❌ 在 NPU 使用 GPU 专用参数（num_stages/num_warps/num_ctas）
- ❌ 用 PyTorch 而非 Triton 实现算子
- ❌ 不测试算子正确性 / 不在 NPU 上测试

## 常见陷阱

| 陷阱 | 症状 | 解决 |
|------|------|------|
| 计算逻辑错 | 输出不符预期 | 伪代码描述并确认 |
| UB 溢出 | "ub overflow" | 减小 BLOCK_SIZE |
| coreDim 超限 | "coreDim > UINT16_MAX" | 增大 BLOCK_SIZE 或设 `TRITON_ALL_BLOCKS_PARALLEL=1` |
| 精度损失 | FP16 不准确 | 归约前升 FP32 |
| 索引不够 | D-cache 报错 | 超大 shape 用 int64 |
