---
name: external-gitcode-ascend-triton-operator-code-review
description: 静态检视 Triton 算子代码质量（Host+Device 侧），面向 Ascend NPU。发现潜在 bug、API 误用和性能隐患。仅关注静态代码分析。关键词：code
  review、代码检视、静态分析。
original-name: triton-operator-code-review
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# Triton 算子静态代码检视（Ascend NPU）

## 检视原则

- **Ascend 特有约束优先**：聚焦硬件差异，不检查 Triton 通用知识
- **Mask 零容错**：Ascend 对越界访问零容忍

### 严重性分级

| 级别 | 含义 | 典型问题 |
|------|------|---------|
| **P0** | 必定崩溃或错误结果 | Mask 遗漏、核类型错配、Atomic 循环死锁 |
| **P1** | 高概率精度/功能问题 | 归约未升精度、Softmax 未减 max |
| **P2** | 性能/可维护性 | 冗余访存、BLOCK 未对齐 |

## 参考资源加载

| 阶段 | 加载 | 不要加载 |
|------|------|---------|
| Phase 1: Host 侧 | [`ascend-triton-api-constraints.md`](references/ascend-triton-api-constraints.md) | dtype-matrix |
| Phase 2: Device 侧 | [`ascend-api-dtype-matrix.md`](references/ascend-api-dtype-matrix.md) | test-patterns |
| 逐项核对 | [`code-review-checklist.md`](references/code-review-checklist.md) | — |
| 参考官方实现 | [`ascend-test-patterns.md`](references/ascend-test-patterns.md) | — |
| 确认 API 签名/参数 | [`triton-api-reference.md`](../triton-operator-shared/references/triton-api-reference.md) | — |

**加载触发**：需要确认某个 tl/libdevice API 的签名、参数或可用数据类型时，完整阅读 `triton-api-reference.md`。

## 检视工作流

### Phase 1: Host 侧

| 检查项 | 识别方式 | 级别 |
|--------|---------|------|
| 硬编码核数 | `grid = (20,)` 等字面量 | P0 |
| 核类型错配 | 含 `tl.dot` 的 kernel 用了 `num_vectorcore` | P0 |
| 矩阵乘法退化为逐元素 | 无 `tl.dot`，用 Vector Core 逐元素乘加实现 matmul/GEMV | P0 |
| BLOCK_SIZE 非 `tl.constexpr` | 声明检查 | P1 |
| 矩阵运算 BLOCK 非 16 倍数 | 数值检查 | P2 |

**核类型**：含 `tl.dot` → AI Core (`num_aicore`)；逐元素/归约/激活 → Vector Core (`num_vectorcore`)。注意：矩阵乘法类算子（含 GEMV）必须用 `tl.dot` + AI Core，Cube Core 吞吐远高于 Vector Core。

### Phase 2: Device 侧

**Mask 完整性（P0）**：所有 `tl.load`/`tl.store` 必须有 `mask=` 或使用 `make_block_ptr`。

**数据类型（P0-P1）**：`tl.dot` 输入仅支持 int8/fp16/fp32/bf16；`dot_scaled` 有条件支持（bf16/fp16 lhs/rhs, int8 scale, fp32 out）；`permute`/`trans` 不支持 int64。

**精度处理（P1）**：FP16/BF16 归约前 `.to(tl.float32)`；Softmax 必须减 max。

**归约类算子单 pass 检查（P1）**：GroupNorm/LayerNorm/RMSNorm 等算子，检查是否对同一数据多次 load（先算统计量，再 load 做归一化）。当 D ≤ UB 时应单 pass：一次 load 后 `tl.sum(x, 1)` 在 UB 内完成全部计算。双 pass 可慢 5x。

**控制流（P0）**：Triton IR 使用 MLIR 结构化控制流（`scf.for`/`scf.while`），不支持 early exit。
- ❌ `for/while` 循环内 `return` — 编译错误："Cannot have return statements inside while or for"（包括子函数中的 return）
- ❌ `for` 循环内 `break` — 编译错误："unsupported AST node type: Break"

**Tensor 索引（P0）**：Triton tensor 不支持 Python 风格 `[]` 下标操作。
- ❌ `tensor[i] = val` — AssertionError
- ❌ `val = tensor[i]` — AssertionError
- ❌ `tensor[i:j]` 切片 — 编译错误

**代码模式**：
- ❌ `for ... : tl.atomic_cas/or/xor/and/xchg(...)` — 可能死锁（P0）
- ❌ 多核 kernel 中使用 `tl.atomic_add` 返回值（P0）
- ❌ kernel 内 `import numpy`（P0）
- ⚠️ `tensor[i].item()` 在 Host 热路径 — 触发 CPU-NPU 同步（P2）
- ⚠️ 归约类算子存在多个 `for ... : tl.load(x_ptr + ...)` 循环 — 双 pass 反模式（P1）
- ⚠️ Post-dot 操作未用 `tl.parallel(bind_sub_block=True)` — `tl.dot` 后有逐元素操作但无并行分片（P2）
- ⚠️ 整数类型转换溢出 — `tl.cast` 到 int8/int16 未指定 `overflow_mode`（P1）

### Phase 3: 性能隐患（P2）

- 同一 ptr 多次 `tl.load` → 冗余 GM 访问
- `tl.arange(0, N) * stride`（stride > 1）→ 非连续访存
- `pid` 直接映射 block 无循环 → 负载不均
- kernel 内对辅助张量（cos/sin 等）使用 broadcast stride 计算偏移（`b * stride0 + n * stride1 + ...`）→ 建议改为 host 侧 expand+contiguous，统一 `row * D + col` 偏移
- 归约类算子 weight/bias 未预展开，kernel 内用 `weight_ptr + ch_idx` gather → 建议 host 侧展开为连续布局

## 反模式清单（NEVER）

### Host
- ❌ 硬编码核数 / 矩阵乘法用 `num_vectorcore` / BLOCK_SIZE 非 `tl.constexpr`
- ❌ 矩阵乘法（含 GEMV）用 Vector Core 逐元素乘加实现——必须用 `tl.dot` + AI Core

### Device
- ❌ `tl.load`/`tl.store` 无 mask / `tl.dot` 输入 int32/int16/int64 / `dot_scaled`
- ❌ `atomic_or/xor/and/xchg/cas` 在 for 循环内 / kernel 内第三方库
- ❌ FP16/BF16 归约不升 FP32 / Softmax 不减 max
- ❌ 整数截断 cast 未指定 `overflow_mode`
- ❌ `for/while` 循环内 `return` / `break` — Triton 结构化控制流不支持 early exit（含子函数中的 return）
- ❌ `tensor[i]` 索引操作（读取/赋值/切片）— 用 `tl.where` / `tl.gather` / `tl.extract_slice` 替代

## 输出

按 [`code-review-report-template.md`](references/code-review-report-template.md) 格式输出报告。
