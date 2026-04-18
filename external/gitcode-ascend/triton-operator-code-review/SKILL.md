---
name: external-gitcode-ascend-triton-operator-code-review
description: 静态检视 Triton 算子代码质量（Host 侧 + Device 侧），面向 Ascend NPU。当用户需要通过阅读代码发现潜在 bug、API
  误用和性能隐患时使用。核心能力：(1)Ascend API 约束合规检查 (2)Mask 完整性验证 (3)精度处理审查 (4)代码模式识别。注意：本 Skill
  仅关注静态代码分析，编译期和运行时问题由其他 Skill 处理。
original-name: triton-operator-code-review
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-18'
synced-commit: 9f4c6c19a042f03239a07ac2f3196fb590d0a114
license: UNKNOWN
---

# Triton 算子静态代码检视（Ascend NPU）

## 检视原则

- **Ascend 特有约束优先**：Agent 已懂 Triton 通用知识，聚焦 Ascend 硬件差异
- **仅做静态分析**：只通过阅读代码发现问题，不涉及编译期/运行时
- **Mask 零容错**：Ascend 对越界访问零容忍，这是最致命的差异点

### 严重性分级

检视发现的问题按以下级别分类，报告时必须标注：

| 级别 | 含义 | 典型问题 |
|------|------|---------|
| **P0 致命** | 必定导致错误结果或崩溃 | Mask 遗漏、核类型错配、Atomic 循环死锁 |
| **P1 严重** | 高概率导致精度或功能问题 | 归约未升精度、dot 无累加器、Softmax 未减 max |
| **P2 建议** | 影响性能或可维护性 | 冗余访存、非连续访存、BLOCK 未对齐 |

## 检视工作流

### Phase 1: Host 侧检视

**MANDATORY - READ ENTIRE FILE**：在检视 Host 侧前，完整阅读 [`ascend-triton-api-constraints.md`](references/ascend-triton-api-constraints.md)。

#### 1.1 Grid 配置（P0）

| 检查项 | 如何在代码中识别 |
|--------|-----------------|
| 硬编码核数 | `grid = (20,)` 或 `grid = (24,)` 等字面量 |
| 核类型错配 | 含 `tl.dot` 的 kernel 使用了 `num_vectorcore` |
| Grid 维度 | 使用 2D/3D Grid 但无必要（推荐 1D） |

**核类型速查**：

| 算子类型 | 应该用 | 获取方式 |
|----------|--------|----------|
| 含 `tl.dot` | AI Core | `get_device_properties(device)["num_aicore"]` |
| 逐元素/归约/激活 | Vector Core | `get_device_properties(device)["num_vectorcore"]` |

```python
# ❌ P0：硬编码 + 核类型错配
core_num = driver.active.utils.get_device_properties(device)["num_vectorcore"]
grid = (20,)  # 但 kernel 中使用了 tl.dot

# ✅ 正确
core_num = driver.active.utils.get_device_properties(device)["num_aicore"]
grid = (min(core_num, triton.cdiv(n_elements, BLOCK_SIZE)),)
```

#### 1.2 Block Size 配置（P1-P2）

| 检查项 | 级别 |
|--------|------|
| BLOCK_SIZE 未声明为 `tl.constexpr` | P1 |
| 矩阵运算 BLOCK_M/N/K 非 16 倍数 | P2（Cube 单元粒度） |
| BLOCK_K 未对齐 `kalign = 32 // dtype_bytes` | P2 |

### Phase 2: Device 侧检视

#### 2.1 Mask 完整性（P0）

**Ascend 对越界访问零容错**。搜索所有 `tl.load`/`tl.store`，确认每个都满足以下之一：
- 有 `mask=` 参数（`tl.load` 还需 `other=`）
- 使用 `make_block_ptr`（自动处理边界）

```python
# ❌ P0：缺少 mask
x = tl.load(x_ptr + offsets)

# ✅ 显式 mask
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# ✅ make_block_ptr（自动处理）
block_ptr = tl.make_block_ptr(base=ptr, shape=(M, N), ...)
x = tl.load(block_ptr)
```

#### 2.2 数据类型合规（P0-P1）

**MANDATORY - READ ENTIRE FILE**：首次检视时，完整阅读 [`ascend-api-dtype-matrix.md`](references/ascend-api-dtype-matrix.md)。

| 代码模式 | 问题 | 级别 |
|----------|------|------|
| `tl.dot(a_int32, b_int32)` | 输入仅支持 int8/fp16/fp32/bf16 | P0 |
| `dot_scaled(...)` | 不支持 | P0 |
| `permute`/`trans` 用 int64 | 不支持 | P0 |
| `tl.dot(a, b)` 无显式 `out_dtype` | 浮点默认 fp32、int8 仅 int32 可选，显式指定非必要 | P2 |
| `permute`/`trans` 3D (2,1,0) | 兼容性风险 | P1 |

#### 2.3 精度处理（P1）

```python
# ❌ P1：FP16 直接归约 → 应先 .to(tl.float32)
sum_x = tl.sum(x_fp16, axis=-1)

# ❌ P1：Softmax 未减最大值 → 数值不稳定
exp_x = tl.exp(x)

# ✅ 正确精度模式
x_fp32 = x_fp16.to(tl.float32)
sum_x = tl.sum(x_fp32, axis=-1)

# out_dtype 浮点默认 fp32、int8 仅 int32 可选，显式指定非必要
acc = tl.dot(a, b, acc)

max_x = tl.max(x, axis=-1, keepdims=True)
exp_x = tl.exp(x - max_x)
```

#### 2.4 代码模式（P0-P2）

| 代码模式 | 问题 | 级别 |
|----------|------|------|
| `for ... : tl.atomic_cas/or/xor/and/xchg(...)` | 不支持在 loop 中，可能死锁 | P0 |
| 多核 kernel 中 `tl.atomic_add` 返回值被使用 | 不支持多核 add + 保存中间结果 | P0 |
| `import numpy` 在 kernel 中 | kernel 内不可调用第三方库 | P0 |
| `for i in range(N):` 在 kernel 中（loop 次数少且固定） | 可考虑 `tl.static_range`，但 loop 数较大时收益不明显甚至劣化，不应盲目替换 | P2 |
| `tensor[i].item()` 在 Host 热路径 | 触发 CPU-NPU 同步 | P2 |

### Phase 3: 性能隐患检视（P2）

| 代码特征 | 隐患 |
|----------|------|
| 同一 ptr 多次 `tl.load` | 冗余 GM 访问 |
| `tl.arange(0, N) * stride`（stride > 1） | 非连续访存 |
| `pid` 直接映射到 block，无核间循环分配 | 负载不均衡 |

## 反模式清单（NEVER）

### Host 侧
- ❌ 硬编码核数 `grid = (20,)` — P0
- ❌ 矩阵乘法用 `num_vectorcore`（含 `tl.dot` 应用 AI Core）— P0
- ❌ BLOCK_SIZE 不是 `tl.constexpr` — P1

### Device 侧
- ❌ `tl.load`/`tl.store` 无 `mask=`（也无 `make_block_ptr`）— P0
- ❌ `tl.dot` 输入用 int32/int16/int64 — P0
- ❌ `dot_scaled`（不支持）— P0
- ❌ `atomic_or/xor/and/xchg/cas` 在 `for` 循环体内 — P0
- ❌ kernel 内调用第三方库 — P0
- ❌ FP16/BF16 归约不升精度到 FP32 — P1
- ⚠️ `tl.dot` 无显式 `out_dtype`（浮点默认 fp32、int8 仅 int32 可选，非必要）— P2
- ❌ Softmax 不减最大值 — P1
- ⚠️ `for i in range(N):` 可考虑 `tl.static_range`，但仅 loop 次数少且固定时有收益；loop 数较大时可能劣化，不强制要求 — P2

## 检视报告

检视完成后，按 [`code-review-report-template.md`](references/code-review-report-template.md) 输出报告。

## 参考资源

### 按需加载

| 工作流阶段 | 加载文档 | 不要加载 |
|-----------|---------|---------|
| Phase 1: Host 侧 | [`ascend-triton-api-constraints.md`](references/ascend-triton-api-constraints.md) | dtype-matrix, test-patterns |
| Phase 2: Device 侧 | [`ascend-api-dtype-matrix.md`](references/ascend-api-dtype-matrix.md) | test-patterns |
| 逐项核对 | [`code-review-checklist.md`](references/code-review-checklist.md) | test-patterns, dtype-matrix |
| 需要参考官方实现 | [`ascend-test-patterns.md`](references/ascend-test-patterns.md) | — |

**加载原则**：只加载当前检视阶段需要的文档，不要一次加载所有文档。

### 官方文档
- [Triton-Ascend 官方仓库](https://gitcode.com/Ascend/triton-ascend)
- [Triton 官方文档](https://triton-lang.org/main/index.html)
