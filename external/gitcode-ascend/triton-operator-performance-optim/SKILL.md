---
name: external-gitcode-ascend-triton-operator-performance-optim
description: 优化 Ascend NPU 亲和的 Triton 算子性能。解决 UB 溢出、提高 Cube 利用率、Tiling 策略设计。关键词：性能优化、performance
  optimization、tiling、UB。
original-name: triton-operator-performance-optim
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---

# Triton 算子性能优化（Ascend NPU）

## 底线（不可突破）

1. **精度**：优化后 rtol=1e-3, atol=1e-3 对齐 PyTorch-NPU。不通过则回退。
2. **泛化性**：支持原有所有输入形状和 dtype，不能 hardcode 特定尺寸。

**优先级**：正确性 > 泛化性 > 性能。

## 优化工作流

### Phase 0: 算法审视

优化前先审视算法本身。低效算法再优化也有先天不足。

### Phase 1: 分层评估

1. **快速筛选**：`time.time()` 测端到端（覆盖小/中/大），达标则完成
2. **精确诊断**：不达标时用 `msprof` 测 kernel 侧耗时，定位真正瓶颈

### Phase 2: 瓶颈优化

| 瓶颈 | 优化重点 |
|------|---------|
| Memory-Bound | 向量化访存、UB 缓存复用、算子融合 |
| Compute-Bound | Cube 适配、Block 尺寸调优 |
| Latency-Bound | 增大并行度、减少同步 |

**基础四板斧**（按顺序）：Block/Grid Size → 连续访存 → UB 复用 → 编译时常量

**加载**：[`optimization-patterns.md`](references/optimization-patterns.md), [`ascend-terminology.md`](references/ascend-terminology.md)

### Phase 3: 硬件特化

- **Cube**：BLOCK_M/N/K 为 16 倍数，累加器 FP32
- **UB**：缓冲区总大小 < 192KB，单值缓冲区 32B 对齐
- **Grid**：1D Grid ≤ 物理核数，核内循环处理多行

**加载**：[`triton-ascend-api.md`](references/triton-ascend-api.md), [`tiling-strategies.md`](references/tiling-strategies.md)

### Phase 4: 高级优化（按需）

算子融合、Double Buffer

### Phase 5: 验证（MANDATORY）

精度 + 泛化性 + 性能 + 端到端回归

## 反模式清单（NEVER）

- ❌ 仅凭单一规模数据做优化决策
- ❌ 端到端不达标时直接优化 kernel（应先 msprof 确认瓶颈）
- ❌ 为性能牺牲精度 / hardcode 破坏泛化性
- ❌ FP16 直接归约 / 非 16 倍数 BLOCK 做矩阵乘
- ❌ BLOCK_SIZE 超 UB（192KB）/ 非连续访存
- ❌ 热路径用 `tensor.item()`（触发 CPU-NPU 同步）
- ❌ **循环内用 if 分支修改变量**（Triton 编译为 masked 操作，灾难性性能下降）
- ❌ **2D Tiling 只算数据 buffer 的 UB**（必须包含 offset/mask/index 数组）
- ❌ **用预计算的 offset tensor 做 2D broadcasting**（触发编译器 addptr 多用户 assertion）
- ❌ **kernel 内用 broadcast stride 访问辅助张量**（cos/sin 等），应改为 host 侧 expand+contiguous。expand 额外内存 < 非连续访存的性能损失

## 检查清单

- [ ] 精度对齐 PyTorch-NPU（rtol=1e-3, atol=1e-3）
- [ ] 非对齐维度和边界通过
- [ ] 性能测试覆盖小/中/大
- [ ] grid ≤ 物理核数，BLOCK_SIZE 为编译时常量
- [ ] 缓冲区 < 192KB，所有 load/store 有 Mask
- [ ] 归约升 FP32，矩阵乘 BLOCK 为 16 倍数
