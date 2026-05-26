---
name: external-cannbot-ops-pypto-op-perf-tune-tune-incore
description: PyPTO 算子核内性能调优技能。通过分析单 task 的实现指令及 operation，完成核内的性能调优，包括指令级优化、核内流水优化、特殊
  Shape 处理等。当用户需要进行核内性能调优、单 task 耗时分析、指令级优化时使用此技能。触发词：核内性能调优、单 task 优化、指令级优化、核内流水、Operation
  实现优化。
original-name: tune-incore
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# PyPTO 算子核内性能调优

## 概述

核内性能调优通过分析单 task 的实现指令及 operation，完成核内的性能调优。适用于深度性能调优后仍需要进一步优化的场景。

## 前置条件

1. **完成深度性能调优**：泳道图分析和合图调优已完成
2. **精度校验通过**：确保算子计算正确
3. **识别出单 task 瓶颈**：通过泳道图定位到耗时较长的 task

## 调优方向

### 1. 特殊 Shape 处理

#### 1.1 小 Shape 矩阵乘优化

当矩阵 Shape 较特殊时，可以使用 Vector 操作提前处理输入矩阵。

**案例**：左右矩阵 Shape 分别为 (884736, 16) 和 (16, 16) 的矩阵乘

```python
def matmul_kernel(a, b, out):
    # 构造 c：将四个重复的右矩阵在对角线拼成 (64, 64)
    pypto.set_vec_tile_shapes(64, 64)
    d = pypto.full([16, 16], 0.0, pypto.DT_BF16)
    c1 = pypto.concat([b, d, d, d], 1)
    c2 = pypto.concat([d, b, d, d], 1)
    c3 = pypto.concat([d, d, b, d], 1)
    c4 = pypto.concat([d, d, d, b], 1)
    c = pypto.concat([c1, c2, c3, c4], 0)

    # a 变形
    a = pypto.reshape(a, [221184, 64])

    # 矩阵乘
    pypto.set_pass_options(cube_l1_reuse_setting={-1: 9})
    pypto.set_cube_tile_shapes([512, 512], [64, 64], [64, 64], True)
    e = pypto.matmul(a, c, pypto.DT_BF16)
    e = pypto.reshape(e, [884736, 16])
    pypto.assemble(e, [0, 0], out)
```

**效果**：从 500us 优化到 40us

### 2. 增加冗余计算避免冗余依赖

通过增加冗余计算来避免冗余依赖和搬运。

**案例**：GLM MoE Fusion

```python
# 将 e_score_bias_2d 复制 tile_batch 份后进行 cast 操作
# 使每一份的 cast 都和对应 batch 的其他操作进行了合图
# 避免一对多的子图依赖，减少调度开销和搬运

e_score_bias_2d_tile = pypto.tensor([tile_batch, ne], e_score_bias_2d.dtype, "e_score_bias_2d_tile")
for tmp_idx in range(tile_batch):
    pypto.assemble(e_score_bias_2d, [tmp_idx, 0], e_score_bias_2d_tile)
e_score_bias_2d_cast = pypto.cast(e_score_bias_2d_tile, tile_logits_fp32.dtype)
```

### 3. 尾轴长度优化

尽量避免处理尾轴长度较小的 Tensor。

**解决方案**：
- 使用 concat、transpose 或 reshape 等 Operation 来增大尾轴
- 设置较大的 TileShape

### 4. L2 Cache 策略

设置合理的 L2 Cache Mode，对于只访问一次的 Global Memory 数据设置其访问状态为不进入 L2 Cache。

```python
# 设置 L2 Cache 策略
tensor.set_cache_policy(...)
```

### 5. TileOperation 实现检查

当进行上述优化后算子性能仍然较差时，需要考虑 TileOperation 本身实现是否较差。

**排查方法**：
1. 构造单独 Operation 的用例
2. 与 Ascend C 小算子的性能对比
3. 确认性能较差后检查是否使用了更优的指令

## 性能优化建议库

### 建议 1：小 Shape 矩阵乘

**问题**：矩阵 Shape 特殊，性能较差

**解决方案**：
- 使用 Vector 操作提前处理输入矩阵
- 通过 concat/reshape 调整 Shape

**代码示例**：
```python
# 构造标准 Shape 的矩阵
c = pypto.concat([...], ...)
a = pypto.reshape(a, [new_shape])
```

### 建议 2：尾轴过小

**问题**：Operation 输入 Tensor 尾轴较小

**解决方案**：
- 使用 concat 增大尾轴
- 使用 transpose 调整轴顺序
- 使用 reshape 调整 Shape

### 建议 3：冗余依赖

**问题**：一对多的子图依赖，增加调度开销

**解决方案**：
- 增加冗余计算使每个分支独立
- 使用 `sg_set_scope` 合并子图

### 建议 4：L2 Cache 效率低

**问题**：L2 Cache 命中率低

**解决方案**：
- 使用 L2 亲和调度
- 设置合理的 L2 Cache Mode

### 建议 5：Operation 实现效率低

**问题**：TileOperation 本身实现较差

**解决方案**：
- 与 Ascend C 小算子性能对比
- 检查是否使用更优指令
- 考虑使用其他 Operation 组合替代

## 调优流程

```
┌────────────────────────────────────────────┐
│                核内性能调优流程            │
├────────────────────────────────────────────┤
│                                            │
│  1. 定位瓶颈 task                          │
│     └─ 通过泳道图找到耗时最长的 task       │
│                                            │
│  2. 分析 task 特征                         │
│     ├─ 输入输出 Shape                      │
│     ├─ Operation 类型                      │
│     └─ 依赖关系                            │
│                                            │
│  3. 选择优化策略                           │
│     ├─ 特殊 Shape → Vector 预处理          │
│     ├─ 尾轴过小 → concat/transpose/reshape │
│     ├─ 冗余依赖 → 增加冗余计算             │
│     ├─ L2 Cache → Cache 策略优化           │
│     └─ Operation → 检查实现/替代方案       │
│                                            │
│  4. 应用优化                               │
│     └─ 每次只修改一个参数                  │
│                                            │
│  5. 验证                                   │
│     ├─ 重新编译运行                        │
│     ├─ 检查精度                            │
│     └─ 对比性能数据                        │
│                                            │
│  6. 迭代直到达到目标性能                   │
│                                            │
└────────────────────────────────────────────┘
```

## 常见问题

### Q1: 何时需要进行核内性能调优？

A: 当深度性能调优后，泳道图显示某个或某些 task 耗时明显过长，且无法通过 Stitch、TileShape、合图等方式优化时。

### Q2: 如何判断 Operation 实现效率低？

A:
1. 构造单独 Operation 的测试用例
2. 与 Ascend C 小算子性能对比
3. 如果差距明显，说明 Operation 实现可能需要优化

### Q3: 增加冗余计算会影响精度吗？

A: 不会。冗余计算是指增加一些不影响最终结果的计算（如复制数据），目的是优化调度和合图，不会改变计算逻辑。

## 参考资料

- [性能调优文档](../../../../docs/tutorials/debug/performance.md)
- [GLM MoE Fusion 案例](../../../../models/glm_v4_5/glm_moe_fusion.py)
- [MLA Prolog Quant 案例](../../../../models/deepseek_v32_exp/mla_prolog_quant_impl.py)
