---
name: external-gitcode-ascend-triton-operator-code-gen
description: 根据算子设计文档生成 Ascend NPU 的 Triton kernel 代码。当用户需要实现 Triton 算子 kernel、将需求文档转化为可执行代码时使用。核心能力：(1)解析需求文档确认计算逻辑
  (2)设计 tiling 分块策略 (3)生成高性能 kernel 代码 (4)生成测试代码验证正确性。
original-name: triton-operator-code-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-18'
synced-commit: 9f4c6c19a042f03239a07ac2f3196fb590d0a114
license: UNKNOWN
---

# Triton 算子代码生成

## 核心原则

**计算逻辑 → Tiling 策略 → 代码实现**

这个顺序绝对不可颠倒。错误的计算逻辑会导致完全错误的结果，错误的 tiling 策略会导致性能问题或内存溢出。

## 参考资源加载路由表

**MANDATORY - 按需加载**：根据任务阶段加载对应的参考文档

| 阶段 | 必须加载 | 不要加载 |
|------|----------|----------|
| 理解需求文档 | 无 | 所有 references |
| 确认计算逻辑 | 无 | 所有 references |
| 设计 Tiling 策略 | [`hardware-architecture.md`](references/hardware-architecture.md) | `templates.md` |
| 生成 Kernel 代码 | [`templates.md`](references/templates.md) | `hardware-architecture.md` |
| 生成测试代码 | 无 | 所有 references |

## 工作流程

### 阶段 1：理解需求文档

提取：数学公式、输入输出规格、约束条件、Tiling 策略

### 阶段 2：确认计算逻辑

1. 用伪代码描述计算过程
2. 确认数据依赖关系
3. 确认精度处理（归约操作必须使用 FP32）

**输出**：计算逻辑确认（必须与用户确认）

### 阶段 3：设计 Tiling 策略

**MANDATORY - READ ENTIRE FILE**：在设计 Tiling 策略之前，你必须完整阅读 [`hardware-architecture.md`](references/hardware-architecture.md)。

**绝对不要设置任何行数限制。**

**核间切分原则（必须遵循）**：

1. **grid = 物理核数**：保证利用每个核，避免资源浪费
2. **核内均衡负载**：每个核自己计算要处理哪些数据，实现负载均衡

```python
core_num = get_npu_aicore_num()  # 或 get_npu_vectorcore_num()

grid = (core_num,)  # 原则1：grid必须等于物理核数

@triton.jit
def xxx_fwd(
    ......
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)

    total_blocks = num_block_m * num_block_n

    # 原则2：核内循环处理多任务，每个核自己计算要处理的数据
    for block_idx in range(pid, total_blocks, num_core):
        pid_m = block_idx // num_block_n
        pid_n = block_idx % num_block_n
```

**UB空间计算**：
```
UB 总大小: 192KB (A2/A3)
安全 BLOCK_SIZE = (196608 - 32) / (缓冲区数量 × 数据类型大小) × 0.8
```

### 阶段 4：生成 Kernel 代码

**MANDATORY - READ ENTIRE FILE**：在生成代码之前，你必须完整阅读 [`templates.md`](references/templates.md)。

**绝对不要设置任何行数限制。**

根据算子类型选择对应模板灵活参考：

| 算子类型 | 特征 | 核心类型 | 模板 |
|----------|------|----------|------|
| 归约类 | sum/max/min 归约 | vector core | 模板 1 |
| GEMM | tl.dot() 矩阵乘法 | AI core | 模板 2 |
| 激活函数 | 逐元素计算 | vector core | 模板 3 |
| 损失函数 | softmax + reduction | vector core | 模板 4 |
| 索引变换 | 索引计算、条件分支 | vector core | 模板 5 |
| 注意力 | QK^T + SV 多阶段 | AI core | 模板 6 |
| MoE | 门控机制 | vector core | 模板 7 |
| 后处理 | 简单数据变换 | vector core | 模板 8 |
| 卷积 | 状态更新、滑动窗口 | AI core | 模板 9 |

### 阶段 5：生成测试代码

## 反模式清单（NEVER）

- ❌ 不确认计算逻辑就开始写代码
- ❌ 忽略 UB 大小限制（192KB）
- ❌ 归约操作不使用 FP32 精度
- ❌ 使用 int64 数据类型（性能极差）
- ❌ grid 大小超过 65535
- ❌ 在 kernel 中使用第三方库
- ❌ 一个元素一个元素地计算
- ❌ 过度复杂的优化（如对角线分核）
- ❌ 调用第三方函数获取核数
- ❌ 混淆 Vector Core 和 Cube Core 的用途
- ❌ 使用pytorch而不用triton实现算子
- ❌ 不测试算子的正确性
- ❌ 不在npu上测试算子
- ❌ 不确保测试标杆的准确性
- ❌ grid大小不等于物理核数（违反核间切分原则1）
- ❌ 核间负载不均衡（违反核间切分原则2）

## 常见陷阱

| 陷阱 | 症状 | 解决方案 |
|------|------|----------|
| 计算逻辑错误 | 输出结果与预期不符 | 用伪代码描述计算过程，与用户确认 |
| UB 溢出 | 运行时报错 "ub overflow" | 计算缓冲区总大小，减小 BLOCK_SIZE |
| coreDim 超限 | 运行时报错 "coreDim can't be greater than UINT16_MAX" | 增大 BLOCK_SIZE 或设置 `TRITON_ALL_BLOCKS_PARALLEL=1` |
| 精度损失 | FP16 输入时结果不准确 | 归约操作前升精度到 FP32 |
| 索引长度不够 | D-cache报错 | 在超大shape下int32索引长度不足，需要换成int64 |

## 检查清单

### 计算逻辑
- [ ] 数学公式理解正确
- [ ] 伪代码与公式一致
- [ ] 边界条件处理正确
- [ ] 数据类型转换正确

### Tiling 策略
- [ ] grid = 物理核数（原则1）
- [ ] 核内循环处理多任务，负载均衡（原则2）
- [ ] UB 空间计算正确
- [ ] BLOCK_SIZE 选择合理

### Kernel 实现
- [ ] 核数获取函数正确调用
- [ ] 指针计算正确
- [ ] mask 处理正确
- [ ] 精度处理正确（归约用 FP32）
- [ ] 无第三方库依赖

### 测试代码
- [ ] PyTorch 参考实现正确
- [ ] 测试用例覆盖多种形状
- [ ] 测试用例覆盖多种数据类型
- [ ] 精度容差设置合理
- [ ] 执行测试代码，确保算子正确运行
