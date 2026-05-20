# Block Size Scaling 优化模式

## 概述

在所有指令级优化策略（constexpr 静态化、int32 向量加法、load 重排、向量比较）应用完毕后，通过倍增 `BLOCK_SIZE` 来探索更大分块带来的性能收益。更大的 BLOCK_SIZE 可以减少 kernel launch 次数、提高单次计算的数据吞吐量，但超过硬件 UB 容量后会导致 verify 失败。

**本策略是 latency-optimizer 的最终优化步骤，必须在其他策略之后执行。**

## 触发条件

**当代码满足以下全部条件时触发：**

1. Kernel 中存在**单维度** `BLOCK_SIZE` 参数（如 `BLOCK_SIZE`、`XBLOCK`）
2. 该参数在 forward() 中以字面量或变量赋值（可静态解析其数值）
3. Kernel **不含多维分块参数**（如同时存在 `BLOCK_SIZE_M`、`BLOCK_SIZE_N` 则跳过）

**不触发的场景：**

- 多维 BLOCK_SIZE（BLOCK_M / BLOCK_N / BLOCK_K 同时存在）
- BLOCK_SIZE 由 `triton.autotune` 的 configs 管理（autotune 自身已覆盖搜索）
- 无法从代码中静态解析出当前 BLOCK_SIZE 数值

## 分析输出

当触发条件满足时，latency-optimizer 在常规优化代码之外，额外输出 **Block Size Scaling 计划**，包含以下信息：

### 1. 参数识别

```
block_size_param_name: "BLOCK_SIZE"       # kernel 签名中的参数名
current_value: 1024                        # 当前值
forward_assignment_location: "第 42 行"    # forward() 中赋值位置
grid_dependency: "triton.cdiv(n, BLOCK_SIZE)"  # grid 计算是否依赖此参数
```

### 2. 候选值序列

从当前值开始，每次 ×2：

```
candidates: [2048, 4096, 8192, 16384, 32768, 65536]
```

上限为 65536（Triton Ascend 单块元素数硬限制：< 65536）。

### 3. 代码改写规则

对每个候选值 `C`，需要修改的位置：

| 位置 | 改写方式 |
|------|---------|
| forward() 中 `BLOCK_SIZE = <原值>` | 替换为 `BLOCK_SIZE = C` |
| forward() 中 kernel 调用的 `BLOCK_SIZE=BLOCK_SIZE` | 无需改动（跟随变量） |
| forward() 中 grid 计算 `triton.cdiv(n, BLOCK_SIZE)` | 无需改动（跟随变量） |
| kernel 签名 `BLOCK_SIZE: tl.constexpr` | 无需改动 |

**关键约束**：只改 forward() 中的赋值，不改 kernel 内部逻辑。Kernel 已参数化，改调用参数即可。

## 调用方行为

latency-optimizer 输出 scaling 计划后，由主流程（Phase 4.6）逐个候选值执行：

```
for candidate in candidates:
    改写代码（按上述规则）
    verify → 失败则 break（到达硬件上限）
    benchmark → 记录 latency
全部完成后，选 latency 最低的版本作为最优
```

## 示例

### 原始代码（latency-optimizer 常规优化后）

```python
class ModelNew(torch.nn.Module):
    def forward(self, x):
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        output = torch.empty_like(x)
        my_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
        return output
```

### Scaling 计划输出

```
block_size_param_name: BLOCK_SIZE
current_value: 1024
candidates: [2048, 4096, 8192, 16384, 32768, 65536]
改写位置: forward() 第 4 行 "BLOCK_SIZE = 1024"
```

### 候选值 2048 的改写结果

```python
class ModelNew(torch.nn.Module):
    def forward(self, x):
        n = x.numel()
        BLOCK_SIZE = 2048  # 仅此行改变
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        output = torch.empty_like(x)
        my_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
        return output
```

## 关键点

1. **最终步骤**：必须在其他优化策略全部应用后执行，不可提前
2. **单维度限定**：仅处理单维度 BLOCK_SIZE，多维场景跳过
3. **只改参数不改逻辑**：kernel 内部代码不变，只改 forward() 中的赋值
4. **上限 65536**：Triton Ascend 硬限制，候选值不得超过此值
5. **实验驱动**：不做理论估算，verify 失败即到达硬件上限，停止搜索

## 性能收益

增大 BLOCK_SIZE 可以：
- 减少 grid 数量，降低 kernel launch 和调度开销
- 增大单次 tl.load/tl.store 的数据量，提高访存效率
- 更充分利用 UB 容量（Ascend910B: 192KB）

但过大会导致：
- 超出 UB 容量，verify 失败
- Grid 过小（< AI Core 数量），并行度不足，性能下降
