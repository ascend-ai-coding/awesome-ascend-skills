# P16 Playbook: Cost-Driven Core Partition

> 本 Playbook 为**强制流程**。采纳 P16 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P16 的核心是**通过计算代价模型（对齐开销、轴权重）替代简单均分，实现多粒度层次化负载均衡**。

## Step 1: 定位关键结构

```bash
grep -n "GetBlockNum|BLOCK_DIM|coreNum|blockIdx|SplitCore" \
    shared/original/op_host/*_tiling.cpp > /tmp/p16_locations.txt
grep -n "tileSize|ubFactor|Tiling|elementsPerCore" \
    shared/original/op_host/*_tiling.cpp >> /tmp/p16_locations.txt
grep -n "CalcCost|CostTable|alignCoef|alignBasic" \
    shared/original/op_host/*_tiling.cpp >> /tmp/p16_locations.txt
grep -n "NORMAL_BLOCK|TAIL_BLOCK|FD|Reduce" \
    shared/original/op_host/*_tiling.cpp >> /tmp/p16_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前分核策略（均分/代价模型**：文件 + 行号
- **代价函数位置**：文件 + 行号
- **尾块处理方式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 分核策略 | `?` (均分) | 代价驱动 | `op_host/*_tiling.cpp:L?` |
| 代价模型 | `?` (无) | alignCoef + 轴权重 | `op_host/*_tiling.cpp:L?` |
| 粒度 | `?` (单级) | Batch→Row→Block | `op_host/*_tiling.cpp:L?` |
| 尾块 | `?` (等分) | 代价感知跳过 | `op_host/*_tiling.cpp:L?` |
| 容忍度 | `?` (无) | 有（防碎片化） | `op_host/*_tiling.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整代价模型（CalcCost + CalcCostTable + 多级分配）**。
- **形态 β — 仅尾块代价感知**：简化版，只修正尾块不均。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
int64_t CalcCost(uint32_t basicM, uint32_t basicS2) {
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U;
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U;
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2);
}

BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
    uint32_t s1GTailSize, uint32_t s2TailSize) {
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U :
        CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U :
        CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ?
        0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}
```

### 3C. Variant Notes

- 对齐系数（16/64）和轴权重（6/10）需按硬件特性校准。
- Mask/稀疏模式需在代价表中置零无效块。
- 跨核归约（FD）需独立负载均衡路径。

## Step 4: 约束复核

- Host 端 Tiling 计算复杂度显著增加
- 代价模型参数不准会导致退化
- 适合高频核心算子，不适合简单算子

## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "CalcCost|CostTable" modified_files/op_host/*_tiling.cpp  # >=1
grep -cE "alignCoef|alignBasic" modified_files/op_host/*_tiling.cpp  # >=1
grep -cE "NORMAL_BLOCK|TAIL_BLOCK" modified_files/op_host/*_tiling.cpp  # >=1
grep -cE "elementsPerCore|totalElements / coreNum" modified_files/op_host/*_tiling.cpp  # ==0（或注释）
grep -cE "GetBlockNum|BLOCK_DIM" modified_files/op_host/*_tiling.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 模型参数不准 | 校准 alignCoef 和轴权重 |
| Tiling 耗时增加 | 限制枚举核数范围 |
| 碎片化 | 启用容忍度机制 |
| 简单算子负收益 | 回退均分 |

---

**完成清单**：
```
[P16 Playbook Completion]
Step 1: done (/tmp/p16_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: Host 端 Tiling 计算复杂度显著增加; 代价模型参数不准会导致退化; 适合高频核心算子，不适合简单算子: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
