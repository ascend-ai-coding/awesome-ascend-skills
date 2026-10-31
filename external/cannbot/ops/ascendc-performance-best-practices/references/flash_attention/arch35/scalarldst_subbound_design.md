# SCALARLDST 减载优化设计

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

## 1. 优化目标

针对 FIA 在 SCALAR_BOUND 下由 SCALARLDST 主导的场景，减少热循环内成员变量、栈变量和查表访问带来的 scalar load/store。该策略适用于 `SCALARLDST% >= 95%` 或 `SCALARLDST% > SCALAR%` 的 case。

已验证效果：

| 指标 | Baseline | SCALARLDST 优化后 | 变化 |
|------|---------:|------------------:|-----:|
| 最长核 case cycle | 18353 | 18076 | -1.51% |
| Cube 核 SCALARLDST busy | 81.30% | 81.14% | -0.16pp |
| 头开销 cycles | 3205 | 3102 | -3.21% |

## 2. 架构概览

SCALARLDST 主导时，继续把 ALU 计算改成成员预计算可能只是在“少一次乘法”和“多一次成员 load”之间对冲。优化重点应转为减少 load/store 本身。

| 问题场景 | 典型代码形态 | 优化方向 |
|----------|--------------|----------|
| 一次性初始化守卫留在热循环 | `if (!ready_) Update(...)` | 初始化前置，删除热循环守卫 |
| 准常量分支每轮读成员 | `if (pow2_) x >> shift_ else x / divisor_` | 入口捕获为局部 `const` 或模板化 |
| 查表替代 ALU 后增加 load | `maskTable_[len]` | 小集合改为 `switch` 常量 |
| 成员字段反复访问 | `this->coreWsBase_` / `this->blockSize_` | 函数局部 `const` 别名 |

## 3. 关键参数 / 修改点

| 策略 | 修改点 | 约束 |
|------|--------|------|
| MR1 lazy-init 守卫清理 | 删除热循环入口的 `if (!stridesReady_) UpdateStride(...)` | 上层入口必须显式完成 `UpdateStride` |
| MR2 准常量局部捕获 | 将 `*_Pow2_`、`*_Shift_`、`*_Mask_` 在 operator 入口读入局部 `const` | 若编译器未消除 load，再考虑模板化 |
| MR3 查表改常量分支 | `SetBlockReduceMask` 使用 `switch` 常量替代成员数组 | 保持边界语义不变 |
| MR4 workspace 成员局部化 | `coreWsBase_`、`coreWs1_`、`blockStackNum_` 改读局部别名 | 只替换函数内不变字段 |
| MR5 循环内成员局部化 | `maxKVStackLen`、`blockSize` 等入口读一次 | 防止过多局部变量造成寄存器 spill |

## 4. 核心计算循环 / 关键代码

### MR1：删除热循环 lazy-init 守卫

```cpp
// 优化前：每次 operator 调用都读 ready flag
if (!stridesReady_) {
    UpdateStride(strideKV, blockSize);
}

// 优化后：入口前显式初始化，热循环内不再读 flag
UpdateStride(strideKV, blockSize);
BlockMmad(...);
```

### MR2：准常量字段局部捕获

```cpp
const bool tileNNumPow2Local = tileNNumPow2_;
const uint32_t tileNNumShiftLocal = tileNNumShift_;
const uint32_t tileNNumMaskLocal = tileNNumMask_;
const uint32_t tileNNumPerBaseBlockLocal = tileNNumPerBaseBlock_;

for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
    if (tileNNumPow2Local) {
        nL1DivTile = nL1Idx >> tileNNumShiftLocal;
        nL1ModTile = nL1Idx & tileNNumMaskLocal;
    } else {
        nL1DivTile = nL1Idx / tileNNumPerBaseBlockLocal;
        nL1ModTile = nL1Idx % tileNNumPerBaseBlockLocal;
    }
}
```

### MR3：小规模查表改 switch 常量

```cpp
__aicore__ inline void SetBlockReduceMask(int32_t len) {
    uint64_t mask;
    switch (len) {
        case 1: mask = 0x0101010101010101ULL; break;
        case 2: mask = 0x0303030303030303ULL; break;
        case 3: mask = 0x0707070707070707ULL; break;
        case 4: mask = 0x0F0F0F0F0F0F0F0FULL; break;
        case 5: mask = 0x1F1F1F1F1F1F1F1FULL; break;
        case 6: mask = 0x3F3F3F3F3F3F3F3FULL; break;
        case 7: mask = 0x7F7F7F7F7F7F7F7FULL; break;
        case 8: mask = 0xFFFFFFFFFFFFFFFFULL; break;
        default:
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
    }
    AscendC::SetVectorMask<int8_t>(mask, mask);
}
```

## 5. 效果与适用边界

| 场景 | 建议 |
|------|------|
| `SCALARLDST% >= 95%` 或高于 `SCALAR%` | 优先做 MR1-MR5 |
| `SCALAR%` 更高且 ALU 运算密集 | 使用 scalar precompute，不优先做 MR |
| 低精度 mask 表项较多 | 先脚本生成常量，禁止手推边界值 |
| 局部变量过多导致 spill | 回退部分局部化，保留收益最大的字段 |

风险红线：

- 不把 `stackSeqCount % (PRE_LAUNCH + 1U)` 改成位与，因为 `PRE_LAUNCH + 1U == 3`。
- 不改 TilingData 结构、host tiling 字段和事件同步链。
- 删除 lazy-init 守卫前，必须保证所有调用路径已显式初始化 stride。
