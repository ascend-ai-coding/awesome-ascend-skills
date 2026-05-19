# MatMul 类算子 Tiling 场景路由

> 本文档用于 **场景判定**和**Tiling 策略选择**。确定场景后，按链接进入对应详细文档。
>
> 当前覆盖范围：**mxfp8 quantized matmul + eltwise 融合**（Div / Mul / Add / Relu / Cast / 组合算子）。
> 其它 matmul 形态（fp16/bf16 matmul、BatchMatmul 等）仍在规划中。

---

## 场景判定流程

```
给定：算子需求（含 mxfp8 量化 + eltwise 后处理？）

Step 0 — 算子分类：
  ├─ mxfp8 matmul + eltwise 融合 → 本文件 §Tiling 设计要素
  └─ 其它 matmul 形态 → 规划中（暂无正式指南，建议参考 Ascend C 官方示例）
```

> MatMul 融合的工程落地、角色路径等 Engineering 级细节
> 请见 `/ascendc-direct-invoke-template` skill 的
> `references/matmul_fusion_kernel/`（代码级模板）。

---

## Architect 路由（mxfp8 + eltwise 融合场景确定后）

1. 以 `references/matmul/matmul-fusion-design-template.md` 为骨架撰写 DESIGN.md
2. 按模板 §5 确认清单逐项自检（§5.3 含 [SAMPLE] 重评清单）

---

## Tiling 设计要素

以下四要素是所有 MatMul 融合算子 Tiling 设计必须完成的，与 `SKILL.md` 中通用设计要素一一对应。

### 1. 多核切分策略

**核心问题**：任务如何分配给多个 AI Core？

| 项目 | 说明 |
|------|------|
| 切分维度 | M × N 二维切分（K 轴在核内迭代，不参与核间切分） |
| 调度方式 | 蛇形调度（BlockScheduler），参考工程 `matmul_kernel_fused.h` |
| 单核任务量 | `singleCoreM × singleCoreN`（由 SWAT Tiling 引擎自动计算） |
| 核数计算 | `usedCoreNum = CeilDiv(M, singleCoreM) × CeilDiv(N, singleCoreN)` **强制动态计算**，禁止硬编码 |
| 负载均衡 | BlockScheduler 蛇形分配保证同 row/col 的核工作量一致 |

**输出**：
- [ ] M × N 二维切分方案
- [ ] `singleCoreM / singleCoreN` 取值
- [ ] 核数计算正确（动态，非硬编码）

---

### 2. UB 切分策略

**核心问题**：单次能处理多少数据？UB 如何布局？

#### 2.1 Tile 尺寸

| 项目 | 说明 |
|------|------|
| baseM / baseN / baseK | SWAT Tiling 引擎根据 L1/UB 容量与算子约束优化得出 |
| 分 tile 条件 | `singleCoreM × singleCoreN > baseM × baseN` 时核内需多次 tile 迭代 |
| K 轴迭代 | 核内 L1 滚动深度（`depthA1 / depthB1`），K 轴不参与核间切分 |

#### 2.2 UB 布局与 SPLIT_M

MatMul 融合算子使用 Fixpipe SPLIT_M 将 L0C 输出拆分给两个 AIV，
UB 布局受以下约束：

```
UB offset 0 ┌───────────┐
            │ cLocal_   │  matmul 结果（L0C → UB via Fixpipe SPLIT_M）
            │           │  布局为 ND [curMPad, curNUbAlign]，其中：
            │           │    curMPad  = (curM + 1) & ~1        ODD-M 向上取偶
            │           │    curNUbAlign = AlignUp(curN, 8)    ODD-N 对齐 32B
            ├───────────┤
            │ dLocal_   │  第二路输入暂存（stage buffer）
            ├───────────┤
            │ resultLocal_│ 计算结果暂存（stage buffer）
            └───────────┘
```

**ODD-M 处理**：当 M 为奇数时，Fixpipe DUAL_DST_SPLIT_M 要求 M 为偶数才能等分给双 AIV。
须将 L0C/UB 视图的 M 向上取偶（`curMPad`），MMAD 仍用原始 `curM`。
Epilogue 的 SubBlockIdx 偏移逻辑仅访问有效行，padding 行不会被写出到 GM。

**ODD-N 对齐**：UB 侧 Fixpipe 的 row stride 必须对齐到 32B（即 8 个 float）。
若 `curN` 非 8 倍数（如 N=1/4/15/33），须将 UB 视图的 N 向上对齐到 8 的倍数（`curNUbAlign`），
Fixpipe 每行仅填充 `curN` 列，尾部对齐列保持未触碰，
Epilogue 通过 `DataCopyPad` 按 `rowBytes = curN × sizeof(float)` 仅写出有效列。

**SPLIT_M 偏移**：Epilogue 计算 GM 偏移时须追加：
```cpp
offset += AscendC::GetSubBlockIdx() * halfM * N;
```
其中 `halfM = CeilDiv(curM, 2)`。

#### 2.3 UB 容量自检

```
UB 总用量 = matmulArea + stageNum × stageSize_
matmulArea  = CeilDiv(baseM, taskRation) × baseNAlign  (SplitM 修正)
stageSize_  = min(剩余 UB / stageNum / sizeof(DataType) / baseNAlign × baseNAlign, matmulArea)
```

必须满足 `UB 总用量 ≤ TOTAL_UB_SIZE`（不同芯片 UB 容量不同，见 [SAMPLE] S1）。

**输出**：
- [ ] baseM/baseN/baseK 取值
- [ ] ODD-M/ODD-N 对齐方案
- [ ] SPLIT_M 偏移公式
- [ ] UB 容量自检通过

---

### 3. Buffer 规划

**核心问题**：需要哪些 buffer？各多大？

#### AIC 侧 Buffer

| Buffer | 用途 | 大小 | 位置 | 标注 |
|--------|------|------|------|------|
| L1_A | A 矩阵滚动窗口 | `depthA1 × baseM × baseK × sizeof(AType)` | L1 | `[PATTERN]` |
| L1_B | B 矩阵滚动窗口 | `depthB1 × baseN × baseK × sizeof(BType)` | L1 | `[PATTERN]` |
| L0A | A 分块计算 | `baseM × baseK × sizeof(AType)` | L0A | `[PATTERN]` |
| L0B | B 分块计算 | `baseN × baseK × sizeof(BType)` | L0B | `[PATTERN]` |
| L0C | 累加结果 | `dbL0C × baseM × baseN × sizeof(float)` | L0C | `[PATTERN]` |
| L1_ScaleA/B | mxfp8 scale（如需） | `baseM × sizeof(float)` / `baseN × sizeof(float)` | L1 | `[EXTEND]` |

> AIC 侧 buffer 由 SWAT Tiling 引擎统一管理（Tensor API），标注 `[PATTERN]` 的区域 **禁止业务修改**。

#### AIV 侧 UB Buffer（静态偏移分配）

| Buffer | 用途 | 大小 | 标注 |
|--------|------|------|------|
| UB_L0COut | L0C → UB Fixpipe 输出 | `baseMPad × baseNUbAlign × sizeof(float)` (含 ODD 对齐) | `[PATTERN]` |
| UB_Eltwise | 第二路输入暂存 | `stageSize_ × sizeof(DataType)` | `[USER]` |
| UB_Cast | Cast 输出暂存（如需） | `stageSize_ × sizeof(OutputType)` | `[USER]` |

> **融合算子禁止使用 TPipe 管理 UB**，采用静态偏移分配。
>
> `stageNum` 按输入路数选择：
> - 1（Relu/Cast 等无第二输入） → 仅 `resultLocal_`
> - 2（Div/Mul/Add 等单第二输入） → `secondLocal_` + `resultLocal_`
> - 3（Mul+Add 等双第二输入） → `secondLocal_` + `thirdLocal_` + `resultLocal_`

**输出**：
- [ ] AIC L1/L0 buffer 已确认（`[PATTERN]` 不改）
- [ ] AIV UB buffer 布局已确定
- [ ] `stageNum` 已按输入路数正确设置
- [ ] 总 UB 用量 ≤ 目标芯片 UB_SIZE

---

### 4. 分支场景覆盖

**核心问题**：需要处理哪些不同场景？

| 分支维度 | 条件 | 处理策略 |
|---------|------|---------|
| 数据类型 | A/B dtype ∈ {mxfp8}, C/中间 dtype ∈ {float, half, bf16} | `[MODIFY]` 修改类型别名；同步修改 `sizeof`、`ALIGN_ELEM` |
| 转置组合 | `transA/transB` 可选 NN/NT/TN/TT | Host 侧按 trans 选择 `RowMajor/ColumnMajor`；L1 Layout 始终 `Nz/Zn`（Cube 硬件要求） |
| 大 shape | M × N 远大于 `singleCoreM × singleCoreN` | 多核 M×N 切分 + 核内 K 轴 L1 滚动 |
| 小 shape | M/K/N 较小 | 减少核数；baseM/baseN 可适当缩小以匹配 |
| M 尾块（ODD-M） | `curM` 为奇数 | UB/L0C 视图 M 向上取偶；Epilogue 仅访问有效行 |
| N 尾块（ODD-N） | `curN` 非 8 倍数 | UB 视图 N 对齐到 32B；DataCopyPad 按实际 `curN` 读写 |
| K 对齐 | mxfp8 场景 | Host 侧 padding K 到 64 倍数（mxfp8 Cube 硬件要求） |

**输出**：
- [ ] dtype 组合已覆盖
- [ ] 转置组合已规划（如适用）
- [ ] 大/小 shape 策略明确
- [ ] ODD-M/ODD-N 尾块处理已就绪
- [ ] K 对齐已确保

---

## 参考资源

| 资源 | 位置 | 用途 |
|------|------|------|
| MatMul 融合设计模板 | `references/matmul/matmul-fusion-design-template.md` | DESIGN.md 编写骨架（API 映射、工程结构、确认清单、[SAMPLE] 重评） |
| MatMul 融合参考工程 | `/ascendc-direct-invoke-template` → `references/matmul_fusion_kernel/` | 代码级模板（Kernel / Epilogue / BlockMmad / run.sh）— Developer 使用 |
| NPU 架构配置 | `npu-arch` skill | UB_SIZE 等芯片相关常量 |
