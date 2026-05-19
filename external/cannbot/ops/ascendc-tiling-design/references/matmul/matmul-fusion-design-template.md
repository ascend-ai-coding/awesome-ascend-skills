# {OperatorName} 算子设计文档

> **本模板仅用于** mxfp8 matmul + eltwise 融合场景。
>
> Tiling 设计详见 `references/matmul/patterns.md`。
> 参考工程：`/ascendc-direct-invoke-template` → `references/matmul_fusion_kernel/`。
>
> 标注约定：
> `[PATTERN]` 固定结构禁改；`[USER]` 按算子语义填写；`[SAMPLE]` 样例需重评；
> `[MODIFY]` 必改点；`[EXTEND]` 扩展点；`[CONFIG]` 配置点。

---

## 1. 概述

### 1.1 需求类型

- **特定用例**：用户明确指定具体的 M / K / N 与 dtype。
- **通用**：未指定 shape，需支持通用 M × K × N 矩阵乘法。

### 1.2 基本信息

| 项目 | 内容 |
|------|------|
| 算子名称 | |
| 算子类别 | MatMul + {EltwiseType}（mxfp8 量化 × eltwise 融合） |
| 算子模式 | MIX（AIC + AIV；L0C → UB Fixpipe 恒定路径） |
| 需求类型 | 特定用例 / 通用 |
| 支持数据类型 | A = {AType}, B = {BType}, C = {CType} |
| 支持芯片 | Ascend950（mxfp8 仅 Ascend950 硬件支持） |
| 特殊约束 | |
| 参考工程 | `/ascendc-direct-invoke-template` → `references/matmul_fusion_kernel/` |

---

## 2. 算子设计

### 2.1 数学公式

```
输入:
  A       - shape [M, K], dtype = {AType}            # mxfp8
  B       - shape [K, N], dtype = {BType}            # mxfp8
  scaleA  - mxfp8 scale for A                        # [PATTERN]
  scaleB  - mxfp8 scale for B                        # [PATTERN]
  D       - shape [M, N], dtype = {EltwiseType}      # [USER] 可选第二输入；单输入算子（如 Relu）删除

输出:
  C       - shape [M, N], dtype = {CType}

数学公式:
  C = f(dequant(matmul(A * scaleA, B * scaleB)), D)  # f 为 Div / Mul / Add / Relu / Cast / ...
```

### 2.2 API 映射

> **核心约束**：
> - Matmul 部分**使用 Tensor API**，融合场景**无需修改**。
> - **设计重点为 Vector 侧 Epilogue**，使用 AscendC API（`DataCopy` / `Add` / `Mul` / `Cast` 等）。

#### 2.2.1 Cube 数据通路（AIC 侧，固定模式）

> Matmul 部分已固定使用 Tensor API，融合场景无需修改。
> 数据通路：GM→L1→L0→MMAD→L0C→UB(Fixpipe)，详见参考工程 `block_mmad_swat.h`。

| 阶段 | Tensor API | 标注级别 |
|------|-----------|---------|
| GM→L1→L0→MMAD | `Te::Copy` / `Te::Mad` | `[PATTERN]` 固定 |
| L0C→UB | `Te::Fixpipe<FIXPIPE_TRAIT_SPLIT_M>` | `[PATTERN]` 固定 |

#### 2.2.2 Vector 数据通路（AIV 侧 Epilogue，设计重点）

> AIV 侧 Epilogue 由自定义类实现，不使用枚举或预设分支。
> Epilogue 开发详见 `/ascendc-direct-invoke-template` → `references/matmul_fusion_guide.md` §3（Epilogue 开发详解）。

##### API 映射表

| 阶段 | 数学操作 | ascendc API | 关键参数 | 标注级别 | 官方文档 |
|------|---------|------------|---------|---------|---------|
| GM → UB | 融合输入搬运 | `DataCopy(dLocal, gmD, len)` 或 `DataCopyPad(...)` | 32 字节对齐；`curN < N` 时必须用 `DataCopyPad` | `[USER]` | [DataCopy](asc-devkit/docs/api/context/data_copy.md) / [DataCopyPad](asc-devkit/docs/api/context/data_copy_pad.md) |
| 计算 | 融合操作 | `Div` / `Mul` / `Add` / `Relu` / ... | float 操作；`count` 为有效数据长度 | `[USER]` | [AscendC API](asc-devkit/docs/api/context/) |
| Cast | 类型转换（如需） | `Cast(dst, src, roundMode, count)` | `CAST_NONE` / `CAST_RINT` | `[USER]` | [Cast](asc-devkit/docs/api/context/cast.md) |
| UB → GM | 结果写回 | `DataCopy(gmC, out, len)` 或 `DataCopyPad(...)` | 32 字节对齐；`curN < N` 时必须用 `DataCopyPad` | `[USER]` | [DataCopy](asc-devkit/docs/api/context/data_copy.md) / [DataCopyPad](asc-devkit/docs/api/context/data_copy_pad.md) |

##### API 语义验证表

**每个 Vector 侧 API 调用前，必须填写验证表**。

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| DataCopy / DataCopyPad | GM 连续 / UB 连续；32B 对齐；`curN < N` 时带 stride | GM↔UB 搬运；非对齐时填充 | `DataCopy(dst, src, len)` 或 `DataCopyPad(dst, src, blockLen, DataCopyExtParams{...})` | DataCopy 要求 32B 对齐；DataCopyPad 要求 `blockLen` 为 dataBlock（32B）倍数 | ✅/❌ | [链接] |
| Div / Mul / Add / Relu | UB 连续；vector 寄存器 | 逐元素二元/一元运算 | `Div(dst, src0, src1, count)` 等 | 输入输出 dtype 一致；`count` 为有效长度 | ✅/❌ | [链接] |
| Cast | UB 连续 | 类型转换 | `Cast(dst, src, roundMode, count)` | 目标 dtype 支持；roundMode 选择正确 | ✅/❌ | [链接] |

**验证清单**（每个 API 必须完成）：
- [ ] 1. 数据布局确认（内存排列、连续性、对齐）
- [ ] 2. 功能需求明确（操作类型、维度、输出格式）
- [ ] 3. 已查阅官方文档（提供链接）
- [ ] 4. 匹配验证（数据布局与 API 能力匹配、限制条件满足）
- [ ] 5. 已记录验证过程

**验证方法**：
```
问题 1：数据布局是什么？
    ├─ 内存如何排列？（GM 连续 / UB 连续 / 带 stride）
    ├─ 是否对齐？（32 字节对齐 / 非对齐）
    └─ 输入输出格式？（vector / 标量）

问题 2：需要什么操作？
    ├─ 操作类型？（copy / elementwise / cast）
    ├─ 操作维度？
    └─ 特殊要求？（roundMode / stride）

问题 3：API 能实现吗？
    ├─ 查阅官方文档了吗？
    ├─ API 适用场景对吗？
    ├─ 满足 API 限制吗？
    └─ 有更好的选择吗？
```

**验证公式**：`正确使用 = (数据布局 ∈ API 支持范围) AND (满足所有限制条件) AND (无更好选择)`

**参数使用规则**：
| 参数位置 | 用有效长度 | 用对齐长度 |
|---------|-----------|-----------|
| DataCopyPad blockLen / 计算 API count | ✓ | ✗ |
| UB 数据偏移 / Buffer 大小 | ✗ | ✓ |

> **DataCopyPad 关键说明**：`curN < N` 时必须使用 `DataCopyPad + DataCopyExtParams` 显式设置 `srcGap/dstGap`。`DataCopyExtParams` 中 **GM 侧单位为字节**，**UB 侧单位为 dataBlock（32B）**——两者不可混用。

##### 核间同步

| 方向 | 机制 | 参考工程常量 | 标注级别 |
|------|------|------------|---------|
| AIC → AIV | `CrossCoreSetFlag` + `CrossCoreWaitFlag` | `CvSync::AIC_TO_AIV_FLAG`（`include/epilogue/cv_sync_constants.h`） | `[PATTERN]` |
| AIV → AIC | `CrossCoreSetFlag` + `CrossCoreWaitFlag` | `CvSync::AIV_TO_AIC_FLAG`（同上） | `[PATTERN]` |

> **重要**：CrossCoreFlag 常量必须复用参考工程 `cv_sync_constants.h`，禁止自定义 Flag ID。

### 2.3 数据流

```
A [M,K] {AType} ──┐
B [K,N] {BType} ──┤
                  ▼
            ┌────────────────┐
            │  AIC (Cube)     │
            │ GM→L1→L0→MMAD  │  Tensor API 分层
            │ L0C→UB (Fix)   │  Fixpipe SPLIT_M 恒定路径
            └────────┬───────┘
                     │ CrossCoreSetFlag (AIC→AIV)
                     ▼
            ┌────────────────┐
            │  AIV (Vector)   │
            │ GM→UB: D       │  DataCopy / DataCopyPad
            │ Cast + Eltwise │  ascendc API
            │ UB→GM: C       │  DataCopy
            └────────┬───────┘
                     │ CrossCoreSetFlag (AIV→AIC)
                     ▼
              C [M,N] {CType}
```

### 2.4 核心计算步骤

```
1. Host Tiling — 计算 baseM/baseN/baseK、L1 depth、核数（SWAT Tiling 引擎）
2. Kernel Launch — <<<usedCoreNum, stream>>> 启动 MIX Kernel（AIC/AIV 共入口）
3. BlockScheduler — 蛇形多核调度，分配 tile 坐标
4. AIC BlockMmad — K 轴迭代:
   4.1 GM→L1: A(Nz), B(Zn)         — Te::Copy(CopyGM2L1{})
   4.2 L1→L0: A(L0A), B(L0B)        — Te::Copy(CopyL12L0A/B{})
   4.3 MMAD: float 累加到 L0C        — Te::Mad(MmadMxAtom{})
   4.4 L0C→UB: Fixpipe<SPLIT_M>     — 恒定路径，无条件分支
   4.5 CrossCoreSetFlag(AIC→AIV)    — 通知数据就绪
5. AIV Epilogue:
   5.1 CrossCoreWaitFlag(AIC→AIV)   — 等待数据就绪
   5.2 GM→UB: D 张量                 — DataCopy（单输入算子删除）
   5.3 Eltwise 计算                  — ascendc API
   5.4 UB→GM: 结果写回               — DataCopy
   5.5 CrossCoreSetFlag(AIV→AIC)    — 通知消费完成
6. CrossCoreWaitFlag(AIV→AIC) — AIC 等待 AIV，进入下一 tile
```

### 2.5 内存管理

#### AIC 侧 Buffer

| Buffer | 用途 | 大小 | TPosition | 标注级别 |
|--------|------|------|-----------|---------|
| L1_A | A 矩阵滚动 | depthA1 × baseM × baseK × sizeof({AType}) | L1 | `[PATTERN]` |
| L1_B | B 矩阵滚动 | depthB1 × baseN × baseK × sizeof({BType}) | L1 | `[PATTERN]` |
| L0A | A 分块计算 | baseM × baseK × sizeof({AType}) | L0A | `[PATTERN]` |
| L0B | B 分块计算 | baseN × baseK × sizeof({BType}) | L0B | `[PATTERN]` |
| L0C | 累加结果 | dbL0C × baseM × baseN × sizeof(float) | L0C | `[PATTERN]` |

> mxfp8 场景另需规划 ScaleA / ScaleB 的 L1 Buffer。

#### AIV 侧 UB Buffer（静态偏移分配）

| Buffer | 用途 | 大小 | 标注级别 |
|--------|------|------|---------|
| UB_L0COut | L0C → UB Fixpipe 输出（cLocal_） | `matmulArea × sizeof(float)` | `[PATTERN]` |
| UB_Eltwise | D 张量暂存（dLocal_） | `stageSize_ × sizeof({EltwiseType})` | `[USER]` |
| UB_Cast | Cast 输出暂存（cLocalTmp_） | `stageSize_ × sizeof({CType})` | `[USER]` |

> **融合算子禁止使用 TPipe 管理 UB**，采用静态偏移分配。

**matmulArea 计算公式**（SPLIT_M 修正）：
```
l1NAlign   = CeilDiv(baseN, ALIGN_ELEM) × ALIGN_ELEM    // N 对齐到 32B
taskRation = AscendC::GetTaskRation()                    // SplitM 拆分比（=2）
l1MSplit   = CeilDiv(baseM, taskRation)                  // 每个 Vector 核分到的 M
matmulArea = l1MSplit × l1NAlign                        // matmul 结果占用 UB 元素数
```

**stageSize_ 计算公式**：
```
lastUBBytes = UB_SIZE - matmulArea × sizeof(DataType)   // 剩余 UB 字节数
usableElems = lastUBBytes / stageNum / sizeof(DataType)  // 每 stage 可用元素数
stageSize_  = min(usableElems / l1NAlign × l1NAlign, matmulArea)  // 对齐后取 min
```

**stageNum 选择**：
| stageNum | 输入路数 | 典型算子 | UB 分区 |
|----------|---------|---------|---------|
| 1 | 0（无额外输入） | Relu、GeLU、Cast | cLocal_ + cLocalTmp_ |
| 2 | 1（单第二路输入） | Div、Mul、Add | cLocal_ + dLocal_ + cLocalTmp_ |
| 3 | 2（双第二路输入） | Mul+Add | cLocal_ + dLocal1_ + dLocal2_ + cLocalTmp_ |

**UB 容量自检**：`matmulArea × sizeof(float) + stageNum × stageSize_ × sizeof(DataType) ≤ TOTAL_UB_SIZE`

---

## 3. 架构设计

> 详细的 Tiling 设计策略（多核切分、UB 切分、Buffer 规划、分支覆盖）与决策理由
> 见 `/ascendc-tiling-design` → `references/matmul/patterns.md`。
> 本节摘录模板要点，Architect 按 `[USER]` / `[MODIFY]` 填写后即构成完整的 DESIGN.md。

### 3.1 多核切分

| 项目 | 说明 |
|------|------|
| 切分维度 | M × N 二维切分（K 轴在核内迭代） |
| 单核任务量 | singleCoreM × singleCoreN（SWAT Tiling 自动计算） |
| 核数 | **强制动态计算**：`usedCoreNum = CeilDiv(M, singleCoreM) × CeilDiv(N, singleCoreN)` |
| 负载均衡 | 蛇形调度（BlockScheduler `[PATTERN]`） |

### 3.2 UB / L1 切分

| 项目 | 说明 |
|------|------|
| L1 容量 | SWAT Tiling 引擎管理 |
| L1 滚动深度 | depthA1 / depthB1（Tiling 引擎计算） |
| baseM / baseN / baseK | Tiling 引擎优化 |
| UB 容量 | `matmulArea × sizeof(float) + stageNum × stageSize_ × sizeof(DataType)`（见 §2.5 计算公式） |
| 是否分 tile | singleCoreM × singleCoreN > baseM × baseN 时核内多次迭代 |

### 3.3 分支场景

| 分支条件 | 处理策略 |
|---------|---------|
| 数据类型 | `[MODIFY]` 修改类型别名 |
| 转置组合 | Host 侧按 `transA/transB` 选择 `RowMajor/ColumnMajor`；L1 Layout 始终 `Nz/Zn`（Cube 硬件要求） |
| 大 shape | 多核 M × N 切分 + 核内 K 迭代，L1 滚动 |
| 小 shape | 减少核数，baseM/baseN 适当缩小 |
| M / N 尾块 | BlockScheduler 蛇形调度自动处理 `[PATTERN]` |
| K 对齐 | Host 侧 padding（mxfp 场景 K 到 64 倍数） |

### 3.4 关键实现要点

#### Kernel 入口（MIX 模式）

```cpp
// [PATTERN] MIX Kernel 入口（参考工程 matmul_kernel_fused.h；Developer 不应修改结构）
__global__ __aicore__ void {OperatorName}FusedKernel(GM_ADDR dA, GM_ADDR dB,
    GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dOutput,
    GM_ADDR dD,  // [USER] 单输入算子可删除
    const QuantMatmulTilingData tilingData)
{
    if ASCEND_IS_AIV {
        RunAIV(params);   // [PATTERN] AIV: Epilogue 入口
    } else {
        RunAIC(params);   // [PATTERN] AIC: BlockMmad 入口
    }
}
```

#### AIC 侧（L0C → UB 恒定路径）

```cpp
// [PATTERN] AIC 侧：BlockMmad + Fixpipe SPLIT_M（无条件分支）
for (k_iter = 0; k_iter < kL1Iter; ++k_iter) {
    // GM→L1→L0→MMAD（Tensor API，[PATTERN]，Developer 不修改）
}
// [PATTERN] L0C → UB（唯一输出路径）
auto ubTensor = Te::MakeTensor(Te::MakeUBmemPtr<float>(0), layoutUB);
Te::Fixpipe<FIXPIPE_TRAIT_SPLIT_M>(ubTensor, l0C, Te::FixpipeParams{FINAL_ACCUMULATION});
// [PATTERN] AIC → AIV 同步（使用 cv_sync_constants.h 常量）
CrossCoreSetFlag<mode, pipe>(CvSync::AIC_TO_AIV_FLAG + countId);
CrossCoreWaitFlag<mode, pipe>(CvSync::AIV_TO_AIC_FLAG + countId);
```

#### AIV 侧 Epilogue

Epilogue 开发详见 `/ascendc-direct-invoke-template` → `references/matmul_fusion_guide.md` §3（Epilogue 开发详解），包含：
- §3.1 计算流程（MTE2→V→MTE3 三对 SetFlag/WaitFlag 同步顺序）
- §3.2 UB 分配策略（matmulArea / stageSize_ 计算公式）
- §3.3 三接口合约（Init / GetTensor / operator()）
- §3.4 SPLIT_M 与 ODD-M/ODD-N 处理
- §3.5 CV 同步机制
- §3.6 DataCopyPad 与 stride 处理

---

## 4. 工程结构

基于 `references/matmul_fusion_kernel/`（扁平目录）：

```
operators/{operator_name}/
├── src/
│   └── matmul_fused_swat.cpp         # [MODIFY]+[USER] Host Launcher（替换 MyEpilogue）
├── include/
│   ├── kernel/
│   │   └── matmul_kernel_fused.h     # [PATTERN] MIX Kernel 模板（不改）
│   ├── block/
│   │   ├── block_mmad_swat.h         # [PATTERN] L0C→UB Fixpipe SPLIT_M 单路径
│   │   ├── block_mmad.h              # [PATTERN] BlockMmad 基类
│   │   └── block_scheduler.h         # [PATTERN] 蛇形调度
│   ├── epilogue/
│   │   ├── cv_sync_constants.h       # [PATTERN] CV 同步常量
│   │   ├── div_epilogue.h            # [USER]+[SAMPLE] 主样板（mxfp8 + Div）
│   ├── policy/*.h                    # [PATTERN] Dispatch Policy
│   ├── tile/*.h                      # [PATTERN] 硬件指令封装（不改）
│   ├── tiling/
│   │   ├── tiling_data.h             # [MODIFY]+[EXTEND] Tiling 参数
│   │   └── tiling_swat.h             # [CONFIG] SWAT Tiling 引擎
│   └── utils/
│       ├── constants.h               # [CONFIG]
│       ├── hardware_constants.h      # [PATTERN]+[SAMPLE] UB_SIZE 等
│       └── tiling_key.h              # [EXTEND] 转置模式枚举
├── common/                           # [PATTERN] 跨算子共享工具
├── scripts/
│   ├── gen_data.py                   # [MODIFY]+[USER] 测试数据生成
│   └── verify_result.py              # [MODIFY] 精度验证
├── third_party/
│   └── tensor_api/                   # [PATTERN] Tensor API 头文件
├── docs/
│   ├── DESIGN.md                     # 本模板
│   ├── PLAN.md                       # 开发计划
│   └── REVIEW.md                     # Reviewer 输出
├── CMakeLists.txt                    # [MODIFY]+[CONFIG]（仅融合 target）
└── run.sh                            # [MODIFY]
```

---

## 5. 确认清单

### 5.1 通用检查

- [ ] 多核切分策略已确定（M × N 二维切分，蛇形调度）
- [ ] Buffer 规划已完成（AIC L1/L0 + AIV UB）
- [ ] 分支场景已覆盖（转置、大/小 shape、尾块、对齐）
- [ ] `[MODIFY]` / `[EXTEND]` / `[USER]` API 映射已验证
- [ ] `[PATTERN]` API 已确认无需修改
- [ ] `[SAMPLE]` 项已按 §5.3 [SAMPLE] 重评清单 逐项评估
- [ ] Vector 侧 API 语义验证已完成（§2.2.2 验证表 + 验证清单）

### 5.2 融合算子专项

- [ ] MIX 模式（`__aicore__` + `ASCEND_IS_AIC/AIV` 分发）
- [ ] L0C → UB Fixpipe 恒定路径（`FIXPIPE_TRAIT_SPLIT_M`，无 `enableFusion` 开关）
- [ ] CrossCoreFlag 使用参考工程 `cv_sync_constants.h` 常量
- [ ] AIV Epilogue PipeBarrier 仅在跨 pipe 依赖处添加；Epilogue 内**无** `CrossCoreWaitFlag`
- [ ] 工程**不包含**单算子残留
- [ ] `CMakeLists.txt` 仅含融合 target（`matmul_fused_swat`）
- [ ] `mmadParams` 仅 5 个 GM 地址（`ubAddr` / `enableFusion` 已删除）

### 5.3 [SAMPLE] 重评清单

| 编号 | 样例项 | 风险 | 处理要求 |
|------|--------|------|---------|
| S1 | `UB_SIZE` | 不同芯片 UB 容量不同 | 按目标芯片重定义 |
| S2 | `using DataType = float` | dtype 迁移后链路不一致 | 同步修改 alias 与 `sizeof` |
| S3 | `stageNum = 2` | 输入路数变化导致布局错配 | 按 1/2/3 路重设 |
| S4 | `ALIGN_ELEM = 32/sizeof(float)` | dtype 改变后对齐错误 | 改为 `sizeof(DataType)` |
| S5 | `rowBytes = ... * sizeof(float)` | stride 字节数错误 | 按输出 dtype 重算 |
| S6 | `divisor*` 命名 | 语义漂移 | 按业务重命名 |
| S7 | batch=1 假设 | 批量场景不成立 | 超范围场景单独扩展 |
| S8 | RowMajor 假设 | 布局变更导致偏移错误 | 保持 RowMajor 或全链路调整 |
| S9 | `sizeD = m*n*sizeof(...)` | 广播输入 shape 不匹配 | 同步改 Host 与 Epilogue |
| S10 | `input_d.bin` 文件名 | 语义不匹配 | 按算子语义改名并同步脚本 |

