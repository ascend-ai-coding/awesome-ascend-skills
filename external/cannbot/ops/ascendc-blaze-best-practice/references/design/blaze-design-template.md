# {operator_name} Blaze 算子设计文档

> ⚠️ **SubAgent 生成此文档时必须替换以下占位符**：
> - `{operator_name}` → 实际算子名称

---

## 0. 概述

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | |
| 算子类别 | MatMul / MatMul+Vector 融合 / MX 量化 MatMul / GroupMatmul |
| 需求类型 | 特定用例（shape=[...], dtype=...） / 通用 |
| 支持数据类型 | A = , B = , C = |
| 支持芯片 | Ascend950（DAV_3510） |
| 特殊约束 | |

### 0.2 用户原始需求

| # | 需求内容 |
|---|---------|
| 1 | |
| 2 | |

---

## 1. 算子设计

### 1.1 数学公式

```
输入:
  A - shape [M, K], dtype =
  B - shape [K, N], dtype =
输出:
  C - shape [M, N], dtype =

数学公式:
  C = f(A, B)
```

### 1.2 数据流

> 数据流应紧跟数学公式展开，先说明逻辑计算如何映射到 GM/L1/L0/UB/GM，再继续设计 Kernel、Epilogue、Tiling 和工程目录。

**纯 Matmul**：
```
GM → L1 → L0A/L0B → MMAD → L0C → Fixpipe → GM
```

**普通 C+V 融合场景**：
```
GM → L1 → L0A/L0B → MMAD → L0C → Fixpipe(SPLIT_M) → UB → Epilogue → GM
```

> **SplitM 说明**：`DUAL_DST_SPLIT_M` 将 L0C 的 M 行对半切分到两个 AIV SubBlock。每个 AIV 从各自 UB offset 0 读取半份数据（UB 读取不需要 sub-block 偏移），GM 读写需要 sub-block 偏移。

**MX C+V / Grouped C+V 等特殊场景**：
请结合 `/ascendc-blaze-best-practice` 中对应场景文档，补充 scale / groupList / context 等额外输入在数据流中的位置。

### 1.3 Kernel 入口与组件组装

> 参考 `/ascendc-blaze-best-practice` skill 的 `references/development/step2-kernel-design.md`。

**开发路径决策**：

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 开发路径 | blaze_custom / blaze 库 | |
| Dispatch mode | NO_FULL_LOAD / A_FULL_LOAD / MatmulWithScaleMx | |
| Tiling 引擎 | MatmulTilingSwat / MatmulTilingAFullLoad / QuantMatmulTilingSwat | |

**Kernel 签名**：

| 项目 | 值 |
|------|------|
| 修饰符 | `__cube__` / `__mix__(aicCount, aivCount)` |
| GM_ADDR 参数 | dA, dB, [dScaleA, dScaleB,] [dBias,] dC, tilingData |
| 模板参数 | TransA, TransB [, CubeFormat] |

**参考文档**：

| 路径 | 文档位置 |
|------|---------|
| blaze_custom | `/ascendc-blaze-best-practice` → `references/modules/blaze-custom/` |
| blaze 库 | `/ascendc-blaze-best-practice` → `references/modules/blaze-library/` |

**组件组装**：

| 层 | 代码库 | 选用组件 | 来源文件 | 说明 |
|---|--------|---------|---------|------|
| Launcher | | | | 入口文件，按 trans/format 分发 |
| Kernel | | | | 循环驱动 |
| BlockMmad | | | | L1/L0/MMAD 流水 |
| Scheduler | | | | 多核调度 |
| DispatchPolicy | | | | dispatch tag |
| Epilogue（融合场景） | | | | AIV 侧后处理 |
| tensor_api | - | `Te::Copy/Mmad/Fixpipe` | `third_party/tensor_api/` | 硬件指令 |

### 1.4 Epilogue 设计（融合场景）

> 参考 `/ascendc-blaze-best-practice`：
> - `references/scenarios/fusion-matmul-development.md`
> - `references/modules/blaze-custom/development/epilogue-dev-guide.md`
> - `references/modules/blaze-custom/development/epilogue-membase-design.md`
> - `references/modules/blaze-custom/development/epilogue-regbase-design.md`
> 非融合场景本节简写“不涉及 Vector Epilogue”即可。

**Epilogue 基本信息**：

| 项 | 值 |
|----|----|
| 是否融合场景 | 是 / 否 |
| Epilogue 类型 | 无 / RegBase / MemBase / 自定义 |
| 开发路径 | blaze_custom / 不涉及 |
| Epilogue 文件 | `op_kernel/include/blaze_custom/epilogue/{epilogue_name}.h` / 不涉及 |
| 接口形式 | `Init` / `GetTensor` / `operator()` / 不涉及 |

> 普通 MatMul + Vector 融合使用 `blaze_custom` 路线，自定义 Epilogue 必须放在 `op_kernel/include/blaze_custom/epilogue/` 下，不要放在 `op_kernel/` 顶层。

**公式分析**：

| 步骤 | 操作 | 输入类型 | 输出类型 |
|------|------|---------|---------|
| ① | Cast | L0CType → ComputeType | |
| ② | | | |
| ③ | Cast | ComputeType → OutputType | |

**dtype 链路**：
```
L0CType ──Cast──▶ ComputeType ──[公式链]──▶ ComputeType ──Cast──▶ OutputType
```

**额外输入**：

| 输入 | shape | dtype | broadcast | LoadDist |
|------|-------|-------|-----------|----------|
| | | | | |

**UB 分区布局**：

| 分区顺序 | Buffer 名称 | 存储内容 | 数据类型 | 大小 / 行数 | 生命周期 | 说明 |
|---------|-------------|----------|----------|-------------|----------|------|
| 1 | `cLocal_` | L0C2UB 的 MatMul 结果 | `L0CType` | `matmulAreaBytes` 或 `splitMRows × nAlign` | 固定，不可释放 | 由 C 部分 Fixpipe 写入，Epilogue 消费 |
| 2 | `extraBufA_` | 额外输入 A（如 per-channel scale / bias） | `ComputeType` / 输入 dtype | `nAlign × sizeof(T)` 或按设计填写 | tile 内复用 / stage 内复用 | 列依赖输入通常按 tile 加载 |
| 3 | `extraBufB_` | 额外输入 B（如 per-token scale / row-dependent input） | `ComputeType` / 输入 dtype | `stageRows × ...` 或按设计填写 | stage 级 | 行依赖输入通常按 stage 加载 |
| 4 | `tmpBuf_` | 中间结果（仅 MemBase 需要） | `ComputeType` / `OutputType` | 按设计填写 | stage 级 | RegBase 一般不需要此块 |
| 5 | `outBuf_` | 输出 staging | `OutputType` | `stageRows × nAlignOut × sizeof(OutputType)` | stage 级 | `DataCopyPad` 写回 GM 前的暂存区 |

约束要求：

- `cLocal_` 优先级最高，必须先保证 L0C2UB 的 MatMul 结果完整落入 UB。
- RegBase 路径优先将中间值保留在 `RegTensor`，减少 `tmpBuf_` 占用。
- MemBase 路径应显式列出 `tmpBuf_` 是否存在及其大小。
- 所有依赖 M 维的额外输入，必须说明其是否按 stage 加载，以及 GM offset 是否受 SplitM / `GetSubBlockIdx()` 影响。

**UB 预算汇总**：

| 项 | 公式 / 数值 | 说明 |
|----|-------------|------|
| `nAlignL0C` | `ceil(baseN / (32/sizeof(L0CDataType))) * (32/sizeof(L0CDataType))` | UB 行对齐宽度，**不是** L0C cube 边长 16 |
| `splitMRows` | `ceil(baseM / GetTaskRation())` | 单个 AIV 的 MatMul 结果行数 |
| `matmulAreaBytes` | `splitMRows * nAlignL0C * sizeof(L0CDataType)` | L0C2UB 结果区；行步长是 `nAlignL0C` |
| `extraBufBytes` | | 所有额外输入 staging 总和 |
| `tmpBufBytes` | | MemBase 中间结果区；RegBase 可填 `0` |
| `outBufBytes` | | 输出 staging |
| `remainBytes` | `TOTAL_UB_SIZE - matmulAreaBytes` | 供 Epilogue 使用的剩余空间 |
| `totalEpilogueBytes` | `extraBufBytes + tmpBufBytes + outBufBytes` | Epilogue 总占用 |
| 可行性结论 | `totalEpilogueBytes <= remainBytes` / 否 | 是否满足 UB 预算 |

**同步指令清单**：

| 同步方向 | 指令 | 用途 | 必须性 |
|----------|------|------|--------|
| AIC → AIV | `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIC 通知 AIV 可以消费 UB 中 MatMul 结果 | C+V 必需 |
| AIV → AIC | `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIV 通知 AIC 可以继续覆盖 UB | C+V 必需 |
| MTE2 → V | `SetFlag/WaitFlag<HardEvent::MTE2_V>` | 额外输入搬入完成后再开始 Vector 计算 | 必需 |
| V → MTE2 | `SetFlag/WaitFlag<HardEvent::V_MTE2>` | Vector 用完输入 buffer 后允许下一轮覆盖 | 按 stage 复用时必需 |
| V → MTE3 | `SetFlag/WaitFlag<HardEvent::V_MTE3>` | Vector 计算完成后才能启动写回 | 必需 |
| MTE3 → V | `SetFlag/WaitFlag<HardEvent::MTE3_V>` | 写回结束后允许下一轮 Vector 或 buffer 复用 | 多轮 stage 时必需 |

**CopyL0C2UB Trait 选择**：

| Trait | 适用场景 | UB 数据分布 |
|-------|---------|------------|
| `CopyL0C2UBSplitMTrait`（`DUAL_DST_SPLIT_M`） | `__mix__(1,2)` 标准 C+V | M 对半分片，各 AIV 从各自 UB offset 0 读取 |
| `CopyL0C2UBNonSplitTrait`（`DUAL_DST_DISABLE`） | 单 AIV 调试 | 全量数据在 UB offset 0 |

> `__mix__(1,2)` C+V 场景必须使用 `CopyL0C2UBSplitMTrait`。`matmul_block_mmad.h` 参考模板已默认使用 SplitMTrait。

**SplitM 偏移参考表**：

| 操作 | 是否需要 SubBlock 偏移 | 公式 |
|------|----------------------|------|
| UB 读取 cLocal_ | **否** | `cLocal_.GetPhyAddr() + row * nAlign` |
| GM 读取 row-dependent input | 是 | `stageM0 = tileM0 + GetSubBlockIdx() * halfM + stageRowOffset` |
| GM 写回 output | 是 | `gmRowOffset = subM0 * N + tileN0` |

> **关键**：GM 侧 offset 需要加 sub-block 偏移，UB 侧不需要。`DUAL_DST_SPLIT_M` 硬件已自动分片，UB 读取从 offset 0 开始。localRows=0 时（如 curM=1 时 V1）early return 即可，CV 同步由 kernel 层处理。

**DataCopyPad stride 单位**：

| 方向 | srcStride | dstStride |
|------|-----------|-----------|
| GM → UB | bytes | 32 字节单位 |
| UB → GM | 32 字节单位 | bytes |

> 当 UB 行按 `nAlign` 对齐排布时（`nAlign * sizeof(T)` 是 32 的倍数），UB 侧 stride 恒等于 0，直接传 `0`。

**Epilogue 实现伪代码（必须含同步指令）**：

> 本节必须给出可实现级伪代码，而不只是数学公式。伪代码至少应覆盖：
> - `Init` / `GetTensor` / `operator()` 三接口的职责
> - SplitM / `GetTaskRation()` / `GetSubBlockIdx()` 对行数和 offset 的影响
> - extra input 的 GM offset 与加载顺序
> - `MTE2 / V / MTE3` 的同步指令
> - 输出写回顺序
> - tail tile / ODD-M / ODD-N 的处理要点

```cpp
Init(params, baseM, baseN, problemShape):
    // 1. 绑定额外输入和输出 GM 地址
    // 2. 计算 cLocal_ / extraBuf / tmpBuf / outBuf 的 UB 偏移
    //    matmulAreaBytes = splitMRows * nAlignL0C * sizeof(L0CDataType)
    //    （行步长是 nAlignL0C，不是 L0C cube 边长 16）
    // 3. 预发射首轮反向依赖：
    //    SetFlag<V_MTE2>(eventID) — 每个 stage 级 extra input buffer 各一个
    //    SetFlag<MTE3_V>(0) — output buffer
    //    注意：tile 级与 stage 级 extra input 使用不同 eventID（如 0 和 1）

GetTensor():
    // 返回 L0C2UB 的目标 UB Tensor（通常为 cLocal_）

operator()(blockShape, gmOffset, flagId):
    // 1. 计算 SplitM 行数：
    //    halfM = ceilDiv(curM, GetTaskRation())
    //    localRows = (curM odd) ? (halfM - GetSubBlockIdx()) : halfM
    //    若 localRows <= 0: return  // V1 无数据，CV 同步由 kernel 层处理
    // 2. 计算 GM offset（需要 sub-block 偏移）：
    //    subM0 = tileM0 + GetSubBlockIdx() * halfM
    // 3. UB 读取从 offset 0（不需要 sub-block 偏移，SplitM 已硬件分片）
    // 4. 按 stage 循环处理 localRows

    // ---- tile 级 extra input（加载一次，跨 stage 只读复用）----
    WaitFlag<V_MTE2>(tileEventID)
    DataCopyPad(extraBufA_, ...)  // GM→UB, dstStride=0 (UB 32B 对齐)
    SetFlag<MTE2_V>(tileEventID); WaitFlag<MTE2_V>(tileEventID)

    for each stage:
        // stage 级 extra input（每 stage 覆盖，GM offset 含 sub-block 偏移）
        WaitFlag<V_MTE2>(stageEventID)
        DataCopyPad(extraBufB_, ...)  // GM→UB, dstStride=0
        SetFlag<MTE2_V>(stageEventID); WaitFlag<MTE2_V>(stageEventID)

        WaitFlag<MTE3_V>(0)  // 等上一轮 MTE3 读完 outBuf_

        __VEC_SCOPE__ / Vector compute:
            // 从 cLocal_ offset 0 读取（SplitM 已硬件分片）
            // Load / Cast / [USER COMPUTE] / Store

        SetFlag<V_MTE2>(stageEventID)  // 通知可覆盖 extraBufB_

        SetFlag<V_MTE3>(0); WaitFlag<V_MTE3>(0)
        DataCopyPad(outputGM[subM0*N + tileN0 + stageOffset*N], ...)  // UB→GM, srcStride=0
        SetFlag<MTE3_V>(0)  // 通知可覆盖 outBuf_

    SetFlag<V_MTE2>(tileEventID)  // 通知可覆盖 extraBufA_

~Destructor():
    // 排空所有反向依赖的 WaitFlag
```

### 1.5 Tiling 设计

> 必须加载 `/ascendc-blaze-best-practice`，优先查阅 `references/tiling/tiling-selection.md`，结合当前场景查找可复用的 Blaze tiling 实现，并写明具体来源文件。禁止只写“使用 MatmulTilingSwat”而不说明文件和字段映射。

| 项目 | 值 |
|------|------|
| 场景 | 基础 MatMul / 普通 C+V / MX C+V / GroupMatMul |
| 复用 tiling 实现 | MatmulTilingSwat / MatmulTilingAFullLoad / QuantMatmulTilingSwat / Group... / 自定义派生 |
| 来源文件 | `/ascendc-blaze-best-practice` 中的具体文件 |
| 是否需要改造 | 是 / 否 |
| 改造原因 | dtype / layout / full-load / group / epilogue buffer / 其他 |
| TilingData 目标文件 | `op_tiling/{operator_name}_tiling_data.h` |
| Tiling 计算目标文件 | `op_tiling/{operator_name}_tiling.h` |
| baseM | |
| baseN | |
| baseK | |
| L1 depth (depthA1/depthB1) | |
| kL1 | |
| usedCoreNum | **强制动态计算** |
| dbL0c | |

**TilingData 字段规划**：

| 字段 | 类型 | 说明 |
|------|------|------|
| m, n, k | uint32_t | 矩阵维度 |
| baseM, baseN, baseK | uint32_t | tile 尺寸 |
| ... | | |

**Tiling 字段映射**：

| DESIGN 字段 | 对应 Blaze Params 字段 | 来源结构体/文件 | 说明 |
|-------------|------------------------|----------------|------|
| baseM | | | |
| baseN | | | |
| baseK | | | |
| kL1 | | | |
| usedCoreNum | | | |
| l0cDB | | | |

### 1.6 工程目录目标设计

> 本节基于 §1.3 组件组装、§1.4 Epilogue 设计和 §1.5 Tiling 设计推导最终工程目录结构。目标目录必须以 `/ascendc-direct-invoke-template` 的 Blaze 样例工程结构为基础，并结合本算子的开发场景确定，不要混用 Add/Vector 模板的 `op_host/` 结构。

#### 1.6.1 模板来源与开发场景

| 项 | 值 |
|----|----|
| 模板 skill | `/ascendc-direct-invoke-template` |
| 样例目录 | `references/matmul_blaze_template/` |
| 样例默认场景 | MX 量化 MatMul 示例 |
| 本算子场景 | 纯 MatMul / 普通 C+V 融合 / MX C+V / GroupMatMul |
| 目标开发路径 | blaze library / blaze_custom / 受控组合 |
| 是否需要自定义 Epilogue | 是 / 否 |
| 是否需要 PyTorch 接入 | 是 / 否 |

#### 1.6.2 目标目录结构

根据本算子实际选用的库和模块填写目标目录。普通 C+V 融合示例：

```text
operators/{operator_name}/
├── CMakeLists.txt
├── run.sh
├── README.md
├── {operator_name}.cpp
├── test_{operator_name}_torch.py
├── common/
│   ├── acl_utils.h
│   ├── common_utils.h
│   └── io_utils.h
├── op_kernel/
│   ├── {operator_name}_kernel.h
│   ├── {operator_name}_kernel.cpp
│   └── include/
│       ├── blaze/
│       ├── tensor_api/
│       └── blaze_custom/
│           ├── block/
│           ├── kernel/
│           ├── policy/
│           ├── utils/
│           └── epilogue/
│               └── {epilogue_name}.h
├── op_tiling/
│   ├── {operator_name}_tiling_data.h
│   └── {operator_name}_tiling.h
├── op_extension/
│   ├── {operator_name}_torch.cpp
│   ├── register.cpp
│   └── ops.h
└── scripts/
    ├── gen_data.py
    ├── golden.py
    └── verify_result.py
```

目录约束：

| 路径 | 约束 |
|------|------|
| `op_kernel/include/blaze/` | 外部 Blaze 依赖，可保留，不做业务改写 |
| `op_kernel/include/tensor_api/` | 外部 tensor_api 依赖，可保留，不做业务改写 |
| `op_kernel/include/blaze_custom/` | 自定义 Blaze 组件路径，只有选择 blaze_custom、C+V 融合或受控组合时使用 |
| `op_kernel/include/blaze_custom/epilogue/` | 普通 C+V 融合自定义 Epilogue 必须放在此处 |
| `op_kernel/` 顶层 | 只放目标 Kernel 入口/Wrapper，不放自定义 Epilogue 组件 |
| 工程根目录 | Blaze 样例 launcher 与 torch 测试在根目录改造，不引入 `op_host/` |
| `scripts/` | 只保留目标算子的输入生成、Golden 和验证脚本 |

#### 1.6.3 样例工程改造映射

复制 `references/matmul_blaze_template/` 后，必须按下表将样例业务文件改造成目标算子文件。不能仅从 CMake 中移除后保留未使用的样例业务文件。

| 样例文件/目录 | 本算子处理方式 | 目标文件/目录 | 依据 |
|---------------|----------------|---------------|------|
| `test_matmul_blaze.cpp` | 重命名并改写 / 删除 | `{operator_name}.cpp` | 根目录 launcher |
| `test_matmul_blaze_torch.py` | 重命名并改写 / 删除 | `test_{operator_name}_torch.py` | PyTorch 测试入口 |
| `op_kernel/matmul_blaze_kernel.h` | 重命名并改写 / 删除 | `op_kernel/{operator_name}_kernel.h` | Kernel 类型链 |
| `op_kernel/matmul_blaze_kernel.cpp` | 重命名并改写 / 删除 | `op_kernel/{operator_name}_kernel.cpp` | Kernel wrapper |
| `op_extension/matmul_blaze_torch.cpp` | 重命名并改写 / 删除 | `op_extension/{operator_name}_torch.cpp` | PyTorch extension |
| `op_extension/register.cpp` | 改写后保留 / 删除 | `op_extension/register.cpp` | 注册目标算子 |
| `op_extension/ops.h` | 改写后保留 / 删除 | `op_extension/ops.h` | 声明目标算子 |
| `op_tiling/matmul_tiling_data.h` | 重命名并改写 / 删除 | `op_tiling/{operator_name}_tiling_data.h` | TilingData |
| `op_tiling/matmul_tiling_stub.h` | 改写为正式 tiling / 删除 | `op_tiling/{operator_name}_tiling.h` | mock tiling 不进入正式实现 |
| `scripts/gen_data_mxfp8.py` | 改写 / 删除 | `scripts/gen_data.py` | 非 MX 场景不得保留 MX 语义 |
| `scripts/gen_data_mxfp4.py` | 改写 / 删除 | `scripts/gen_data.py` 或删除 | 非 FP4 场景不得保留 |
| `scripts/verify_result.py` | 改写后保留 | `scripts/verify_result.py` | 精度验证 |
| `common/` | 保留 | `common/` | ACL/IO 工具 |
| `op_kernel/include/blaze/` | 保留 | `op_kernel/include/blaze/` | 外部依赖 |
| `op_kernel/include/tensor_api/` | 保留 | `op_kernel/include/tensor_api/` | 外部依赖 |
| `op_kernel/include/blaze_custom/` | 保留并扩展 / 删除 | `op_kernel/include/blaze_custom/` | 自定义组件 |

#### 1.6.4 CMake target 源文件设计

| Target | 源文件 |
|--------|--------|
| `{operator_name}` | `{operator_name}.cpp`; `op_kernel/{operator_name}_kernel.cpp` |
| `{operator_name}_ops` | `op_kernel/{operator_name}_kernel.cpp`; `op_extension/{operator_name}_torch.cpp`; `op_extension/register.cpp` |

未列入上述 target 的样例业务源文件不应保留在工程中；`blaze/`、`tensor_api/`、`blaze_custom/` 公共头文件除外。

---

## 2. 确认清单

- [ ] 数据流已明确（§1.2）
- [ ] Kernel 入口签名已明确（§1.3）
- [ ] 开发路径已决策：blaze_custom / blaze 库（§1.3）
- [ ] 各层组装组件已选定（§1.3）
- [ ] 融合场景：UB 分区已规划（§1.4）
- [ ] Tiling 引擎已选择（§1.5）
- [ ] 工程目录目标和样例改造映射已明确（§1.6）
- [ ] 分支场景已覆盖

### C+V 融合专项确认

- [ ] CopyL0C2UB Trait 已选为 `CopyL0C2UBSplitMTrait`（`DUAL_DST_SPLIT_M`）
- [ ] UB 读取 cLocal_ 不含 sub-block 偏移（SplitM 已硬件分片）
- [ ] `matmulAreaBytes` 行步长为 `nAlignL0C`（非 L0C cube 边长 16）
- [ ] DataCopyPad UB 侧 stride 传 0（nAlign 保证 32B 对齐），GM 侧 stride 传 bytes
- [ ] tile 级与 stage 级 extra input 使用不同 eventID（避免假依赖）
- [ ] Init 预发射所有反向 SetFlag，析构排空所有 WaitFlag
- [ ] `localRows=0` 边界场景已处理（early return，CV 同步由 kernel 层处理）
- [ ] per-call `nAlign` 从 `blockShapeN` 计算（非 Init 时 `baseN`，正确处理 tail tile）
