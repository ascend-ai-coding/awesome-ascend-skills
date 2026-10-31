# Ascend C Kernel 侧编码规范

<适用>
语言: C++
侧别: Kernel
领域: true
触发: AscendC::, pipe.InitBuffer, DataCopy, DataCopyPad, AllocTensor, FreeTensor, CrossCoreSetFlag, CrossCoreWaitFlag, SetFlag, WaitFlag, EnQue, DeQue, PipeBarrier, ASCENDC_TPL_ARGS_DECL, ASCENDC_TPL_DTYPE_DECL, ASCENDC_TPL_DATATYPE_DECL, TILING_KEY_IS, _apt, __VEC_SCOPE__, asc_vf_call, __simd_vf__, __simt_vf__
默认启用: true
适用场景: ascendc kernel 侧编码规范（KERNEL-1~5 API 使用与硬件约定 + KERNEL-6~8 TilingKey 注册/Kernel 派发/命名约定 + KERNEL-9 核内同步推荐 Lock/Unlock + KERNEL-10 SetFlag/WaitFlag 配对 + KERNEL-11 禁用 PipeBarrier<PIPE_ALL> + KERNEL-12 VF 函数实现方式 + KERNEL-13 VF 函数内禁分支）
介绍: Kernel 侧 AscendC API 使用与工程约定规范（数据搬运推荐 DataCopyPad/内存管理配对/API 参数限制/核间与核内同步配对/硬件参数 + TilingKey 注册/Kernel 派发/命名约定 + VF 编程），由 ascendc-api.md 与 ascendc-perf.md 合并而来，另自 ascendc-op-conventions.md 迁入 3 条（重编号 KERNEL-6~8）；日落检查独立于 ascendc-sunset-api.md
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

> **适用范围**：Kernel 侧（Device 侧）
>
> **Host 侧不适用**：Host 侧代码（tiling/图推断/aclnn）不使用 Kernel 侧 Ascend C API。
>
> **合并说明**：本文件由 ascendc-api.md（API 使用类）、ascendc-perf.md（PERF-1/2/4、PREC-1/3 → 性能/精度类）及 ascendc-op-conventions.md 条款 1/2/4（→ TilingKey 注册/Kernel 派发/命名类，条款 1 侧别标记改 [适用: Kernel]——其检查对象 op_kernel/**/*tiling_key*.h 属 Kernel 侧）合并而来；perf 其余条款已废弃。合并时删除 2 条 API 条款（原 KERNEL-5/11），2026-09-16 重排编号消除空洞：旧 KERNEL-6~10 → KERNEL-5~9、旧 KERNEL-12 → KERNEL-10、旧 KERNEL-13~17 → KERNEL-11~15、旧 KERNEL-18~20 → KERNEL-16~18，历史引用按此映射换算。2026-09-28 删除 10 条（原 KERNEL-1/2/4/6/8/9/11/13/14/15：GlobalTensor::SetValue/GetValue 禁用、std:: 计算函数禁用、Compare 256 字节对齐、动态内存分配、Cast RoundMode、blockLen 单位差异、循环内逐元素操作、PipeBarrier、流水线同步、FP16 中间精度保护），并重排编号消除空洞：旧 KERNEL-3→1、5→2、7→3、10→4、12→5、16→6、17→7、18→8、19→9、20→10、21→11，历史引用按此映射换算。同日 KERNEL-4 拆分：核内 SetFlag/WaitFlag 配对检查独立成条（新增 KERNEL-10），KERNEL-4 全名化仅管核间 CrossCore* 配对，原 KERNEL-10/11 顺延为 KERNEL-11/12，历史引用按此换算。同日新增 KERNEL-11：禁止使用 PipeBarrier<PIPE_ALL>（自原已删 KERNEL-13 的 PIPE_ALL 部分提炼，依据官方 PipeBarrier 文档约束与 PR#0309/PR#3019 真实案例），原 KERNEL-11/12 顺延为 KERNEL-12/13，历史引用按此换算。

## 检视前置要求

> **重要**：Agent 在检视本条例时，必须先使用 `ascendc-docs-search` skill 获取相关 API 的最新文档，确认 API 参数、限制、对齐要求等信息是否与条例描述一致。若文档与条例有差异，以最新 API 文档为准，并记录差异供后续更新条例。

### 需要查阅文档的 API

| 条例编号 | 涉及 API | 查阅重点 |
|---------|---------|---------|
| KERNEL-1 | `DataCopy`, `DataCopyPad` | 确认 DataCopy 对齐限制与 blockLen 单位、DataCopyPad 参数与 pad 机制 |
| KERNEL-2 | `AllocTensor`, `FreeTensor` | 确认内存管理配对要求 |
| KERNEL-3 | 向量计算类 API（高维切分形态） | 确认 repeatTime 参数类型与上限、count 形态可用性 |
| KERNEL-4 | `CrossCoreSetFlag`, `CrossCoreWaitFlag` | 确认同步配对规则和提前退出风险 |
| KERNEL-5 | 系统变量 API（`GetBlockNum`/`GetDataBlockSizeInBytes`/`GetRuntimeUBSize` 等）+ 目标代际硬件规格 | 确认代际核数/UB/L1/L2 值、GetRuntimeUBSize 等的代际支持范围 |
| KERNEL-9 | `SetFlag/WaitFlag`, `Mutex::Lock/Unlock` | 确认目标代际 Lock/Unlock 支持情况与 Mutex 约束 |
| KERNEL-10 | `SetFlag/WaitFlag` | 确认 HardEvent 模板参数与 eventID 配对约束、连续 SetFlag 限制 |
| KERNEL-11 | `PipeBarrier`（PIPE_ALL） | 确认 PIPE_ALL 性能约束与各流水类型（PIPE_V/PIPE_MTE2/PIPE_MTE3）适用场景 |
| KERNEL-12 | `__VEC_SCOPE__`, `__simd_vf__`/`__simt_vf__`, `asc_vf_call` | 确认目标代际 VF 限定符与 asc_vf_call 签名约束 |
| KERNEL-13 | 向量级 VF（`__simd_vf__`）体内分支（if/switch/?:） | 确认目标代际 VF 分支与硬件循环生成约束（仅 `__simd_vf__`，`__simt_vf__` 豁免） |

---

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| KERNEL-1 | 推荐使用 DataCopyPad，不推荐使用 DataCopy | 数据搬运 | 中 | [适用: Kernel] |
| KERNEL-2 | AllocTensor/FreeTensor 必须配对使用 | 内存管理 | 高 | [适用: Kernel] |
| KERNEL-3 | repeatTime 限制（≤255） | API限制 | 中 | [适用: Kernel] |
| KERNEL-4 | CrossCoreSetFlag/WaitFlag 必须对称 | 核间同步 | 高 | [适用: Kernel] |
| KERNEL-5 | 禁止写死硬件参数 | 性能优化 | 高 | [适用: Kernel] |
| KERNEL-6 | ASCENDC_TPL_ARGS_DECL 禁止包含算子数据类型 | TilingKey注册 | 高 | [适用: Kernel] |
| KERNEL-7 | Kernel 代码禁止使用 TILING_KEY_IS 宏 | Kernel派发 | 高 | [适用: Kernel] |
| KERNEL-8 | Kernel 文件禁止 _apt 后缀 | 命名规范 | 高 | [适用: Kernel] |
| KERNEL-9 | 核内同步推荐使用 Lock/Unlock，SetFlag/WaitFlag 不推荐使用 | 核内同步 | 中 | [适用: Kernel] |
| KERNEL-10 | SetFlag/WaitFlag 必须配对对称 | 核内同步 | 高 | [适用: Kernel] |
| KERNEL-11 | 禁止使用 PipeBarrier<PIPE_ALL> | 核内同步 | 中 | [适用: Kernel] |
| KERNEL-12 | 禁止使用 __VEC_SCOPE__，VF 任务须以具名 VF 函数 + asc_vf_call 实现 | API黑名单 | 高 | [适用: Kernel] |
| KERNEL-13 | 向量级 VF（__simd_vf__）实现内部禁止使用 if 分支判断 | 性能优化 | 高 | [适用: Kernel] |

---

## 检视检查清单

使用以下清单快速检查代码是否满足 API 最佳实践：

- [ ] **KERNEL-1**: 是否使用了 DataCopy？（不推荐——建议改用 DataCopyPad 自动处理对齐与尾块；非对齐属正确性缺陷）
- [ ] **KERNEL-2**: AllocTensor/FreeTensor 是否配对？分支/提前 return 路径是否都有 Free？
- [ ] **KERNEL-3**: 高维切分形态 API 的 repeatTime 是否可能超过 255？（优先改用 count 形态）
- [ ] **KERNEL-4**: CrossCoreSetFlag/WaitFlag 是否一一配对？所有代码路径（含提前 return）是否都有 SetFlag？是否与 Matmul 高阶 API 混用？同一 flagId 是否超过 15 次？
- [ ] **KERNEL-5**: 是否写死核数/UB/L1/L2/Cache 等代际可变硬件参数？（改用系统 API 或 TilingData 下发；DataBlock 32B 须具名）
- [ ] **KERNEL-9**: 目标代际是否为 950？若是，核内同步是否使用了 SetFlag/WaitFlag？（不推荐，建议迁移至 Lock/Unlock，非强制）
- [ ] **KERNEL-10**: 核内 SetFlag/WaitFlag 是否配对？（模板参数+eventID 一致、所有路径双向可达、不共用 eventID、无连续 SetFlag）
- [ ] **KERNEL-11**: 是否使用了 PipeBarrier<PIPE_ALL>？（禁止——改用细粒度 Barrier 或队列/事件同步）
- [ ] **KERNEL-12**: 是否使用了 __VEC_SCOPE__？（禁止——提取为 __simd_vf__/__simt_vf__ 具名函数并用 asc_vf_call 启动）
- [ ] **KERNEL-13**: 向量级 VF 函数（__simd_vf__，含 callee）实现内部是否有运行时 if/switch/三目分支？（__simt_vf__ 除外；禁止——用 if constexpr/外提/无分支计算替代）

---

## KERNEL-1: 推荐使用 DataCopyPad，不推荐使用 DataCopy

**严重级别**：中

### 问题描述

数据搬运（GM ↔ UB）推荐统一使用 `DataCopyPad`，**不推荐继续使用 `DataCopy`**：新增搬运代码一律使用 `DataCopyPad`；存量已正确对齐的 `DataCopy` 可保留，建议在重构或功能迭代时逐步迁移（非强制）。

不推荐 `DataCopy` 的机理（其参数约束天然易错）：

1. **硬性 32 字节对齐要求**：`DataCopy` 要求数据量必须 32 字节对齐，非对齐直接导致数据错误（无运行时保护）；`DataCopyPad` 通过 `padParams` 自动处理对齐与尾块填充
2. **blockLen 单位为 32 字节块数**：`DataCopy` 配套 `DataCopyParams` 的 `blockLen` 单位是 DataBlock（32 字节），而 `DataCopyPad` 配套 `DataCopyExtParams` 的 `blockLen` 单位是字节——同一结构体在不同 API 中语义不同，是最常见的混淆来源

`DataCopy` 的对齐要求（非对齐即数据错误）：

| 数据类型 | 对齐元素数 | 最小对齐字节数 |
|---------|-----------|--------------|
| half (2 bytes) | 16 | 32 |
| float (4 bytes) | 8 | 32 |
| int32_t (4 bytes) | 8 | 32 |

### 不推荐示例

```cpp
// ⚠️ 不推荐：DataCopy 需自行保证 32 字节对齐，且 blockLen 单位为 32 字节块数
AscendC::DataCopy(xLocal, xGm, 4);              // 4 个 float = 16 字节，非对齐 → 数据错误
DataCopyParams dp;
dp.blockLen = totalCount * sizeof(float) / 32;  // 单位为 32 字节块数，极易误写为字节数
dp.blockCount = 1;
DataCopy(dst, src, dp);
```

### 推荐示例

```cpp
// ✅ 推荐：统一使用 DataCopyPad，自动处理对齐，blockLen 单位为字节
AscendC::DataCopyExtParams copyParams;
copyParams.blockLen = cols * sizeof(float);     // 单位：字节，语义直观
copyParams.blockCount = dealRowCount;
AscendC::DataCopyPad(xLocal, xGm, copyParams);

// ✅ 搬入需要尾块 padding 时配套 padParams
AscendC::DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

### 检视方法

**步骤1 — 定位**：`grep -nE '\bDataCopy\s*\(' {file}`（词边界天然排除 `DataCopyPad`）。

**步骤2 — 范围过滤**：PR 模式仅报 diff 新增/修改的 `DataCopy` 调用（存量未动代码按 methodology 范围外规则不进正文）；文件检视模式全报。

**步骤3 — 意见输出**（FAIL + confidence HIGH，措辞为建议性）：
- problem_desc：使用 `DataCopy` 搬运数据——需硬性保证 32 字节对齐且 `blockLen` 单位为 32 字节块数，易错；推荐改用 `DataCopyPad`（自动处理对齐，单位为字节）
- fix_suggestion：按推荐示例给出 `DataCopyPad` 改写；若为存量已对齐代码，注明可保留、建议迭代时迁移
- **措辞升级**：若该调用数据量不保证 32 字节对齐（如尾块/非整块搬运），属正确性缺陷——意见中标注「非对齐将导致数据错误，须立即整改」

**步骤4 — 前置核实**：检视时通过 `/ascendc-docs-search` 核实 `DataCopy`/`DataCopyPad` 的参数约束与对齐要求（见本文件「检视前置要求」）。

---

## KERNEL-2: AllocTensor/FreeTensor 必须配对使用

**严重级别**：高

### 问题描述

使用队列管理内存时，AllocTensor 和 FreeTensor 必须配对调用，否则会导致内存泄漏。

### 错误示例

```cpp
// ❌ 错误1：AllocTensor 后忘记 FreeTensor
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
// ... 使用 xLocal
// 忘记 FreeTensor
```

```cpp
// ❌ 错误2：提前 return 绕过 FreeTensor
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
if (isTailBlock) {
    return;  // ❌ xLocal 未 Free，inQueueX 槽位泄漏，后续 AllocTensor 阻塞
}
AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);
inQueueX.FreeTensor(xLocal);
```

### 正确示例

```cpp
// ✅ 正确1：AllocTensor/FreeTensor 配对
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
// ... 使用 xLocal
inQueueX.FreeTensor(xLocal);
```

```cpp
// ✅ 正确2：所有退出路径先释放再退出
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
if (isTailBlock) {
    inQueueX.FreeTensor(xLocal);  // 提前退出前先 Free
    return;
}
AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);
inQueueX.FreeTensor(xLocal);
```

### 检视方法

**步骤1 — 定位**：`grep -n "AllocTensor\|FreeTensor" {file}`，记录每个调用点的行号、队列对象（如 `inQueueX`）与变量名（Alloc 的返回变量 / Free 的入参变量）。

**步骤2 — 范围过滤**：PR 模式仅保留 diff 新增/修改的调用点所在的函数/循环体（存量未动代码按 methodology 范围外规则不进正文）；文件检视模式全量进入步骤 3。

**步骤3 — 按队列分组核对**：将调用点按队列对象分组，核对同一循环体/阶段内 `AllocTensor → EnQue` 与 `DeQue → FreeTensor` 是否成组出现。以下两类均为**合法配对，勿误报**：
- 标准三段式流水线：Free 的是 **DeQue 返回的变量**（如 `xIn`），而非 Alloc 时的变量名（`xLocal`）——两者为同一 buffer 槽位
- 直接使用范式：`AllocTensor → 使用 → FreeTensor`，不经过 EnQue/DeQue

**步骤4 — 路径完备性检查**（核心，逐 Alloc 点执行）：从每个 `AllocTensor` 调用点沿代码向下，至其所在函数/循环体的**全部出口**（函数末尾、`return`、`break`、`continue`）逐路径核对：
1. **无绕过**：任何路径不得跳过该 tensor 的配对调用——提前退出路径漏 Free 泄漏队列槽位（后续 AllocTensor 阻塞），漏 EnQue 则对侧 DeQue 永久等待
2. **无重复释放**：同一路径不得对同一 tensor Free 两次（如分支内已 Free、分支外又统一 Free）
3. **队列一致**：FreeTensor 的队列对象必须与该 tensor 的 AllocTensor/DeQue 队列同一，跨队列 Free 属错误

**步骤5 — 意见输出**（FAIL + confidence HIGH）：
- problem_desc：指明具体缺陷路径与变量（如「xLocal（L120 Alloc）在 if (edgeCase) return（L128）路径下未 FreeTensor，inQueueX 槽位泄漏，后续 AllocTensor 将阻塞」）
- fix_suggestion：在提前退出路径补齐配对调用，或参照标准三段式流水线重构（CopyIn 段 Alloc+EnQue，使用段 DeQue+Free）

---

## KERNEL-3: repeatTime 限制（≤255）

**严重级别**：中

### 问题描述

向量计算类 API 的 **tensor 高维切分形态**（`mask + repeatTime + repeatParams`）中，`repeatTime` 参数类型为 `uint8_t`，**由于硬件限制不能超过 255**——矢量计算单元每次迭代读取连续 8 个 DataBlock（256Bytes）计算，repeatTime 表示迭代次数。

> **依据**（asc-devkit 官方文档 `SIMD-API/basic_api/memory_vector_compute/SIMD_compute/high_dim_split.md`）：「由于硬件限制，repeatTime不能超过255」。

### 受影响 API

| API/参数 | 类型 | 上限 | 是否受 255 限制 |
|---------|------|------|---------------|
| 向量计算类 API 高维切分形态（`Add/Sub/Mul/Div/Exp/Sqrt/Cast/Duplicate/Compare/GatherMask` 等）的 `repeatTime`/`repeatTimes` 参数 | `uint8_t` | 255 | ✅ 是 |
| `DataCopy/DataCopyPad` 的 `blockCount` | `uint16_t` | 4095（取值范围 [0, 4095]） | ❌ 否 |
| `DataCopy/DataCopyPad` 的 `blockLen` | `uint16_t` | 65535（单位 DataBlock 32B） | ❌ 否 |

### 推荐做法

优先使用**连续计算形态**（count 入参，`int32_t`），无 255 限制、无需手工分批：

```cpp
// ✅ 推荐：连续计算形态，count 为参与计算的元素个数（int32_t）
AscendC::Add<T>(dstLocal, src0Local, src1Local, count);
AscendC::Exp<T>(dstLocal, srcLocal, count);
```

### 处理方案

必须使用高维切分形态时，repeatTime > 255 需分批调用：

```cpp
// 分批处理
constexpr uint32_t MAX_REPEAT = 255;
uint32_t repeatTime = ...;
uint32_t fullBatches = repeatTime / MAX_REPEAT;
uint32_t remainder = repeatTime % MAX_REPEAT;

for (uint32_t i = 0; i < fullBatches; i++) {
    // 调用 API，repeatTime = MAX_REPEAT
}
if (remainder > 0) {
    // 调用 API，repeatTime = remainder
}
```

### 检视方法

**步骤1 — 定位**：`grep -nE 'repeatTime|RepeatParams' {file}`，定位高维切分形态调用（`mask, repeatTime, repeatParams` 签名）。

**步骤2 — 范围过滤**：PR 模式仅报 diff 新增/修改的调用（存量未动代码按 methodology 范围外规则不进正文）；文件检视模式全报。

**步骤3 — 判定**：
1. repeatTime 实参是否可能超过 255（如由 `CeilDiv(totalCount, elementsPerRepeat)` 等计算得出，超过将按 `uint8_t` 截断，实际计算量错误）
2. 同一计算是否可直接改用连续计算形态（count 入参）替代
3. 排除不受限参数：`DataCopy/DataCopyPad` 的 blockCount/blockLen（[0,4095]/[0,65535]）不适用本条

**步骤4 — 意见输出**（FAIL + confidence HIGH）：
- problem_desc：repeatTime 超过 255（硬件限制），按 uint8_t 截断导致实际迭代次数错误
- fix_suggestion：给出分批调用示例；若可用连续计算形态则建议直接改写为 count 入参

---

## KERNEL-4: CrossCoreSetFlag/CrossCoreWaitFlag 必须配对对称

**严重级别**：高

### 问题描述

多核算子中，Cube 核和 Vec 核通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 进行跨核同步。若 CrossCoreSetFlag 和 CrossCoreWaitFlag 不一一对应，或存在某条代码路径跳过了 CrossCoreSetFlag，对应的 CrossCoreWaitFlag 将永久阻塞，导致死锁。

> **边界**：本条仅覆盖核间同步（`CrossCore*` API，Cube/Vec 跨核）；核内 `SetFlag`/`WaitFlag` 配对正确性归 KERNEL-10，核内同步 Lock/Unlock 迁移建议归 KERNEL-9。

> **官方约束**（8.5.0）：
> - `CrossCoreSetFlag` 内部已被 Matmul 高阶 API 使用，**禁止二者混用**，否则有 flagId 冲突风险
> - **同一 flagId 计数器最多设置 15 次**

### 核心原则

1. **每个 CrossCoreWaitFlag 必须有且仅有一个对称的 CrossCoreSetFlag**
2. **所有能到达 CrossCoreWaitFlag 的代码路径，都必须能触发对应的 CrossCoreSetFlag**
3. **提前 return、条件分支不能跳过 CrossCoreSetFlag**
4. **禁止与 Matmul 高阶 API 混用**：Matmul 高阶 API 内部使用了 CrossCoreSetFlag 进行核间同步控制，若开发者同时使用 CrossCoreSetFlag/WaitFlag 和 Matmul 高阶 API，会有 flagId 冲突的风险
5. **同一 flagId 计数器最多设置 15 次**：超过此限制行为未定义

### 同步配对示意

```
Cube 侧                          Vec 侧
─────────────────────────────────────────────────
ComputeMm1 末尾:
  CrossCoreSetFlag(syncC1V1)  →  ComputeVec1 开头:
                                   CrossCoreWaitFlag(syncC1V1)  ✅

ComputeVec1 末尾:
  CrossCoreSetFlag(syncV1C2)  →  ComputeMm2 开头:
  ←  CrossCoreWaitFlag(syncV1C2)                              ✅

ComputeMm2 末尾:
  CrossCoreSetFlag(syncC2V2)  →  ComputeVec2 开头:
                                   CrossCoreWaitFlag(syncC2V2)  ✅
```

### 错误示例

```cpp
// ❌ 危险：提前 return 跳过了 SetFlag
void ComputeMm1(...) {
    if (edgeCase) {
        return;  // Vec1 的 CrossCoreWaitFlag(syncC1V1) 永久等待 → 死锁！
    }
    // ... 正常计算 ...
    CrossCoreSetFlag(syncC1V1);  // 只有正常路径才执行
}
```

### 正确示例

```cpp
// ✅ 正确：所有路径都 SetFlag
void ComputeMm1(...) {
    if (edgeCase) {
        CrossCoreSetFlag(syncC1V1);  // 提前退出前必须 Set
        return;
    }
    // ... 正常计算 ...
    CrossCoreSetFlag(syncC1V1);
}
```

### 双缓冲 loop 偏移下溢风险

使用双缓冲时，常见 `(loop - 1) % N` 的计算。当 `loop` 为 `uint32_t` 且等于 0 时，`loop - 1` 会下溢为 `UINT32_MAX`，导致错误的 buffer 索引。

```cpp
// ❌ 危险：loop=0 时 uint32_t 下溢
uint32_t inIdx = (info.loop - 1) % preLoadNum;

// ✅ 正确：首次迭代单独处理
if (info.isFirstIteration) {
    // 使用默认初始值，不读上一轮
} else {
    uint32_t inIdx = (info.loop - 1) % preLoadNum;  // loop >= 1，安全
}
```

### 检视方法

**步骤1 — 定位**：`grep -nE 'CrossCore(Set|Wait)Flag' {file}`，记录行号、flag 变量名与所在函数（核间同步通常分处 Cube/Vec 两侧文件或函数）。

**步骤2 — 范围过滤**：PR 模式仅报 diff 新增/修改的调用（存量未动代码按 methodology 范围外规则不进正文）；文件检视模式全报。

**步骤3 — 配对核对**：
1. 列出所有 `CrossCoreSetFlag` 和 `CrossCoreWaitFlag` 调用，按 flag 变量名配对
2. 对每个 `CrossCoreSetFlag`，向上追踪是否存在 `if/return/break` 绕过它的路径
3. 检查是否有 `(loop - k) % N` 形式的 buffer 索引，确认 `loop < k` 时有保护分支
4. 检查是否同时调用了 Matmul 高阶 API（如 `Matmul`、`MatmulSimple`）
5. 统计同一 flagId 的 `CrossCoreSetFlag` 调用次数，确认不超过 15 次

**步骤4 — 意见输出**（FAIL + confidence HIGH）：
- problem_desc：指明不配对的 flag 变量、绕过路径与行号及后果（对侧 CrossCoreWaitFlag 永久阻塞 → 死锁）
- fix_suggestion：在所有提前退出路径补齐 CrossCoreSetFlag；Matmul 混用/flagId 超限场景给出整改说明

---

## KERNEL-5: 禁止写死硬件参数

**严重级别**：高

### 问题描述

核数、UB/Cache 等硬件参数随芯片代际变化，Kernel 侧写死会导致代码不可移植、跨代际运行出错。**代际可变参数禁止写死**；架构不变量（如 DataBlock 32B）也应以具名常量或系统 API 引用，禁止裸数字参与对齐计算。

### 硬件参数定义

| 类别 | 典型标识符 | 判定 |
|------|-----------|------|
| 核数 | coreNum/core_num、aicNum/aic_num、aivNum/aiv_num、blockDim（核数语义） | 代际可变，禁止写死 |
| UB 大小 | ubSize/UB_SIZE、ubSizeBytes、localMemSize/local_mem_size | 代际可变，禁止写死 |
| Block 粒度 | blockSize/BLOCK_SIZE（DataBlock 32B）、对齐字节数 | 架构不变量——须具名常量或 `GetDataBlockSizeInBytes()`，禁止裸 32 |
| 片上缓存 | l0aSize/l0bSize/l0cSize、l1Size/l1_size、l2Size、cacheSize | 代际可变，禁止写死 |

> **合法来源**：① Kernel 系统变量 API——`GetBlockNum()`（核总数）、`GetBlockIdx()`、`GetDataBlockSizeInBytes()`（DataBlock 大小）、`GetRuntimeUBSize()`/`GetUBSizeInBytes()`（UB 大小，**仅 950 系列支持**，其余代际返回 0）、`GetArchVersion()`；② TilingData 字段下发（Host 侧经 GetPlatformInfo 计算传递，其获取与校验归 HOST-3/HOST-4）。

### 错误示例

```cpp
// ❌ 错误1：写死核数（代际可变）
uint32_t blockDim = 24;                     // Ascend910B 核数，其他代际错误

// ❌ 错误2：写死 UB 大小（代际可变）
constexpr uint32_t UB_SIZE = 192 * 1024;    // 910B UB 规格，950 系列不同

// ❌ 错误3：写死 Cache/L1 大小（代际可变）
constexpr uint32_t L1_SIZE = 512 * 1024;

// ❌ 错误4：裸 32 参与 Block 对齐计算（架构不变量也不得裸写）
uint32_t blockNum = totalBytes / 32;
```

### 正确示例

```cpp
// ✅ 核数：系统 API 动态获取
uint32_t blockDim = static_cast<uint32_t>(AscendC::GetBlockNum());

// ✅ Block 粒度：系统 API 或具名常量
constexpr uint32_t BLOCK_SIZE = AscendC::GetDataBlockSizeInBytes();  // DataBlock 32B

// ✅ UB 大小：950 系列用系统 API（官方示例形态，按 UB 容量计算 tile）
if (totalLength > AscendC::GetRuntimeUBSize() / sizeof(half) / 2) {
    tileLength = AscendC::GetRuntimeUBSize() / sizeof(half) / 2;
} else {
    tileLength = totalLength;
}
// 其余代际：由 Host 侧经 GetPlatformInfo 计算后写入 TilingData 字段下发（见 HOST-3/HOST-4）
uint32_t ubSize = tilingData.ubSize;
```

### 检视方法

**步骤1 — 候选定位（保召回）**：
- 名称信号：`grep -inE 'core[_]?num|aic[_]?num|aiv[_]?num|ub[_]?(size|len|bytes)|local.?mem|unified.?buf|block[_]?size|l0[abc]?[_]?size|l1[_]?size|l2[_]?size|cache[_]?size' {file}`
- 数值信号：`grep -nE '=\s*[0-9]+\s*\*\s*1024' {file}`（KB/MB 换算形态）；典型核数值（20/24/26/32/40/48/50 等）赋给核数语义变量
- Read 代码概要「常量清单」表（常量/值/行号/用途），与上述 grep 结果合并去重

**步骤2 — 代际值表比对（判别核心）**：从 API 预研报告/代码概要「芯片架构参数」表确认目标代际；缺失时用 `/npu-arch` skill 查询该代际核数、UB/L1/L2 大小、对齐要求。候选常量值（含 1024 换算）等于目标代际硬件值 → 强违规证据；等于其他代际规格值 → 同样违规（跨代际写死，不可移植）；与任何代际硬件规格无关 → 进入步骤 3 用途判定。

**步骤3 — 用途与来源判定（消误报）**：
- 违规：裸常量/宏直接用作 `pipe.InitBuffer` 大小、blockDim、循环切分边界、对齐计算，且数值命中硬件规格
- 豁免：值来自 `tilingData.*` 字段读取（概要「TilingData 值域溯源」表佐证）；来自系统 API 返回值；DataBlock 32B 的**具名**常量；`sizeof(T)` 参与的换算
- 边界排除：Host 侧 platform 参数获取与校验归 HOST-3/HOST-4；通用裸数字风格问题归 GEN-4.2（本条仅在数值命中硬件规格时报出）

**步骤4 — 意见输出**（FAIL + confidence HIGH）：
- problem_desc：指明参数类别（核数/UB/Block/Cache）、写死值与目标代际实际值的差异、不可移植影响
- fix_suggestion：三选一——改用系统 API（注明 GetRuntimeUBSize 等的代际支持范围）/ 经 TilingData 字段由 Host 下发 / 架构不变量改为具名 constexpr 并注释架构依据

---

## KERNEL-6: ASCENDC_TPL_ARGS_DECL 禁止包含算子数据类型 `[适用: Kernel]`

> **注意**：`ASCENDC_TPL_DTYPE_DECL` 定义的 **INT 索引类型**（如 INDEX_DTYPE: `TPL_INT32` / `TPL_INT64`）仍允许出现，本条只关注算子**运算数据类型**（fp16/bf16/fp32/int8 等）。

**严重级别**：高

**问题描述**

`ASCENDC_TPL_ARGS_DECL` 宏中直接使用 `ASCENDC_TPL_DTYPE_DECL` 或 `ASCENDC_TPL_DATATYPE_DECL` 声明算子数据类型（fp16、bf16、fp32 等），会使得数据类型的枚举值直接暴露为 TilingKey 枚举组合。这违反 TilingKey 不耦合算子语义类型的设计原则。正确做法是将 dtype 差异编码为整型 TilingKey（如 `schMode`），由 Host 侧 `ChooseTilingKey` 统一决策和派发。

**检查点**

| 序号 | 检查项 | 判断依据 |
|------|--------|---------|
| 1 | `ASCENDC_TPL_ARGS_DECL` 内是否出现 `ASCENDC_TPL_DTYPE_DECL` | 查找是否声明了 fp16/bf16/fp32 等运算数据类型枚举 |
| 2 | `ASCENDC_TPL_ARGS_DECL` 内是否出现 `ASCENDC_TPL_DATATYPE_DECL` | 查找是否通过 `C_DT_*` / `ASCENDC_TPL_INPUT(n)` 绑定了输入 dtype |
| 3 | `ASCENDC_TPL_SEL` 内是否使用 `ASCENDC_TPL_DTYPE_SEL` / `ASCENDC_TPL_DATATYPE_SEL` | 确认选择器未展开数据类型的枚举组合 |

**注意**：以下不视为违规：
- `ASCENDC_TPL_DTYPE_DECL` 用于 **索引/中间类型**（如 `INDEX_DTYPE: TPL_INT32, TPL_INT64`），属于合理用途

**反例**

```
optim/apply_rms_prop/op_kernel/arch35/apply_rms_prop_tiling_key.h
activation/relu6/op_kernel/arch35/relu6_tiling_key.h
activation/gelu_v2/op_kernel/arch35/gelu_v2_struct.h
optim/apply_adamax/op_kernel/arch35/apply_adamax_tiling_key.h
loss/sigmoid_cross_entropy_with_logits/op_kernel/arch35/sigmoid_cross_entropy_with_logits_struct.h
```

```cpp
// 反例1: ASCENDC_TPL_DATATYPE_DECL 直接声明 fp16/fp32/bf16 作为 TilingKey（relu6_tiling_key.h）
ASCENDC_TPL_ARGS_DECL(Relu6,
    ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_BF16, ASCENDC_TPL_INPUT(0))
);

// 反例2: ASCENDC_TPL_DTYPE_DECL 直接声明 fp16/bf16/fp32 作为 TilingKey（sigmoid_cross_entropy_with_logits_struct.h）
ASCENDC_TPL_ARGS_DECL(SigmoidCrossEntropyWithLogits,
    ASCENDC_TPL_DTYPE_DECL(dType, TPL_FP16, TPL_BF16, TPL_FP32)
);

// 反例3: ASCENDC_TPL_DATATYPE_DECL 绑定输入0的 dtype（apply_rms_prop_tiling_key.h）
ASCENDC_TPL_ARGS_DECL(ApplyRmsProp,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);
```

**正例**

```
pooling/adaptive_avg_pool2d/op_kernel/arch35/adaptive_avg_pool2d_struct.h
pooling/max_pool_grad/op_kernel/arch35/max_pool_grad_struct.h
pooling/avg_pool_v2_grad/op_kernel/arch35/avg_pool_v2_grad_tiling_key.h
pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_pool3d_tiling_struct.h
```

```cpp
// 正例1: 全部用 ASCENDC_TPL_UINT_DECL 整型编码，无 dtype 声明
// （pooling/adaptive_avg_pool2d/op_kernel/arch35/adaptive_avg_pool2d_struct.h）
#define TPL_INT32_UINT32 0
#define TPL_INT64_UINT64 1
#define TPL_SMALL_KERNEL 0
#define TPL_BIG_KERNEL 1
#define TPL_SIMT_KERNEL 2
ASCENDC_TPL_ARGS_DECL(AdaptiveAvgPool2d,
    ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 2, ASCENDC_TPL_UI_LIST, TPL_SMALL_KERNEL, TPL_BIG_KERNEL, TPL_SIMT_KERNEL),
    ASCENDC_TPL_UINT_DECL(DTYPE_MODE, 3, ASCENDC_TPL_UI_LIST, TPL_INT32_UINT32, TPL_INT64_UINT64),
    ASCENDC_TPL_UINT_DECL(NC_FACTOR, 1, ASCENDC_TPL_UI_LIST, TPL_NC_FACTOR_64, TPL_NC_FACTOR_128),
    ASCENDC_TPL_UINT_DECL(BIG_KERNEL_COPY_MODE, 1, ASCENDC_TPL_UI_LIST, TPL_BIG_KERNEL_NDDMA, TPL_BIG_KERNEL_COPYPAD),
);

// 正例2: 仅有 ASCENDC_TPL_UINT_DECL，无任何 dtype 声明（max_pool_grad_struct.h）
// op_kernel/arch35/max_pool_grad_struct.h ✅ 全部用 UINT
```

**检视方法**

```bash
# 1. 在 tiling_key 文件中查找 ASCENDC_TPL_DTYPE_DECL 或 ASCENDC_TPL_DATATYPE_DECL
grep -n "ASCENDC_TPL_DTYPE_DECL\|ASCENDC_TPL_DATATYPE_DECL" op_kernel/**/*tiling_key*.h op_kernel/**/*struct*.h 2>/dev/null

# 2. 对命中条目逐一判断是否属于算子运算数据类型（排除 INDEX_DTYPE 等索引类型）
# 3. 同时检查 ASCENDC_TPL_SEL 中是否有对应的 ASCENDC_TPL_DTYPE_SEL / ASCENDC_TPL_DATATYPE_SEL
```

**逐步检视**：
1. 在 `tiling_key.h` / `struct.h` 中找到所有 `ASCENDC_TPL_ARGS_DECL` 宏
2. 检查其参数列表中是否有 `ASCENDC_TPL_DTYPE_DECL(...)` 或 `ASCENDC_TPL_DATATYPE_DECL(...)`
3. 对命中的条目判断：声明的类型值是否包含 fp16/bf16/fp32/int8 等**算子运算数据类型**——若是则为违规
4. 排除 `INDEX_DTYPE: TPL_INT32, TPL_INT64` 等索引/中间类型定义（非算子运算数据类型）

---

## KERNEL-7: Kernel 代码禁止使用 TILING_KEY_IS 宏 `[适用: Kernel]`

**严重级别**：高

**问题描述**

Kernel 入口函数（`op_kernel/*.cpp`）中使用 `TILING_KEY_IS(N)` 宏按硬编码的 TilingKey 数值分支选择不同实现路径，直接耦合了 Kernel 实现与具体 TilingKey 值。正确做法是通过模板参数（如 `uint32_t schMode`）利用 `if constexpr` 实现编译期分支派发，Kernel 逻辑不感知 TilingKey 具体数值。

**检查点**

| 序号 | 检查项 | 判断依据 |
|------|--------|---------|
| 1 | Kernel `.cpp` 中是否出现 `TILING_KEY_IS(` | 搜索 Kernel 入口函数中的 `TILING_KEY_IS` 调用 |
| 2 | 是否使用硬编码数值（如 `TILING_KEY_IS(0)`）或宏常量数值（如 `TILING_KEY_IS(800001)`） | 判断是否直接引用了具体 TilingKey 值 |

**反例**

```
pooling/max_pool_with_argmax/op_kernel/max_pool_with_argmax.cpp
pooling/max_pool_v3/op_kernel/max_pool_v3_apt.cpp
pooling/avg_pool_v2/op_kernel/avg_pool_v2_apt.cpp
quant/dynamic_quant_v2/op_kernel/dynamic_quant_v2.cpp
quant/flat_quant/op_kernel/flat_quant.cpp
rnn/single_layer_lstm_grad/op_kernel/single_layer_lstm_grad.cpp
```

```cpp
// 反例1: 大量硬编码宏常量的 TILING_KEY_IS 分支（max_pool_with_argmax.cpp）
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C 800001
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_PAD 800002
// ...
__global__ __aicore__ void max_pool_with_argmax(GM_ADDR x, GM_ADDR y, GM_ADDR argmax,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C)) {        // ❌
        // ...
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_PAD)) {  // ❌
        // ...
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_NANPROP)) {  // ❌
        // ...
    }
    // ... 更多 TILING_KEY_IS 分支
}

// 反例2: 在 Kernel 中使用 TILING_KEY_IS 按硬编码数值分支（quant/flat_quant.cpp）
__global__ __aicore__ void flat_quant(...)
{
    if (TILING_KEY_IS(1)) {                    // ❌
        // ...
    } else if (TILING_KEY_IS(2)) {             // ❌
        // ...
    } else if (TILING_KEY_IS(3)) {             // ❌
        // ...
    }
}

// 反例3: LSTM grad 中大量 TILING_KEY_IS 硬编码组合（rnn/single_layer_lstm_grad.cpp）
__global__ __aicore__ void single_layer_lstm_grad(...)
{
    if (TILING_KEY_IS(0)) {                    // ❌
    } else if (TILING_KEY_IS(1)) {             // ❌
    } else if (TILING_KEY_IS(10)) {            // ❌
    } // ... 更多分支
}
```

**正例**

```
pooling/max_pool_grad/op_kernel/max_pool_grad.cpp
conv/conv3d_transpose_v2/op_kernel/conv3d_transpose_v2.cpp
conv/quant_conv3d/op_kernel/quant_conv3d.cpp
pooling/max_pool3d_grad/op_kernel/max_pool3d_grad.cpp
conv/conv3d_backprop_input_v2/op_kernel/conv3d_backprop_input_v2.cpp
```

```cpp
// 正例1: 使用模板参数 + if constexpr 编译期分支（max_pool_grad.cpp）
// Kernel 通过模板参数 KERNEL_MODE/FORMAT/INDICES_DTYPE 做编译期分支，无 TILING_KEY_IS
template <
    uint64_t KERNEL_MODE = TPL_SIMT_KERNEL, uint64_t FORMAT = TPL_NCHW_FORMAT,
    uint64_t INDICES_DTYPE = TPL_INT32, uint64_t IS_CHECK_RANGE = TPL_NO_CHECK_RANGE>
__global__ __aicore__ void max_pool_grad(
    GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaxPoolGradWithArgmaxSimtTilingCommonData);

    if constexpr (KERNEL_MODE == TPL_SIMT_KERNEL) {            // ✅ 模板参数编译期分支
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxSimtTilingCommonData, tilingData, tiling);
        // ...
    }
}

// 正例2: 模板参数分支，无 TILING_KEY_IS（conv3d_transpose_v2.cpp）
__global__ __aicore__ void conv3d_transpose_v2(GM_ADDR x, GM_ADDR filter, GM_ADDR y, ...)
{
    // ...
    if constexpr (FORMAT_Y == FORMAT_NCDHW) {                 // ✅ 模板参数编译期分支
        // ...
    }
}

// 正例3: 模板参数直接传入，无 TILING_KEY_IS（quant_conv3d.cpp）
__global__ __aicore__ void quant_conv3d(GM_ADDR x, GM_ADDR filter, ...)
{
    // ...
    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {  // ✅
        // ...
    }
}
```

**检视方法**

```bash
# 1. 在 Kernel 源文件中查找 TILING_KEY_IS
grep -rn "TILING_KEY_IS(" op_kernel/*.cpp op_kernel/**/*.cpp 2>/dev/null

# 2. 确认命中文件为 Kernel 入口函数（包含 __global__ __aicore__ 的 .cpp）
# 3. 排除 op_host 目录下的文件（tiling 代码使用 TILING_KEY_IS 是合理的）
```

**逐步检视**：
1. 在 `op_kernel/` 目录下搜索 `TILING_KEY_IS(` 的所有出现
2. 排除 `op_host/` 目录下的文件（Host 侧允许使用）
3. 检查 Kernel 入口函数是否改用 `if constexpr (schMode == ...)` 基于模板参数编译期分支
4. 若 Kernel 仅做简单透传无分支，检查模板参数是否为 `uint32_t schMode` 或 `ASCENDC_TPL_KERNEL_TYPE`

---

## KERNEL-8: Kernel 文件禁止 _apt 后缀 `[适用: Kernel]`

> **排除**：ops-transformer 仓暂不遵守此规则。

**严重级别**：高

**问题描述**

`op_kernel/` 下的 Kernel 入口文件应使用标准命名 `xxx.cpp`，禁止添加 `_apt` 后缀（如 `xxx_apt.cpp`）。`_apt` 后缀源自 Adaptive Tiling Pattern 的旧写法，现已不推荐使用。Kernel 入口文件命名为 `{op_name}.cpp` 即可，无需额外后缀。

**检查点**

| 序号 | 检查项 | 判断依据 |
|------|--------|---------|
| 1 | `op_kernel/` 目录下是否存在 `*_apt.cpp` 文件 | 文件名为违规 |

**正例**

```
pooling/max_pool_grad/op_kernel/max_pool_grad.cpp
pooling/avg_pool3_d/op_kernel/avg_pool3_d.cpp
pooling/max_pool_with_argmax/op_kernel/max_pool_with_argmax.cpp
pooling/max_pool_grad_with_argmax/op_kernel/max_pool_grad_with_argmax.cpp
pooling/adaptive_max_pool3d/op_kernel/adaptive_max_pool3d.cpp
conv/quant_conv3d/op_kernel/quant_conv3d.cpp
```

```cpp
// 正例: 标准命名，无需 _apt 后缀
// pooling/avg_pool3_d/op_kernel/avg_pool3_d.cpp ✅
// pooling/max_pool_grad/op_kernel/max_pool_grad.cpp ✅
// pooling/max_pool_with_argmax/op_kernel/max_pool_with_argmax.cpp ✅
```

**反例**

```
pooling/avg_pool_v2/op_kernel/avg_pool_v2_apt.cpp
pooling/max_pool_v2/op_kernel/max_pool_v2_apt.cpp
pooling/max_pool_v3/op_kernel/max_pool_v3_apt.cpp
pooling/avg_pool/op_kernel/avg_pool_apt.cpp
pooling/avg_pool3_d/op_kernel/avg_pool3_d_apt.cpp
pooling/avg_pool_grad/op_kernel/avg_pool_grad_apt.cpp
pooling/adaptive_avg_pool2d/op_kernel/adaptive_avg_pool2d_apt.cpp
pooling/adaptive_max_pool3d/op_kernel/adaptive_max_pool3d_apt.cpp
conv/deformable_offsets/op_kernel/deformable_offsets_apt.cpp
quant/dynamic_quant_v2/op_kernel/dynamic_quant_v2_apt.cpp
```

```cpp
// 反例1: 文件名带 _apt 后缀（avg_pool_v2_apt.cpp）❌
// 文件路径: pooling/avg_pool_v2/op_kernel/avg_pool_v2_apt.cpp

// 反例2: 文件名带 _apt 后缀（avg_pool_apt.cpp）❌
// 文件路径: pooling/avg_pool/op_kernel/avg_pool_apt.cpp

// 反例3: 文件名带 _apt 后缀（avg_pool3_d_apt.cpp）❌
// 文件路径: pooling/avg_pool3_d/op_kernel/avg_pool3_d_apt.cpp
// 注意: 同目录下 avg_pool3_d.cpp 是正例 ✅
```

**检视方法**

```bash
# 1. 在 op_kernel 目录下搜索 _apt 后缀文件
find . -path '*/op_kernel/*_apt.cpp' ! -path '*/tests/*' 2>/dev/null

# 2. 排除 ops-transformer 仓的文件
```

**逐步检视**：
1. 在 `op_kernel/` 目录下搜索文件名含 `_apt.cpp` 的文件
2. 检查文件是否在 ops-transformer 仓中（ops-transformer 暂不适用此规则）
3. 对非 ops-transformer 的 `_apt.cpp` 文件标记为违规

---

## KERNEL-9: 核内同步推荐使用 Lock/Unlock，SetFlag/WaitFlag 不推荐使用 `[适用: Kernel]`

**严重级别**：中

### 问题描述

Ascend 950PR/950DT 起提供 Mutex 机制（`AscendC::Mutex::Lock` / `AscendC::Mutex::Unlock`，头文件 `basic_api/kernel_common.h`）实现核内多流水同步，**新代码推荐优先使用 Lock/Unlock**。`SetFlag/WaitFlag` 为 ISASI 类别接口（不能保证跨硬件版本兼容），**不推荐继续使用，但非禁止**——存量代码可保留，建议在重构或功能迭代时逐步迁移。

> **代际门控（重要）**：Lock/Unlock 仅 Ascend 950PR/950DT 支持（A3/A2/310B/310P/910 等均不支持）。仅当目标代际为 950 时本条报出迁移建议；其余代际 SetFlag/WaitFlag 为可用选择，本条 PASS 并注明「该代际不支持 Lock/Unlock，SetFlag/WaitFlag 可继续使用」。

**迁移价值**（SetFlag/WaitFlag 的典型陷阱）：

| 陷阱 | 来源 |
|------|------|
| 提前 return 路径漏 WaitFlag，event token 永久占用 | ops-transformer PR#4021 |
| 多事件共用同一 eventID，同步信号互相覆盖 | ops-transformer PR#8802 |
| SetFlag 与 WaitFlag 模板参数/事件 ID 不匹配引发 timeout | 官方约束 |
| 相同流水+相同 eventID 连续 SetFlag 属未定义行为，再执行 `PipeBarrier<PIPE_ALL>` 会卡死 | 官方约束 |

### 不推荐示例

```cpp
// ⚠️ 不推荐（仅 950 代际）：核内 MTE2→V 流水同步使用 SetFlag/WaitFlag
AscendC::DataCopy(xLocal, srcGm[tile * TILE_LEN], TILE_LEN);
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);   // ⚠️ 建议迁移至 Lock/Unlock
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
AscendC::Add(zLocal, xLocal, yLocal, TILE_LEN);
```

### 推荐示例

```cpp
// ✅ 950 代际：Mutex 三段流水同步（MTE2 搬入 → V 计算 → MTE3 搬出）
uint8_t mutexId = AscendC::AllocMutexID();   // TPipe-TQue 框架范式必须申请/释放
AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);    // 搬入段
AscendC::DataCopy(xLocal, srcGm[tile * TILE_LEN], TILE_LEN);
AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

AscendC::Mutex::Lock<PIPE_V>(mutexId);       // 计算段（等待对应搬入完成）
AscendC::Add(zLocal, xLocal, yLocal, TILE_LEN);
AscendC::Mutex::Unlock<PIPE_V>(mutexId);

AscendC::Mutex::Lock<PIPE_MTE3>(mutexId);    // 搬出段
AscendC::DataCopy(dstGm[tile * TILE_LEN], zLocal, TILE_LEN);
AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId);
AscendC::ReleaseMutexID(mutexId);
```

### 迁移注意事项（官方约束）

1. Lock 与 Unlock 必须严格成对使用，pipe 与 id 完全一致，Unlock 必须在 Lock 之后，否则未定义行为
2. 相同 id 的 Lock/Unlock 不得嵌套使用（无论 pipe 是否相同）
3. 相同 id+pipe 的两对连续 Lock/Unlock **不能**实现单流水内指令间同步——单流水内同步使用 `PipeBarrier`
4. MutexID 管理：TPipe-TQue 框架范式必须通过 `AllocMutexID`/`ReleaseMutexID` 申请释放；静态 Tensor 范式可自行管理（建议使用 0-27，28-31 系统预留）。自行管理时**不得**与 `TPipe`/`TBufPool`/`TQue`/`TQueBind` 的 InitBuffer/AllocTensor/FreeTensor/EnQue/DeQue 等资源管理接口同时使用（MutexID 冲突导致未定义行为）

### 检视方法

**步骤1 — 代际确认**：从 API 预研报告/代码概要「芯片架构参数」确认目标代际。非 950 代际 → 本条 PASS，注明「该代际不支持 Lock/Unlock，SetFlag/WaitFlag 可继续使用」。

**步骤2 — 定位**：`grep -nE '\b(SetFlag|WaitFlag)\s*<' {file}`（词边界天然排除 `CrossCoreSetFlag`/`CrossCoreWaitFlag`——核间同步归 KERNEL-4，本条不涉及）。

**步骤3 — 范围过滤**：PR 模式仅报 diff 新增/修改的 SetFlag/WaitFlag 调用（存量未动代码按 methodology 范围外规则不进正文）；文件检视模式全报。

**步骤4 — 意见输出**（FAIL + confidence HIGH，措辞为建议性）：
- problem_desc：核内同步使用 SetFlag/WaitFlag（ISASI 类别，不保证跨硬件兼容）；目标代际为 950，已提供 Lock/Unlock，**不推荐继续使用，建议迁移（非强制，存量代码可保留）**
- fix_suggestion：按推荐示例给出 Lock/Unlock 改写，并提示迁移注意事项 1-4
- **去重**：若该调用点已由 KERNEL-10 报出 SetFlag/WaitFlag 配对缺陷，本条跳过（整改缺陷时顺带迁移），避免同一位置重复意见

---

## KERNEL-10: SetFlag/WaitFlag 必须配对对称 `[适用: Kernel]`

**严重级别**：高

### 问题描述

核内多流水同步（如 MTE2→V、V→MTE3）使用 `SetFlag`/`WaitFlag`（`HardEvent` 模板参数 + eventID）时，二者必须严格配对对称，否则死锁、timeout 或未定义行为。

> **边界**：本条管核内 `SetFlag`/`WaitFlag` 配对**正确性**（全代际）；核间 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 配对归 KERNEL-4；950 代际 Lock/Unlock 迁移建议归 KERNEL-9。

**核心陷阱**：

| 陷阱 | 后果 | 来源 |
|------|------|------|
| 提前 return/break 路径漏 WaitFlag | event token 永久占用 | ops-transformer PR#4021 |
| 多事件共用同一 eventID | 同步信号互相覆盖 | ops-transformer PR#8802 |
| SetFlag 与 WaitFlag 模板参数（HardEvent）/eventID 不匹配 | timeout/死锁 | 官方约束 |
| 相同流水+相同 eventID 连续 SetFlag | 未定义行为（再执行 `PipeBarrier<PIPE_ALL>` 会卡死） | 官方约束 |

### 核心原则

1. **配对一致**：每个 `WaitFlag<HardEvent, id>` 必须有配对的 `SetFlag<HardEvent, id>`——模板参数与 eventID **完全一致**
2. **双向路径完备**：任何路径绕过 `SetFlag` 会使 `WaitFlag` 永久阻塞（死锁）；任何路径跳过 `WaitFlag` 会使 event token 永久占用
3. **eventID 独占**：多个同步事件不得共用同一 eventID（信号互相覆盖）
4. **禁连续 Set**：相同流水+相同 eventID 禁止连续两次 `SetFlag`（未定义行为）
5. **（次要）eventID 生命周期**：经 `AllocEventID` 申请的 eventID 须在同步完成后 `ReleaseEventID` 释放

### 错误示例

```cpp
// ❌ 错误1：提前 return 跳过 WaitFlag，event token 永久占用
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2);
if (edgeCase) {
    return;  // ❌ 未执行配对的 WaitFlag，event token 永久占用
}
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2);

// ❌ 错误2：多个同步事件共用同一 eventID，信号互相覆盖
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);   // 事件 A：数据 A 就绪
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
// ... 另一同步事件 B 复用同一 EVENT_ID0：
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);   // ❌ 与事件 A 共用 ID，配对关系被破坏

// ❌ 错误3：SetFlag 与 WaitFlag 模板参数不匹配
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);  // ❌ HardEvent 不匹配 → timeout
```

### 正确示例

```cpp
// ✅ 正确：Set/Wait 模板参数与 eventID 一致、所有路径成对，用毕释放 eventID
event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
AscendC::DataCopy(xLocal, srcGm[tile * TILE_LEN], TILE_LEN);
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
AscendC::Add(zLocal, xLocal, yLocal, TILE_LEN);
GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
```

### 检视方法

**步骤1 — 定位**：`grep -nE '\b(SetFlag|WaitFlag)\s*<' {file}`（词边界天然排除 `CrossCoreSetFlag`/`CrossCoreWaitFlag`——核间同步归 KERNEL-4）。

**步骤2 — 范围过滤**：PR 模式仅报 diff 新增/修改的 SetFlag/WaitFlag 调用（存量未动代码按 methodology 范围外规则不进正文）；文件检视模式全报。

**步骤3 — 配对核对**：
1. 按 `HardEvent` 模板参数 + eventID 将 Set/Wait 配对——模板参数或 eventID 不一致即 FAIL
2. 对每个 `WaitFlag` 向上追踪：是否存在绕过配对 `SetFlag` 的路径（if/return/break/continue）→ 死锁
3. 对每个 `SetFlag` 向下追踪：是否存在跳过配对 `WaitFlag` 的路径 → event token 永久占用
4. 检查多个同步事件是否共用同一 eventID → 信号互相覆盖
5. 检查相同流水+相同 eventID 是否连续 `SetFlag` 两次 → 未定义行为
6. （次要）`AllocEventID`/`ReleaseEventID` 是否配对释放

**步骤4 — 意见输出**（FAIL + confidence HIGH）：
- problem_desc：指明不配对的调用点、eventID、路径与行号及后果（死锁 / token 占用 / 信号覆盖 / UB）
- fix_suggestion：补齐配对调用、改用独立 eventID；950 代际可顺带提示迁移 Lock/Unlock（见 KERNEL-9）

---

## KERNEL-11: 禁止使用 PipeBarrier<PIPE_ALL> `[适用: Kernel]`

**严重级别**：中

### 问题描述

`PipeBarrier<PIPE_ALL>()` 会阻塞**所有流水线**中所有先前提交的指令，粒度过粗、打断流水并行、影响执行效率。禁止使用；应按依赖关系选择细粒度同步或队列机制。

> **官方约束**（asc-devkit `SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.md`）：「`PipeBarrier<PIPE_ALL>()`会等待所有流水线中所有先前提交的接口完成，这会对性能产生影响。若仅阻塞单条流水线即可解决问题，应避免随意调用`PipeBarrier<PIPE_ALL>()`。」

> **边界**：本条管 PIPE_ALL 粒度禁止（其他以 PIPE_ALL 为模板参数的同步调用同此原则）；核内同步 Lock/Unlock 迁移建议归 KERNEL-9；SetFlag/WaitFlag 配对归 KERNEL-10；核间 CrossCore 同步归 KERNEL-4。细粒度（如 `PIPE_V`）冗余 Barrier 属防御性编程，不告警。

### 替代方案

| 同步场景 | 正确做法 |
|---------|---------|
| 单流水内指令依赖（如连续 Vector 计算） | `PipeBarrier<PIPE_V>()`；开启 cce-auto-sync 编译选项时由编译器自动插入、950 系列 PIPE_V 间由硬件保序（无需手动） |
| DataCopy 目的地址重叠（GM） | `PipeBarrier<PIPE_MTE3>()` 串行化搬出流水 |
| DataCopy 目的地址重叠（UB） | `PipeBarrier<PIPE_MTE2>()` 串行化搬入流水 |
| 跨流水依赖（MTE2→V、V→MTE3） | EnQue/DeQue 队列同步或 SetFlag/WaitFlag（KERNEL-10）；950 代际可用 Mutex Lock/Unlock（KERNEL-9） |

### 错误示例

```cpp
// ❌ 禁止：本段仅 V 流水内依赖，PIPE_ALL 会阻塞所有流水
AscendC::Adds<T>(xLocal, xLocal, 1.0f, count);
AscendC::PipeBarrier<PIPE_ALL>();   // 应改为 PipeBarrier<PIPE_V>()
AscendC::Mul<T>(yLocal, xLocal, val, count);

// ❌ 禁止：DataCopyPad 后接 Cast，只需 MTE2→V 同步
DataCopyPad(castLocal, gradOutputGM[...], gParams, {false, 0, 0, 0});
AscendC::PipeBarrier<PIPE_ALL>();   // 打断其它流水，影响执行效率
Cast(...);
```

### 正确示例

```cpp
// ✅ 正确：细粒度同步（单流水内依赖）
AscendC::Adds<T>(xLocal, xLocal, 1.0f, count);
AscendC::PipeBarrier<PIPE_V>();     // 仅阻塞 V 流水
AscendC::Mul<T>(yLocal, xLocal, val, count);

// ✅ 正确：队列同步（跨流水依赖）
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
inQueueX.EnQue(xLocal);             // 标记就绪
auto xIn = inQueueX.DeQue<T>();     // 阻塞等待 MTE2 完成
```

### 真实检视案例（均被人工采纳）

案例1 — ops-transformer PR#0309，`op_kernel/rope_with_sin_cos_cache_f_bf16.h:631`：

> 评论：「建议此处不要使用 PipeBarrier<PIPE_ALL>()，应该使用 V 等 MTE2 同步指令。否则会打断其它流水，影响执行效率。」

```cpp
 631 |     PipeBarrier<PIPE_ALL>();   // ❌ 此处为 local tensor 间复制，只需 PIPE_V/PIPE_MTE2
```

案例2 — ops-math PR#3019，`op_kernel/*.h:268`：

> 评论：「PipeBarrier<PIPE_ALL> 同步冗余，会影响流水」

```cpp
 268 |     PipeBarrier<PIPE_ALL>();   // ❌ DataCopyPad 后接 Cast，只需 MTE2→V 同步
```

### 检视方法

**步骤1 — 定位**：`grep -n 'PIPE_ALL' {file}`，命中即存在 PIPE_ALL 粒度同步（以 `PipeBarrier<PIPE_ALL>()` 为主要形态）。

**步骤2 — 范围过滤**：PR 模式仅报 diff 新增/修改的调用（存量未动代码按 methodology 范围外规则不进正文）；文件检视模式全报。

**步骤3 — 判定**：命中即 FAIL，无场景豁免；确需全流水阻塞的极端场景，要求作者注释说明依据并提请人工复核。同时识别该位置的依赖类型（单流水内/地址重叠串行化/跨流水），选择对应替代方案。

**步骤4 — 意见输出**（FAIL + confidence HIGH）：
- problem_desc：使用 `PipeBarrier<PIPE_ALL>()` 阻塞所有流水（官方文档明示性能影响，应避免），打断流水并行、影响执行效率
- fix_suggestion：按替代方案表改写——单流水内依赖用细粒度 PipeBarrier（PIPE_V/PIPE_MTE2/PIPE_MTE3），跨流水依赖用 EnQue/DeQue 队列同步或事件/互斥锁同步

---

## KERNEL-12: 禁止使用 __VEC_SCOPE__，VF 任务必须以具名 VF 函数（__simd_vf__/__simt_vf__）+ asc_vf_call 实现 `[适用: Kernel]`

**严重级别**：高

### 问题描述

`__VEC_SCOPE__` 是旧的内联作用域写法（在调用处直接内联 VF 计算体），**现行官方文档（asc-devkit）函数限定符体系中已不存在该机制**，属于待淘汰写法，禁止使用。

VF（Vector Function）任务的正确实现方式为**具名 VF 函数 + asc_vf_call 启动**：

| 环节 | 正确用法 |
|------|---------|
| 定义 | 具名 VF 函数 + 限定符：`__simd_vf__`（向量级并行）/ `__simt_vf__`（线程级并行），返回类型必须为 void |
| 启动 | `asc_vf_call<funcName>(dim3 threadNums, args...)`（头文件 `simt_api/common_functions.h`，threadNums 总数 ≤ 2048，建议 32 的倍数） |

**与 SIMT-1-5 的边界**：SIMT-1-5（ascendc-simt.md）管旧 C++ 风格调用 `Simt::VF_CALL` → `asc_vf_call` 的转换；本条管 `__VEC_SCOPE__` 内联块 → 具名 VF 函数化。两者独立判定，同现可双报。

### 错误示例

```cpp
// ❌ __VEC_SCOPE__ 内联作用域（旧写法，现行官方文档已无此机制）
void Process(...)
{
    __VEC_SCOPE__
    {
        // VF 计算体直接内联在调用处
        ...
    }
}
```

### 正确示例

```cpp
// ✅ 定义具名 VF 函数（向量级 __simd_vf__ / 线程级 __simt_vf__）
__simd_vf__ inline void XxxVF(__ubuf__ int32_t* buf, int32_t colFactor)
{
    // VF 计算体
}

// ✅ 通过 asc_vf_call 启动
asc_vf_call<XxxVF>(dim3{THREAD_NUM, 1, 1}, buf, colFactor);
```

官方完整示例（线程级，asc-devkit 混合编程）：

```cpp
__simt_vf__ inline void add_simt(__gm__ float* dst, __gm__ float* src0, __gm__ float* src1)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = src0[idx] + src1[idx];
}

__global__ __vector__ void add_custom(__gm__ float* x, __gm__ float* y, __gm__ float* z)
{
    asc_vf_call<add_simt>(dim3{1024, 1, 1}, z, x, y);
}
```

### 真实检视案例

案例1（ops-transformer PR#2725）：

> 评论：「【严重】__VEC_SCOPE__ 当前已经不用了，需要整改」

案例2（ops-transformer PR#3124）：

> 评论：【一般】VF要整改，不能使用"  __VEC_SCOPE__"

案例3（ops-transformer PR#5291）：

> 评论：「VF函数建议统一使用 __simd_vf__ 实现」——作者回复：「将于下一笔PR整改」

案例4（ops-transformer PR#7092）：

> 评论：「建议：__VEC_SCOPE__ 替换为 __simd_vf__ 会比较符合现在的编程方式？」

### 注意事项

1. VF 函数约束（官方）：返回类型必须为 void；入参仅支持内置数据类型及其指针（指针必须指向 GM 或 UB，用 `__gm__`/`__ubuf__` 修饰）
2. `__launch_bounds__(N)` 可在 VF 入口函数上配置启动线程上限（N ∈ [1, 2048]），建议与启动时 dim3 线程数一致
3. 线程级 VF（`__simt_vf__`）内只能调用 `__simt_callee__`/`__callee__` 函数；向量级 VF（`__simd_vf__`）内只能调用 `__simd_callee__`/`__callee__` 函数

### 检视方法

**步骤1 — 定位**：`grep -n "__VEC_SCOPE__" {file}`，命中即存在旧写法。

**步骤2 — 范围过滤**：PR 模式仅报 diff 新增/修改的调用（存量未动代码按 methodology 范围外规则不进正文）；文件检视模式全报。

**步骤3 — 意见输出**（FAIL + confidence HIGH）：
- problem_desc：使用 __VEC_SCOPE__ 内联旧写法实现 VF 任务，现行官方文档已淘汰该机制
- fix_suggestion：按正确示例将计算体提取为具名 VF 函数（__simd_vf__/__simt_vf__），并通过 asc_vf_call 启动；VF 函数约束见注意事项

**步骤4 — 前置核实**：检视时通过 `/ascendc-docs-search` 核实目标代际的 VF 限定符与 asc_vf_call 签名（见本文件「检视前置要求」）。

---

## KERNEL-13: 向量级 VF 函数实现内部禁止使用 if 分支判断 `[适用: Kernel]`

**严重级别**：高

### 问题描述

**向量级** VF（Vector Function）函数实现内部（`__simd_vf__` 修饰的函数体，**含被内联进 VF 的 callee 子函数**）禁止使用运行时分支判断。机理：

- 分支判断引入 cmp（比较）指令，严重影响性能（ops-nn PR#4803）
- 含分支的循环体**无法生成硬件循环**，丧失编译器循环优化（ops-nn PR#2847）
- 部分场景直接不支持在 VF 内写 if 判断（ops-math PR#2868）

> **例外：`__simt_vf__`（线程级并行）VF 及其 callee 不受本条约束**。SIMT 按线程独立执行标量控制流（`int idx = threadIdx.x + blockIdx.x * blockDim.x;` 形态），线程块按固定规格（dim3）启动，实际数据边界必须由线程级分支守卫（如 `if (idx >= count) return;`）保护——线程级分支是 SIMT 编程模型的常规形态。本条机理（cmp 指令、硬件循环生成）仅针对向量级执行。

**禁止对象与豁免**：

| 分支形式 | 判定 |
|---------|------|
| `if (...)` / `if-else` | FAIL |
| `switch-case` | FAIL |
| 三目运算符 `?:`（同样生成 cmp/select 指令） | FAIL |
| `if constexpr`（编译期分支，不生成运行时 cmp，VF 模板类型分发的标准写法） | 豁免 |
| `#if` 等预处理指令（编译期） | 豁免 |
| `__simt_vf__` 修饰的 VF 函数及其 callee（线程级并行） | 豁免（本条仅约束向量级 `__simd_vf__`） |

### 错误示例

```cpp
// ❌ VF 函数体内使用运行时 if 分支
__simd_vf__ inline void XxxVF(__ubuf__ int32_t* buf, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        if (buf[i] > 0) {          // ❌ 引入 cmp 指令、无法生成硬件循环
            buf[i] = buf[i] * 2;
        } else {
            buf[i] = 0;
        }
    }
}

// ❌ callee 子函数内的运行时分支同样禁止（会被内联进 VF 体）
__simd_callee__ inline int32_t ClampVal(int32_t v)
{
    return v > 0 ? v : 0;          // ❌ 三目同样是运行时分支
}
```

### 正确示例

```cpp
// ✅ 1. if constexpr 编译期类型分发（豁免）
template <typename T>
__simd_vf__ inline void XxxVF(__ubuf__ T* buf, int32_t n)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        // half 路径（编译期展开，不生成运行时 cmp）
    } else {
        // float 路径
    }
}

// ✅ 2. 运行时分支移到 VF 外：调用方按条件选择不同 VF 入口
__simd_vf__ inline void XxxVFPos(__ubuf__ int32_t* buf, int32_t n) { /* 正数路径 */ }
__simd_vf__ inline void XxxVFOther(__ubuf__ int32_t* buf, int32_t n) { /* 其余路径 */ }
// 调用方（分支在 VF 外）：
if (cond) { asc_vf_call<XxxVFPos>(dim3{N, 1, 1}, buf, n); }
else      { asc_vf_call<XxxVFOther>(dim3{N, 1, 1}, buf, n); }

// ✅ 3. 数据级分支改为无分支计算（掩码/极值运算等，视算法而定）
```

### 真实检视案例

案例1（ops-nn PR#2847）：

> 评论：「vec_scope里面有ifelse会无法生成硬件循环」

案例2（ops-nn PR#4803）：

> 评论：「vf里的if操作严重影响性能，直接改成for循环即可」
> 评论：「建议VF内部不要使用if操作指令，会引入cmp指令，严重影响性能」

案例3（ops-math PR#2868）：

> 评论：「这个为啥拆开了，vf里面不支持写if判断」

### 注意事项

1. `if constexpr` 是 VF 模板函数类型分发的标准写法（编译期展开，不生成运行时 cmp），豁免——真实代码中广泛使用且未被检视批评
2. `__simd_vf__` 调用的 callee（`__simd_callee__`/`__callee__`）会被内联进 VF 体，其内部的运行时分支同样阻碍硬件循环生成，一并禁止；`__simt_vf__` 调用的 callee（`__simt_callee__`/`__callee__`）随 SIMT VF 一并豁免
3. 与 KERNEL-12 联动：旧 `__VEC_SCOPE__` 写法整体由 KERNEL-12 报；本条只管具名 VF 函数（及 callee）实现内部
4. 本条依据社区检视共识与代码生成机理（官方 API 文档未显式约束），检视时结合 `/ascendc-docs-search` 确认目标代际编译器行为

### 检视方法

**步骤1 — 定位 VF 实现范围**：`grep -n "__simd_vf__" {file}` 找**向量级** VF 函数起始行，按大括号配对确定函数体范围，并追踪其调用的 callee（`__simd_callee__`/`__callee__`）；`__simt_vf__`（线程级）及其 callee 不在本条范围（豁免）。

**步骤2 — 体内分支检索**：在函数体范围内检索 `if`、`switch`、`?:`；逐条排除豁免项：`if constexpr`（编译期）、`#if`（预处理）。

**步骤3 — 范围过滤**：PR 模式仅报 diff 新增/修改的行（存量未动代码按 methodology 范围外规则不进正文）；文件检视模式全报。

**步骤4 — 意见输出**（FAIL + confidence HIGH）：
- problem_desc：VF 函数（或其 callee 子函数）实现内部使用运行时分支判断，引入 cmp 指令且无法生成硬件循环
- fix_suggestion：按正确示例三选一——if constexpr 编译期分发 / 分支外提到调用方 / 无分支计算改写

---
