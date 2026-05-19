# MatMul 融合算子开发指南

> 适用范围：`mxfp8 matmul + eltwise`（Div/Mul/Add/Relu/Cast/激活函数等）。
> 本文档说明融合算子工程的架构设计理念、各层角色与职责、以及派生算子的开发适配步骤。

## 1. 整体架构

### 1.1 MIX 模式与数据流

融合算子采用 **MIX 模式**（AIC + AIV 共入口）。
单次 Kernel launch 同时调度 AIC 和 AIV 两类运算单元，通过 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 在循环体内分发。

```
A [M,K] mxfp8 ──┐
B [K,N] mxfp8 ──┤
                 ▼
           ┌─────────────────────────────────┐
           │  Kernel (matmul_kernel_fused.h) │  MIX 入口，tile 循环 + CV 协调
           │  AIC/AIV 统一循环                │
           ├───────────────┬─────────────────┤
           │  AIC (Cube)    │  AIV (Vector)    │
           │  Tensor API    │  ascendc API     │
           │                │                  │
           │ GM→L1(Nz/Zn)   │                  │
           │ L1→L0A/L0B     │ CrossCoreWait    │
           │ MMAD → L0C     │ ← Flag(AIC→AIV)  │
           │ Fixpipe→UB     │                  │
           │ CrossCoreSet   │ Eltwise(Div/…)   │
           │ Flag(AIC→AIV)  │ UB→GM            │
           │                │ CrossCoreSet     │
           │ CrossCoreWait  │ Flag(AIV→AIC)    │
           │ ← Flag(AIV→AIC)│                  │
           └───────────────┴─────────────────┘
                         ▼
               C [M,N] {OutputType}
```

**核心约束**：
- Matmul 部分**必须使用 Tensor API**（`Te::Copy` / `Te::Mad` / `Te::Fixpipe`）
- L0C → UB 采用 Fixpipe SPLIT_M 模式
- CV 同步由 Kernel 层统一管理（AIC↔AIV 的 CrossCoreSetFlag/WaitFlag）
- Epilogue 仅负责融合计算 + 写回 GM + 发送 AIV→AIC

### 1.2 内存层次

| 层次 | 位置 | 用途 | 管理方式 |
|------|------|------|---------|
| GM | Host ↔ Device | 输入 A/B/Scale/D，输出 C | aclrtMemcpyAsync |
| L1 | Cube 侧缓存 | A/B 矩阵滚动窗口 (depthA1/depthB1) | SWAT Tiling 引擎 |
| L0A/L0B | Cube 侧寄存器 | A/B 分块计算 (baseM×baseK / baseN×baseK) | Tensor API |
| L0C | Cube 侧累加器 | float 累加结果 (dbL0C × baseM × baseN) | Tensor API |
| UB | Vector 侧统一缓存 | matmul 结果 + Epilogue 输入/输出 | 静态偏移分配，禁止 TPipe |

> AIC L1/L0 由 SWAT Tiling 引擎自动管理，`[PATTERN]` 区域禁止业务修改。
> AIV UB 采用静态偏移分配——`cLocal_` 固定在 offset 0（matmul 结果），后续按 `stageSize_` 划分 Epilogue 缓冲区。

### 1.3 角色边界

| 层 | 文件 | 职责 |
| --- | --- | --- |
| Kernel | `include/kernel/matmul_kernel_fused.h` | tile 循环、AIC/AIV 分发、CrossCore 协调 |
| BlockMmad | `include/block/block_mmad_swat.h` | AIC 侧 MMAD 与 Fixpipe（L0C→UB） |
| Epilogue | `include/epilogue/*_epilogue.h` | AIV 侧融合计算、写回 GM、发送 `AIV→AIC` |

### 1.4 Tiling 基础概念

| 概念 | 含义 | 来源 |
|------|------|------|
| `baseM / baseN / baseK` | 单次 tile 的 Cube 计算尺寸 | SWAT Tiling 引擎 |
| `singleCoreM / singleCoreN` | 单核任务量 | SWAT Tiling 引擎 |
| `usedCoreNum` | 动态核数：`CeilDiv(M,singleCoreM) × CeilDiv(N,singleCoreN)` | Host 侧动态计算 |
| `depthA1 / depthB1` | L1 滚动深度（K 轴迭代次数） | SWAT Tiling 引擎 |
| 蛇形调度 | BlockScheduler 沿 M×N 二维蛇形分配 tile | `block_scheduler.h` |

> 详细的 Tiling 设计策略（多核切分、UB 切分、Buffer 规划、分支覆盖）见
> `/ascendc-tiling-design` → `references/matmul/patterns.md`。

---

## 2. 工程文件适配指南

### 2.1 最小改动步骤

0. 复制参考工程为算子目录：
   ```bash
   cp -rT references/matmul_fusion_kernel operators/{operator_name}
   cd operators/{operator_name}
   ```
1. 复制 `include/epilogue/div_epilogue.h` 到 `<your>_epilogue.h`。
2. 修改 `operator()` 中 Step c 的计算语句（Div / Mul / Add / Relu / …）。
3. 在 `src/matmul_fused_swat.cpp` 替换 `#include` 与 `using MyEpilogue = ...`。
4. 在 `scripts/gen_data.py` 同步修改 `OUTPUT_DTYPE`、`gen_fusion_inputs()`、`compute_golden()`。
5. 编译运行并进行精度验证。

### 2.2 Host Launcher 适配（`src/matmul_fused_swat.cpp`）

Host Launcher 是融合算子的主入口文件，负责 ACL 初始化、Tiling 计算、内存管理、Kernel 启动。核心结构：

```
main()
  ├── 1. 尺寸解析（m, k, n）
  ├── 2. Tiling 计算（QuantMatmulTilingSwat → tilingData）
  ├── 3. ACL 初始化（AclRtSession + stream）
  ├── 4. Host/Device 内存分配 + H2D 拷贝
  ├── 5. Kernel Launch（<<<usedCoreNum, stream>>>）
  └── 6. D2H 拷贝 + 写出输出
```

**适配修改点**（搜索 `[EXTEND]` / `[MODIFY]` / `[USER]`）：

| 修改点 | 位置 | 说明 |
|--------|------|------|
| 数据类型别名 | `using AType/BType/CType/OutputType/DivisorType = …` | 按算子修改 dtype |
| Epilogue 类引用 | `#include "epilogue/div_epilogue.h"` + `using MyEpilogue = …` | 替换为自实现类 |
| Epilogue Params | `params.epilogueParams = { … }` | 按 Epilogue 的 Params 结构填充 |
| 第二路输入内存 | `sizeD` + `hD` + `dD` 分配/销毁/读写 | 无第二输入时删除整段 |
| Kernel 入口签名 | `FusedKernelEntry(…, dDivisor, …)` | 参数语义与 Epilogue 对齐 |

**场景速查**：

| 场景 | 操作 |
|------|------|
| **Relu/GeLU/Cast**（无第二输入） | 删除 `sizeD`/`hD`/`dD` 内存分配/释放/读写、`input_d.bin` 读取、Kernel 签名中的 `dDivisor` 参数 |
| **Mul+Add**（双第二输入） | 新增第三路输入的 `size/host/dev` 内存路径 + `input_e.bin` 读取 + `params.epilogueParams` 追加第三个 GM 地址 |
| **Cast 输出**（OutputType ≠ CType） | 修改 `OutputType`、`sizeC` 字节数、写出 `hC` 的逻辑与之对齐 |

**参考**：`params` 结构包含 6 子字段（`problemShape` / `mmadParams` / `l1Params` / `schParams` / `qbmmParams` / `epilogueParams`），其中 `mmadParams` 仅 5 个 GM 地址（A/B/C/ScaleA/ScaleB）。

### 2.3 CMakeLists.txt 适配

```cmake
# 修改点：
project(matmul_fusion_kernel LANGUAGES CXX ASC)  # 工程名称
set(ASCENDC_COMPILE_OPTS "…" "--npu-arch=dav-3510" …)  # bisheng 编译架构，非运行时 NPU
add_executable(matmul_fused_swat src/matmul_fused_swat.cpp)  # target 名 + 源文件
```

**验证**：`cmake .. && make` 全程在 Host CPU 完成，与运行时 NPU 无关。

### 2.4 gen_data.py 适配

脚本按三段式组织，每段有明确修改点：

| 段 | 函数 | 修改点 |
|----|------|--------|
| 输入生成 | `gen_matmul_inputs(m,k,n)` | `[PATTERN]` 不改（mxfp8 A/B/Scale 生成 + dequant matmul golden） |
| 融合输入 | `gen_fusion_inputs(m,n)` | `[EXTEND]` 按算子语义生成第二路输入（Relu 无第二路输入可返回空） |
| Golden 计算 | `compute_golden(matmul_out, fusion_tensors)` | `[EXTEND]` 实现融合公式（Div/Mul/Add/Relu/…） |

**注意**：`OUTPUT_DTYPE` 必须与 Host Launcher 中 `OutputType` 一致。

### 2.5 verify_result.py 适配

验证采用**双阶段检查**：逐点相对误差 + 超阈值点比例，单阈值 `torch.allclose` 可能漏检精度问题。

| 修改点 | 说明 | 示例（float32） |
|--------|------|---------------|
| `DATA_TYPE` | 必须与 `gen_data.py` 的 `OUTPUT_DTYPE` 一致 | `np.float32` / `np.float16` |
| `POINT_ERROR_TOL` | 逐点相对误差上限（硬门禁） | `1e-1` |
| `RATIO_POINT_ERROR_TOL` | 逐点绝对误差上限（用于比例统计） | `1e-3` |
| `ERROR_RATIO_TOL` | 超阈值点比例上限（mxfp8 量化误差随 K 累积，≤10%） | `1e-1` |

**判定逻辑**：`point_error_count == 0 AND error_ratio <= ERROR_RATIO_TOL`

### 2.6 run.sh 适配

| 修改点 | 说明 |
|--------|------|
| `OP_NAME` | 必须与 `CMakeLists.txt` 的 `add_executable` target 名称一致 |

> `run.sh` 含 NPU 芯片检测（`npu-smi`），非 Ascend950 机器自动跳过运行。

### 2.7 其他文件速查

| 文件 | 标注 | 何时需要修改 |
|------|------|-------------|
| `include/utils/hardware_constants.h` | `[SAMPLE]` `UB_SIZE` | 切换芯片架构时重定义（dav-3510 = 248KB） |
| `include/tiling/tiling_data.h` | `[EXTEND]` | Epilogue 需要非 float 的 per-launch 参数（如 dtype 大小）时，在结构体末尾追加字段 |
| `include/utils/tiling_key.h` | `[EXTEND]` | 算子支持转置（transA/transB）时，按需扩展 `TransposeMode` 枚举 |

### 2.8 派生算子改动矩阵

| 片段 | 归属 | 改动策略 |
| --- | --- | --- |
| `Init/GetTensor/operator` 框架 | `[PATTERN]` | 保持结构 |
| SPLIT_M 切分与 stage 循环 | `[PATTERN]` | 保持框架 |
| Step a GM→UB 输入读取 | `[USER]` | 按是否有第二输入保留/删除 |
| Step c 计算语句 | `[USER]` | 必改 |
| 末尾 `CrossCoreSetFlag` | `[PATTERN]` | 保持 |
| `stageNum` / `DataType` / `sizeof(float)` | `[SAMPLE]` | 必须重评 |

常见场景：
- Mul/Add：替换 Step c；保留双输入读取。
- Relu：删除第二输入读取；`stageNum` 通常为 1。
- Cast 输出：新增 cast 缓冲并同步 Host 与脚本 dtype。
- Div+Relu：两步计算并保留必要 `PipeBarrier<PIPE_V>`。

---

## 3. Epilogue 开发详解

### 3.1 计算流程

Epilogue 在每个 tile 被 Kernel 层调用一次（`operator()`）。matmul 结果已由 AIC 侧 Fixpipe 写入 UB offset 0，Epilogue 在此基础上做融合后处理。

由于整个 tile 可能超过 UB 容量，Epilogue 内部将 tile 按 `stageSize_` 切分为多个 stage 逐段处理：

```
operator() 被调用
  │
  ├── 1. 计算当前 tile 的有效 M（SPLIT_M 拆分）
  └── while stageOffset < inputSize:
        │
        ├── Step a: GM → UB（MTE2 pipe）
        │     第二路输入 D 从 GM 搬运到 dLocal_
        │     DataCopyPad + DataCopyExtParams 处理 stride
        │
        ├── Sync: MTE2 → V（SetFlag + WaitFlag）
        │     等待 DGMA 搬运完成，数据对 Vector 单元可见
        │
        ├── Step b: 融合计算（V pipe）
        │     cLocalTmp_ = f(cLocal_[stageOffset], dLocal_)
        │     f ∈ {Div, Mul, Add, Relu, Cast, ...}
        │
        ├── Sync: V → MTE3（SetFlag + WaitFlag）
        │     计算完成，允许 MTE3 读结果
        │
        ├── Step c: UB → GM（MTE3 pipe）
        │     结果从 cLocalTmp_ 写回 GM
        │     DataCopyPad + DataCopyExtParams 处理 stride
        │
        ├── Sync: MTE3 → MTE2（下一轮 Step a 的前置）
        │
        └── stageOffset += curStageSize

  └── 末尾: CrossCoreSetFlag(AIV→AIC) 通知 AIC 本轮 Epilogue 完成
```

**各 Step 的 pipe 对应**：

| Step | Pipe | 操作 | 依赖前置同步 |
|------|------|------|------------|
| a | MTE2 | GM → UB 搬运 | `WaitFlag(MTE3_MTE2)` |
| b | V | 融合计算 | `WaitFlag(MTE2_V)` |
| c | MTE3 | UB → GM 搬运 | `WaitFlag(V_MTE3)` |

> 同步顺序不可颠倒或省略。省略某对 SetFlag/WaitFlag 会导致数据竞争（读到半成品数据）。

### 3.2 UB 分配策略

AIV 侧 UB 由三组 `LocalTensor<DataType>` 按静态偏移瓜分：

```
UB offset 0 ┌─────────────────────────┐
            │  cLocal_                │  matmul 结果（AIC Fixpipe 写入，只读）
            │  matmulArea 个元素      │  = CeilDiv(baseM, taskRation) × baseNAlign
            ├─────────────────────────┤
  ubOffset  │  dLocal_                │  第二路输入缓冲（stageNum ≥ 2 时存在）
            │  stageSize_ 个元素      │  stageNum=1 时此区域不存在
            │                         │
            ├─────────────────────────┤  ← 若有第三路输入（stageNum=3），此处再分配
            │  cLocalTmp_             │  计算结果缓冲（必有）
            │  stageSize_ 个元素      │
            └─────────────────────────┘ UB_SIZE
```

**matmulArea 计算公式**（Init 中执行）：

```cpp
l1NAlign = CeilDiv(baseN, ALIGN_ELEM) × ALIGN_ELEM    // N 对齐到 32B
taskRation = AscendC::GetTaskRation()                   // SplitM 拆分比（=2）
l1MSplit = CeilDiv(baseM, taskRation)                   // 每个 Vector 核分到的 M
matmulArea = l1MSplit × l1NAlign                       // matmul 结果占用 UB 元素数
```

> SplitM 场景下一个 Cube 计算拆给两个 Vector 核各自消费，因此每个 Vector 核的 UB 只需容纳 **一半 M** 的 matmul 结果。

**stageSize 计算公式**：

```cpp
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

**stageNum 对性能的影响**：
- stageNum 越大 → 每区 stageSize_ 越小 → 同一 tile 需要更多轮 stage 循环 → 吞吐降低
- stageNum=1 时剩余 UB 全给 cLocalTmp_，通常一轮循环即可覆盖整个 tile
- `stageSize_ ≤ matmulArea` 保证单轮 stage 的 UB 用量不超过 matmul 结果区，避免覆盖未处理数据

### 3.3 三接口合约

Kernel 层（`matmul_kernel_fused.h`）通过模板参数注入 Epilogue 类型，严格依赖以下三接口：

```cpp
class MyEpilogue {
public:
    struct Params {};
    using BlockShape   = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    __aicore__ inline void Init(Params const&, int64_t baseM, int64_t baseN, ProblemShape&);
    __aicore__ inline auto GetTensor();
    __aicore__ inline void operator()(BlockShape const&, int64_t gmOffset, int64_t flagId);
};
```

- **`Init`**：UB 分 stage + 绑定 GM。Kernel 层在 tile 循环前调用一次。
- **`GetTensor`**：返回 matmul 结果的 UB 起点（`cLocal_`），BlockMmad Fixpipe 写入此处。
- **`operator()`**：每个 tile 调用一次。执行 §3.1 的 stage 循环 + 末尾 `CrossCoreSetFlag(AIV→AIC)`。

### 3.4 SPLIT_M 与 ODD-M/ODD-N

- SPLIT_M 场景下，GM 偏移必须追加 AIV 子块偏移：
  ```cpp
  offset += AscendC::GetSubBlockIdx() * halfM * N;
  // halfM = CeilDiv(blockShapeM, GetTaskRation())
  ```
- **ODD-M**：M 为奇数时，UB/L0C 视图 M 向上取偶（`curMPad`），MMAD 仍用原始 `curM`，Epilogue 仅访问有效行。
- **ODD-N**：N 非 8 倍数时，UB 视图 N 对齐到 32B（`curNUbAlign`），Fixpipe 每行仅填 `curN` 列，Epilogue 通过 DataCopyPad 按实际 `curN` 读写。

> 详细对齐策略与容量自检公式见 `/ascendc-tiling-design` → `references/matmul/patterns.md` §2。

### 3.5 CV 同步

- Kernel 层负责 AIC→AIV 等待与通知（`CrossCoreWaitFlag` / `CrossCoreSetFlag`）。
- Epilogue 仅在末尾发送 `CrossCoreSetFlag(AIV→AIC, PIPE_MTE3)`，不在内部执行 `CrossCoreWaitFlag`。
- Coprocessor 间同步使用 `PIPE_S`（系统流水），Flag ID 复用 `cv_sync_constants.h`。

### 3.6 DataCopyPad 与 stride

`curN < N` 时必须使用 `DataCopyPad + DataCopyExtParams` 显式设置 `srcGap/dstGap`。

**关键**：`DataCopyExtParams` 中 **GM 侧单位为字节**，**UB 侧单位为 dataBlock（32B）**——两者不可混用。
ODD-N 场景下 UB row stride 对齐与 dataBlock 单位换算的详细推导见 `/ascendc-tiling-design` → `references/matmul/patterns.md` §2.2。

---

## 4. 常见问题与修复

| 现象 | 根因 | 修复动作 |
| --- | --- | --- |
| AIV 卡死 | Epilogue 内错误执行 `CrossCoreWaitFlag` | 删除 Epilogue 内 Wait，保留 Kernel 同步 |
| M 后半区异常 | 漏写 `SubBlockIdx * halfM * N` 偏移 | 按 §3.4 补齐偏移 |
| 边缘 tile 错行 | 未使用 `DataCopyPad` 处理 stride | 按 §3.6 改造 |
| 随机值/越界 | `stageSize_` 计算或 stage 循环错误 | 重算并分 stage |
| drain 卡死 | 未发送 `AIV→AIC` flag | 在 `operator()` 末尾补齐 |
