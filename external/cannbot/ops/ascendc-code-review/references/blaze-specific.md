# Blaze 领域代码检视规则

<适用>
语言: C++
侧别: Host, Kernel
领域: true
触发: blaze/, Blaze::, GemmUniversal, BlockMmad, BlockScheduler, BlockEpilogue, DispatchPolicy, dispatch_policy, ScheduleType, NotifyVector, WaitForVector, NotifyCube, WaitForCube, MatmulWithScale
默认启用: true

排除场景: third_party/ 及 Third_Party_* 第三方代码, 纯构建系统改动(仅 CMakeLists.txt/cmake/build.sh 且不含 blaze 组件引用), 纯 Python/CSV/Markdown 改动且 PR 无 C++ 差异
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

> **检视边界（必读）**：本规则集只针对 PR **新增/修改行**出具红线意见；存量代码的**风格类**问题（命名、include 风格、变量就近等）不提示，避免对历史代码刷屏。存量代码中涉及**正确性/编译/同步/兼容性**的问题仍需提示，但单独标注"存量，建议随本次改动一并处理"。

> **Blaze = Basic Linear Algebra optimiZed Engine**（NPU 高性能矩阵乘加速引擎）
>
> header-only 的 CUBE 公共底座框架，为 Matmul / GroupedMatmul / QBMM / WQMM 等矩阵乘类算子提供 Kernel 端公共计算组件（数据搬运、MMAD、多核调度、后处理），不涉及 aclnn 入口与 Host 端 Tiling 逻辑。
>
> **规则分类概览**：
>
> | 分类 | 规则范围 | 核心检视关注点 |
> |------|---------|-------------|
> | 架构与扩展 | BLAZE-01~03 | Kernel/Block/Tile 职责分层、核心类模板参数兼容性、策略定义与特化注册 |
> | 编码约束 | BLAZE-04~12 | 高阶 API 禁令、头文件路径与自包含、命名空间、架构宏、接口可见性、static_assert、Params 结构、目录约束 |
> | 命名规范 | BLAZE-13~14 | 文件命名（层级_算子_类型_策略）、标识符命名风格 |
> | 性能优化 | BLAZE-15 | 变量就近定义、局部变量优先（寄存器压力） |
> | AIC/AIV 同步 | BLAZE-16 | Mix 模式核间同步配对、syncID 保留区间 |
> | 交付件完整性 | BLAZE-17~19 | 四目录交付件齐套、UT 增补与不可变、编译验证命令 |
> | Tile 层接口 | BLAZE-20~21 | tag 分派、tensor 推导类型与形状、实现封装（基类私有继承）、单一全特化、对外接口严格控制（仅聚合头 + 最小 public 接口） |
>
> **核心特点**：header-only 模板库（全部组件以 .h 提供）、分层抽象（Kernel→Block→Tile 逐层下沉）、策略驱动（DispatchPolicy + ScheduleType 编译期派发）、dtype/Layout 类型参数化（编译期生成最优路径）、Mix 模式 AIC+AIV 融合、MX/A8W8/权重量化、StreamK/SplitK 调度、L1/L0 双缓冲流水

---

> **适用场景**：ops-tensor 仓（Blaze 框架）及其使用方的 C++ 代码检视
>
> **权威信息源**：
> - 本 Skill 静态规则提炼自 ops-tensor 仓《编程规范》（`CODING_CONVENTIONS.md`）与《贡献指南》（`CONTRIBUTING.md`）。若仓内文档更新，以仓内文档为准，并建议同步更新本 Skill
> - Blaze 组件 API 签名与用法：以仓库 `docs/API/` 与 `include/blaze/` 头文件注释为准，禁止凭记忆推断组件接口
> - AscendC 原生 API（高阶 API 判定、同步原语、数据搬运约束等）：通过 `/ascendc-docs-search` 查询当前 CANN 版本官方文档
>
> **Blaze 组件识别方法**：
> - **include 路径**：`#include "blaze/..."`（如 `blaze/gemm/kernel/kernel_universal.h`）
> - **命名空间**：`Blaze::`（`Blaze::Gemm` / `Blaze::Epilogue` / `Blaze::Attention`）
> - **核心类**：`GemmUniversal`（Kernel 层入口）、`BlockMmad` / `BlockScheduler`（Block 层）、`BlockEpilogue*`（Epilogue 层）、`DispatchPolicy` / `ScheduleType`（Policy 层）
> - **策略结构体**：`MatmulWithScaleMx` / `MatmulWithScaleFixpipeQuant` / `MatmulWithScaleMix` 等（`gemm/policy/dispatch_policy.h`）
> - **Mix 同步原语**：`NotifyVector` / `WaitForVector` / `NotifyCube` / `WaitForCube`（`gemm/utils/sync.h`）
> - **动态识别**：代码中出现疑似 Blaze 组件但不在上述列表（如新增策略结构体），检查其是否位于 `include/blaze/` 或引用 `blaze/` 头文件，是则纳入检视范围

---

## 术语表

| 术语 | 含义 |
|------|------|
| Blaze | Basic Linear Algebra optimiZed Engine，NPU 高性能矩阵乘加速引擎，构建 CUBE 公共底座 |
| header-only | 纯头文件库：所有组件以 .h 提供，无独立编译单元，因此头文件自包含与 include 路径规范尤为关键 |
| Kernel 层 | `gemm/kernel/`、`attention/kernel/`，完整算子内核入口，负责多核任务分布调度，组合 Block + Epilogue + Scheduler |
| Block 层 | `gemm/block/`，BlockMmad（单核块矩阵乘）与 BlockScheduler（多块任务调度） |
| Tile 层 | `gemm/tile/`，细粒度搬运/补零/Scale 处理/计算原语；`arch35/` 子目录存放 Ascend950 特有实现 |
| Epilogue 层 | `epilogue/block/`，L0C→输出的后处理（量化、激活、StreamK 汇聚等）；无后处理时复用 `BlockEpilogueEmpty` |
| Policy 层 | `gemm/policy/dispatch_policy.h`，定义 DispatchPolicy 策略结构体与 ScheduleType 空标签类型 |
| DispatchPolicy | 策略结构体（如 `MatmulWithScaleMx`），携带 `FULL_LOAD_MODE` / `IS_ATOMIC_ADD` / `NON_CONTIGUOUS_TYPE` 等编译期配置 |
| ScheduleType | 空 struct 标签类型（如 `KernelMmadWithScaleMx`），Kernel/Block 特化通过 `enable_if` + `is_same_v` 匹配 |
| GemmUniversal | Kernel 层通用入口类模板，主模板 static_assert 兜底，按 `DispatchPolicy::ScheduleType` 特化 |
| ProblemShape / BlockShape / TileShape | 全局问题形状 → 单核 Block 处理形状 → L0 切分形状，逐层下沉 |
| MMAD | Cube 矩阵乘指令，A(L0A) × B(L0B) → C(L0C) |
| L1 / L0A / L0B / L0C | 片上存储层级：L1（单核，双缓冲）→ L0A/L0B（Cube 输入）→ L0C（Cube 输出） |
| UB | Unified Buffer，AIV 核本地存储 |
| Fixpipe | L0C→UB/GM 搬运通路，量化/反量化可在 fixpipe 阶段融合 |
| 分形布局 | NZ / ZN / ZZ 等 fractal 数据排布，适配 Cube 16×16 块计算；ND/DN 为连续布局；`ext` 后缀（nd_ext/dn_ext）支持更灵活的 stride 配置 |
| c0_element | `tensor_api/utils` 提供的 C0 对齐元素数接口，替代 AscendC 高阶 API（如 `AuxGetC0Size`） |
| tensor_api | `include/tensor_api/` 张量结构抽象（Layout/Shape/Coord/make_tensor），Blaze 的公共依赖 |
| Mix 模式 | AIC（Cube 计算）+ AIV（权重预处理/量化后处理）融合执行模式 |
| AIC / AIV | AI Core Cube 核 / Vector 核 |
| StreamK / SplitK | K 轴切分调度策略；SplitK 常配合原子加（IS_ATOMIC_ADD）或 workspace 汇聚 |
| MX 量化 | MxFP4（fp4x2_e2m1_t / fp4x2_e1m2_t）/ MxFP8（fp8_e5m2_t / fp8_e4m3fn_t）+ E8M0 scale（scaleA/scaleB） |
| A8W8 | INT8 激活 + INT8 权重的 fixpipe 量化方案（per-channel/per-group scale） |
| iterbatch | batch 维迭代优化调度（kernel_matmul_iterbatch 等） |
| ASC_DEVKIT_MAJOR | CANN 开发套件版本宏；>=9 使用拆分头文件（kernel_basic_intf.h），否则 kernel_operator.h |
| `__NPU_ARCH__` / `__CCE_AICORE__` | 设备侧编译架构宏；`__CCE_AICORE__` 已废弃、统一使用 `__NPU_ARCH__`。bisheng 混合编译（host/device 双边）时 host 侧不定义，成员初始化表达式须避免依赖其保护的符号 |
| 声明头（decl header） | `kernel_universal_decl.h` / `block_mmad_decl.h`（gemm / attention 各一），仅存放主模板声明 + static_assert 兜底；聚合头与全部特化片段 include 它，消除循环包含、恢复片段自包含 |
| tile 聚合头 / 转发头 | `blaze/gemm/tile/datamove.h`（数据搬运）/`compute.h`（计算）为聚合头，使用方 include 聚合头；`copy_gm_to_l1.h` 等顶层转发头仅为兼容保留、不得新增，实现位于 `arch35/` |
| tag 分派 | tile 层接口范式：算法 tag（`GeluTanhTag`/`GeluErfTag`/`B32ToFpTag` 等）作为类模板第一分派轴，主模板仅声明、按 tag 全特化；类型/形状从张量推导，仅非类型参数参与分派 |
| 双缓冲 | double buffer，L1/L0 两个 buffer 交替实现搬入-计算-搬出重叠 |
| always_false_v | 主模板兜底 trait，配合 static_assert 报"未实现该 DispatchPolicy"的编译错误 |

---

## PR 差异→规则速查

> **使用方法**：查看 PR 差异代码中的关键词，匹配下表，仅阅读对应分类的规则。无需一次阅读全部规则。

| 差异关键词 | 对应分类 | 应读规则 | 典型搜索命令 |
|-----------|---------|---------|------------|
| 新增 `kernel_*.h` / `block_mmad_*.h` / `block_scheduler_*.h` / `block_epilogue_*.h` 文件 | 架构扩展 + 命名 + 交付件 | BLAZE-01~03, 13, 17~19 | `git diff --name-only \| grep -E "(kernel\|block_mmad\|block_scheduler\|block_epilogue)_"` |
| `tile/` 下 `.h` 新增/修改 / `copy_traits` / `GeluTanhTag` / `GeluErfTag` / `B32ToFpTag` / `DataTypeIn` / `DataTypeOut` / `detail` 命名空间 | Tile 层接口 | BLAZE-20~21 | `git diff --name-only \| grep "tile/"` |
| `dispatch_policy.h` / `ScheduleType` / `MatmulWithScale` | 架构与扩展 | BLAZE-02~03 | `grep -rn "ScheduleType\|MatmulWithScale" include/blaze/gemm/policy/` |
| `kernel_universal_decl.h` / `block_mmad_decl.h` / 聚合头 `kernel_universal.h` / `block_mmad.h` / 回头 include 聚合头 | 架构与扩展 | BLAZE-03, 05~06 | `grep -rn "kernel_universal_decl\|block_mmad_decl\|kernel_universal.h\|block_mmad.h" include/blaze/` |
| `AscendC::Aux` / `AuxGetC0Size` / `VECTOR_REG_WIDTH` / `asc_get_vf_len` | 编码约束 | BLAZE-04 | `grep -rn "AscendC::Aux\|AuxGetC0Size\|GetL1BufferSize\|VECTOR_REG_WIDTH\|asc_get_vf_len" include/blaze/` |
| `#include "../` / `#include "./` / 裸文件名 include / `impl/` / `tile/arch35/` / `datamove.h` / `compute.h` / `kernel_operator.h` / `ASC_DEVKIT_MAJOR` | 编码约束 | BLAZE-05 | `grep -rn '#include "\.\./\|#include "\./\|/impl/\|tile/arch35/' include/blaze/` |
| 新增/修改 `.h` 头文件 | 编码约束 | BLAZE-05~06 | `git diff --name-only \| grep "include/blaze/.*\.h$"` |
| `using namespace` | 编码约束 | BLAZE-07 | `grep -rn "using namespace" include/blaze/` |
| `__NPU_ARCH_` / `__CCE_AICORE__` / `TOTAL_UB_SIZE` / `static constexpr` 成员 | 编码约束 | BLAZE-08 | `grep -rn "__NPU_ARCH_\|__CCE_AICORE__\|TOTAL_UB_SIZE" include/blaze/` |
| `public:` / `Init(` / `Run(` / `operator()` | 编码约束 | BLAZE-09 | `grep -rn -A3 "public:" include/blaze/gemm/kernel/` |
| `static_assert` | 编码约束 | BLAZE-10 | `grep -rn "static_assert" include/blaze/` |
| `struct Params` / `Params(` | 编码约束 | BLAZE-11 | `grep -rn "struct Params" include/blaze/` |
| `prelogue` | 编码约束 | BLAZE-12 | `grep -rni "prelogue" include/` |
| `NotifyVector` / `WaitForVector` / `NotifyCube` / `WaitForCube` / `asc_sync_` / `ASC_IS_AIC` / `ASC_IS_AIV` / `MIX_FLAG` / `FIXPIPE_.*FLAG` | AIC/AIV 同步 | BLAZE-16 | `grep -rn "NotifyVector\|WaitForVector\|NotifyCube\|WaitForCube\|asc_sync_\|ASC_IS_AIC\|ASC_IS_AIV" include/blaze/` |
| `examples/` / `gen_data.py` / `verify_result.py` / `.csv` / `.conf` | 交付件 | BLAZE-17, 19 | `git diff --name-only \| grep examples/` |
| `tests/ut/op_kernel/` / `TEST_F` / `KERNEL_RUN_KF` | 交付件 | BLAZE-18~19 | `git diff --name-only \| grep tests/ut/` |

---

## 领域判定规则

> **Blaze** 是 ops-tensor 仓内置的 NPU 高性能矩阵乘加速引擎（header-only CUBE 底座），服务于矩阵乘类算子的 Kernel 端计算组件。
>
> Agent 审阅代码时，先使用以下规则判定代码是否属于 Blaze 领域。满足 **至少一个核心特征** 且 **不属于排除场景** 时，启用本规则集。

### 核心特征（满足任一即可）

| 特征编号 | 特征描述 | 检测方法 |
|---------|---------|---------|
| **C1** | 差异文件位于 Blaze 组件目录 | 文件路径匹配 `include/blaze/**`（gemm / attention / epilogue） |
| **C2** | 代码引用 Blaze 组件 | `#include "blaze/..."`，或使用 `Blaze::` 命名空间 / `GemmUniversal` / `BlockMmad` / `BlockScheduler` 等核心类 |
| **C3** | ops-tensor 仓新算子交付 PR | PR 同时变更 `include/blaze/` + `examples/` + `tests/ut/op_kernel/`（新算子交付件模式） |

### 排除场景（即使命中关键字也不启用）

| 场景 | 说明 |
|------|------|
| 第三方代码 | `third_party/`、`Third_Party_*` 目录下的外部代码 |
| 纯构建改动 | 仅 CMakeLists.txt / cmake / build.sh 且不涉及 blaze 组件引用 |
| 纯脚本/文档改动 | 仅 Python / CSV / Markdown 且 PR 无任何 C++ 差异 |

### 常见误报场景排除

| 误报场景 | 特征 | 处理方式 |
|----------|------|----------|
| 仅使用 tensor_api 未用 Blaze | 代码仅 include `tensor_api/`、使用 `asc::te::`，无任何 `blaze/` 引用 | tensor_api 是公共张量抽象层，非 Blaze 专属，不启用本规则集 |
| UT/示例中实例化 Kernel | tests/examples 组装 GemmUniversal 等组件 | 启用，但组件编码规则（BLAZE-04~12）从宽，重点检 BLAZE-10 模板参数组合与 BLAZE-17~19 交付件 |
| 其他仓库复用 Blaze | 非 ops-tensor 仓 include `blaze/` 头文件 | 启用组件使用类规则（BLAZE-04~10, 16），交付件规则（BLAZE-17~19）仅对 ops-tensor 仓生效 |

### 场景子分类（判定为 Blaze 后，选取规则子集）

| 子分类 | 特征 | 应读规则 |
|--------|------|---------|
| 新算子全套交付 | 新增 kernel_ / block_mmad_ / block_scheduler_（+ epilogue / tile）文件 | BLAZE-01~19 全量 |
| 新增 DispatchPolicy 策略 | 修改 dispatch_policy.h + 新增特化实现 | BLAZE-01~03, 05~11, 13 |
| Mix 模式（AIC+AIV 融合） | NotifyVector/WaitForVector、权重预处理、量化后处理 | BLAZE-01, 04~11, 16 |
| 量化算子（MX/A8W8/WQMM） | scaleA/scaleB、E8M0、fixpipe quant | BLAZE-01, 04~11, 16 |
| 存量算子修复/优化 | 修改已有组件文件 | BLAZE-02, 04~09, 11, 14~16, 18 |
| 交付件补充 | examples / tests / docs 目录变更 | BLAZE-17~19 |

---

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| BLAZE-01 | Kernel/Block/Tile 分层职责划分 | 架构分层 | 中 | [适用: Kernel] |
| BLAZE-02 | 核心类模板参数稳定性 | 接口兼容 | 高 | [适用: Kernel] |
| BLAZE-03 | 策略定义与特化注册 | 策略扩展 | 中 | [适用: Kernel] |
| BLAZE-04 | 禁止直接调用 AscendC 高阶 API | API使用 | 高 | [适用: Kernel] |
| BLAZE-05 | 头文件引用路径规范 | 头文件管理 | 高 | [适用: Kernel] |
| BLAZE-06 | 头文件自包含 | 头文件管理 | 高 | [适用: Kernel] |
| BLAZE-07 | 全局作用域禁止 using namespace | 命名空间 | 高 | [适用: Kernel] |
| BLAZE-08 | 架构宏统一与成员初始化约束 | 编译配置 | 高 | [适用: Kernel] |
| BLAZE-09 | 对外接口可见性控制 | 接口设计 | 中 | [适用: Kernel] |
| BLAZE-10 | static_assert 编译期校验 | 类型校验 | 中 | [适用: Kernel] |
| BLAZE-11 | Params 结构体约束 | 数据结构 | 高 | [适用: Kernel] |
| BLAZE-12 | 禁止独立 prelogue 目录 | 目录结构 | 中 | [适用: Kernel] |
| BLAZE-13 | 文件命名规范（层级_算子_类型_策略） | 文件命名 | 中 | [适用: Kernel] |
| BLAZE-14 | 标识符命名风格 | 标识符命名 | 中 | [适用: Kernel] |
| BLAZE-15 | 变量生命周期与寄存器压力 | 性能优化 | 中 | [适用: Kernel] |
| BLAZE-16 | AIC/AIV 同步配对一致性 | 并发同步 | 高 | [适用: Kernel] |
| BLAZE-17 | 新算子交付件完整性 | 交付管理 | 高 | [适用: All] |
| BLAZE-18 | UT 增补与不可变约束 | 测试完备 | 高 | [适用: All] |
| BLAZE-19 | 编译与 UT 验证 | 编译验证 | 中 | [适用: All] |
| BLAZE-20 | Tile 层接口风格统一 | 接口设计 | 中 | [适用: Kernel] |
| BLAZE-21 | Tile 层对外接口严格控制 | 接口边界 | 高 | [适用: Kernel] |

> **严重级别说明**：高 = 红线（必须整改）；中 = 设计原则（应当整改）。

---

## 一、架构与扩展规则

> **触发条件**：PR 差异中出现 `include/blaze/` 组件文件新增/修改、`dispatch_policy.h`、`GemmUniversal`、`BlockMmad`、`ScheduleType`
>
> **检视原则**：Blaze 的核心价值在于分层抽象与策略驱动的编译期派发。检视时追踪 ProblemShape→BlockShape→TileShape 的数据流下沉，确认各层职责不越界、扩展点走正规注册路径。

### BLAZE-01: Kernel/Block/Tile 分层职责划分 `[Kernel]` `[设计原则]`

**规则**：各层职责严格划分——Kernel 层（`gemm/kernel/`、`attention/kernel/`）只负责多核间任务分布调度并确定单核 BlockShape；BlockMmad 层（`gemm/block/`）负责单核内数据块处理（L0 Tile 切分与 MMAD 迭代）；Tile 层（`gemm/tile/`）只提供细粒度原语。

**检测模式**：
- 检查 `gemm/kernel/` 新增/修改文件：是否出现逐 tile 的 L0 迭代细节（LoadA/LoadB、L0A/L0B 双缓冲循环）
- 检查 `gemm/block/` 新增/修改文件：是否出现多核切分逻辑（GetBlockNum、按 coreIdx 计算 BlockShape、多核任务分布）

**判断标准**：
- Kernel 层包含 L0 Tile 级迭代细节 → **风险**（职责下沉不足，Kernel 无法复用）
- BlockMmad 中出现多核任务分布/BlockShape 按核数切分 → **风险**（职责上浮，多算子无法组合）
- Kernel 只做调度 + BlockShape 计算，BlockMmad 只做块内 Tile 切分 → **安全**

**正确示例**：
```cpp
// Kernel 层：根据 problemShape 和核数计算 blockShape，组合组件
void operator()(const Params& params) {
    auto blockShape = ComputeBlockShape(params.problemShape, coreNum);
    blockMmad_(blockShape, /* ... */);
}

// BlockMmad 层：在 blockShape 内做 L0 Tile 切分，不关心多核切分
void operator()(/* blockShape, ... */) {
    for (/* tile 迭代 */) { /* L0 搬运 + MMAD */ }
}
```

---

### BLAZE-02: 核心类模板参数稳定性 `[Kernel]` `[红线]`

**规则**：禁止修改 `BlockMmad`、`GemmUniversal` 等核心类定义的模板参数（增删、重排、中间插入）。这些模板参数是经过严格验证的公共接口，变更破坏向前兼容性。确有必要时新参数只能追加到参数列表**末尾**。

**检测模式**：
- 定位 PR diff 中 `class GemmUniversal<...>` 与 `class BlockMmad<...>` 主模板定义的 `template <...>` 参数列表变更
- 比对参数数量与顺序：`GemmUniversal` 标准签名 `ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_, Enable_`；`BlockMmad` 标准签名 `DispatchPolicy_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_`

**判断标准**：
- 删除或重排已有模板参数 → **风险**（所有既有特化与使用方编译失败）
- 在参数列表中间插入新参数 → **风险**（位置参数错位）
- 新参数追加到末尾且带默认值 → **安全**
- 特化文件中的特化模板参数宏（如 `QBMM_MX_KERNEL_CLASS_TEM_PARAMS`）随主模板同步调整 → 正常，不违规

**错误示例**：
```cpp
// 错误：在已有模板参数列表中插入新参数
template <class AType_, class BType_, class NewParam_, class CType_>  // 插入中间，破坏 API 兼容性
class BlockMmad {};
```

**正确示例**：
```cpp
// 正确：新参数追加到模板参数列表末尾
template <class AType_, class BType_, class CType_, class NewParam_>
class BlockMmad {};
```

---

### BLAZE-03: 策略定义与特化注册 `[Kernel]` `[设计原则]`

**规则**：新增算法变体必须走完整的策略注册链路：① 在 `gemm/policy/dispatch_policy.h` 定义 ScheduleType 空标签 struct 与 DispatchPolicy 策略结构体；② 实现类模板特化并用 `enable_if_t<is_same_v<ScheduleType, ...>>` 守卫匹配；③ 特化片段**必须注册到聚合头底部**（GemmUniversal 聚合头 `kernel_universal.h`、BlockMmad 聚合头 `block_mmad.h`，gemm / attention 各一）；④ 主模板声明**抽取到独立声明头** `kernel_universal_decl.h` / `block_mmad_decl.h`（gemm / attention 各一），聚合头与全部特化片段均 include 声明头，主模板保留 `static_assert(always_false_v<...>)` 兜底。

**禁止反向依赖与循环包含**：特化片段不得回头 `#include` 聚合头（`kernel_universal.h` / `block_mmad.h`），否则形成仅靠 `#pragma once` 工作的循环包含、破坏片段自包含；片段只应 include 声明头。使用方（example / UT / kernel 层）只 include 聚合头，不得绕开聚合头单独 include 片段头（不得依赖"片段回包含聚合头"才能编译）。

> **特别强调：BlockMmad 与 GemmUniversal 完全一致**。Block 层 `BlockMmad` 的聚合规则与 Kernel 层 `GemmUniversal` **逐条相同**：`block_mmad_*.h` 特化片段**只 include `block_mmad_decl.h`**，**不得**回头 `#include "block_mmad.h"`；`block_mmad.h` 是纯聚合头，才逐个 include 全部片段。最典型的红线错误是：在 `block_mmad_xxx.h` 片段内写 `#include "blaze/gemm/block/block_mmad.h"`（回头 include 聚合头 → 循环包含）。同理，`block_mmad_decl.h` 只放 `BlockMmad` 主模板声明 + static_assert 兜底，不含任何特化实现。

**聚合外例外**：仅 `kernel_wqmm_mix_weight_prologue.h`（定义独立 `GmmWeightQuantMxKernel` 类、非 GemmUniversal 特化）与 `kernel_gmm_fixpipe_quant.h`（废弃改名兼容头）不参与聚合，使用方直接 include 对应头即可。

**检测模式**：
- PR 新增 `kernel_*.h` / `block_mmad_*.h` 文件时，检查 `dispatch_policy.h` 是否同步新增对应 ScheduleType 与策略结构体
- 检查聚合头（`kernel_universal.h`、`block_mmad.h`）底部是否 include 新特化片段
- 检查新增特化片段内是否回头 `#include "kernel_universal.h"` / `#include "block_mmad.h"`（循环包含），应改为 include 声明头 `kernel_universal_decl.h` / `block_mmad_decl.h`
- 检查是否存在绕开聚合头单独 include 片段头的使用方（example/UT/kernel 层）
- 检查特化是否带 `enable_if_t<AscendC::Std::is_same_v<ScheduleTypeTag, typename BlockMmad::DispatchPolicy::ScheduleType>>` 守卫

**判断标准**：
- 新特化未在聚合头注册 → **风险**（使用方 include 聚合头时命中主模板兜底报错）
- 特化片段回头 include 聚合头 → **风险**（循环包含，仅靠 `#pragma once` 维系，破坏自包含；应改 include 声明头）
- 片段未 include 声明头 → **风险**（不自包含，仓外直接 include 片段无法编译）
- 使用方绕开聚合头单独 include 片段头 → **风险**（依赖回包含才能编译，违反聚合头设计）
- 新特化无 enable_if ScheduleType 守卫 → **风险**（特化歧义，编译失败或匹配错误实现）
- dispatch_policy.h 未定义对应策略 → **风险**（无法通过 DispatchPolicy 派发）
- 注册链路完整、声明头一致、主模板兜底保留 → **安全**

**错误示例（BlockMmad 回头 include 聚合头是红线）**：
```cpp
// ❌ 错误：block_mmad_matmul_basic.h 片段回头 include 聚合头 → 循环包含、不自包含
#include "blaze/gemm/block/block_mmad.h"   // 应改为 #include "blaze/gemm/block/block_mmad_decl.h"

template <class DispatchPolicy, class AType, class LayoutA, class BType, class LayoutB,
    class CType, class LayoutC, class BiasType, class LayoutBias>
class BlockMmad<...> { /* ... */ };
```

**正确示例**：
```cpp
// ① dispatch_policy.h：定义标签与策略
struct KernelMmadWithScaleMx {};  // ScheduleType 标签
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, ...>
struct MatmulWithScaleMx {
    using ScheduleType = KernelMmadWithScaleMx;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
};

// ② 特化片段（kernel_matmul_basic.h）：只 include 声明头即可，无需 include 其他片段/聚合头
#include "blaze/gemm/kernel/kernel_universal_decl.h"

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelMmadWithScaleMx, typename BlockMmad::DispatchPolicy::ScheduleType>>> {
    // ...
};

// ③ 声明头 kernel_universal_decl.h：仅主模板声明 + static_assert 兜底（不含任何特化实现）
template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler, class Enable>
class GemmUniversal {
    static_assert(always_false_v<Enable>, "Unsupported GemmUniversal DispatchPolicy");
};

// ④ 聚合头 kernel_universal.h：include 声明头 + 全部特化片段（不再定义主模板）
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/kernel/kernel_matmul_basic.h"   // ← 聚合头才逐个 include 片段；片段本身见 ②
// ... 其余全部 GemmUniversal 特化片段 ...

// ⑤ BlockMmad 侧完全同理（同一条规则，千万别回头 include block_mmad.h）：
//    block_mmad_*.h 片段：只 include 声明头 block_mmad_decl.h
#include "blaze/gemm/block/block_mmad_decl.h"

template <class DispatchPolicy, class AType, class LayoutA, class BType, class LayoutB,
    class CType, class LayoutC, class BiasType, class LayoutBias>
class BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelMmadWithScaleMx,
        typename DispatchPolicy::ScheduleType>>> {
    // ...
};

//    block_mmad.h 聚合头：include 声明头 + 全部片段（不含主模板定义）
#include "blaze/gemm/block/block_mmad_decl.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"  // ← 聚合头逐个 include 片段；片段本身见上
// ... 其余全部 BlockMmad 特化片段 ...
```

---

## 二、编码约束规则

> **触发条件**：PR 差异涉及 `include/blaze/` 下 `.h` 文件的修改（include 语句、命名空间、类定义、struct 定义、目录/文件新增）

### BLAZE-04: 禁止直接调用 AscendC 高阶 API `[Kernel]` `[红线]`

**规则**：禁止直接调用 AscendC 高阶 API 接口（如 `AuxGetC0Size`），应使用 `tensor_api/utils` 下的等价封装接口（如 `asc::te::c0_element`）。寄存器向量长度统一使用 `asc_get_vf_len()`（而非 `AscendC::VECTOR_REG_WIDTH`）；核数/核 ID 统一使用 C API `asc_get_sub_block_num()` / `asc_get_sub_block_id()`（而非其他封装）。

**检测模式**：
- 定位 `AscendC::Aux` / `AuxGetC0Size` / `GetL1BufferSize` 等高阶 API 调用
- 定位 `AscendC::VECTOR_REG_WIDTH` 常量使用（应替换为 `asc_get_vf_len()`）
- 定位核信息获取方式，确认是否使用 `asc_get_sub_block_num` / `asc_get_sub_block_id`
- 对疑似高阶 API，使用 `/ascendc-docs-search` 查询确认其类别与推荐替代接口

**判断标准**：
- 直接调用 AscendC 高阶 API → **风险**（绕过 tensor_api 抽象，架构演进时无法隔离适配）
- 使用 `AscendC::VECTOR_REG_WIDTH` → **风险**（应替换为 `asc_get_vf_len()`，避免混合编译与代际差异问题）
- 使用 `asc::te::c0_element<T>` / `asc_get_vf_len()` / `asc_get_sub_block_*` 等封装接口 → **安全**

**错误示例**：
```cpp
// 错误：直接调用 AscendC 高阶 API
constexpr uint32_t c0Size = AscendC::AuxGetC0Size<AType>();
```

**正确示例**：
```cpp
// 正确：使用 tensor_api/utils 封装
constexpr uint32_t c0Size = asc::te::c0_element<AType>;
```

---

### BLAZE-05: 头文件引用路径规范 `[Kernel]` `[红线]`

**规则**：`#include` 必须使用从 `blaze` 根目录开始的绝对路径（如 `blaze/gemm/block/block_mmad.h`），禁止 `../`、`./` 相对路径；同目录裸文件名引用（如 `#include "block_mmad.h"`、`#include "kernel_universal.h"`）为废弃存量写法，**新增/修改代码必须全路径化**。禁止直接引用 `impl/` 目录下的内部实现文件。涉及 AscendC 基础头文件时，使用 `ASC_DEVKIT_MAJOR` 宏做版本隔离。

**tile 层 include 约定**：使用方 include tile 聚合头 `blaze/gemm/tile/datamove.h`（数据搬运）/`compute.h`（计算）；**不得直接 include `blaze/gemm/tile/arch35/` 实现头**（绕过聚合头）。顶层转发头（`copy_gm_to_l1.h`、`copy_gm_to_ub.h`、`copy_weight_ub_to_l1.h`、`pad_mx_kl1.h`、`shift_w4_to_w8.h`）仅为兼容保留的转发入口，**不得再新增**，新代码 include 聚合头。

**检测模式**：
- 定位 `#include "../` 与 `#include "./` 形式的相对路径引用
- 定位同目录裸文件名引用（`#include "block_mmad.h"` / `#include "kernel_universal.h"` 等）
- 定位 include 路径含 `/impl/` 或 `tile/arch35/` 的引用
- 定位 tile 顶层转发头的新增（转发头不得新增）与直接 include（应改 include 聚合头 datamove.h/compute.h）
- 定位 `kernel_operator.h` 引用，确认是否被 `#if ASC_DEVKIT_MAJOR >= 9` 宏包裹（>=9 应使用 kernel_basic_intf.h 等拆分头文件）

**判断标准**：
- `../` 或 `./` 相对路径 → **风险**（编译依赖 include 搜索顺序，重构即断）
- 同目录裸文件名引用 → **风险**（新增/修改代码未全路径化，应统一为 `blaze/` 根路径）
- 引用 `impl/` 内部实现头 → **风险**（破坏公开/内部边界）
- 直接 include `tile/arch35/` 实现头 → **风险**（绕过聚合头约定，应改 include `datamove.h`/`compute.h`）
- 新增 tile 顶层转发头 → **风险**（转发头仅为兼容保留，不得新增）
- `kernel_operator.h` 未做 ASC_DEVKIT_MAJOR 版本隔离 → **风险**（新版本编译耗时且无隔离）
- 跨目录引用以 `blaze/` 开头、tile 使用聚合头 → **安全**

**正确示例**：
```cpp
// 跨目录引用：从 blaze 根目录开始的绝对路径
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/utils/common_utils.h"

// AscendC 基础头文件版本隔离
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
```

---

### BLAZE-06: 头文件自包含 `[Kernel]` `[红线]`

**规则**：头文件必须自包含（self-contained）——自身需要的所有依赖在自己的文件内引入，不依赖使用者前置 `#include`。Blaze 为 header-only 库，任何头文件都可能被使用方直接 include，此约束比常规项目更严格。聚合体系下的**特化片段头**同样必须自包含：通过 include 对应声明头（`kernel_universal_decl.h` / `block_mmad_decl.h`）获得主模板声明，不得依赖聚合头内的 include 顺序。

**检测模式**：
- 对新增/修改的头文件，检查其用到的符号（类/函数/宏）所需的 include 是否齐全
- 对特化片段头，检查是否 include 声明头（而非依赖聚合头/回包含）
- 可用单独编译该头文件做冒烟验证：`echo '#include "blaze/xxx/yyy.h"' > /tmp/smoke.cpp && 编译`

**判断标准**：
- 头文件使用的符号依赖使用者前置 include → **风险**（使用方 include 顺序不同即编译失败）
- 特化片段未 include 声明头、依赖聚合头 include 顺序 → **风险**（不自包含，仓外直接 include 片段编译失败）
- 所有依赖自引入且有 `#pragma once` → **安全**

**正确示例**：
```cpp
// my_header.h —— 自身依赖自己引入
#pragma once
#include "kernel_operator.h"  // 或按 ASC_DEVKIT_MAJOR 隔离
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze { namespace Gemm {
    // 使用 common_utils.h 中的符号，无需使用者前置引入
} }
```

---

### BLAZE-07: 全局作用域禁止 using namespace `[Kernel]` `[红线]`

**规则**：禁止在全局作用域（头文件 namespace 之外）使用 `using namespace` 引入较大的命名空间（如 `using namespace AscendC`、`using namespace asc::te`）。允许函数作用域内局部引入单个名称或使用全限定名。

**检测模式**：
- 定位 `using namespace`，确认其是否位于函数体内；header-only 库中头文件全局作用域的 using namespace 污染所有使用方

**判断标准**：
- 头文件全局作用域 `using namespace` → **风险**（命名冲突污染所有 include 方）
- 函数内 `using asc::te::shape;` 局部引入单个名称 → **安全**
- 全限定名 `AscendC::GlobalTensor<int64_t>` → **安全**

---

### BLAZE-08: 架构宏统一与成员初始化约束 `[Kernel]` `[红线]`

**规则**：架构宏统一使用 `__NPU_ARCH__`，禁止新代码使用已废弃的 `__CCE_AICORE__`（含注释中引用）。禁止在类成员变量初始化表达式中使用被 `__NPU_ARCH__` 宏保护的函数或常量——bisheng `--asc-aicore-lang` 混合编译（host/device 双边）时 host 侧不定义这些宏，成员初始化表达式在 host 侧同样需要解析，导致编译失败。架构相关初始化只能放入 `__aicore__` 类方法内。类内 `static constexpr` 成员若依赖仅 device 侧可用常量（如 `AscendC::TOTAL_UB_SIZE`、`AscendC::VECTOR_REG_WIDTH`），必须下沉到使用处的函数体局部 `constexpr`（或在 `__aicore__` 方法内），避免 host pass 解析失败。

**检测模式**：
- 定位 `__CCE_AICORE__` 使用（含注释），确认是否为新增/修改行（存量按文件头"检视边界"单独标注）
- 定位类成员变量声明处的初始化表达式（`类型 成员_ = 表达式;`），可用 `grep -nE '^\s+\w+ \w+_ = ' 文件` 快速筛查
- 定位类内 `static constexpr` 成员，追踪其取值表达式是否依赖架构宏保护的常量或仅 device 侧可用符号
- 追踪表达式中的函数/常量定义是否被 `#if defined(__NPU_ARCH__)` 包裹

**判断标准**：
- 新代码使用 `__CCE_AICORE__` → **风险**（已废弃，统一使用 `__NPU_ARCH__`）
- 成员初始化表达式调用仅设备侧可用的函数/常量 → **风险**（host 侧混合编译报 use of undeclared identifier）
- 类内 `static constexpr` 成员取值依赖架构宏/仅 device 常量 → **风险**（host pass 解析失败，应下沉到函数体局部 constexpr）
- 架构相关初始化放在 `__aicore__` 方法（如 Init）中、常量下沉函数体 → **安全**

**错误示例**：
```cpp
template <class AType_, class LayoutA_>
class MyKernel {
private:
    // GetArchDepValue() 仅在 __NPU_ARCH__ 下可用，CPU/host 侧编译会失败
    uint32_t archValue_ = GetArchDepValue();
};

// 错误：类内 static constexpr 依赖仅 device 侧可用常量，host pass 解析失败
class MyEpilogue {
private:
    static constexpr uint32_t UBOffset_ = AscendC::TOTAL_UB_SIZE;  // 应下沉到函数体
};
```

**正确示例**：
```cpp
template <class AType_, class LayoutA_>
class MyKernel {
private:
    uint32_t archValue_;

    __aicore__ void Init() {
        archValue_ = GetArchDepValue();  // __aicore__ 方法内编译器忽略 host 侧解析
    }
};

// 正确：常量下沉到使用处函数体局部 constexpr
class MyEpilogue {
private:
    __aicore__ void Run() {
        constexpr uint32_t ubOffset = AscendC::TOTAL_UB_SIZE;  // device pass 内解析
    }
};
```

---

### BLAZE-09: 对外接口可见性控制 `[Kernel]` `[设计原则]`

**规则**：严格控制对外接口，不对外开放的类方法和成员变量必须 `private`。对 Kernel 类而言，只有 `operator()` 接口对外开放，`Init` 和 `Run` 都应当设置为 `private`。

**检测模式**：
- 定位 Kernel/Block 类的 `public:` 段，检查是否暴露 Init / Run 或内部辅助方法

**判断标准**：
- Kernel 类 public 暴露 Init/Run → **风险**（使用方绕过完整生命周期调用，状态未初始化）
- 仅 `operator()`（及必要的类型别名/Params）public → **安全**

**正确示例**：
```cpp
template <class AType_, class LayoutA_>
class MyKernel {
public:
    __aicore__ inline void operator()(const Params& params) {
        Init(params);
        Run(params);
    }

private:
    __aicore__ void Init(const Params& params) { /* ... */ }
    __aicore__ void Run(const Params& params) { /* ... */ }
};
```

---

### BLAZE-10: static_assert 编译期校验 `[Kernel]` `[设计原则]`

**规则**：各层接口须显式增加编译时 `static_assert` 校验，不支持的场景直接报错提示：输入 dtype、shape、format（Layout）校验；不支持后处理的 kernel 需增加后处理校验。错误信息须说明具体约束。

**检测模式**：
- 新增 Kernel/Block 模板特化时，检查是否含 dtype/Layout/组合的 static_assert（常用 `AscendC::Std::is_one_of_v` / `is_same_v`）
- 主模板兜底 `static_assert(always_false_v<...>, "...")` 是否保留（见 BLAZE-03）

**判断标准**：
- 新特化无任何 static_assert，不支持的 dtype/Layout 组合静默实例化 → **风险**（错误延迟到运行期或产生错误结果）
- static_assert 信息为空泛字符串（如 "error"）→ **补充**具体约束说明
- 校验齐全且信息明确 → **安全**

**正确示例**：
```cpp
static_assert(
    (AscendC::Std::is_one_of_v<AType, fp8_e4m3fn_t, fp8_e5m2_t> ||
     AscendC::Std::is_one_of_v<AType, fp4x2_e2m1_t, fp4x2_e1m2_t>) &&
    AscendC::Std::is_one_of_v<CType, half, bfloat16_t, float>,
    "Unsupported (AType, CType) combination");
static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
    "LayoutA must be nd_ext_layout_ptn/dn_ext_layout_ptn.");
```

---

### BLAZE-11: Params 结构体约束 `[Kernel]` `[红线]`

**规则**：`Params` 结构体不得提供自定义构造函数（只能默认构造）；新增参数字段必须追加到结构体**最后**。Kernel/Block/Scheduler 各层 Params 按 members 排布在 host/device 间按偏移传递，字段插入中间破坏二进制兼容。

**检测模式**：
- 定位 `struct Params` 定义变更：检查是否新增自定义构造函数；新字段是否插入已有字段之前

**判断标准**：
- Params 提供带参构造函数 → **风险**（禁用默认构造，使用方聚合初始化失败）
- 新字段插入已有字段中间 → **风险**（字段偏移变化，host/device 数据错位）
- 默认构造 + 新字段追加末尾 → **安全**

**错误示例**：
```cpp
struct Params {
    GM_ADDR aGmAddr;
    uint32_t newField;    // 禁止：插入已有字段之前
    uint64_t mL1;

    Params(GM_ADDR a, uint64_t l1)  // 禁止：自定义构造函数
        : aGmAddr(a), mL1(l1) {}
};
```

**正确示例**：
```cpp
struct Params {
    GM_ADDR aGmAddr;
    uint64_t mL1;
    uint64_t mHbm;
    uint32_t newField;    // 新字段放在最后
};
```

---

### BLAZE-12: 禁止独立 prelogue 目录 `[Kernel]` `[设计原则]`

**规则**：`blaze` 代码中不单独增加 `prelogue` 目录，前处理逻辑放在各自的 `kernel` 文件中。

**检测模式**：
- 检查 PR 是否新增 `prelogue/` 目录或 `prelogue_` 前缀文件

**判断标准**：
- 新增 `blaze/**/prelogue/` 目录或 `prelogue_*.h` 文件 → **风险**（破坏既定目录结构）
- 前处理逻辑放入对应 kernel 文件内 → **安全**

---

## 三、命名规范规则

> **触发条件**：PR 新增/重命名 `include/blaze/` 下的组件文件，或修改类型/函数/变量/常量声明
>
> **注意**：本仓命名约定（源自 CODING_CONVENTIONS.md）与通用 C++ 风格存在差异（如函数名大驼峰），检视命名问题时**以本节规则优先**，抑制通用风格规则与仓规范冲突的意见。

### BLAZE-13: 文件命名规范 `[Kernel]` `[设计原则]`

**规则**：组件文件名遵循 `层级_算子_类型_策略` 格式：层级取 `kernel` / `block_mmad` / `block_scheduler` / `block_epilogue` / `tile`；算子取 `matmul` / `batchmatmul` / `qbmm` / `qgmm` / `wqmm` / `wqgmm` / `tbmm` / `tqbmm` / `syrk` 等；类型可选（默认不填表示 cube）：`mx`（MX 量化纯 cube 模板）/ `mx_mix`（MX 量化 + Mix 模板）/ `mix`（非 MX 量化的 Mix 模板）；策略可选（默认 basic）：`streamk` / `split_k` / `al1_full_load` / `broadcast` / `iterbatch_broadcast` / `fixpipe_quant` / `fixpipe_opti` / `activation_quant` / `without_batch` / `l0c_pingpong` 等，多子策略用 `_` 连接，策略含数据类型时数据类型放最前。

**检测模式**：
- 对新增文件名拆解各字段：`^(kernel|block_mmad|block_scheduler|block_epilogue|tile)_{算子}(_{类型})?(_{策略})?\.h$`
- 对照仓内已有合规文件（kernel_matmul_streamk.h、block_mmad_qbmm_mx.h、kernel_qbmm_mx_mix.h、block_mmad_matmul_basic_split_k.h）核对风格

**判断标准**：
- 文件名无法按 层级_算子_类型_策略 拆解 → **风险**（目录检索与组件定位困难）
- 字段顺序错误（如策略放在算子前、类型放在策略后）→ **修正**
- 命名与实现变体不符（如文件名带 mx 但实现无 MX scale 处理）→ **风险**（误导）
- 符合格式且与已有文件风格一致 → **安全**

**合规示例**：
```text
kernel_matmul_basic.h               // 非量化 + basic
kernel_qbmm_mx.h                    // MX 量化纯 cube
kernel_qbmm_mx_mix.h                // MX 量化 + Mix 模板
kernel_qbmm_pertensor_streamk.h     // pertensor + streamk
block_mmad_matmul_basic_split_k.h   // 非量化 + basic_split_k
```

---

### BLAZE-14: 标识符命名风格 `[Kernel]` `[设计原则]`

**规则**：标识符命名遵循仓级约定：类模板参数大驼峰 + `_` 后缀；类型名（类/结构体/枚举/别名/命名空间）大驼峰；函数名（全局/成员）大驼峰；常量/枚举值/宏全大写下划线；类成员变量小驼峰 + `_` 后缀；全局变量 `g_` 前缀小驼峰；局部变量/函数参数/结构体成员小驼峰。

**检测模式**：
- 对 diff 中新增声明逐类核对下表

| 类别 | 命名风格 | 正确示例 | 错误示例 |
|------|----------|---------|---------|
| 类模板参数 | 大驼峰 + `_` 后缀 | `AType_`, `FullLoadMode_` | `a_type` |
| 类型名 | 大驼峰 | `BlockMmad`, `ErrorCode` | `block_mmad`, `PARAMS` |
| 函数名 | 大驼峰 | `ComputeResult()` | `compute_result()`, `initParams()` |
| 常量/枚举值/宏 | 全大写下划线 | `TRANS_A`, `MAX_SIZE` | `transA` |
| 类成员变量 | 小驼峰 + `_` 后缀 | `dataSize_` | `m_dataSize`, `data_size` |
| 全局变量 | `g_` + 小驼峰 | `g_maxCount` | `gMaxCount` |
| 局部变量/参数/struct 成员 | 小驼峰 | `inputSize` | `input_size` |

**判断标准**：
- 新增标识符与上表不符 → **修正**（存量代码不改，仅约束新增）
- 命名风格正确 → **安全**

---

## 四、性能优化规则

> **触发条件**：PR 差异修改 `__aicore__` 计算函数体（变量声明、循环体）

### BLAZE-15: 变量生命周期与寄存器压力 `[Kernel]` `[设计原则]`

**规则**：变量应在首次使用处就近定义，避免过早声明；优先使用局部变量替代成员变量暂存中间状态。Kernel 侧活跃变量超过寄存器个数时编译器通过 Store/Load 溢出到栈内存，直接损失性能。

**检测模式**：
- 检查函数开头集中声明但远端才使用的变量
- 检查仅作循环内中间暂存的状态是否被提升为类成员变量（如 `tempKIter_`）

**判断标准**：
- 变量声明与首次使用之间隔大量无关代码 → **风险**（寄存器溢出）
- 循环内临时状态提升为成员变量 → **风险**（整个类生命周期占用寄存器）
- 就近定义 + 局部变量 → **安全**

**错误示例**：
```cpp
class BlockMmad {
private:
    int32_t tempKIter_;  // 整个类生命周期占用寄存器

    __aicore__ void Run() {
        int32_t result = 0;      // 过早定义
        // ... 大量不涉及 result 的中间计算 ...
        Compute(result);
    }
};
```

**正确示例**：
```cpp
class BlockMmad {
private:
    __aicore__ void Run() {
        // ... 中间计算 ...
        int32_t result = ComputeIntermediate();  // 首次使用时定义
        for (int k = 0; k < kSize; ++k) {
            int32_t tempKIter = k;  // 仅循环内占用寄存器
        }
    }
};
```

---

## 五、AIC/AIV 同步规则

> **触发条件**：PR 差异中出现 `NotifyVector` / `WaitForVector` / `NotifyCube` / `WaitForCube` / `asc_sync_` / `MIX_FLAG` / `FIXPIPE_.*FLAG`
>
> **检视原则**：Mix 模式（AIC+AIV 融合）通过 `gemm/utils/sync.h` 的同步原语握手。检视时追踪 AIC 与 AIV 两侧的调用路径，确认通知/等待成对、flag 一致、每条执行路径次数匹配。

### BLAZE-16: AIC/AIV 同步配对一致性 `[Kernel]` `[红线]`

**规则**：AIC/AIV 交叉通知必须成对且参数一致：AIC 侧 `NotifyVector(id)` 与 AIV 侧对 id 的等待配对；AIV 侧 `NotifyCube(id)` 与 AIC 侧 `WaitForVector(id)` 配对；双方 Mode / Pipe / isMixCV1V2 参数必须一致；自定义 syncID 必须落在合法区间且避开保留区间；每条核执行路径的 arrive/wait 总次数必须匹配。同步原语须按核类型守卫：`NotifyVector` / `WaitForVector` 仅限 AIC 核（`ASC_IS_AIC` 分支），`NotifyCube` / `WaitForCube` 仅限 AIV 核（`ASC_IS_AIV` 分支），避免 MIX kernel 一侧子编译看到另一侧专用同步内建函数（`PIPE_V`、flag id 范围非法）。

**检测模式**：
- 定位所有 `NotifyVector` / `WaitForVector` / `NotifyCube` / `WaitForCube` 调用，核对 id 与模板参数（Mode/Pipe/NpuArch）
- 检查 `gemm/utils/sync.h` 中同步原语定义是否按核类型守卫（`NotifyVector`/`WaitForVector` 定义内 `if ASC_IS_AIC`、`NotifyCube`/`WaitForCube` 定义内 `if ASC_IS_AIV`）
- 追踪 AIC 与 AIV 两条核执行路径，统计各自 arrive/wait 调用点，逐路径比较
- 对自定义 id，比对保留区间

**判断标准**：
- 通知无对应等待（或反向）→ **风险**（数据未就绪即消费，结果错误或死等挂死）
- `NotifyVector`/`WaitForVector` 未限 AIC 核、或 `NotifyCube`/`WaitForCube` 未限 AIV 核 → **风险**（MIX 子编译看到另一侧专用同步内建函数，flag id 范围非法）
- 同通道两侧 id 不一致 → **风险**（同步失效）
- isMixCV1V2 只在一侧为 true → **风险**（950 场景镜像 flag 只有一边 arrive，另一边挂死）
- 自定义 id 与保留区间冲突 → **风险**（flag 语义踩踏）
- AIC 与 AIV 路径 arrive/wait 次数不匹配 → **风险**（挂死）
- 成对、id 一致、参数一致、核类型守卫正确、路径匹配 → **安全**

**检视步骤（Mix 握手核对法）**：
1. 列出 AIC 与 AIV 两侧从入口到 return 的独立执行路径（按 coreIdx / 条件分支展开）
2. 对每条路径收集所有 `Notify*` / `Wait*` 调用点，记录 id、isMixCV1V2、Mode、Pipe
3. 按 flag 通道配对：**0 通道** = `NotifyVector` ↔ `WaitForCube`（AIC→AIV）；**1 通道** = `NotifyCube` ↔ `WaitForVector`（AIV→AIC）
4. 逐路径比对同一通道两侧 arrive/wait 次数与参数是否一致；自定义 syncID 须避开保留区间（见下表）且 < `MIX_FLAG_ID_MAX(16)`

**syncID 保留区间**（以 `gemm/utils/sync.h` 常量为准）：

| 常量 | 值 | 含义 |
|------|-----|------|
| `MIX_AIC_SYNC_AIV_FLAG` | 0 | AIC→AIV 默认握手 flag |
| `MIX_AIV_SYNC_AIC_FLAG` | 1 | AIV→AIC 默认握手 flag |
| `FIXPIPE_AIV_ACK_FLAG_BASE` | 4 | fixpipe UB 槽位复用通知（AIV→AIC），实际 id = base + slot |
| `FIXPIPE_AIC_READY_FLAG_BASE` | 6 | fixpipe UB 槽位数据就绪通知（AIC→AIV），实际 id = base + slot |
| `FIXPIPE_MAX_SLOT_COUNT` | 2 | fixpipe 双槽，占用 id {4,5} 与 {6,7} |
| `MIX_FLAG_ID_MAX` | 16 | 自定义 id 上限（isMixCV1V2 时 id + 16 为镜像 flag，故自定义 id 需 < 16） |

**正确示例**：
```cpp
// AIC 侧：通知 AIV 权重已就绪，随后等待 AIV 反馈
Blaze::Gemm::Sync::NotifyVector(MIX_AIC_SYNC_AIV_FLAG, isMixCV1V2);
// ... AIV 侧以相同 Mode/Pipe 等待同一 flag，处理完毕后 ...
Blaze::Gemm::Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, isMixCV1V2);

// AIV 侧（与上面对应）：isMixCV1V2 参数必须与 AIC 侧一致
Blaze::Gemm::Sync::WaitForCube(MIX_AIC_SYNC_AIV_FLAG);
Blaze::Gemm::Sync::NotifyCube(MIX_AIV_SYNC_AIC_FLAG);
```

**检视要点补充**：Mode（SYNC_MODE_BLOCK=0 / INTER=2 / INTRA=4）与 Pipe（AIC 侧默认 PIPE_FIX、AIV 侧默认 PIPE_V）取值需与同 flag 的既有调用保持一致，不得同 flag 混用不同 Mode/Pipe。

---

## 六、交付件完整性规则

> **触发条件**：PR 涉及新算子交付（新增 `include/blaze/` 组件文件）、或变更 `examples/` / `tests/ut/op_kernel/` / `docs/API/`
>
> **检视原则**：ops-tensor 仓新算子交付件分散在四个目录，检视时按清单核对齐套性，缺件直接提示补充。

### BLAZE-17: 新算子交付件完整性 `[Host/Kernel]` `[红线]`

**规则**：新算子 PR 必须四件套齐套交付：

| 交付件 | 路径 | 必选/按需 |
|--------|------|----------|
| Blaze 组件头文件 | `include/blaze/gemm/kernel/`（Kernel 层入口）、`include/blaze/gemm/block/`（BlockMmad + BlockScheduler）、`include/blaze/gemm/policy/dispatch_policy.h`（策略扩展） | 必选 |
| Tile 原语 / Epilogue / Utils | `include/blaze/gemm/tile/`、`include/blaze/epilogue/block/`（无后处理复用 BlockEpilogueEmpty）、`include/blaze/gemm/utils/` | 按需 |
| 接口文档 | `docs/API/`（Kernel/Block/Tile/Epilogue 各层按新增组件补充） | 必选（有新增组件时） |
| 端到端样例 | `examples/${op}/`：CMakeLists.txt（ops_example_add_executable 注册）、scripts/gen_data.py + verify_result.py、${op}_${variant}/（样例 .cpp + .conf（INI 格式）+ .csv 用例表 + README.md） | 必选 |
| Kernel UT | `tests/ut/op_kernel/${op}/`：CMakeLists.txt（AddOpTestCase）、test_${op}.cpp（TEST_F + KERNEL_RUN_KF）、${op}.cpp（if constexpr 按 OP_TYPE 分发）、${op}_${variant}.h（变体 wrapper）、${op}_tiling_data.h | 必选 |

**检测模式**：
- PR 新增 `include/blaze/gemm/kernel/kernel_${op}_*.h` 时，检查同一 PR 是否包含对应 `examples/${op}/`、`tests/ut/op_kernel/${op}/`、`docs/API/` 变更
- 样例目录核对：gen_data.py（数据生成 + CPU golden）、verify_result.py（NPU vs CPU 精度比对）、CSV 用例表、README.md 是否齐备

**判断标准**：
- 新算子只交组件头文件，无 examples / UT / 文档 → **风险**（不满足上库要求，缺件清单直接列出）
- 样例缺 gen_data.py / verify_result.py / CSV / README 任一 → **风险**
- 四件套齐套 → **安全**

---

### BLAZE-18: UT 增补与不可变约束 `[Host/Kernel]` `[红线]`

**规则**：新增功能必须补充对应 UT；已有 UT 不能修改（不得改已有测试用例的断言、入参与期望结果）。

**检测模式**：
- PR 新增功能（新组件/新策略/新分支）时，检查 `tests/ut/op_kernel/` 是否有对应新增用例
- PR 修改 `tests/ut/op_kernel/` 下已有文件（test_*.cpp、*_tiling_data.h、变体 wrapper）时，区分"修改存量用例"与"新增变体文件"

**判断标准**：
- 新增功能无对应 UT → **风险**（违反上库要求）
- 修改已有 UT 的用例/断言以适配新行为 → **风险**（存量用例不可变；新增行为应新增用例）
- 仅新增测试文件/用例 → **安全**

---

### BLAZE-19: 编译与 UT 验证 `[Host/Kernel]` `[设计原则]`

**规则**：上库前必须完成两类编译验证：样例编译与 Kernel UT 编译执行。Agent 无法在检视环境执行时，须在报告中输出验证命令提示开发者执行。

**验证命令**：
```bash
bash build.sh --examples --ops=${op}        # 样例编译（examples/ 端到端）
bash build.sh --opkernel -u --ops=${op}     # Kernel UT 编译并执行（tests/ut/op_kernel/）
```

**检测模式**：
- PR 涉及组件头文件变更时，报告中附带上述命令与说明（${op} 取 examples/ 或 tests/ut/op_kernel/ 下的算子目录名）

**判断标准**：
- 报告未含编译验证提示 → **补充**
- 已附带验证命令 → **安全**

---

## 七、Tile 层接口约定

> **触发条件**：PR 差异涉及 `include/blaze/gemm/tile/`（或 `epilogue/**/tile/`）下 `.h` 文件的新增/修改
>
> **检视原则**：tile 层接口统一遵循「tag 选择语义、tensor 决定类型与形状、非类型参数做分派」范式（dequant.h 为基准）。检视时确认类型/形状信息均从张量推导、不出现冗余显式参数；算法变体走 tag 全特化；内部实现被编译期强制封装。

### BLAZE-20: Tile 层接口风格统一 `[Kernel]` `[设计原则]`

**规则**：tile 层接口统一遵循「tag 选择语义、tensor 决定类型与形状、非类型参数做分派」范式，具体五点：

- **① tag 分派**：算法变体以算法 tag（如 `GeluTanhTag` / `GeluErfTag`、`B32ToFpTag`）作为类模板第一分派轴，主模板仅声明、按 tag 全特化实现（与 dequant.h 组织方式对齐），禁止用类级 bool/type 参数模拟算法分派。
- **② tensor 决定类型与形状**：输入/输出元素类型从张量元素类型推导（`get_attribute_element_type`），tile 形状从 `NDExt layout.shape()` 读取。**禁止**冗余显式参数：类型模板参数（`DataTypeIn` / `DataTypeOut` / `BiasDtype`）与推导值被 `static_assert` 强制一致、信息量为零；形状参数（`mSize` / `nSize`）与构造 layout 时重复传参，均应删除。
- **③ 非类型参数做分派**：真正影响分派的非类型参数（如 `kPreciseDiv`、`x2Gran`/`x1Gran`/`hasBias`）保留为模板参数，其余下沉推导。
- **④ 实现封装**：共享内部实现通过「基类 + protected 静态成员 + 私有继承」编译期强制封装，外部不继承基类即无法触达内部实现。**禁止**仅依赖 `detail` 命名空间约定防护（外部可直接调用 detail）。
- **⑤ 单一全特化**：tile 拷贝算子 `copy_traits` 采用单一全特化直接继承默认 trait 的 4 参实现；若算子所有消费点仅 `make_copy(Op{})` 单参形式，**禁止**为无实际用途的自定义 trait 保留 2 参偏特化入口（YAGNI）。

**检测模式**：
- 新增/修改 `gemm/tile/`（或 `epilogue/**/tile/`）头文件，检查算法变体是否 tag 分派（主模板仅声明 + tag 全特化）
- 检查类模板是否携带可从张量推导的冗余类型参数（`DataTypeIn` / `DataTypeOut` / `BiasDtype`）
- 检查 Run 等接口是否携带可从 `layout.shape()` 读取的形状参数（`mSize` / `nSize`）
- 检查共享实现是否用基类私有继承封装，而非 `detail` 命名空间
- 检查 `copy_traits` 是否保留无实际调用方的 2 参偏特化（应合并为单一全特化）

**判断标准**：
- 类型/形状参数冗余可推导 → **风险**（信息量为零的 static_assert 或重复传参，应删除并改为推导）
- 算法变体用类级 bool/type 参数模拟分派而非 tag 全特化 → **修正**（统一为 tag 分派）
- 共享实现仅 `detail` 命名空间防护 → **风险**（约定无编译期强制，应改基类私有继承）
- 为无用自定义 trait 保留 2 参偏特化 → **补充**（YAGNI，应合并单一全特化）
- 遵循 tag 分派 + tensor 推导 + 非类型分派 + 基类封装 + 单一全特化 → **安全**

**错误示例**：
```cpp
// 错误：冗余类型参数 DataTypeIn/DataTypeOut，可由张量推导，且被 static_assert 强制一致
template <typename DataTypeIn, typename DataTypeOut>
class Gelu {
    static_assert(IsSame<DataTypeIn, typename GetElementType<InTensor>::type>::value, "...");  // 信息量为零
};

// 错误：形状参数 mSize/nSize 与构造 layout 时重复传参
template <int32_t mSize, int32_t nSize>
void Run(const srcTensor_t& src, dstTensor_t& dst);
```

**正确示例**：
```cpp
// 正确：算法 tag 第一分派轴，主模板仅声明，按 tag 全特化
template <typename GeluTag>
struct Gelu;  // 主模板仅声明

// 元素类型从张量推导，形状从 layout.shape() 读取，仅保留非类型分派参数
template <>
struct Gelu<GeluTanhTag> {
    template <bool kPreciseDiv = false>
    __aicore__ void Run(const srcTensor_t& src, dstTensor_t& dst) {
        auto [rows, cols] = src.layout().shape();  // 形状从 layout 读取，不传 mSize/nSize
        using InType = typename GetElementType<srcTensor_t>::type;  // 类型从张量推导
        // ...
    }
};
```

---

### BLAZE-21: Tile 层对外接口严格控制 `[Kernel]` `[红线]`

**规则**：tile 层对外接口面严格收窄，仅允许两类公共入口，其余一律内部化：

1. **文件级**：外部（kernel/block 层、examples/UT、仓外使用方）只能通过聚合头 `blaze/gemm/tile/datamove.h`（数据搬运）/`compute.h`（计算）访问 tile 组件。单个 tile 实现头、`arch35/` 实现头、顶层转发头（`copy_gm_to_l1.h` 等 5 个）均为内部实现，外部不得直接 include（include 路径机械细节见 BLAZE-05）；顶层转发头仅兼容保留，**不得新增**。
2. **类级**：对外 tile 类仅暴露最小公共接口（`Run` / `Copy` 等必要入口 + 必要类型别名），其余成员、辅助方法、共享实现一律 `private` 或经基类私有继承封装（见 BLAZE-20 ④），不得对外暴露内部实现细节。

新增 tile 组件必须**注册进聚合头**（`datamove.h` / `compute.h`），否则对外不可达。

**检测模式**：
- 检查 tile 聚合头是否覆盖全部对外符号（新增 tile 组件是否注册进 `datamove.h` / `compute.h`）
- 检查是否存在外部（kernel/block/example/UT）直接 include 单个 tile 头 / `arch35/` 实现头 / 顶层转发头
- 检查是否新增顶层转发头
- 检查对外 tile 类的 `public:` 段是否仅含最小入口（`Run`/`Copy` + 类型别名）

**判断标准**：
- 新增 tile 组件未注册进聚合头 → **风险**（对外不可达，使用方 include 聚合头拿不到组件）
- 外部直接 include 单个 tile 头 / `arch35/` 实现头 / 顶层转发头 → **风险**（绕过聚合头，破坏对外接口边界）
- 新增顶层转发头 → **风险**（扩大对外接口面，转发头仅兼容保留不得新增）
- tile 类 `public` 暴露内部实现细节/辅助方法 → **风险**（对外接口未收窄，应 `private` 或基类私有继承封装）
- 仅聚合头对外 + 组件注册齐全 + 类接口最小 → **安全**

**正确示例**：
```cpp
// ① 对外入口只有聚合头：datamove.h / compute.h
//    datamove.h —— 数据搬运类 tile 组件聚合
//    compute.h —— 计算类 tile 组件聚合
// 新增 tile 组件必须注册进对应聚合头（禁止外部 include 单个实现头/arch35/转发头）
#include "blaze/gemm/tile/datamove.h"

// ② 对外类仅暴露最小公共接口，内部实现私有化
template <typename Tag>
struct CopySlice {
public:
    __aicore__ void Run(const srcTensor_t& src, dstTensor_t& dst);  // 唯一对外入口
private:
    // 内部实现细节（辅助方法、中间状态）一律 private
};
```

---

## 搜索关键词总表

| 规则分类 | 搜索关键词 |
|---------|-----------|
| 架构与扩展 | `include/blaze/`, `kernel_`, `block_mmad_`, `block_scheduler_`, `GemmUniversal`, `BlockMmad`, `BlockScheduler`, `dispatch_policy`, `DispatchPolicy`, `ScheduleType`, `MatmulWithScale`, `enable_if_t`, `always_false_v`, `kernel_universal_decl`, `block_mmad_decl` |
| 编码约束 | `AscendC::Aux`, `AuxGetC0Size`, `VECTOR_REG_WIDTH`, `asc_get_vf_len`, `asc_get_sub_block_num`, `asc_get_sub_block_id`, `#include "../`, `#include "./`, `/impl/`, `tile/arch35/`, `datamove.h`, `compute.h`, `kernel_operator.h`, `ASC_DEVKIT_MAJOR`, `using namespace`, `__NPU_ARCH_`, `__CCE_AICORE__`, `TOTAL_UB_SIZE`, `#pragma once`, `static_assert`, `struct Params`, `prelogue` |
| 命名规范 | `kernel_.*\.h`, `block_mmad_.*\.h`, `block_epilogue_.*\.h`, `template <class .*_>` |
| Tile 层接口 | `copy_traits`, `GeluTanhTag`, `GeluErfTag`, `B32ToFpTag`, `Dequant`, `DataTypeIn`, `DataTypeOut`, `BiasDtype`, `mSize`, `nSize`, `detail`, `make_copy`, `datamove.h`, `compute.h`, `copy_gm_to_l1.h`, `copy_gm_to_ub.h`, `copy_weight_ub_to_l1.h`, `pad_mx_kl1.h`, `shift_w4_to_w8.h` |
| 性能优化 | `__aicore__.*void`, 成员变量 `_ =`（初始化表达式） |
| AIC/AIV 同步 | `NotifyVector`, `WaitForVector`, `NotifyCube`, `WaitForCube`, `asc_sync_`, `ASC_IS_AIC`, `ASC_IS_AIV`, `MIX_AIC_SYNC_AIV_FLAG`, `MIX_AIV_SYNC_AIC_FLAG`, `MIX_FLAG_ID_MAX`, `FIXPIPE_AIV_ACK_FLAG_BASE`, `FIXPIPE_AIC_READY_FLAG_BASE`, `FIXPIPE_MAX_SLOT_COUNT` |
| 交付件 | `examples/`, `gen_data.py`, `verify_result.py`, `.csv`, `.conf`, `ops_example_add_executable`, `tests/ut/op_kernel/`, `TEST_F`, `KERNEL_RUN_KF`, `AddOpTestCase`, `docs/API/` |

---

## Agent 行为指南

### 领域判定阶段
使用核心特征（C1~C3）判定代码是否属于 Blaze 领域；仅使用 tensor_api 而无 blaze/ 引用的代码不启用本规则集。遇到疑似 Blaze 组件但不确定（如新策略结构体），检查其是否位于 `include/blaze/` 或引用 `blaze/` 头文件，确认后纳入检视范围。

### 规则匹配阶段
对照差异代码关键词，参照「PR 差异→规则速查」定位规则子集。新算子交付 PR 必读 BLAZE-17 交付件清单并逐项核对。

### 意见输出原则
- 只对 PR 新增/修改行出具意见；存量代码风格不提示（见文件头部"检视边界"）
- 同一根因合并输出：同一 PR 内相同根因的问题（如同一类命名偏差、同一变量的多处同类问题）合并为一条意见，不逐处刷屏

### 权威信息源使用原则
- 本文件静态规则提炼自 ops-tensor 仓 `CODING_CONVENTIONS.md` / `CONTRIBUTING.md`，与仓内文档不一致时以仓内文档为准，并建议更新本 Skill
- Blaze 组件 API 签名以仓库 `docs/API/` 与头文件为准，禁止凭记忆推断
- AscendC 原生 API 约束（高阶 API 判定、同步原语参数）使用 `/ascendc-docs-search` 动态确认
- syncID 保留区间以 `gemm/utils/sync.h` 常量实际值为准，本文件数值仅供快速参考

### 命名规则冲突处理
本仓命名约定（函数大驼峰、成员变量 `_` 后缀等）与通用 C++ 风格规则存在差异时，**以本文件（仓级 CODING_CONVENTIONS）优先**，抑制通用风格规则与仓规范冲突的检视意见。

### 版本演进应对
- `ASC_DEVKIT_MAJOR` 阈值随 CANN 版本递增，头文件隔离方案以 `/ascendc-docs-search` 查询结果为准
- 新硬件代际的 Tile 实现放入 `tile/arch{代际}/` 子目录（如 arch35），策略结构体与 ScheduleType 随算子族扩展持续新增，检视时以 dispatch_policy.h 当前内容为准
- 交付件目录结构随仓演进调整时，以 `CONTRIBUTING.md` 当前版本为准

---

## 规则版本

- **v1.3**（2026-10-22）：应「严格控制 tile 层对外接口」要求新增 BLAZE-21「Tile 层对外接口严格控制」（红线）：文件级仅聚合头 `datamove.h`/`compute.h` 对外、转发头/arch35/单个实现头内部化、新增组件须注册进聚合头；类级仅暴露最小 public 接口、内部实现 private/基类私有继承封装；条款数 20→21，同步更新规则分类概览、快速索引、速查表、搜索关键词表
- **v1.2**（2026-10-22）：依据 ops-tensor 仓 PR #544（tile 层接口统一简化：copy_traits 合并为单一全特化、Dequant/Gelu 移除冗余 BiasDtype/DataTypeIn/DataTypeOut/mSize/nSize 参数、Gelu tag 分派全特化、共享实现 detail 命名空间下沉为 GeluImplBase 基类）新增 BLAZE-20「Tile 层接口风格统一」：tag 分派、tensor 推导类型与形状、非类型参数分派、基类私有继承封装、单一全特化；条款数 19→20，同步更新规则分类概览、快速索引、速查表、搜索关键词表、术语表
- **v1.1**（2026-10-22）：依据 ops-tensor 仓 PR #433（`__CCE_AICORE__` 统一为 `__NPU_ARCH__`、类内 `static constexpr` 下沉函数体）、#507（混合编译安全：`AscendC::TOTAL_UB_SIZE`/`VECTOR_REG_WIDTH` 下沉、`asc_get_vf_len()`/`asc_get_sub_block_*` 替代、sync.h 同步原语核类型守卫 `ASC_IS_AIC`/`ASC_IS_AIV`、include 全路径化）、#541（聚合头抽取声明头 `kernel_universal_decl.h`/`block_mmad_decl.h`、tile 转发头/聚合头约定、禁直接 include arch35 实现头）完善：BLAZE-03 重写为「声明头 + 聚合头」注册模型并补循环包含/回包含禁令与聚合外例外；BLAZE-04 补 `VECTOR_REG_WIDTH`→`asc_get_vf_len()`、`asc_get_sub_block_*`；BLAZE-05 全路径化转强制并新增 tile 聚合头/转发头/arch35 约定；BLAZE-06 补片段自包含（include 声明头）；BLAZE-08 改名「架构宏统一与成员初始化约束」并补 `__CCE_AICORE__` 统一、`static constexpr` 下沉函数体；BLAZE-16 补同步原语核类型守卫；同步更新术语表、速查表、搜索关键词表
- **v1.0**（2026-10-09）：依据 ops-tensor 仓 CODING_CONVENTIONS.md（命名约定 + 编码约束 2.1~2.15）、CONTRIBUTING.md（新算子交付件 + 上库要求）、include/blaze 代码结构（分层架构、dispatch_policy、sync.h）提炼，共 19 条规则
