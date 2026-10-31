# 算子架构与可复用框架（910B/A2 / aclnn 单算子 API）

本文档承载 910B（A2）MC2 skill 的"算子架构子能力"——以已验证功能正常的 [`all_gather_matmul/`](all_gather_matmul) 为蓝本，讲清单算子工程的真实目录分工、aclnn 注册模型，以及 `mc2_common/` 这套可跨算子复用的框架（即"哪些部分通用、可总结为经验"）。读完应能回答：一个新 910B MC2 算子工程长什么样？哪些文件改、哪些文件原样复用？

> 官方依据（CANN 官方仓库 <https://gitcode.com/cann/asc-devkit>）：通算融合指南 `docs/guide/算子实践参考/SIMD算子实现/融合算子编程/通算融合/算子实现.md`（A2 样例 `all_gather_matmul_v2`）；HCCL/Matmul 高阶 API 见 `docs/api/SIMD-API/高阶API/` 对应目录。本 skill 仅覆盖 A2。

> **关键事实**：通算融合算子**不支持 Kernel 直调（`<<<>>>`）与入图（GE）开发，仅支持单算子 API 调用**（`算子实现.md:3`）。故 910B MC2 工程是 **aclnn 单算子 API 工程**（`build.sh` → `.run` 包 + `libcust_opapi.so`），不是 `<<<>>>` 直调工程。本 skill 早期版本误标为"Kernel 直调"，已更正。

## 1. 工程总览（真实结构）

```
all_gather_matmul/
├── build.sh                       # 构建入口：bash build.sh -n x_all_gather_matmul [-c ascend910b]
├── CMakeLists.txt                 # 顶层：project(cann_ops_adv)；ASCEND_COMPUTE_UNIT 默认 ascend910b
├── cmake/                         # 构建框架（config/func/intf/intf_pub/modules/Findalog/scripts）
├── x_all_gather_matmul/           # 目录名必须 == OpDef 类名 snake_case（XAllGatherMatmul）
│   ├── CMakeLists.txt             # 算子级：host include 根、vendored mc2_common include、target_sources
│   ├── op_host/
│   │   ├── x_all_gather_matmul_def.cpp          # 算子定义（IRD）：DataType({FP16,BF16}) → opc 自动烘焙双 dtype
│   │   ├── x_all_gather_matmul_proto.cpp        # infershape（调 mc2_common_infershape）
│   │   ├── op_api/
│   │   │   ├── aclnn_x_all_gather_matmul.cpp    # aclnn 两段式 C 接口（GetWorkspaceSize + launch）
│   │   │   └── aclnn_x_all_gather_matmul.h
│   │   └── op_tiling/
│   │       ├── x_all_gather_matmul_tiling_base.{cpp,h}   # tiling 基类（Mc2CcTilingConfig 填充等）
│   │       ├── x_all_gather_formulaic_tiling.{cpp,h}     # XAllGatherPlusMM（继承 OneCalcOneCommBase）
│   │       └── arch22/
│   │           ├── x_all_gather_matmul_tiling_a2a3.{cpp,h}   # A2/A3 具体子类（SocVersion::SOC910_B）
│   │           └── x_all_gather_formulaic_tiling_a2a3.{cpp,h}
│   ├── op_kernel/
│   │   ├── x_all_gather_matmul.cpp            # kernel 入口 x_all_gather_matmul；GetHcclContext<HCCL_GROUP_ID_0>
│   │   ├── x_all_gather_matmul_base.h        # 公共基类
│   │   ├── x_all_gather_matmul_full_mesh.h   # AIC-only 融合主类（HCCL + MatmulCompute）
│   │   ├── x_all_gather_matmul_tiling.h      # device 侧 tiling 结构体（首字段 Mc2InitTiling）
│   │   └── x_all_gather_matmul_tiling_key.h  # tiling key
│   └── mc2_common/                           # 可复用框架（vendor 进来，跨算子原样复用）
│       ├── op_host/{mc2_common_infershape.*, op_tiling/*, utils/*}
│       ├── op_kernel/{mc2_matmul_block*, mc2_matmul_compute.h, mc2_nd_to_nz.h, mc2_tiling_struct.h}
│       └── utils/{mc2_hcom_topo_info.*, ops_utils.h}
├── examples/                     # 多进程精度测试（golden/run_test.sh/single_server_*；调用 PTA 前须先按路线 B 生成）
└── README.md
```

构建产物：`bash build.sh -n x_all_gather_matmul` → 单次构建即烘焙 **fp16 + bf16 双二进制**（opc 按 `_def.cpp` 的 `DataType({DT_FLOAT16, DT_BF16})` 列表自动 per-dtype 烘焙，运行时按 tensor dtype 经 `multi_kernel` 选择，无需为 bf16 单独重建）→ `build/custom_opp_<arch>.run`（含 `libcust_opapi.so`，导出 `aclnnXAllGatherMatmul` 公共 C 接口）。运行前 `export ASCEND_CUSTOM_OPP_PATH=<vendors/custom_opp 绝对路径>`。

- `-n`/`ASCEND_OP_NAME` 的取值必须等于**外层算子目录名**：`func.cmake` 遍历子目录，取含 `CMakeLists.txt` 的目录 basename 作 `OP_NAME`，再与 `ASCEND_OP_NAME` 比对过滤（`func.cmake:15-31`）。省略 `-n` 时默认 `ALL`（`CMakeLists.txt:9`），编译全部算子。多个算子用分号分隔并加引号。
- 默认 SoC `ascend910b`（`CMakeLists.txt:8`）；顶层 `project(cann_ops_adv)`（未指定 VERSION）；`VENDOR_NAME` 默认 `custom_opp`（`CMakeLists.txt:10`），CPack 产物名 `custom_opp_${CMAKE_SYSTEM_PROCESSOR}.run`（`:552`），安装树暂存于 `build/_CPack_Packages/Linux/External/<run 名>/packages/vendors/${VENDOR_NAME}`。
- `.run` **只在 `-b host` 模式下**被 build.sh 额外拷贝到 `output/`（`build.sh:303-309`）；默认模式产物留在 `build/`。
- `build.sh` 的 `--help` 只列了 `-h/-n/-c/--tiling_key/--verbose`，但实际还解析 `-p|--package-path`、`-b|--build`、`--ccache`、`--parent_job`、`--enable_host_tiling`、`--op_build_tool`、`--ascend_cmake_dir`、`--msdebug`、`--ops-compile-options`、`--tiling-key`、`--clang`、`-f|--changed_list`、`--disable-check-compatible`。

> 经验：dtype 不靠 `-DDTYPE_*` 宏（`x_all_gather_matmul/CMakeLists.txt:109-115` 注释明确），由 opc 自动烘焙；硬编码 `-DDTYPE_*` 会退化为单 dtype。
>
> 经验：op_type 必须带自定义前缀（此处 `X`），否则与 CANN 内置 `AllGatherMatmul` 撞注册表 key，构建与调用都"成功"但实际跑的是内置实现，kernel 里的 printf 不会输出。蓝本据此把整条链路（OpDef 类名、公式化 tiling 类、device tiling 类、aclnn 符号、kernel 入口、PTA op_type）统一加了 `X` 前缀。

## 2. op_host 分工（host 侧）

| 文件 | 职责 |
|------|------|
| `x_all_gather_matmul_def.cpp` | 算子定义（IRD）：`OP_ADD(XAllGatherMatmul)`（`:78`）；输入 `x1`/`x2`/`bias`（bias OPTIONAL）、输出 `y`/`gather_out`；属性 7 个且**全 snake_case**：`group`(String,必选)/`is_trans_a`(Bool,false)/`is_trans_b`(Bool,false)/`gather_index`(Int,0)/`comm_turn`(Int,0)/`rank_size`(Int,0)/`is_gather_out`(Bool,true)；五个张量的 `DataType({DT_FLOAT16, DT_BF16})` 驱动双 dtype 烘焙；`this->MC2().HcclGroup("group")`（`:74`） |
| `x_all_gather_matmul_proto.cpp` | `IMPL_OP_INFERSHAPE(XAllGatherMatmul).InferShape(...).InferDataType(...)`（infershape **与 infer dtype 一起注册**）；infershape 调 `AllGatherMatmulCommonInferShape(context, GATHER_OUT_V1)`（见 §4）→ `y=[dimM*rankSize, dimN]`、`gather_out=[dimM*rankSize, dimKX1]`（`is_gather_out=false` 时为 `[0]`） |
| `op_api/aclnn_x_all_gather_matmul.{cpp,h}` | **aclnn 两段式 C 接口**：`aclnnXAllGatherMatmulGetWorkspaceSize(x1, x2, bias, group, gatherIndex, commTurn, streamMode, output, gatherOut, workspaceSize, executor)` + `aclnnXAllGatherMatmul(workspace, workspaceSize, executor, stream)`；`group` 为 `const char*`（由 torch_npu 分布式创建的 HCCL 通信域名）；`output`/`gatherOut` 由调用方预分配、在第一段就要传入 |
| `op_tiling/x_all_gather_matmul_tiling_base.{cpp,h}` | tiling 基类（508 行）：`Mc2CcTilingConfig(group, tilingData->param.commtype, algConfig)`（`:444`）填充 `mc2InitTiling`/`mc2CcTiling`（`:449-450`）；`param.commtype` 在 `:478` 置 `AicpuComType::HCCL_CMD_ALLGATHER`；`SetSkipBufferWindowCopy`（`:448`）按 `gatherLen` 选 `MC2_BUFFER_TYPE_DEFAULT`/`MC2_BUFFER_TYPE_OUTPUT` |
| `op_tiling/x_all_gather_formulaic_tiling.{cpp,h}` | `XAllGatherPlusMM : public OneCalcOneCommBase`（`:30`，算子专属公式化 tiling） |
| `op_tiling/arch22/x_all_gather_*_a2a3.{cpp,h}` | `XAllGatherPlusMMA2A3 : public XAllGatherPlusMM`（`x_all_gather_formulaic_tiling_a2a3.h:23`）——**A2 具体子类**，实例化 `SocVersion::SOC910_B`（`x_all_gather_matmul_tiling_a2a3.cpp:53`）；`algConfig = "AllGather=level0:fullmesh"`（`:46`）；`IMPL_OP_OPTILING(XAllGatherMatmul)`（`:69`） |

## 3. op_kernel 分工（device 侧 / AIC-only）

核心事实（grep 实证，[`x_all_gather_matmul_full_mesh.h`](all_gather_matmul/x_all_gather_matmul/op_kernel/x_all_gather_matmul_full_mesh.h)）：

- **整算子 AIC-only**：HCCL prepare/Wait/Finalize 与 Matmul 都在 **`if ASCEND_IS_AIC { … }`** 正向门控内执行（`:75,101,220`）；AIV 不执行算子主体。这与通算融合指南的 `ASCEND_IS_AIV` 早返回 + `#define ASCENDC_CUBE_ONLY` 蓝本等价（`算子实现.md:451-457,549`）。**不是** `g_coreType==AIV` 门控。**Finalize 前需跨核同步**：`HcclFinalize()`（`:218`）用 `CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6)`+`CrossCoreWaitFlag(EVENT_ID_6)`（`:224-225`）后 `hccl_.Finalize()`（`:226`）——非 `SyncAll<true>()` 但效果等同。
- **HCCL 对象**：`Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;`（`:46`，无 config 模板参=DEFAULT，运行时由 `ASCEND_IS_AIC` 门控）。
- **生命周期（V2）**：`hccl_.InitV2(contextGM, tilingData); hccl_.SetCcTilingV2(offsetof(Mc2Tiling::XAllGatherMatmulTilingData, mc2CcTiling));`（`:57-58`）。蓝本已用 V2；V1 `Init`/`SetCcTiling` 已废弃（A2 仍支持），详见 [`comm_hccl.md`](comm_hccl.md) §2。
- **AllGather**：主块 `hccl_.AllGather<true>(...)`（`:89`）恒发；尾块（`:92`）仅在 `cfg.tailCnt > 0` 时发（`:91` 条件）。`<true>`=prepare 时同步通知服务端。
- **融合循环**：`MatmulKernelCompute` 中逐 tile `hccl_.Wait(handleId)`（`:143`）→ 跳过本 rank（`:149-150`）→ `mm.Compute(index)`（`:161`，AIC 计算）；L2CACHE 变体 `MatmulKernelComputeL2Cache` 同结构，`Wait` 在 `:188`。
- **Finalize**：`HcclFinalize()`（`:218` 封装，`:220` AIC 门控）内 `CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6)`（`:224`）+ `CrossCoreWaitFlag(EVENT_ID_6)`（`:225`）→ `hccl_.Finalize()`（`:226`，默认 = `Finalize<true>`）。
- **context**：`GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();`（`x_all_gather_matmul.cpp:71`，index∈{0,1} 最多 2 通信域）。

| 文件 | 职责 |
|------|------|
| `x_all_gather_matmul.cpp` | kernel 入口：`extern "C" __global__ __aicore__ void x_all_gather_matmul(aGM, bGM, biasGM, cGM, gatherOut, workspaceGM, tilingGM)`（`:61-62`，**不是 `REGISTER_KERNEL`**）；`REGISTER_TILING_DEFAULT(Mc2Tiling::XAllGatherMatmulTilingData)`（`:64`）；取 HCCL context；dispatch `XAllGatherMatmulFullMesh` |
| `x_all_gather_matmul_base.h` | 公共基类（成员/初始化 + `MatmulKernelLocal()`，`:134-150`） |
| `x_all_gather_matmul_full_mesh.h` | AIC-only 融合主类（230 行）：HCCL 生命周期 + `MatmulCompute` 调用 + 本地 rank 先算 |
| `x_all_gather_matmul_tiling.h` | device 侧 tiling 类 `XAllGatherMatmulTilingData`（**`Mc2InitTiling` 必为首字段**） |
| `x_all_gather_matmul_tiling_key.h` | tiling key 定义 |

> 本地 rank 先算：AllGather 输出含本卡数据，`InnerProcess()` 先调 `this->MatmulKernelLocal()`（`:104`）算本卡，与首轮通信重叠；随后的远端循环里再用 `if (rank == this->rankId_) { continue; }`（`:149-150`）跳过本 rank，避免重复计算——通算两层流水的关键（`算子实现.md:111,514`）。注意跳过逻辑在 `MatmulKernelCompute` 内，`MatmulKernelLocal()` 自身不含该判断。

## 4. mc2_common 可复用框架（"经验"）

`x_all_gather_matmul/mc2_common/`（**在算子目录内部，不是工程根下的独立目录**）是 vendor 自上游 `ops-transformer/mc2`（<https://gitcode.com/cann/ops-transformer>，未在 cannbot 挂载；蓝本已剥离为自包含示例工程，vendor 副本即权威、不依赖上游仓库）的**通用框架**，跨 MC2 算子原样复用。新算子只改 `op_host`/`op_kernel` 的算子专属文件，`mc2_common/` 基本不动。分四块：

### 4.1 op_host/op_tiling（host 侧 tiling 基础设施）
- `formulaic_tiling_datatype.h`：核心枚举/常量。`enum class SocVersion{SOC910_B,SOC310_P,SOC910_93,SOC910_B4,SOC950}`、`enum class KernelType{ALL_REDUCE,ALL_GATHER,REDUCE_SCATTER,ALL_TO_ALL,REDUCE_SCATTER_VIA_ALL_TO_ALL,ALL_REDUCE_VIA_TWO_SHOT}`、`enum class MatmulCalcType{FP16,QUANT,ANTI_QUANT}`、`enum class HCCLType{FULL_MESH,DOUBLE_RING,SWITCH,RING_RANK2_310P,RING_RANK4_310P}`、`enum class TopoType{STANDARD_CARD(4P),EIGHT_P(8P)}`、`struct MatmulParameters/.../HCCLFittingParameters/CutResult/TileArguments/TilingBestBaseBlock`（默认 baseM=256/baseN=256/baseK=128，`:157-159`）。
- `mc2_tiling_struct.h`（host 副本）：**trimmed**，仅留常量 `COMM_ALG_FULL_MESH=1`、`KVALUE_MIN=256`、`KVALUE_MAX=65535`；注释指向 kernel 副本。
- `mc2_tiling_utils.{h,cpp}`（`namespace mc2tiling`）：`GetNpuArch`（用 `platform_ascendc::PlatformAscendC::GetCurNpuArch`，替代已移除的 `GetSocVersion`）、`GetMaxWindowSize`（读 `HCCL_BUFFSIZE`，默认 200MB，`cpp:209-222`）、`supportedRankSizeSet`（**DAV_2201={1,2,4,8}**，`h:205-209`）、`GetRankSize`（调 `Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup`）、`ConvertGeTypeToHcclType`/dtype-size maps、`CheckSuppportedFormat`（仅 `ge::FORMAT_ND`）。**另含 A2 相关容量常量**：`AICPU_NUM_BLOCKS_A2=6`（`h:79`）、`ALL_GATHER_HCCL_MEM_LIMIT=256MB`（`h:87`）、`ALL_GATHER_HCCL_NUM_LIMIT=16`（`h:88`）。> 注意：`CheckSuppportedFormat` 只认 ND，但通用入参校验 `CommonParamCheck` 走的是另一张更宽的 `SUPPORTED_FORMAT` 表（含 NCL/NCDHW/NHWC 等，`h:211-213`），两者不要混淆。
- `matmul_formulaic_tiling.{h,cpp}`（`namespace mc2tiling`）：`class MatmulFormulaicTiling`——`GetCubeTiling(TilingArgs&, TCubeTiling&, [TileL2Tiling&])`、`static GetRankSize(group)`、`SetSocVersion`、`GetBaseBlockParm`；常量 `BASE_BLOCK_M/N/K=128/256/64`（`:37-39`）、`L1_SIZE`、`L0C_SIZE_DB_ON=128KB`（`:50`）、`enum MC2_BUFFER_TYPE`（`:117-126`，枚举值**带前缀**：`MC2_BUFFER_TYPE_DEFAULT=0`/`_OUTPUT`/`_WINDOW_IN`/`_WINDOW_OUT`/`_WORKSPACE`/`_INPUT`/`_COMMOUT`/`_END`）、`struct TilingArgs`（`:159-194`，含 cmdType/rankDim/usedCoreNum/orgM/N/K/aicCoreNum/commTurn/commAlg/isATrans/isBTrans/isBias/...）、`struct SoCInfo`（`:263-271`，`platform_ascendc::SocVersion socVersion = ASCEND910B`）。
- `hccl_performance.{h,cpp}`：`class HCCLPerformanceModel`——`CommTime`/`InverseCommTime`（分段抛物→线性）、`GetRankTileNum`（非 ALL_REDUCE 走 `GetFullMeshRankTileNum`，ALL_GATHER→`rankDim-1`；ALL_REDUCE→1，`cpp:102-114`）、`GetLinearThresholdLen`；常量 `HCCL_MIN_TILE_LEN=64KB`（`:21`）、`FULL_MESH_TIME_FACTOR=2.0`（`:24`）、`FITTING_RANK=8`（`:29`）、`LOCAL_REDUCE_FACTOR=0.4`（`:30`）。
- `matmul_performance.{h,cpp}`（`namespace MatmulPerformance`）：`class MatmulPerformanceModel`——`MatmulTime`/`InverseMatmulTime`/`FindCubeUtil`/`GetMatmulGradient`；常量 `COMPUTES_PER_CYCLE=4096`（`:21`）、**`MARK_CORE_NUM_SOC910B=20`**（`:30`）、`CYCLE_PER_MICRO_SEC=1.8*1024`（`:32`）、`MAX_CUBE_UTIL=0.95`（`:54`）、`AVERAGE_CUBE_UTIL=0.75`（`:55`）。
- `hccl_formulaic_tiling.{h,cpp}`：**`class FormPartition`**（`:45`，M 轴切分：`GenerateInitialPartition`/`FitTileLengthDiscrete/Continuous`/...）+ **`class OneCalcOneCommBase`**（`:112`，公式化 tiling 基类，组合 `MatmulPerformanceModel`+`HCCLPerformanceModel`+`FormPartition`）；常量 **`MAX_TILE_CNT=16`**（`:38`）、`commGrowRatio=1.15`（`:39`）、`ALLGATHERMM_COMMTIME_FACTOR=2`（`:40`）、`ALLREDUCE_COMMTIME_FACTOR=2.0517`（`:41`）、`REDUCESCATTER_COMMTIME_FACTOR=2.09136`（`:42`）（详见 [`pipeline_tuning.md`](pipeline_tuning.md) §5）。
- `mc2_common_infershape.{h,cpp}`（`namespace ops`）：`AllGatherMatmulCommonInferShape(context, gatherIndex)`——属性槽常量 `GROUP=0/AG_IS_TRANS_A=1/AG_IS_TRANS_B=2/RANK_SIZE=5/GATHER_OUT_V1=6/GATHER_OUT_V2=8`（`h:24-34`）；输出 `y=[dimM*rankSize,dimN]`（`cpp:112-113`）、`gather_out=[dimM*rankSize,dimKX1]`（`cpp:128-130`）或 `[0]`（`cpp:132-133`）。> 注意两点坑：① 本算子只有 7 个属性（槽位 0–6），`GATHER_OUT_V2=8` 是给属性布局不同的其它 MC2 算子留的，本算子传 `GATHER_OUT_V1`；② 形参名叫 `gatherIndex`，实际索引的是 `is_gather_out` 属性槽，不是 `gather_index` 属性。

### 4.2 op_kernel（device 侧可复用 helper）
- `mc2_matmul_compute.h`：**`template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T> class MatmulCompute`**（`:22-23`）——是 `MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm_` 的封装（`:44`），`Init→SetSingleShape/SetTensorA/B/SetBias→Iterate→GetTensorC` 循环 + **`SetFlag`/`WaitFlag<HardEvent::FIX_M>`** 同步（`:130-132,161-162,195-196`，配 `GetTPipePtr()->FetchEventID`；**不是** `CrossCoreSetFlag`）；`typename BlockType<T>::PARAMS block_` 选块策略。
- `mc2_matmul_block.h`：`class MatmulBaseBlockMC2`——多核块索引/偏移（`mBlockCnt/nBlockCnt/totalBlockCnt`、`GetBlockStartIdx`/`UpdateBlockParams`/`CalcGMOffset`）。行/列序与原子写的实际写法（`:83-92`）：`isRowOrder` 默认 `true`，仅当 `tiling_.N > 5 * tiling_.M` 时置 `false`；`isAtomic` 默认 `false`，仅当 `isTransA` 时置 `true`。
- `mc2_matmul_block_l2cache.h`：`class MatmulBaseBlockL2Cache: public MatmulBaseBlockMC2`（L2 分裂变体）+ **`enum SplitType{DEFAULT=0,L2CACHE=1}`**（`:142-146`）+ `BlockType<DEFAULT|L2CACHE>::PARAMS` 选型（`:149-163`）。
- `mc2_nd_to_nz.h`（580 行）：ND→NZ 转换 + **`g_coreType` 在此合法**（`SET_G_CORE_TYPE_IS_AIV/AIC` 宏定义在 `:22-28`，判定在 `:98`（`!= AIV`）、`:479`（`== AIV`，AIV 转 NZ）、`:531`（`== AIC`，AIC 消费））+ `CastBFtoFloat`（`:529`，`__CCE_AICORE__==220` AIC 分支 `:530-531`）。**注意**：`g_coreType` 是 ND2NZ helper 的 AIV/AIC 分工机制，**不是** HCCL 门控（HCCL 用 `ASCEND_IS_AIC`）。
- `mc2_tiling_struct.h`（kernel 副本）：**full** structs，`namespace Mc2Tiling` 共 7 个：`MC2ServerCfg`/`MC2HcommCfg`（含 `skipBufferWindowCopy` `:37`、`stepSize` `:38`）/`Mc2Msg`（含 `notifyOff` `:64`/`notifyBeginCnt` `:66`/`notifyEndCnt` `:67`）/`RCSTiling`（rankDim/rankID/commtype/tileCnt/tailM/rankM/N/K/isTransposeA/B/gatherLen/aicCoreNum/dataType/...）/`TileL2Tiling`（mL2TileCnt/nL2TileCnt/enableL2Tile/...）/`TileInfo`/`MC2MatmulV3TilingData`；另有 constexpr `COMM_ALG_FULL_MESH`/`KVALUE_MIN`/`KVALUE_MAX`。

### 4.3 utils
- `mc2_hcom_topo_info.{h,cpp}`（`namespace Mc2Hcom`，`class MC2HcomTopology`）：**eager 模式** rank 解析——`HcclLibLoader` dlopen `libhccl.so`（经 `ASCEND_HOME_PATH` 定位 `x86_64-linux/lib64` 或 `aarch64-linux/lib64`，`cpp:39-51`）后 dlsym `HcomGetRankSizeEx`；`CommGetInstSizeByGroup`/`CommGetGroupLocalWindowSize`（失败回退 `GetMaxWindowSize()`）/`TryGetGroupTopoType`（失败时默认 `COMM_MESH`/FullMesh）。三者都有 `#ifdef BUILD_OPEN_PROJECT` 两套分支（`build.sh` 恒传 `-DBUILD_OPEN_PROJECT=ON`）；图模式下 `MC2HcomTopology` 另走 `libhccl_fwk.so`（`cpp:95-110`）。> 经验：本 A2 单算子工程用 eager 路径；多机/非 fullmesh 需 graph 模式（torchair）。> 这也是为什么构建/运行前必须先 `source set_env.sh`——`libhccl.so` 不在默认 `LD_LIBRARY_PATH` 里时，dlopen 与 `import torch_npu` 都会失败。
- `ops_utils.h`（`namespace OpsUtils`）：`Ceil/CeilAlign/CeilDiv/FloorDiv/Aligned/FloorAlign` 等模板工具。
- `hcom_topo_info.h`：vendor 的 `ge::HcomTopoInfo`（topo/rank_size/local_window_size 查询）。

### 4.4 tiling 类层级（算子专属层叠在 mc2_common 之上）
```
OneCalcOneCommBase     (x_all_gather_matmul/mc2_common/op_host/op_tiling/hccl_formulaic_tiling.h:112)
   ↑
XAllGatherPlusMM       (x_all_gather_matmul/op_host/op_tiling/x_all_gather_formulaic_tiling.h:30)   算子专属
   ↑
XAllGatherPlusMMA2A3   (x_all_gather_matmul/op_host/op_tiling/arch22/x_all_gather_formulaic_tiling_a2a3.h:23)  A2 具体子类
   实例化 (x_all_gather_matmul_tiling_a2a3.cpp:53):
     XAllGatherPlusMMA2A3 tileFormulate(args, args.rankDim, KernelType::ALL_GATHER, SocVersion::SOC910_B);
```

## 5. 可复用性经验（新算子起手）

| 改动范围 | 文件 | 说明 |
|---------|------|------|
| **原样复用（vendor 不动）** | `x_{op}/mc2_common/**` | 整套通用框架，跨算子不变 |
| **原样复用** | `cmake/**`、`build.sh`、`CMakeLists.txt` | 构建框架 |
| **改算子名/输入输出/dtype** | `op_host/*_def.cpp`、`*_proto.cpp` | IRD + DataType 列表（驱动双 dtype 烘焙） |
| **改 aclnn 接口** | `op_host/op_api/aclnn_*.{cpp,h}` | 两段式签名 |
| **改 tiling** | `op_host/op_tiling/*_base.{cpp,h}`、`*_formulaic_tiling.{cpp,h}`、`arch22/*_a2a3.{cpp,h}` | 算子专属 + A2 子类 |
| **改融合主体** | `op_kernel/*.cpp`、`*_base.h`、`*_full_mesh.h`、`*_tiling.h`、`*_tiling_key.h` | HCCL 蓝本 + MatmulCompute 调用 |
| **生成 PTA** | 不在本路线提供 | 算子可跑通后走 `ops/torch-ascendc-op-extension` 路线 B |
| **改测试** | `examples/golden.py`、`single_server_*`、`run_test.sh` | golden + 多轮随机 shape/dtype |

> 经验：`mc2_common` 是"算子=通信+计算+复用框架"中的"复用框架"层；新算子的 80% 工作量在 `op_host/op_tiling`（切分策略）与 `op_kernel/*_full_mesh.h`（通算两层流水编排），`mc2_common` 几乎零改动。

## 6. aclnn 注册模型（构建链）

```
op_host/*_def.cpp  ──[OP_BUILD_TOOL/ascendc_impl_build.py]──▶  aclnn op_api（aclnnXxxGetWorkspaceSize + aclnnXxx）自动生成
op_host/op_tiling/* ──[tiling_key + opc]──▶  liboptiling.so（含 tiling key 注册）
op_kernel/*.cpp     ──[opc per-dtype 烘焙]──▶  per-dtype kernel binary（multi_kernel 运行时择）
                                                        ↓
                    install(... DESTINATION packages/vendors/${VENDOR_NAME}/...)
                                                        ↓
                              CPack/makeself ──▶  build/custom_opp_<arch>.run（含 libcust_opapi.so）
                                                        ↓ 运行时
                              export ASCEND_CUSTOM_OPP_PATH=<vendors/custom_opp 绝对路径>
```

- `x_all_gather_matmul/CMakeLists.txt`：vendored mc2_common 的 7 条 `-I` include（`:101-107`：`op_kernel`/`mc2_common/op_kernel`/`mc2_common/op_host`/`mc2_common/op_host/op_tiling`/`mc2_common/utils`/`mc2_common`/`mc2_common/op_api`——最后一条目录在蓝本中并不存在，是无害的空占位）；kernel 编译选项完整为 `--cce-auto-sync=off -DHCCL_COMM -Wno-deprecated-declarations -Wno-error=option-ignored -mllvm -cce-aicore-hoist-movemask=false`（`:116-121`）；host include 根 `_X_AGM_HOST_INC`（`:124-161`）= 10 条工程内路径 + 按 `aarch64-linux`/`arm64-linux`/`${CMAKE_SYSTEM_PROCESSOR}-linux` 探测追加的 5 个 CANN 根（`include/op_common`、`pkg_inc`、`include/external`、`asc/include/tiling`、`include/platform`），显式加是为避免与 intf_pub 的 register/graph 头冲突。`ascendc/include` 属 kernel 侧 `-I`（`:22-28`），不在 host include 根内。
- `optiling` target 的源文件**显式列出**（无 `file(GLOB)` 跨 mc2_common，`:169-180`）：4 个算子专属（`x_all_gather_matmul_tiling_base.cpp`、`x_all_gather_formulaic_tiling.cpp`、`arch22/x_all_gather_matmul_tiling_a2a3.cpp`、`arch22/x_all_gather_formulaic_tiling_a2a3.cpp`）+ 6 个 vendored mc2_common（`hccl/matmul_formulaic_tiling`+`*_performance`+`mc2_tiling_utils`+`mc2_hcom_topo_info`），共 10 个。
- `opsproto`（infershape，`:192-195`）含 `op_host/x_all_gather_matmul_proto.cpp` + vendored `mc2_common/op_host/mc2_common_infershape.cpp`（后者必须在 opsproto 以解析 `AllGatherMatmulCommonInferShape` 符号）。

## 7. 后续阅读

| 想了解 | 读 |
|--------|-----|
| 通信层 HCCL V2 生命周期 | [`comm_hccl.md`](comm_hccl.md) |
| 计算层 AscendC::Matmul | [`matmul_fusion.md`](matmul_fusion.md) |
| 架构心智/AIC-only | [`mc2_architecture.md`](mc2_architecture.md) |
| PTA 接口生成 | `ops/torch-ascendc-op-extension` 路线 B（aclnn 注册） |
| 改造食谱（按文件） | [`codebase_map.md`](codebase_map.md) |
| tileCnt 调优 | [`pipeline_tuning.md`](pipeline_tuning.md) |
