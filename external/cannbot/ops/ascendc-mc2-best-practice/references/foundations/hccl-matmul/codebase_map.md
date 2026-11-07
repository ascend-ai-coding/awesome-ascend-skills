# 参考工程（all_gather_matmul/）改造食谱

本文档是 Agent 在阶段二（开发）的实操指南：从已验证功能正常的基底工程 [`all_gather_matmul/`](all_gather_matmul) 复制起手，按 `[REUSE]` / `[MODIFY]` 标记定点改造为新的 910B（A2）MC2 算子（HCCL 高阶 API + `AscendC::Matmul`，aclnn 单算子 API，无 quant）。

> 工程真实结构详见 [`op_architecture.md`](op_architecture.md)。本 skill 仅覆盖 A2。

读完本文档应能回答：哪些文件不能动？改不同种类的 MC2 算子分别要改哪些文件？每个文件改动的典型 diff 是什么？

---

## 1. 工程总览（真实结构）

```
all_gather_matmul/
├── build.sh                                  # [REUSE]  构建入口：bash build.sh -n <外层算子目录名> [-c ascend910b]
├── CMakeLists.txt                            # [REUSE]  顶层 project(cann_ops_adv)；ASCEND_COMPUTE_UNIT=ascend910b
├── README.md                                 # [MODIFY] 工程说明
├── cmake/                                    # [REUSE]  构建框架
│   ├── config.cmake                          #          ASCEND_CANN_PACKAGE_PATH 解析、路径变量、prepare.sh
│   ├── func.cmake                            #          op_add_subdirectory / add_bin_compile_target 等函数
│   ├── intf.cmake / intf_pub.cmake           #          host INTERFACE 库（include/link/defs）
│   ├── modules/Findalog.cmake               #          alog 查找
│   └── scripts/{check_version_compatible.py, prepare.sh}
├── x_all_gather_matmul/                      # 目录名必须 == OpDef 类名 snake_case（XAllGatherMatmul）
│   ├── CMakeLists.txt                        # [MODIFY] host include 根、vendored mc2_common include、target_sources
│   ├── op_host/
│   │   ├── x_all_gather_matmul_def.cpp       # [MODIFY] 算子定义（IRD）：输入输出/属性/DataType({FP16,BF16})
│   │   ├── x_all_gather_matmul_proto.cpp     # [MODIFY] infershape + infer dtype（调 mc2_common_infershape）
│   │   ├── op_api/aclnn_x_all_gather_matmul.{cpp,h}  # [MODIFY] aclnn 两段式 C 接口
│   │   └── op_tiling/
│   │       ├── x_all_gather_matmul_tiling_base.{cpp,h}    # [MODIFY] tiling 基类（Mc2CcTilingConfig 填充）
│   │       ├── x_all_gather_formulaic_tiling.{cpp,h}      # [MODIFY] XAllGatherPlusMM
│   │       └── arch22/x_all_gather_matmul_tiling_a2a3.{cpp,h}
│   │           + arch22/x_all_gather_formulaic_tiling_a2a3.{cpp,h}  # [MODIFY] A2 子类（SocVersion::SOC910_B）
│   ├── op_kernel/
│   │   ├── x_all_gather_matmul.cpp           # [MODIFY] kernel 入口（extern "C" __global__ __aicore__）；GetHcclContext<HCCL_GROUP_ID_0>
│   │   ├── x_all_gather_matmul_base.h        # [MODIFY] 公共基类（含 MatmulKernelLocal）
│   │   ├── x_all_gather_matmul_full_mesh.h   # [MODIFY] AIC-only 融合主类（HCCL + MatmulCompute）
│   │   ├── x_all_gather_matmul_tiling.h      # [MODIFY] device tiling 类 XAllGatherMatmulTilingData（Mc2InitTiling 首字段）
│   │   └── x_all_gather_matmul_tiling_key.h  # [MODIFY] tiling key
│   └── mc2_common/                           # [REUSE]  vendored 通用框架（跨算子原样复用，详见 op_architecture §4）
│       ├── op_host/{mc2_common_infershape.{cpp,h}, op_tiling/*}
│       ├── op_kernel/{mc2_matmul_block{,_l2cache}.h, mc2_matmul_compute.h, mc2_nd_to_nz.h, mc2_tiling_struct.h}
│       └── utils/{mc2_hcom_topo_info.{cpp,h}, ops_utils.h}
└── examples/                                 # [MODIFY] golden/run_test/single_server_*
    ├── golden.py, test_golden_cpu.py
    ├── single_server_{config.ini,gen_data.py,run.py,check_result.py}
    └── run_test.sh
```

**原则**：`[REUSE]` 文件常规不动（尤其 `mc2_common/`、`cmake/`、`build.sh`、顶层 `CMakeLists.txt`）；`[MODIFY]` 文件按需动。改动量越大，编译/精度风险越高。

> ⚠️ 旧版本 skill 误把 950 样板的 `run.sh`/`cmake/ascend.cmake`/`cmake/hccl.cmake`/`common/`/`include/`/`src/`/`scripts/`/`third_party/` 结构当作 910B 结构——**这些文件在真实工程中不存在**，已删除该虚构描述。

> ⚠️ **蓝本自带的 `all_gather_matmul/README.md` 有几处与工程实际不符**（该 README 属参考实现、本 skill 不修改，以本文档为准）：
> - §1 目录树的顶层名沿用了上游仓名 `x_ops_lite_plus_2_reference/`，且多出一层 `x_all_gather_matmul/` 包裹；实际工程根就是 `all_gather_matmul/`，算子目录只有一层。
> - §1 列了 `算子代码重构报告.md`，**该文件在工程中不存在**。
> - §3 写 `bash build.sh -n op`，`op` 不是有效算子名；正确取值是外层算子目录名 `x_all_gather_matmul`（或省略 `-n` 编译全部）。
> - §5 写"多卡 NPU（≥2，默认 8）"，但 `examples/single_server_config.ini` 里 `rankSize = 2`。

---

## 2. 改造场景速查

| 场景 | op_host | op_kernel | op_tiling | PTA | examples |
|------|---------|-----------|-----------|-----|----------|
| **换 dtype**（加 fp32 等） | `_def.cpp`（DataType 列表）、`_proto.cpp` | 不变 | dtype 枚举映射 | csrc schema/converter dtype | golden/gen_data dtype |
| **换 shape**（M/N/K） | 不变 | 不变 | 公式化 tiling 动态算 | 不变 | gen_data shape |
| **换通信原语**（AllGather→AllReduce） | `_def.cpp`（输出语义）、`aclnn_*` | `_full_mesh.h`（改蓝本 B：一算一通信）、`_base.h` | `_tiling_base.cpp`（`param.commtype = HCCL_CMD_ALLREDUCE`）、arch22 子类的 `KernelType` 与 `algConfig` 字符串 | csrc（算子名/attrs） | golden 语义、check 输出 |
| **加 bias** | `_def.cpp`（bias 输入）、`aclnn_*`（签名） | `_full_mesh.h`（SetBias+EnableBias） | tiling `isBias` | csrc schema（bias 参数） | gen_data（蓝本**暂不支持非 0 bias**，用全零 bias 走 bias 分支） |
| **换卡数**（rankDim） | 不变 | 不变 | tiling rankDim（受 `supportedRankSizeSet` DAV_2201={1,2,4,8} 约束） | 不变 | config.ini rankSize |
| **改流水深度**（tileCnt） | 不变 | `_full_mesh.h`（AllGather repeat / 循环） | tiling `tileCnt` | 不变 | 不变 |
| **L2CACHE 开/关** | 不变 | `BlockType<L2CACHE>` / `MatmulCompute` ComputeWithL2Cache | `enableL2Tile` | 不变 | 不变 |

---

## 3. 关键文件改造食谱

### 3.1 `x_all_gather_matmul/op_host/x_all_gather_matmul_def.cpp`（算子定义）——[MODIFY]

蓝本实际形态（`OP_ADD(XAllGatherMatmul)`，`:78`）：输入 `x1`/`x2`/`bias`（bias OPTIONAL），输出 `y`/`gather_out`，五个张量的 DataType 均为 `{ge::DT_FLOAT16, ge::DT_BF16}`。属性 **7 个，全 snake_case，顺序即 attr 槽位**：

| 槽位 | 名称 | 类型 | 必选 | 默认 |
|---|------|------|------|------|
| 0 | `group` | String | REQUIRED | — |
| 1 | `is_trans_a` | Bool | OPTIONAL | `false` |
| 2 | `is_trans_b` | Bool | OPTIONAL | `false` |
| 3 | `gather_index` | Int | OPTIONAL | `0` |
| 4 | `comm_turn` | Int | OPTIONAL | `0` |
| 5 | `rank_size` | Int | OPTIONAL | `0` |
| 6 | `is_gather_out` | Bool | OPTIONAL | `true` |

典型改动：
1. **算子名/输入输出**：按新算子语义改 IRD；`DataType({DT_FLOAT16, DT_BF16})` 驱动 opc 自动烘焙双 dtype（无需 `-DDTYPE_*`）。op_type **必须带自定义前缀**（蓝本用 `X`），否则与 CANN 内置同名算子（`AllGatherMatmul`）撞注册表 key，你的实现会被静默丢弃。
2. **属性**：槽位顺序被 `mc2_common_infershape.h` 的常量硬编码引用（`GROUP=0`/`AG_IS_TRANS_A=1`/`AG_IS_TRANS_B=2`/`RANK_SIZE=5`/`GATHER_OUT_V1=6`），**增删属性会错位**，改动时须同步这些常量。
3. **HcclGroup**：`this->MC2().HcclGroup("group")`（`:74`）声明通信域属性（host 须先配通信域名，kernel 侧 `GetHcclContext<HCCL_GROUP_ID_0>` 才能取回 context）。

### 3.2 `x_all_gather_matmul/op_host/op_api/aclnn_x_all_gather_matmul.{cpp,h}`（aclnn 接口）——[MODIFY]

aclnn 两段式（蓝本原文，`aclnn_x_all_gather_matmul.h:40-51`）：
```cpp
__attribute__((visibility("default"))) aclnnStatus aclnnXAllGatherMatmulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const char* group,
    int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
    const aclTensor* output, const aclTensor* gatherOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);
__attribute__((visibility("default"))) aclnnStatus aclnnXAllGatherMatmul(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```
改动：按新算子签名调整参数；`group` 为 `const char*`（C 字符串透传）；`output`/`gatherOut` 是**调用方预分配的输出张量**，也出现在第一段签名里（不是只在第二段）。注意签名里的属性是驼峰（`gatherIndex`/`commTurn`/`streamMode`），与 OpDef 的 snake_case 属性名（`gather_index`/`comm_turn`）不同——前者是 aclnn C 形参名，后者是 GE 属性名，两套命名各自独立。PTA 侧 `EXEC_NPU_CMD_V1` 按同序 dlsym——正因为是裸符号名 dlsym，导出名必须带前缀，否则会被内置 `libopapi.so` 的同名符号覆盖。

### 3.3 `x_all_gather_matmul/op_host/op_tiling/`（tiling）——[MODIFY]

- `x_all_gather_matmul_tiling_base.cpp`（508 行）：`Mc2CcTilingConfig(group, tilingData->param.commtype, algConfig)`（`:444`）→ `GetTiling(mc2InitTiling)` + `GetTiling(mc2CcTiling)`（`:449-450`）；`commtype` 在 `:478` 置为 `mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER`（**不是构造处的字面量**，换通信原语改这里）；`SetSkipBufferWindowCopy`（`:448`）由 `gatherLen` 选 `MC2_BUFFER_TYPE_DEFAULT`/`MC2_BUFFER_TYPE_OUTPUT`；`Mc2InitTiling` 必为 TilingData 首字段。
- `arch22/x_all_gather_matmul_tiling_a2a3.cpp`：`algConfig` 字面量 `"AllGather=level0:fullmesh"`（`:46`）；实例化 A2 子类 `XAllGatherPlusMMA2A3 tileFormulate(args, args.rankDim, KernelType::ALL_GATHER, SocVersion::SOC910_B)`（`:53`）；`IMPL_OP_OPTILING(XAllGatherMatmul)`（`:69`）。
> 换通信原语时改 `param.commtype`/`KernelType`/`algConfig`；公式化 tiling 与 perf 模型在 `mc2_common` 内复用（不动）。类名带 `X` 前缀（`XAllGatherPlusMM`/`XAllGatherPlusMMA2A3`），与上游 `ops-transformer` 的无前缀版本区分。

### 3.4 `x_all_gather_matmul/op_kernel/x_all_gather_matmul_full_mesh.h`（融合主类）——[MODIFY]

AIC-only 蓝本（grep 实证，[`x_all_gather_matmul_full_mesh.h`](all_gather_matmul/x_all_gather_matmul/op_kernel/x_all_gather_matmul_full_mesh.h)，全文 230 行）：
```cpp
Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;                  // :46
// Init V2（蓝本已用 InitV2/SetCcTilingV2；V1 Init/SetCcTiling 已废弃）
hccl_.InitV2(contextGM, tilingData);                 // :57
hccl_.SetCcTilingV2(offsetof(Mc2Tiling::XAllGatherMatmulTilingData, mc2CcTiling)); // :58

// HcclPrepare()（:73）
if ASCEND_IS_AIC {                                    // :75  AIC-only 门控（非 g_coreType==AIV）
    handleId_ = hccl_.AllGather<true>(this->aGM_, this->gatherGM_, aTileCnt,
                                      HcclDataType(cfg.dataType), aRankCnt, cfg.tileCnt);   // :89
    if (cfg.tailCnt > 0) {                                                                   // :91 尾块条件下发
        tailHandleId_ = hccl_.AllGather<true>(this->aGM_ + aTileOffset, this->gatherGM_ + aTileOffset,
                                              aTailCnt, HcclDataType(cfg.dataType), aRankCnt, cfg.tailCnt); // :92
    }
}
// InnerProcess()（:99）
if ASCEND_IS_AIC {                                    // :101
    this->MatmulKernelLocal();                        // :104 本地 rank 先算（与首轮通信重叠）
    // MatmulKernelCompute（:141 循环）：逐 tile hccl_.Wait(handleId)（:143）
    //   → 跳过本 rank `if (rank == this->rankId_) continue;`（:149-150）→ mm.Compute(index)（:161）
    // L2CACHE 变体 MatmulKernelComputeL2Cache：同结构，Wait 在 :188
}
// HcclFinalize()（:218 封装，:220 AIC 门控）：CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6)（:224）+ CrossCoreWaitFlag（:225）→ hccl_.Finalize()（:226，默认=true）
```
> 注意事项：`AllGather<true>` 的 `<true>`=prepare 时同步通知服务端；主块必发、尾块仅 `tailCnt>0` 时发（**不是无条件两次 prepare**）；`Wait` 逐 tile 驱动计算；一通信域 Prepare 总调用≤63。`MatmulKernelLocal()` 定义在 `x_all_gather_matmul_base.h:134-150`，本身不含 rank 跳过逻辑；跳过本 rank 的 `continue` 在 `_full_mesh.h` 的 `MatmulKernelCompute` 里。**Finalize 前必跨核同步**：蓝本 `HcclFinalize()` 用 `CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6)`+`CrossCoreWaitFlag(EVENT_ID_6)`（`:224-225`），非 `SyncAll<true>()` 但效果等同。`g_coreType` 不用于 HCCL 门控（仅 `mc2_nd_to_nz.h` 的 ND2NZ 分工用）。详见 [`mc2_architecture.md`](mc2_architecture.md) §4、[`comm_hccl.md`](comm_hccl.md) §2。

### 3.5 `x_all_gather_matmul/op_kernel/x_all_gather_matmul_tiling.h`（device tiling 类）——[MODIFY]

蓝本原文（`:30-42`，是 `class` + `public:`，不是 `struct`）：
```cpp
class XAllGatherMatmulTilingData {
    public:
        Mc2InitTiling mc2InitTiling;   // 必为首字段（≤64B）
        Mc2CcTiling   mc2CcTiling;     // ≤280B；单算子最多 8 个通信任务
        TCubeTiling   tileTiling;      // 主块
        TCubeTiling   tailTiling;      // 尾块
        TCubeTiling   localTiling;     // 本地 rank
        Mc2Tiling::TileL2Tiling  tileL2Tiling;   // 三套 L2 分裂参数，与三套 TCubeTiling 一一对应
        Mc2Tiling::TileL2Tiling  tailL2Tiling;
        Mc2Tiling::TileL2Tiling  localL2Tiling;
        Mc2Tiling::RCSTiling     param;          // rankDim/rankM/N/K/tileCnt/tailM/commtype/...
        Mc2Tiling::XAllGatherSoc socParam;       // 算子专属 SoC 参数
};
```
> 类名须与 kernel 入口 `REGISTER_TILING_DEFAULT(Mc2Tiling::XAllGatherMatmulTilingData)`（`x_all_gather_matmul.cpp:64`）及 `SetCcTilingV2` 的 `offsetof`（`_full_mesh.h:58`）一致，改名时三处同步。

### 3.6 PTA（本蓝本不提供，不在本路线生成）

蓝本不含 `torch_ops_extension/`。算子 `.run` 可跑通后，PTA 必须走 `ops/torch-ascendc-op-extension` **路线 B**（aclnn 注册：`routes/aclnn-registry.md` + `templates/aclnn/`）。契约从本工程 `op_host/*_def.cpp`、`op_api/aclnn_*.h` 采集，禁止在本路线内再写一套绑定层。

命名红线与该 skill 一致：aclnn 导出符号与 GE op_type 必须带自定义前缀（蓝本用 `X`），否则会被内置同名实现静默抢占。

### 3.7 `examples/`（精度测试）——[MODIFY]
- `golden.py`：CPU fp32 参考实现（`torch.cat` 模拟 gather dim0 → `torch.matmul` → +bias，全程 fp32）；`test_golden_cpu.py` 是它的纯 CPU 自测（3 个用例：带 bias / 无 bias 无 gather_out / trans_b）。改算子语义时同步改。
- `single_server_gen_data.py`：per-rank x1 分片 + x2/bias。**bias 语义**：`hasBias=0` 时 bias 为 `None`；`hasBias=1` 时 bias 为**全零向量**——因为本算子暂不支持非 0 bias（`gen_data.py:55-59` 注释），零 bias 用于走通 bias-present 的 tiling/kernel 分支（tiling key 7）而不破坏精度对比。
- `single_server_check_result.py`：容差是**误差元素比例** `err_ratio < 1e-2`，逐元素判定用 `np.isclose(rtol=atol=5e-3)`（bf16 放宽到 `1e-2`），非严格 allclose；**只读 rank-0 的产物**，但 rank-0 上 `output` 与 `gather_out` 都校验。
  > ⚠️ 这两条都**只对蓝本这种先通后算的算子成立**，改造时最容易照抄错的就是这个文件：① 逐元素 `rtol`/`atol` 判据依赖"误差 ∝ |输出|"，先算后通（MM+ReduceScatter/AllReduce）的归约算子不满足，须换整体相对误差 `≤4ε`；② 只查 rank-0 依赖"各卡输出相同"，scatter 类算子须逐 rank 查。判据选择与定量标定见 [`workflow_integration.md`](workflow_integration.md) §1.4.1。
- `run_test.sh`：多轮随机 shape/dtype/bias 循环（M∈[1,64]，K∈{1024,2048,3072,12288}，N∈{1024,2048,3072,3904}，fp16/bf16 与 bias 有无按轮次奇偶交替，`isGatherOut` 恒为 1）。
- `single_server_config.ini`：单轮默认参数——`rankSize=2`、`m=512/k=12288/n=3904`、`dataType=0`（fp16）、`hasBias=0`、`isGatherOut=1`、`rounds=1`、`seed=42`、`masterAddr=127.0.0.1:29500`。换卡数改 `rankSize`。

---

## 4. `x_all_gather_matmul/CMakeLists.txt` 改造

通常改算子名相关项与 target_sources。**不要动**：

- vendored mc2_common 的 7 条 `-I`（`:101-107`）：`op_kernel`、`mc2_common/op_kernel`、`mc2_common/op_host`、`mc2_common/op_host/op_tiling`、`mc2_common/utils`、`mc2_common`、`mc2_common/op_api`。最后一条指向的目录在蓝本中**并不存在**（空占位，无害），复制工程时照抄即可，不必新建。
- kernel 编译选项（`:116-121`，`add_ops_compile_options(OP_NAME XAllGatherMatmul OPTIONS ...)`）完整为：`--cce-auto-sync=off -DHCCL_COMM -Wno-deprecated-declarations -Wno-error=option-ignored -mllvm -cce-aicore-hoist-movemask=false` 再加上述 7 条 `-I` 与 ascendc include。改算子名时 `OP_NAME` 要同步改。
- host include 根 `_X_AGM_HOST_INC`（`:124-161`）：先是 10 条工程内路径（`op_host`、`op_host/op_tiling`、`op_host/op_tiling/arch22`、`op_kernel`、`mc2_common` 及其 5 个子目录），再按 `aarch64-linux`/`arm64-linux`/`${CMAKE_SYSTEM_PROCESSOR}-linux` 逐一探测追加 5 个 CANN 侧根：`include/op_common`、`pkg_inc`、`include/external`、`asc/include/tiling`、`include/platform`（显式加是为避免与 intf_pub 的 register/graph 头冲突）。注意 `ascendc/include` 属于 **kernel** 侧 `-I`（`:22-28`），不在 host include 根里。

```cmake
# optiling target_sources（显式列出 vendored mc2_common，无 file(GLOB)）
target_sources(optiling PRIVATE
    op_host/op_tiling/x_all_gather_matmul_tiling_base.cpp
    op_host/op_tiling/x_all_gather_formulaic_tiling.cpp
    op_host/op_tiling/arch22/x_all_gather_matmul_tiling_a2a3.cpp
    op_host/op_tiling/arch22/x_all_gather_formulaic_tiling_a2a3.cpp
    mc2_common/op_host/op_tiling/hccl_formulaic_tiling.cpp
    mc2_common/op_host/op_tiling/matmul_formulaic_tiling.cpp
    mc2_common/op_host/op_tiling/hccl_performance.cpp
    mc2_common/op_host/op_tiling/matmul_performance.cpp
    mc2_common/op_host/op_tiling/mc2_tiling_utils.cpp
    mc2_common/utils/mc2_hcom_topo_info.cpp)
```

## 5. 构建与运行（替换旧 `run.sh` 虚构流程）

```bash
# 0) 复制起手
cp -r references/all_gather_matmul operators/{op_name}

# 前置：CANN 环境必须先 source，否则 setup.py 里的 import torch_npu 会因找不到 libhccl.so 而失败
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1) 构建算子（.run + libcust_opapi.so）
bash build.sh -n x_{op_name}               # -n 的值必须 == 外层算子目录名；省略则编译全部算子（ASCEND_OP_NAME 默认 ALL）
                                           # 默认 SoC ascend910b；单次烘焙 fp16+bf16 双 dtype
# 产物：build/custom_opp_<arch>.run（仅 `-b host` 模式才额外拷到 output/）
#       CPack 暂存树：build/_CPack_Packages/Linux/External/custom_opp_<arch>.run/packages/vendors/custom_opp
export ASCEND_CUSTOM_OPP_PATH=<vendors/custom_opp 绝对路径>

# 2) PTA：走 torch-ascendc-op-extension 路线 B（aclnn 注册）；蓝本不提供 torch_ops_extension
# 3) 精度测试（多卡 mp.spawn + HCCL；依赖 PTA 已按路线 B 生成）
bash examples/run_test.sh                 # 多轮随机 shape/dtype/bias；RESULT: CHECK PASSED

# 4) 性能（阶段三·3.2）
# msprof task-based，详见 ../../shared/profiling_mc2.md（本路线无需 L2 flush）
```

> 旧版 `bash run.sh`（cmake→gen_data→run→verify 一键）**不存在**；真实流程是 build.sh 构建算子 → 用已有 PTA skill 生成绑定 → examples/run_test.sh 测精度。

> ⚠️ **`ASCEND_CUSTOM_OPP_PATH` 与 `.run` 安装的嵌套陷阱**：`.run` 的 `install.sh` 会在 `ASCEND_CUSTOM_OPP_PATH` 之下再建一层 `vendors/<vendor_name>`。若安装前该变量已指向某个 `.../vendors/custom_opp`，就会装成 `.../vendors/custom_opp/vendors/custom_opp`，而运行时加载的是**外层旧版本**——表现为改了 kernel 重新编译安装后，跑出来的仍是旧代码（新加的 `printf` 不打印）。安装 `.run` 时用 `--install-path=` 显式指定安装根目录，或先 `unset ASCEND_CUSTOM_OPP_PATH`；安装后确认 `find $ASCEND_CUSTOM_OPP_PATH -name "vendors" ` 无嵌套。同一环境里并存"原工程"和"复制出的新工程"时，给新工程改一个不同的 `VENDOR_NAME`（顶层 `CMakeLists.txt:10`）最稳妥。

## 6. 改造清单（阶段二 Developer 工作流）

```
1. cp -r references/all_gather_matmul operators/{op_name}
2. 改名（x_all_gather_matmul → x_{op_name}），下列各处必须同步，漏一处就编译失败或静默走错实现：
   - 外层算子目录名（== OpDef 类名 snake_case，也是 build.sh -n 的取值）
   - x_{op}/CMakeLists.txt 的 add_ops_compile_options(OP_NAME ...) 与 target_sources 文件列表
   - OpDef 类名 / OP_ADD / IMPL_OP_INFERSHAPE / IMPL_OP_OPTILING
   - 公式化 tiling 类名（XAllGatherPlusMM / XAllGatherPlusMMA2A3）
   - device tiling 类名（Mc2Tiling::XAllGatherMatmulTilingData）—— 定义处、kernel 的
     REGISTER_TILING_DEFAULT、_full_mesh.h 的 SetCcTilingV2(offsetof(...)) 三处
   - aclnn 符号名（aclnnXxx / aclnnXxxGetWorkspaceSize）、kernel 入口函数名
   ——op_type 必须保留自定义前缀，且外层目录名 == OpDef snake_case，两条都违反会导致 kernel 静默不编译/不执行
3. 按 §3 改 _def/_proto/op_api/op_tiling[+arch22]/op_kernel[_full_mesh/_base/_tiling/_tiling_key]/examples；PTA 完成后走 torch-ascendc-op-extension 路线 B
4. 冒烟编译：bash build.sh -n x_{op_name}（.run 产出即过）
5. 冒烟测试：bash examples/run_test.sh（小 shape 一轮 CHECK PASSED）
6. 全量精度测试：多轮随机 shape/dtype/bias
7. 写入 PLAN.md "实际改动清单"，供 Reviewer 1.3R/阶段三 核对
```
**禁止**：不复制基底工程从零写文件；改 `[REUSE]` 标记的 `mc2_common/` 文件（除非 Architect 在 DESIGN.md 显式说明）；跳过冒烟直接上全量；引入 `aclshmemx_*`/`hcomm_`/Blaze/quant/`AlltoAllV`/`Finalize<false>`/`SetLocalWorkspace`。

## 7. Reviewer 改动审查清单

| 检查项 | 方法 |
|--------|------|
| 改动文件清单与 DESIGN.md 一致 | `diff -r references/all_gather_matmul operators/{op} --brief` |
| `mc2_common/` 未被修改 | 同上 diff，`mc2_common/` 文件不应出现 |
| 架构=A2 | `grep -i "ascend910b\|dav-2201"` `CMakeLists.txt`/`x_{op}/CMakeLists.txt`；无 910_93/950 |
| 通信走 HCCL 高阶 API | `grep "Hccl<HCCL_SERVER_TYPE_AICPU>\|hccl_\.\(AllGather\|AllReduce\|Wait\|Init\|SetCcTiling\|Finalize\)"` 命中 |
| AIC-only 门控 + 同步 | `grep "ASCEND_IS_AIC"` 命中；非 `g_coreType==AIV` HCCL 门控；Finalize 前有 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 跨核同步（非 SyncAll） |
| 计算走 AscendC::Matmul | `grep "MatmulImpl\|AscendC::Matmul\|MatmulCompute"` 命中；无 Blaze |
| 无 SHMEM | `grep aclshmem` 空 |
| 无 quant | `grep -rni "quant\|qbmm_mx\|copy_scale"` 空 |
| V2 HCCL（蓝本已用） | `grep "InitV2\|SetCcTilingV2"` 命中；V1 `Init`/`SetCcTiling` 应为空（已废弃） |
| Finalize 前跨核同步 | `grep "CrossCoreSetFlag\|CrossCoreWaitFlag"` 命中（`full_mesh.h:224-225`，非 SyncAll，效果等同） |
| PTA 若已生成 | 由 `torch-ascendc-op-extension` 路线 B 产出；蓝本不提供 PTA 目录 |

## 8. 后续阅读

| 想了解 | 读 |
|--------|---|
| 算子架构与可复用框架 | [`op_architecture.md`](op_architecture.md) |
| 各 Step 的具体动作 | [`workflow_integration.md`](workflow_integration.md) |
| MC2 整体架构 | [`mc2_architecture.md`](mc2_architecture.md) |
| HCCL 细节 | [`comm_hccl.md`](comm_hccl.md) |
| Matmul 细节 | [`matmul_fusion.md`](matmul_fusion.md) |
| PTA 细节 | `ops/torch-ascendc-op-extension` 路线 B |
| 性能采集 | [`../../shared/profiling_mc2.md`](../../shared/profiling_mc2.md) |
