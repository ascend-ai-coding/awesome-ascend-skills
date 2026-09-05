# Fused Compute-Communication 算子模板

## 适用条件

- `design.md` 的 `op_kind` 为 `fused_compute_comm`
- 算子同时包含 local compute（如 matmul）和 SHMEM communication（如 reduce-scatter、allgather）
- 需要 AIC/AIV 分核：AIC 做计算（BlockMmad），AIV 做通信（CommBlockEpilogue）

## 使用方法

Agent 生成算子工程时，从本 GUIDE.md 提取模板代码：

1. 扫描本文件，匹配以下格式的章节标题：

   ```text
   ## <relative_path> → <op_name>/<target_path>
   ```

2. 提取该章节下的 fenced code block 内容
3. 将内容写入目标路径（相对于算子工程根目录），例如 `## src/main.cpp → <op_name>/src/main.cpp` 的代码块写入 `<op_name>/src/main.cpp`
4. 替换所有占位符：`<op_name>` → 算子名（snake_case），`<OpName>` → CamelCase，`<OP_NAME>` → UPPER_CASE
5. 按 `docs/design.md` 的 interface、schedule、communication 等字段定制具体参数

## 模板文件 → 目标路径

| 代码块标题（relative_path） | 目标路径 | 语言 |
| --- | --- | --- |
| `CMakeLists.txt` | `<op_name>/CMakeLists.txt` | cmake |
| `src/main.cpp` | `<op_name>/src/main.cpp` | cpp |
| `src/<op_name>_kernel.h` | `<op_name>/src/<op_name>_kernel.h` | cpp |
| `scripts/gen_data.py` | `<op_name>/scripts/gen_data.py` | python |
| `scripts/check_result.py` | `<op_name>/scripts/check_result.py` | python |
| `scripts/run.sh` | `<op_name>/scripts/run.sh` | bash |

baseline 模板（有 baseline 时 Agent 需额外生成，不在本 GUIDE.md 中）：

| 生成目标路径 | 用途 |
| --- | --- |
| `<op_name>/baseline/CMakeLists.txt` | 独立 baseline 编译 target |
| `<op_name>/baseline/src/<op_name>_baseline.cpp` | HCCL/aclnn/拼接 baseline 源码 |
| `<op_name>/baseline/scripts/run_baseline.sh` | baseline runner，供 msprof 包裹 |
| `<op_name>/scripts/perf.sh` | SHMEM runner，供 msprof 包裹；最终性能来自 task_time 解析（实现见 [perf-workflow.md](../../../shmem-ops-performance-eval/references/perf-workflow.md)） |
| `<op_name>/scripts/perf_compare.sh` | 可选；推荐统一用 perf-workflow §1 阶段 C |

## AIC/AIV 分工模式

```text
AIC (计算核)                     AIV (通信核)
    │                                │
    ├─ BlockMmad (matmul)            │  等待 AIC...
    ├─ CrossCoreSetFlag ───────────► ├─ CrossCoreWaitFlag
    │                                ├─ aclshmem_barrier_all()
    │  等待 AIV...                   ├─ CommBlockEpilogue (scatter/gather)
    ├─ CrossCoreWaitFlag ◄─────────  ├─ CrossCoreSetFlag
    ├─ 下一阶段 matmul...            ├─ 下一阶段通信...
```

核心标识：
- AIC: `aicoreIdx = GetBlockIdx()`, `aicoreNum = GetBlockNum()`
- AIV: `aicoreIdx = GetBlockIdx() / GetSubBlockNum()`, `subcoreIdx = GetSubBlockIdx()`
- AIV 只有 subcore 0 (`subcoreIdx == 0`) 执行通信

workspace 流水：`WORKSPACE_STAGES = 2`，`flagAicFinishStore[]` / `flagAivFinishCompute[]` 交替。

## 定制要点

1. 从本 GUIDE.md 提取代码块，写入 `docs/design.md` 指定的目标目录（`src/`、`scripts/` 等）
2. 替换占位符：`<op_name>` / `<OpName>` / `<OP_NAME>`
3. `src/<op_name>_kernel.h`：
   - AIC `operator()<AIC>`：按 design.md `local_compute` 实现 matmul 循环
   - AIV `operator()<AIV>`：按 design.md `communication` 实现 scatter/gather 循环
   - AllReduce 需要两组 epilogue（ReduceScatter + AllGather），参考 `matmul_allreduce.h`
4. `src/main.cpp` 类型组合区域：按 design.md 调整 TileShape、DispatchPolicy、CopyMode、CommBlockShape
5. `src/main.cpp` Host 区域：按 interface 调整 I/O 和 symmetric buffer 大小
6. `scripts/gen_data.py`：按 correctness.oracle 实现 golden

## 变体参考

| 算子类型 | kernel 变化 | 参考 example |
| --- | --- | --- |
| MatmulReduceScatter | 单 comm 阶段（Scatter） | `examples/matmul_reduce_scatter` |
| MatmulAllReduce | 双 comm 阶段（Scatter + Gather） | `examples/matmul_allreduce` |
| AllGatherMatmul | 先通信再计算（AIV 先跑） | `examples/allgather_matmul` |
| DispatchGmmCombine | MIX AIC/AIV 自定义 kernel | `examples/dispatch_gmm_combine` |

## CATLASS 源码参考

生成或调整 L2/L3 compute 段前，先查 CATLASS 真实代码：

- 优先使用 `${SHMEM_REPO}/3rdparty/catlass`
- 缺失时 clone：`git clone https://gitcode.com/cann/catlass.git ${SHMEM_REPO}/3rdparty/catlass`
- 按语义选样例：`13_basic_matmul_tla` / `14_optimized_matmul_tla` / `43_ascend950_basic_matmul`（基础与优化 GEMM），`08_grouped_matmul` / `02_grouped_matmul_slice_m` / `05_grouped_matmul_slice_k`（GroupedMatmul），`27_matmul_gelu` / `28_matmul_silu` / `64_ascend950_matmul_evg`（epilogue/activation），`31_small_matmul`（小矩阵）
- 再查 `include/catlass/gemm/`、`include/catlass/epilogue/` 确认 `BlockMmad`、TileShape、DispatchPolicy、Swizzle、Epilogue 接口

CATLASS 样例只迁移 compute 段模式；通信段、rank 语义、barrier、symmetric buffer 与 `CommBlockEpilogue` 仍以 SHMEM examples/CATCOC 为准。

## 约束

- kernel 类的 `operator()` 通过 `g_coreType` 模板参数自动分发到 AIC/AIV
- Transport 必须通过 CATLASS CommEpilogue 内部的 `aclshmem_*` 完成
- `src/main.cpp` 只做单 PE 编排，不含 golden/checker
- 详细约束见 [SKILL.md](../../SKILL.md) 和 [references/](../../references/)

---

## CMakeLists.txt → `<op_name>/CMakeLists.txt`

```cmake
# ==== 定制：替换 <op_name> 为算子名 ====
# 独立工程构建模式：显式引用 src/ 下源码，include src/ 和 examples/utils/
cmake_minimum_required(VERSION 3.16)
project(<op_name> LANGUAGES CXX)

set(CANN_ENV "$ENV{ASCEND_HOME_PATH}")
if(NOT CANN_ENV)
    message(FATAL_ERROR "ASCEND_HOME_PATH not set; source set_env.sh first")
endif()

set(SHMEM_ROOT "${CANN_ENV}/shmem")
set(EXAMPLE_UTILS "${SHMEM_ROOT}/examples/utils")

include_directories(
    ${CANN_ENV}/include
    ${SHMEM_ROOT}/include
    ${SHMEM_ROOT}/examples/utils
    ${CMAKE_SOURCE_DIR}/src
)

add_executable(<op_name>
    src/main.cpp
)
target_link_libraries(<op_name> shmem ascendcl)
```

---

## src/main.cpp → `<op_name>/src/main.cpp`

```cpp
// ============================================================
// SHMEM Fused Compute-Communication 算子 Host 入口模板
// 来源: shmem/examples/matmul_reduce_scatter/main.cpp 简化
// 使用: 写入 <op_name>/src/main.cpp，
//       替换 <op_name>/<OpName>/<OP_NAME> 占位符，定制 CATLASS 组合参数
// ============================================================

#include <acl/acl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>
#include <cstdint>

// CATLASS
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

// CATCOC (Communication Epilogue)
#include "catcoc/catcoc.h"
#include "catcoc/comm_epilogue/comm_dispatch_policy.h"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.h"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.h"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.h"
#include "catcoc/detail/remote_copy_type.h"

// shmem host
#include "host/shmem_host_def.h"
#include "host/mem/shmem_host_heap.h"
#include "host/init/shmem_host_init.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/team/shmem_host_team.h"

// ==== Kernel 类定义（AIC/AIV 分工）====
#include "<op_name>_kernel.h"

using namespace AscendC;
using namespace Catcoc;

#define CHECK_ACL(call)                                                     \
    do {                                                                    \
        auto _ret = (call);                                                 \
        if (_ret != 0) {                                                    \
            std::cerr << #call << " failed: " << _ret << std::endl;         \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#define CHECK_SHMEM(call)                                                   \
    do {                                                                    \
        int _ret = (call);                                                  \
        if (_ret != 0) {                                                    \
            std::cerr << #call << " failed: " << _ret << std::endl;         \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ---- SHMEM init helper（内联，不依赖 examples/utils）----
static void init_set_attr(int pe_id, int n_pes, uint64_t local_mem_size,
                          const char *ip_port,
                          aclshmemx_uniqueid_t &uid,
                          aclshmemx_init_attr_t &attr)
{
    attr.my_pe      = pe_id;
    attr.n_pes      = n_pes;
    attr.local_mem_size = local_mem_size;
    strncpy(attr.ip_port, ip_port, sizeof(attr.ip_port) - 1);
    attr.ip_port[sizeof(attr.ip_port) - 1] = '\0';

    if (pe_id == 0) {
        aclshmemx_get_uniqueid(&uid);
    }
    attr.uniqueid = uid;
}

// ---- FFTS config helper ----
static uint64_t get_ffts_config()
{
    uint64_t fftsAddr = 0;
    CHECK_ACL(aclrtGetMemInfo(ACL_HBM_MEM, &fftsAddr, nullptr));
    return fftsAddr;
}

// ========================================================================
// CATLASS 类型组合 + Kernel 调用入口
// ========================================================================

// ==== 定制：按 design.md 的 interface 调整 layout 和 element type ====
using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementD = half;

// ==== 定制：按 design.md 的 schedule.tiling 调整 TileShape ====
constexpr bool ENABLE_UNIT_FLAG = true;
constexpr int L1TILEM = 128;
constexpr int L1TILEN = 256;
constexpr int L1TILEK = 256;
constexpr int L0TILEM = 128;
constexpr int L0TILEN = 256;
constexpr int L0TILEK = 64;

using ArchTag = Catlass::Arch::AtlasA2;
using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
using L1TileShape = Catlass::GemmShape<L1TILEM, L1TILEN, L1TILEK>;
using L0TileShape = Catlass::GemmShape<L0TILEM, L0TILEN, L0TILEK>;
using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

using BlockMmad = Catlass::Gemm::Block::BlockMmad<
    MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

// ---- Communication: CommBlockEpilogue ----
// ==== 定制：按 design.md 的 schedule.core_partition 调整 CommCoreSplit ====
constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
constexpr uint32_t SWIZZLE_DIRECTION = 1;
using BlockMmadScheduler =
    Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0, true>;

using RemoteSrcType = CType;
using RemoteDstType = DType;
using CopyDirect = Catcoc::detail::CopyDirect;
// ==== 定制：Scatter 或 Gather ====
using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<
    ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

// ==== 定制：调整 CommBlock 尺寸 ====
constexpr uint32_t COMM_BLOCK_ROWS = 64;
constexpr uint32_t COMM_BLOCK_COLUMNS = 256;
constexpr uint32_t CORE_SPLIT_ROWS = 20;
constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, COMM_BLOCK_COLUMNS>;
using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

constexpr uint32_t UB_STAGES = 2;
constexpr uint32_t SCATTER_TILE_ROWS = 32;
constexpr uint32_t SCATTER_TILE_COLUMNS = 256;
using EpilogueTileShape = Catlass::MatrixShape<SCATTER_TILE_ROWS, SCATTER_TILE_COLUMNS>;
// ==== 定制：CopyMode::Scatter 或 CopyMode::Gather ====
using EpilogueDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<
    UB_STAGES, Catcoc::detail::CopyMode::Scatter>;

using BlockEpilogue = CommEpilogue::Block::CommBlockEpilogue<
    EpilogueDispatch,
    RemoteSrcType, RemoteDstType,
    CommCoreSplit,
    CommBlockShape,
    EpilogueTileShape, TileRemoteCopy, TileScheduler>;

// ---- Fused Kernel（AIC/AIV 分工定义在 <op_name>_kernel.h）----
constexpr uint32_t WORKSPACE_STAGES = 2;
using FusedKernel = Catcoc::DGemm::Kernel::<OpName><
    BlockMmad, BlockEpilogue,
    BlockMmadScheduler, BlockEpilogueScheduler,
    WORKSPACE_STAGES>;

CATLASS_GLOBAL
void Shmem<OpName>(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k)
{
    util_set_ffts_config(fftsAddr);

    uint32_t peIdx = aclshmem_my_pe();
    uint32_t peSize = aclshmem_n_pes();
    if (peSize == 0) return;

    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    // ==== 定制：按通信语义调整输出 layout ====
    // ReduceScatter: m/peSize x n; AllGather: m x n
    LayoutD layoutD{m / peSize, n};

    constexpr uint32_t COMM_INTERVAL = 10;
    typename BlockEpilogue::Params epilogueParams{};
    typename FusedKernel::Params params{
        problemShape, peIdx, peSize,
        COMM_INTERVAL,
        gmA, layoutA,
        gmB, layoutB,
        gmD, layoutD,
        gmSymmetric,
        epilogueParams
    };

    FusedKernel fusedKernel;
    fusedKernel(params);
}

// ========================================================================
// Host 入口
// ========================================================================

constexpr uint32_t BLOCK_NUM = 20;  // ==== 定制：按 schedule.core_partition ====

struct Options {
    int peSize = 2;
    int peId = 0;
    std::string ipPort = "tcp://127.0.0.1:8998";
    uint32_t m = 0, n = 0, k = 0;
    std::string dataPath;
    int perfTimes = 0;
    int warmup = 5;

    int Parse(int argc, char **argv)
    {
        if (argc < 8) {
            std::cerr << "Usage: <op_name> <pe_size> <pe_id> <ip_port> <m> <n> <k> <data_path>"
                      << " [perf_times]" << std::endl;
            return -1;
        }
        peSize   = std::atoi(argv[1]);
        peId     = std::atoi(argv[2]);
        ipPort   = argv[3];
        m        = std::atoi(argv[4]);
        n        = std::atoi(argv[5]);
        k        = std::atoi(argv[6]);
        dataPath = argv[7];
        if (argc > 8) perfTimes = std::atoi(argv[8]);
        return 0;
    }

    std::string GetPath(const std::string &name) const { return dataPath + "/" + name; }
};

template <typename T>
void ReadBinaryFile(const std::string &path, T *dst, size_t bytes)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    f.read(reinterpret_cast<char *>(dst), bytes);
}

template <typename T>
void WriteBinaryFile(const std::string &path, const T *src, size_t bytes)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(src), bytes);
}

int main(int argc, char **argv)
{
    Options opts;
    if (opts.Parse(argc, argv) != 0) return 1;

    int32_t device_id = opts.peId;  // ==== 定制：多卡映射 ====

    // ---- ACL & SHMEM init ----
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_uniqueid_t uid;
    aclshmemx_init_attr_t attributes;
    init_set_attr(opts.peId, opts.peSize, local_mem_size, opts.ipPort.c_str(), uid, attributes);
    CHECK_ACL(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // ---- 分配 I/O ----
    // ==== 定制：按 interface.inputs/outputs 调整 ====
    size_t aSize = (size_t)opts.m * opts.k * sizeof(ElementA);
    size_t bSize = (size_t)opts.k * opts.n * sizeof(ElementB);
    size_t dSize = (size_t)opts.m * opts.n * sizeof(ElementD);
    size_t dSizeScatter = dSize / opts.peSize;

    uint8_t *aDevice, *bDevice, *dDevice;
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&dDevice), dSizeScatter, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *aHost, *bHost, *dHost;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&aHost), aSize));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&bHost), bSize));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&dHost), dSize));

    ReadBinaryFile(opts.GetPath("pe_" + std::to_string(opts.peId) + "_a.bin"), aHost, aSize);
    ReadBinaryFile(opts.GetPath("pe_" + std::to_string(opts.peId) + "_b.bin"), bHost, bSize);
    CHECK_ACL(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- Symmetric buffer ----
    // ==== 定制：按 memory.buffers 调整大小 ====
    size_t symmSize = 204 * 1024 * 1024 * sizeof(__fp16);
    if (symmSize == 0) {
        std::cerr << "symmetric buffer size is zero" << std::endl;
        return 1;
    }
    void *symmPtr = aclshmem_malloc(symmSize);
    if (symmPtr == nullptr) {
        std::cerr << "aclshmem_malloc failed, size=" << symmSize << std::endl;
        return 1;
    }
    uint8_t *symmetricPtr = reinterpret_cast<uint8_t *>(symmPtr);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    // ---- 执行 ----
    auto launch = [&]() {
        uint64_t fftsAddr = get_ffts_config();
        Shmem<OpName><<<BLOCK_NUM, nullptr, stream>>>(
            fftsAddr, aDevice, bDevice, dDevice, symmetricPtr,
            opts.m, opts.n, opts.k);
    };

    // ==== 定制：staging 拷贝函数 —— 将 input_dev 拷贝到 symmetric buffer ====
    // fused 算子的 staging 通常是把 A/B 矩阵从 GM 拷贝到 symmetric buffer
    auto staging_copy = [&]() {
        // ==== 定制：按 design.md 的 memory.staging 实现 GM → symmetric 拷贝 ====
        // 示例：将 A 矩阵拷入 symmetric buffer（实际按 design.md 定制）
        CHECK_ACL(aclrtMemcpy(symmetricPtr, aSize, aDevice, aSize, ACL_MEMCPY_DEVICE_TO_DEVICE));
    };

    if (opts.perfTimes > 0) {
        // warmup（含 staging + barrier + kernel）；msprof 解析时按 WARMUP 跳过
        for (int i = 0; i < opts.warmup; i++) {
            staging_copy();
            aclshmem_barrier_all();
            launch();
            CHECK_ACL(aclrtSynchronizeStream(stream));
        }

        // ---- inline_event timing：保留原有 [PERF] 指标；同一 runner 也可被 msprof 包裹 ----
        double kernel_us_acc = 0;
        auto t0_e2e = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < opts.perfTimes; i++) {
            staging_copy();
            aclshmem_barrier_all();
            auto t0_kern = std::chrono::high_resolution_clock::now();
            launch();
            CHECK_ACL(aclrtSynchronizeStream(stream));
            auto t1_kern = std::chrono::high_resolution_clock::now();
            kernel_us_acc += std::chrono::duration<double, std::micro>(t1_kern - t0_kern).count();
        }
        auto t1_e2e = std::chrono::high_resolution_clock::now();
        double e2e_us = std::chrono::duration<double, std::micro>(t1_e2e - t0_e2e).count()
                        / opts.perfTimes;
        double kernel_us = kernel_us_acc / opts.perfTimes;

        // ==== 定制：按算子类型设置 bus_factor 和 peak_bandwidth ====
        // ReduceScatter/AllGather: (n-1)/n, AllReduce: 2*(n-1)/n, P2P: 1
        double bus_factor = (double)(opts.peSize - 1) / opts.peSize;
        double payload_bytes = (double)aSize;
        double algo_bw = payload_bytes / (e2e_us * 1e-6) / 1e9;
        double bus_bw  = algo_bw * bus_factor;
        double steady_algo_bw = payload_bytes / (kernel_us * 1e-6) / 1e9;
        double kernel_bus_bw  = steady_algo_bw * bus_factor;
        double peak_bw = 196.0;  // ==== 定制：按通信模式和拓扑调整 ====
        double util_pct = kernel_bus_bw / peak_bw * 100.0;

        if (opts.peId == 0) {
            std::cout << "[PERF] source=inline_event"
                      << " warmup=" << opts.warmup
                      << " measure=" << opts.perfTimes
                      << " e2e_us=" << e2e_us
                      << " kernel_us=" << kernel_us
                      << " algo_bandwidth_GBps=" << algo_bw
                      << " e2e_bus_bandwidth_GBps=" << bus_bw
                      << " kernel_bus_bandwidth_GBps=" << kernel_bus_bw
                      << " bandwidth_utilization_pct=" << util_pct
                      << " payload_bytes=" << payload_bytes
                      << " bus_factor=" << bus_factor
                      << " peak_bandwidth_GBps=" << peak_bw
                      << std::endl;
        }

        aclshmemx_get_prof(nullptr, true);
    } else {
        staging_copy();
        aclshmem_barrier_all();
        launch();
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    // ---- 拷回输出 ----
    CHECK_ACL(aclrtMemcpy(dHost, dSizeScatter, dDevice, dSizeScatter, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteBinaryFile(opts.GetPath("output_pe" + std::to_string(opts.peId) + ".bin"), dHost, dSizeScatter);

    // ---- 清理 ----
    aclshmem_free(symmPtr);
    CHECK_ACL(aclrtFreeHost(aHost));
    CHECK_ACL(aclrtFreeHost(bHost));
    CHECK_ACL(aclrtFreeHost(dHost));
    CHECK_ACL(aclrtFree(aDevice));
    CHECK_ACL(aclrtFree(bDevice));
    CHECK_ACL(aclrtFree(dDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_SHMEM(aclshmem_finalize());
    CHECK_ACL(aclrtResetDevice(device_id));
    CHECK_ACL(aclFinalize());

    std::cout << "[SUCCESS] <op_name> completed on PE " << opts.peId << std::endl;
    return 0;
}
```

---

## src/<op_name>_kernel.h → `<op_name>/src/<op_name>_kernel.h`

```cpp
// ============================================================
// SHMEM Fused Compute-Communication Kernel 模板
// 来源: matmul_reduce_scatter.h / matmul_allreduce.h 简化
// 使用: 复制到 <op_name>/src/<op_name>_kernel.h
//       替换占位符，按 design.md 定制 AIC/AIV 逻辑
// ============================================================
#ifndef <OP_NAME>_KERNEL_H
#define <OP_NAME>_KERNEL_H

#include "catcoc/catcoc.h"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "utils/prof/shmemi_prof.h"

namespace Catcoc::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

// ==== 定制：模板参数按 design.md 的 schedule 和 communication 调整 ====
// BlockMmad_               — 计算模块（L1/L0 TileShape + MmadDispatchPolicy）
// BlockEpilogueComm_       — 通信模块（CommBlockEpilogue，Scatter 或 Gather）
// BlockMmadScheduler_      — 计算 block 调度（GemmIdentityBlockSwizzle）
// BlockEpilogueScheduler_  — 通信 block 调度（BlockCommSwizzle）
// WORKSPACE_STAGES_        — workspace 流水级数（通常 2）
template <
    class BlockMmad_,
    class BlockEpilogueComm_,
    class BlockMmadScheduler_,
    class BlockEpilogueScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class <OpName> {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockEpilogueComm = BlockEpilogueComm_;
    using BlockEpilogueCommParams = typename BlockEpilogueComm::Params;
    using ElementD = typename BlockEpilogueComm::ElementDst;
    using LayoutD = typename BlockEpilogueComm::LayoutDst;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockEpilogueScheduler = BlockEpilogueScheduler_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    struct Params {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;

        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementD *ptrD;
        LayoutD layoutD;
        GM_ADDR ptrSymmetric;

        // ==== 定制：如果 AllReduce 需要两组 epilogue params ====
        BlockEpilogueCommParams commParams;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            GemmCoord const &problemShape_, uint32_t rankIdx_, uint32_t rankSize_,
            uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrD_, LayoutD const &layoutD_,
            GM_ADDR ptrSymmetric_,
            BlockEpilogueCommParams const &commParams_
        ) : problemShape(problemShape_), rankIdx(rankIdx_), rankSize(rankSize_),
            commInterval(commInterval_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            ptrSymmetric(ptrSymmetric_),
            commParams(commParams_)
        {
        }
    };

    CATLASS_DEVICE
    <OpName>()
    {
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAicFinishStore[i] = Catlass::Arch::CrossCoreFlag(i);
            flagAivFinishCompute[i] = Catlass::Arch::CrossCoreFlag(i);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    // ================================================================
    // AIC 侧：计算（BlockMmad）
    // ================================================================
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = L1TileShape::ToCoord();
        // ==== 定制：ReduceScatter 需要 / rankSize，AllGather 不需要 ====
        GemmCoord problemShapeInRank = params.problemShape
            / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1);
        BlockMmadScheduler mmadScheduler(problemShapeInRank, blockShape.GetCoordMN());
        uint32_t coreLoops = mmadScheduler.GetCoreLoops() * params.rankSize;

        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        auto layoutSymmetric = Catlass::layout::RowMajor{
            WORKSPACE_STAGES * blockPerComm * L1TileShape::M,
            L1TileShape::N, L1TileShape::N
        };
        auto layoutSymmetricRowLogic =
            Catlass::MakeCoord<int>(WORKSPACE_STAGES, blockPerComm, L1TileShape::M);
        auto layoutSymmetricRow =
            layout::AffineRankN<3>::Packed(layoutSymmetricRowLogic);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageIdx]);
            }

            SHMEMI_PROF_START(0);
            // ==== 定制：按 design.md 的 local_compute 实现 matmul 循环 ====
            uint32_t blockPerCommInRank = blockPerComm / params.rankSize;
            uint32_t actualBlockPerComm = (commIdx == commLoops - 1)
                ? (coreLoops - blockPerComm * commIdx) : blockPerComm;

            uint32_t commBlockOffsetInRank = commIdx * blockPerCommInRank;
            for (uint32_t blockIdxInComm = aicoreIdx;
                 blockIdxInComm < actualBlockPerComm;
                 blockIdxInComm += aicoreNum) {
                uint32_t loopIdxInRank =
                    commBlockOffsetInRank + blockIdxInComm % (actualBlockPerComm / params.rankSize);
                uint32_t targetRankIdx =
                    blockIdxInComm / (actualBlockPerComm / params.rankSize);
                GemmCoord blockCoord = mmadScheduler.GetBlockCoord(loopIdxInRank);
                GemmCoord actualBlockShape = mmadScheduler.GetActualBlockShape(blockCoord);
                GemmCoord offsetCoord = blockCoord * blockShape;

                auto rankOffsetA = problemShapeInRank.GetCoordMK()
                    * Catlass::MakeCoord<uint32_t>(targetRankIdx, 0);
                auto gmBlockA = gmA[params.layoutA.GetOffset(
                    offsetCoord.GetCoordMK() + rankOffsetA)];
                auto gmBlockB = gmB[params.layoutB.GetOffset(
                    offsetCoord.GetCoordKN())];

                // ==== 定制：本 rank 写 gmD，远端写 gmSymmetric ====
                AscendC::GlobalTensor<ElementC> gmBlockC;
                Catlass::layout::RowMajor layoutC;
                if (targetRankIdx == params.rankIdx) {
                    gmBlockC = gmD[params.layoutD.GetOffset(offsetCoord.GetCoordMN())];
                    layoutC = params.layoutD;
                } else {
                    MatrixCoord symOffset{layoutSymmetricRow(
                        Catlass::MakeCoord<int>(stageIdx, blockIdxInComm, 0)), 0};
                    gmBlockC = gmSymmetric[layoutSymmetric.GetOffset(symOffset)];
                    layoutC = layoutSymmetric;
                }

                blockMmad(
                    gmBlockA, params.layoutA,
                    gmBlockB, params.layoutB,
                    gmBlockC, layoutC,
                    actualBlockShape
                );
            }
            SHMEMI_PROF_END(0);

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                flagAicFinishStore[stageIdx]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // ================================================================
    // AIV 侧：通信（CommBlockEpilogue）
    // ================================================================
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        MatrixCoord blockShapeMN = L1TileShape::ToCoordMN();
        GemmCoord problemShapeInRank = params.problemShape
            / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1);
        BlockMmadScheduler mmadScheduler(problemShapeInRank, blockShapeMN);
        uint32_t coreLoops = mmadScheduler.GetCoreLoops() * params.rankSize;

        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        BlockEpilogueComm commEpilogue(resource, params.commParams);

        AscendC::GlobalTensor<ElementC> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
        auto layoutSymmetric = Catlass::layout::RowMajor{
            WORKSPACE_STAGES * blockPerComm * L1TileShape::M,
            L1TileShape::N, L1TileShape::N
        };

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        MatrixCoord commBlockShape = params.commParams.BlockShape();
        MatrixCoord commCoreSplit = params.commParams.CoreSplit();
        BlockEpilogueScheduler commScheduler(commBlockShape, commCoreSplit);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            uint32_t actualBlockInComm =
                Min(blockPerComm, coreLoops - commIdx * blockPerComm);
            uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

            auto actualCommShape = DistMatrixCoord{
                actualBlockInComm * blockShapeMN.row() / params.rankSize,
                blockShapeMN.column(), params.rankSize};
            MatrixCoord loopsInRank = CeilDiv(
                MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);
            uint32_t commAicoreNum = commScheduler.GetRealCore();
            uint32_t commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord stageOffset =
                MatrixCoord{stageIdx * blockPerComm, 0} * blockShapeMN;
            uint32_t mmadStartLoopIdxInComm = commIdx * blockPerCommInRank;

            // ---- 等待 AIC 计算完成 ----
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);

            // ---- 跨设备同步 ----
            aclshmem_barrier_all();

            SHMEMI_PROF_START(1);
            AscendC::SetAtomicAdd<ElementD>();
            AscendC::PipeBarrier<PIPE_ALL>();
            commEpilogue.InitBlockLoop();

            // ==== 定制：按 design.md 的 communication 实现通信循环 ====
            if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIdx;
                     commLoopIdx < commCoreLoops;
                     commLoopIdx += commAicoreNum) {
                    DistMatrixCoord commBlockCoord =
                        commScheduler.GetBlockCoord(commLoopIdx);
                    MatrixCoord blockOffset = commScheduler.GetBlockOffset(
                        DistMatrixCoord{commBlockCoord.GetCoordInRank(),
                                        params.rankIdx});
                    MatrixCoord blockOffsetInRank =
                        commScheduler.GetBlockOffsetInRank(
                            commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape =
                        commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                    uint32_t remoteRankIdx = commBlockCoord.rank();
                    if (remoteRankIdx == params.rankIdx) {
                        continue;
                    }

                    // ==== 定制：源/目的偏移按通信语义调整 ====
                    uint32_t mmadLoopIdx = mmadStartLoopIdxInComm
                        + blockOffsetInRank.row() / blockShapeMN.row();
                    GemmCoord mmadBlockCoordMNK = mmadScheduler.GetBlockCoord(mmadLoopIdx);
                    MatrixCoord mmadBlockCoord = mmadBlockCoordMNK.GetCoordMN();
                    MatrixCoord actualMmadBlockShape =
                        mmadScheduler.GetActualBlockShape(mmadBlockCoordMNK).GetCoordMN();
                    MatrixCoord offsetInMmadBlock = blockOffsetInRank % blockShapeMN;
                    MatrixCoord residueInMmadBlock = actualMmadBlockShape
                        - Min<uint32_t, 2>(actualMmadBlockShape, offsetInMmadBlock);
                    actualCommBlockShape =
                        Min<uint32_t, 2>(actualCommBlockShape, residueInMmadBlock);

                    auto offsetSrc = stageOffset + blockOffset;
                    MatrixCoord mmadBlockOffset = mmadBlockCoord * blockShapeMN;
                    auto offsetDst = mmadBlockOffset + offsetInMmadBlock;

                    auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = layoutSymmetric.GetTileLayout(actualCommBlockShape);
                    auto gmBlockDst = gmD[params.layoutD.GetOffset(offsetDst)];
                    auto layoutBlockDst = params.layoutD.GetTileLayout(actualCommBlockShape);

                    commEpilogue(
                        gmBlockSrc, layoutBlockSrc,
                        gmBlockDst, layoutBlockDst,
                        actualCommBlockShape, remoteRankIdx % params.rankSize
                    );
                }
            }

            commEpilogue.FinalizeBlockLoop();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();
            SHMEMI_PROF_END(1);

            // ---- 跨设备同步 ----
            aclshmem_barrier_all();

            // ==== 定制：AllReduce 在此处添加 AllGather 阶段 ====

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                flagAivFinishCompute[stageIdx]);
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catcoc::DGemm::Kernel

#endif  // <OP_NAME>_KERNEL_H
```

---

## scripts/gen_data.py → `<op_name>/scripts/gen_data.py`

```python
#!/usr/bin/env python3
# ============================================================
# SHMEM Fused Compute-Communication 算子数据生成 + golden 模板
# 来源: shmem/examples/matmul_reduce_scatter gen 模式
# 使用: 复制到 <op_name>/scripts/gen_data.py，
#       按 design.md 的 correctness.oracle 实现 golden
# ============================================================

import argparse
import json
import os
import numpy as np

np.random.seed(42)


def gen_random_data(shape, dtype):
    return np.random.uniform(0.0, 10.0, size=shape).astype(dtype)


# ==== 定制：按 design.md 的 semantics 和 correctness.oracle 实现 golden ====
def compute_golden(n_pes, m, n, k, data_dir, input_dtype, acc_dtype=np.float32):
    """
    示例: MatmulReduceScatter golden
    1. 每个 PE 做 local matmul (A @ B)
    2. AllReduce (sum)
    3. ReduceScatter: 每个 PE 取 output 的 1/n_pes 行
    """
    golden_full = np.zeros((m, n), dtype=acc_dtype)

    for pe in range(n_pes):
        a = np.fromfile(os.path.join(data_dir, f"pe_{pe}_a.bin"), dtype=input_dtype).reshape(m, k)
        b = np.fromfile(os.path.join(data_dir, f"pe_{pe}_b.bin"), dtype=input_dtype).reshape(k, n)
        result = np.matmul(a.astype(acc_dtype), b.astype(acc_dtype))
        golden_full += result

    golden_full = golden_full.astype(input_dtype)

    rows_per_pe = m // n_pes
    for pe in range(n_pes):
        shard = golden_full[pe * rows_per_pe : (pe + 1) * rows_per_pe, :]
        shard.tofile(os.path.join(data_dir, f"golden_pe{pe}.bin"))

    return golden_full


def main():
    parser = argparse.ArgumentParser(description="生成 fused compute-comm 输入和 golden")
    parser.add_argument("--n_pes", type=int, required=True)
    parser.add_argument("--m",     type=int, required=True)
    parser.add_argument("--n",     type=int, required=True)
    parser.add_argument("--k",     type=int, required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    dtype_map = {"float16": np.float16, "float32": np.float32}
    dtype = dtype_map[args.dtype]
    os.makedirs(args.data_dir, exist_ok=True)

    for pe in range(args.n_pes):
        a = gen_random_data((args.m, args.k), dtype)
        b = gen_random_data((args.k, args.n), dtype)
        a.tofile(os.path.join(args.data_dir, f"pe_{pe}_a.bin"))
        b.tofile(os.path.join(args.data_dir, f"pe_{pe}_b.bin"))

    compute_golden(args.n_pes, args.m, args.n, args.k, args.data_dir, dtype)

    config = {
        "n_pes": args.n_pes,
        "m": args.m, "n": args.n, "k": args.k,
        "dtype": args.dtype,
        "rtol": 1e-3,
        "atol": 1e-5,
    }
    with open(os.path.join(args.data_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated data: {args.n_pes} PEs, m={args.m} n={args.n} k={args.k}")


if __name__ == "__main__":
    main()
```

---

## scripts/check_result.py → `<op_name>/scripts/check_result.py`

```python
#!/usr/bin/env python3
# ============================================================
# SHMEM Fused Compute-Communication 精度验证模板
# 使用: 复制到 <op_name>/scripts/check_result.py
#       按 design.md 的 correctness.tolerance 调整阈值
# ============================================================

import argparse
import json
import sys
import numpy as np


def get_dtype(name: str):
    dtype_map = {"float16": np.float16, "float32": np.float32}
    return dtype_map[name]


def check_precision(output, golden, rtol, atol, label=""):
    if output.shape != golden.shape:
        print(f"\033[31m✗ SHAPE MISMATCH {label}: output={output.shape} golden={golden.shape}\033[0m")
        return False

    if np.any(np.isnan(output)):
        print(f"\033[31m✗ NaN DETECTED {label}: {np.sum(np.isnan(output))} NaN values\033[0m")
        return False

    diff = np.abs(output.astype(np.float64) - golden.astype(np.float64))
    golden_abs = np.abs(golden.astype(np.float64))
    rel_error = diff / (golden_abs + 1e-8)

    mask = (rel_error >= rtol) & (diff >= atol)
    error_indices = np.where(mask.flatten())[0]

    if len(error_indices) == 0:
        print(f"\033[32m✓ PASS {label}: all {output.size} elements within tolerance "
              f"(rtol={rtol}, atol={atol})\033[0m")
        return True

    print(f"\033[31m✗ FAIL {label}: {len(error_indices)}/{output.size} elements exceed tolerance\033[0m")
    print(f"  Max absolute error: {np.max(diff):.6e}")
    print(f"  Max relative error: {np.max(rel_error):.6e}")
    print(f"  Mean absolute error: {np.mean(diff):.6e}")

    flat_out = output.flatten()
    flat_gol = golden.flatten()
    for idx in error_indices[:10]:
        print(f"  Index {idx}: output={flat_out[idx]}, golden={flat_gol[idx]}, "
              f"diff={diff.flatten()[idx]:.6e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="验证 fused compute-comm 输出精度")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    with open(f"{args.data_dir}/config.json") as f:
        config = json.load(f)

    dtype = get_dtype(config["dtype"])
    rtol = config.get("rtol", 1e-3)
    atol = config.get("atol", 1e-5)
    n_pes = config["n_pes"]

    all_pass = True
    for pe in range(n_pes):
        output = np.fromfile(f"{args.data_dir}/output_pe{pe}.bin", dtype=dtype)
        golden = np.fromfile(f"{args.data_dir}/golden_pe{pe}.bin", dtype=dtype)
        if not check_precision(output, golden, rtol, atol, label=f"PE {pe}"):
            all_pass = False

    if all_pass:
        print(f"\033[32m✓ ALL {n_pes} PEs PASSED\033[0m")
        return 0
    else:
        print(f"\033[31m✗ SOME PEs FAILED\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## scripts/run.sh → `<op_name>/scripts/run.sh`

```bash
#!/bin/bash
# ============================================================
# SHMEM Fused Compute-Communication 多 PE 启动脚本模板
# 使用: 复制到 <op_name>/scripts/run.sh
#       替换 <op_name> 占位符，按 compile/test contract 调整路径
# ============================================================

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
OP_DIR=$(dirname "$SCRIPT_DIR")
# custom-ops 独立工程：SHMEM 仓库根 = OP_DIR 的上两级
SHMEM_REPO=$(cd "${OP_DIR}/../.." && pwd)
EXEC_BIN="${OP_DIR}/build/bin/<op_name>"
DATA_DIR="${OP_DIR}/data"

# ---- 参数 ----
DEVICE_LIST="${1:-0,1}"
IFS=',' read -ra DEVICE_ID_LIST <<< "$DEVICE_LIST"
PE_SIZE=${#DEVICE_ID_LIST[@]}
M="${2:-1024}"
N="${3:-1024}"
K="${4:-512}"
PERF_TIMES="${5:-0}"
DTYPE="${6:-float16}"
# IPPORT / SHMEM_UID_SESSION_ID 由 setup_shmem_runtime_env → setup_shmem_dynamic_endpoints 分配

echo "=== <op_name> === PE_SIZE=${PE_SIZE} M=${M} N=${N} K=${K} DTYPE=${DTYPE}"

# ---- 生成数据和 golden ----
mkdir -p "${DATA_DIR}"
python3 "${SCRIPT_DIR}/gen_data.py" \
    --n_pes "${PE_SIZE}" \
    --m "${M}" --n "${N}" --k "${K}" \
    --dtype "${DTYPE}" \
    --data_dir "${DATA_DIR}"

# ---- SHMEM 环境（MUST 完整链路；函数体见 env-setup.snippet.md）----
setup_shmem_runtime_env "${SHMEM_REPO}" "${OP_DIR}" || exit 1

if [[ ! -x "${EXEC_BIN}" ]]; then
    echo "Build first: cmake --build ${OP_DIR}/build (see shmem-ops-compile-debug/references/custom-ops-entrypoints.md §1)" >&2
    exit 1
fi

# ---- 启动多 PE 进程 ----
PIDS=()
for (( idx = 0; idx < PE_SIZE; idx++ )); do
    ${EXEC_BIN} "${PE_SIZE}" "${idx}" "${IPPORT}" \
        "${M}" "${N}" "${K}" "${DATA_DIR}" "${PERF_TIMES}" &
    PIDS+=($!)
done

# ---- 等待所有进程 ----
EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || EXIT_CODE=1
done

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Some PE processes failed"
    exit 1
fi

# ---- 验证结果 ----
if [ "${PERF_TIMES}" -eq 0 ]; then
    python3 "${SCRIPT_DIR}/check_result.py" --data_dir "${DATA_DIR}"
    exit $?
fi
```
