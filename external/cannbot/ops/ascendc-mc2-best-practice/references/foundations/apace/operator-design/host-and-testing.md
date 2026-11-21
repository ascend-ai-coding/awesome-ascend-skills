# apace 算子解剖（host 与测试）

> 本文档覆盖 apace 算子的 host 侧与测试侧：CommContext 传递、host 初始化序列、kernel launch、ST 测试工程与性能采集。

## 目录

1. [CommContext 传递模式](#1-commcontext-传递模式)
2. [Host 侧初始化序列（UDMA 模式）](#2-host-侧初始化序列udma-模式)
3. [Kernel Launch](#3-kernel-launch)
4. [perf 模式](#4-perf-模式)
5. [性能采集工程模板（AG ST）](#5-性能采集工程模板ag-st)

---

## 1. CommContext 传递模式

### 模式对比

| 模式 | CommContext | 上下文获取方式 | 支持直调 |
|:---|:---|:---|:---|
| **UDMA** | 第一参数，`__gm__` 指针传递 | Impl::Init 从指针提取 `udmaCtx` 和 `ubmemCtx` | ✅ |
| **HCCL windows** | 不传 | kernel 内部 `GetHcclContext`（依赖框架注入） | ❌ |

> **重要限制**：HCCL windows 模式**无 `__global__` 入口，不支持 CANNBot Kernel 直调工作流**。官网 apace 现有两个算子均为 UDMA 模式。详见 [`architecture.md`](../fundamentals/architecture.md) §10 ④。

### 验收条件

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | UDMA 模式 | 入口签名含 `__gm__ CommContext*` 参数 |
| 2 | Host 构造 | Host 侧构造 CommContext → 写入 GM → 传指针 |
| 3 | CommContext 定义位置 | `CommContext` 聚合体（udmaCtx+ubmemCtx）定义在**本算子 tiling_data.h** 中（PUT 在全局命名空间；AG 在 `Apace::AivComm` 命名空间） |

---

## 2. Host 侧初始化序列（UDMA 模式）

完整顺序（与官网两份 ST `main.cpp` 一致，`runAllToAllMatmul` / `RunAllGatherQuantMatmul`）：

```
fork rankNum 子进程（每进程一个 rank）
  └── RunKernel(rankId)
        0. tiling 推导（GetTilingData + CommTilingData 填充，见 operator-anatomy.md §3）
        1. aclInit + aclrtSetDevice + aclrtCreateStream
        2. TCP 交换 HcclRootInfo（RootInfoExchanger::Exchange，tests/st/utils/root_info_exchanger.h）
        3. HcclCommConfigInit + HcclCommInitRootInfoConfig 创建 HcclComm
        4. CommChannelBuilder::CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag,
                                                   &hostCtx.udmaCtx, &hostCtx.ubmemCtx)
           → 返回 device GM 中的 CommContext*
        5. 跨 rank host barrier（RootInfoExchanger::Barrier()）  ← 强制，不可省略
        6. ReadFile 输入 + aclrtMalloc + aclrtMemcpy（H2D）
        7. <<<>>> launch kernel（devContext 作第一参数）
```

> CreateDeviceContext 内部建链机制（URMA/UBMEM 通道填充、memset 归属、BARRIER_BUF_SIZE）详见 [`communication.md`](../fundamentals/communication.md) §6。

**不可省略的前置/后置条件**：

| 条件 | 原因 |
|:---|:---|
| **CreateDeviceContext 后跨 rank host barrier** | builder 头文件注释明确要求：确保所有 channel 握手完成再 launch，否则 TeamBarrier 轮询远端未就绪 flag 会无限挂死 |
| engine 一致性 | `HcclChannelAcquire` 与 `HcclEngineCtxGet/Create/Copy` 使用同一 engine（`BUILDER_COMM_ENGINE_AIV = 4`，comm_channel_builder.h），否则 ctxTag 复用失效 |
| **ctxTag 变化即换 tag** | `HcclEngineCtxGet` 同 tag 直接复用已建链 context，内容跨 launch 不可变——shape/rank/通信布局变化必须换 ctxTag，否则沿用旧 context 静默错误 |

**staging（workspace）分配（compute-first 算子）**：

| 要点 | 规则 |
|:---|:---|
| 分配方式 | `aclrtMalloc` 分配 GM，作为 kernel 入参（`GM_ADDR workspaceGM`）传入；mm 输出 staging 即 PUT 通信源（零重排，见 [`fusion.md`](../fundamentals/fusion.md) §6.2.4） |
| 大小公式 | `stagingSize = M × N × sizeof(CType)`（= `rankSize × chunkBytes`），填进 tiling 结构体同名字段 |
| T 派生 | 通信轮次 T 由 host 派生：`T0 = max(1, CeilDiv(mSeg, 目标tile行数))`，取最小满足 `T \| mSeg` 的 T（默认无尾块，PUT 钩子 src 偏移限制）；找不到回退 T=1（[`fusion.md`](../fundamentals/fusion.md) §6.2.7） |
| **flagId 范围校验** | flagId 数值范围 0-15（16 通道，超出截断低 4bit；硬件规则见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md` §3）：**计数式**（固定 flagId 递推计数）无 T 数值上限要求（R9 口径：每 flagId 计数器 0-15 衡量未消费积压，紧邻配对下 ≈1-2 不触顶，不设 host 强制数值拒绝）；**轮次索引式**（flagId=tid）须校验轮次索引 < FLAG_ID_MAX(16) 且避开 SyncAll 保留区（`SyncAll<true>` 占 14 → 实际安全上界 13），超限拒绝 launch 并报错（含全参数 + 提示缩小 shape 或调大 tile）。校验位置：tiling 完成后、staging aclrtMalloc 之前 |
| Win 区容量校验 | 计算在前算子 Win 需求 = `M × N × sizeof(CType)` ≤ `HcclGetHcclBuffer` 实测值；通信在前算子 = rankSize ×（data 段 + scale 段）≤ HCCL 内置 buffer（见 operator-anatomy.md §3.5） |

**资源生命周期**：builder 无清理接口，context 按 ctxTag 复用；释放路径与两份 ST 处置差异详见 [`communication.md`](../fundamentals/communication.md) §6。

> HCCL Host API 签名详见 `ascendc-api-best-practices` skill `references/api-hccl-host.md`；字段布局与验收条件详见 [`communication.md`](../fundamentals/communication.md) §5/§6。

---

## 3. Kernel Launch

`__global__` 入口由 Host 侧 `main.cpp` 通过 `<<<>>>` 调用。

### `<<<>>>` 三参数

| 参数 | 含义 | 来源 |
|:---|:---|:---|
| `usedCoreNum` | block 数（AIC 核数） | PUT：`tilingData.tileQbmmTilingData.usedCoreNum`；AG：`tilingData.mmTile.usedCoreNum`（均由 GetTilingData 推导） |
| `nullptr` | shared mem（不显式指定） | — |
| `stream` | ACL stream | `aclrtCreateStream(&stream)` |

### main.cpp 整体结构（两份官网 ST 一致）

```
main()
  ├── fork rankNum 个子进程（每进程一个 rank）
  │   └── RunKernel(rankNum, rankId, ...)
  │       ├── 填充 tiling（GetTilingData + CommTilingData，详见 operator-anatomy.md §3）
  │       ├── aclInit + aclrtSetDevice + aclrtCreateStream
  │       ├── TCP 交换 HcclRootInfo（RootInfoExchanger）
  │       ├── HcclCommInitRootInfoConfig 创建 HCCL comm
  │       ├── CommChannelBuilder::CreateDeviceContext 填充 CommContext
  │       ├── 跨 rank host barrier（RootInfoExchanger::Barrier()，强制，见 §2）
  │       ├── ReadFile 加载 input 数据
  │       ├── aclrtMalloc + aclrtMemcpy（H2D）
  │       ├── launch（precision 单次 / perf 循环）
  │       ├── aclrtMemcpy（D2H）+ WriteFile("npu_out.bin")
  │       └── 资源释放（aclrtFree 输入输出 → HcclCommDestroy → aclrtDestroyStream → aclrtResetDevice → aclFinalize）
  └── waitpid 收集所有子进程状态
```

### 验收条件

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 计时方式 | PUT perf 模式用 `std::chrono::steady_clock`（PERF_LOOP_COUNT=10，统计去掉首轮）；AG perf 模式用 `aclrtEvent`（BENCHMARK_ITERATIONS=20，`aclrtEventElapsedTime` 取平均） |
| 2 | tiling 传递 | `tilingData` 按值传递给 kernel（编译器拷贝到 GM） |
| 3 | CommContext 传递 | `devContext` 按指针传递（`CommContext*` 类型 device 指针，由 `CommChannelBuilder::CreateDeviceContext` 返回，详见 [`communication.md`](../fundamentals/communication.md) §6） |
| 4 | 输出写入 | 两份 ST 在 precision 与 perf 模式下均写 `npu_out.bin` |
| 5 | 模式区分 | 两份 ST 均支持 `precision`（默认）/ `perf` 命令行模式 |

### 3.1 多 rank 工程陷阱（生产实测，dav-3510 / CANN 9.2.0）

| 陷阱 | 现象 | 缓解/修复 |
|:---|:---|:---|
| fork + TCP rootInfo 建链竞态 | msprof 包装后 fork 启动时序变化，非 0 rank 连不上 rank0 listen（10s 重试窗口耗尽）；残留半建链进程假"卡死" | `taskset -c 0` 钉核缓解；`fuser -k 8998/tcp` 清残留端口（run.sh 已内置）；根因（listen 就绪同步）未修，新工程应显式做 rank0 就绪同步 |
| msprof 多设备同采 Task Duration 双峰 | 4-rank profiler 交叉采样开销，同一 case 出现两个量级（如 1.13/3.33ms） | 以 host 计时锚定口径，msprof 数据做 pipe 统计参考；双峰数据不直接进对比表 |
| 采集脚本设备占用误判 | 设备探测逻辑误把 `npu-smi` 设备表行计为占用进程 → 恒判忙 → 采集恒超时 | 探测逻辑只解析进程表行；采集脚本的"环境探测"须单测后再上队列 |
| CMakeCache/CPATH 环境污染 | 系统多版本 CANN 共存时，CMakeCache 缓存旧编译器路径（如 9.1.0 bisheng）→ 编译 `npu_arch_2201` 头缺失等错位报错 | 删 `CMakeCache.txt`/`CMakeFiles` 重 configure；run.sh `unset CPATH`；显式 `export ASCEND_HOME_PATH` |
| 精度验收独立性 | 复用开发态构建/golden 会掩盖问题 | Reviewer 独立构建（校验 md5）、独立生成 golden、独立通信端口（避开开发态占用的 8998） |
| 共享 NPU 队列纪律 | 本地直跑复现实验会与其他开发者任务冲突 | 所有 NPU 任务经 `qrun` 排队（含诊断实验）；一次排队内完成多版本对比（工程化二分开关：环境变量控制禁用 COMM/REDUCE/FLAGC，默认关闭） |

---

## 4. perf 模式

PUT ST（`runAllToAllMatmul` perf 分支）实测：

| # | 项 | 官网现状 |
|:---|:---|:---|
| 1 | 计时 | `std::chrono::steady_clock`，PERF_LOOP_COUNT=10 轮，统计时去掉首轮（avg/min/max） |
| 2 | 同步 | 循环结束后 `aclrtSynchronizeStream` |
| 3 | cacheFlush buffer | 分配 128M×uint16（256MB）并填充 0x0000，**未被任何 kernel 消费（死代码）**——官网尚未实现真正的 L2 flush |
| 4 | boostIn/boostOut | 同样分配并填充（0x3F00）但未被消费（死代码） |

**skill 侧 L2 flush 扩展**（消除前一轮 L2 cache 热度污染，方法论见 [`profiling_mc2.md`](../../../shared/profiling_mc2.md)）：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | L2 flush kernel | 每轮主 kernel 前调用 L2 flush kernel |
| 2 | flush buffer | 大小大于目标架构 L2 容量，填充非零数据 |
| 3 | 同步 | flush 后 `aclrtSynchronizeStream` 确保完成 |
| 4 | 资源释放 | flush buffer 在主循环后释放 |

> ⚠️ L2 flush 为 **skill 侧方法论**，官网仓无 flush kernel 实体；官网 perf 分支的 cacheFlush buffer 是未接线死代码，不能当作已实现的 L2 flush 引用。**只分配 buffer 不调用 kernel = 死代码**（flush 未生效会导致 MTE2 带宽虚高）。

**heavy_add_kernel 实现模板**（已验证形态，可直接复制改造）：

```cpp
// heavy_kernels.h — AIV-only L2 flush kernel：对 > L2 容量的 buffer 做 x+=1 扫描，挤出前一轮热度
#include "kernel_operator.h"

constexpr int32_t HEAVY_BLOCK_NUM = 56;                           // AIV 核数示例值，按目标芯片实际 AIV 核数调整（规格以 npu-arch skill 为唯一知识源）
constexpr int32_t HEAVY_TILE_SIZE = 32 * 1024;                    // 32KB per tile
constexpr int64_t CACHE_FLUSH_ELEM_COUNT = 128L * 1024L * 1024L;  // 示例值 256MB——大小须大于目标芯片 L2 容量（规格以 npu-arch skill 为唯一知识源）

__global__ __aicore__ __vector__ void heavy_add_kernel(GM_ADDR x, int64_t totalLength, int64_t blockLength)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);               // 必须 AIV-only（全 AIV 核参与扫描）
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQue;
    pipe.InitBuffer(inQue, 2, HEAVY_TILE_SIZE);
    pipe.InitBuffer(outQue, 2, HEAVY_TILE_SIZE);
    AscendC::GlobalTensor<half> xGm;
    int64_t blockOffset = blockLength * AscendC::GetBlockIdx();
    xGm.SetGlobalBuffer((__gm__ half *)x + blockOffset);
    // 按 32KB tile 双缓冲流水：搬入 → Add(x, x, 1) → 写回（inplace 扫描）
    for (int64_t i = 0; i + HEAVY_TILE_SIZE / sizeof(half) <= blockLength; i += HEAVY_TILE_SIZE / sizeof(half)) {
        auto t = inQue.AllocTensor<half>();
        AscendC::DataCopy(t, xGm[i], HEAVY_TILE_SIZE / sizeof(half));
        inQue.EnQue(t);
        auto v = inQue.DeQue<half>();
        AscendC::Add(v, v, static_cast<half>(1.0), HEAVY_TILE_SIZE / sizeof(half));
        auto o = outQue.AllocTensor<half>();
        AscendC::DataCopy(o, v, HEAVY_TILE_SIZE / sizeof(half));
        outQue.EnQue(o);
        inQue.FreeTensor(v);
        auto w = outQue.DeQue<half>();
        AscendC::DataCopy(xGm[i], w, HEAVY_TILE_SIZE / sizeof(half));
        outQue.FreeTensor(w);
    }
}
```

> 模板要点：① flush buffer 必须**大于目标架构 L2 容量**（参考工程实测示例值 256MB 足够；L2 确切容量等芯片规格以 npu-arch skill 为唯一知识源）；② 每轮主 kernel 前调用 + `aclrtSynchronizeStream`；③ 尾部不足一个 tile 的部分直接忽略（flush 仅用于热度清除，无正确性语义）；`HEAVY_BLOCK_NUM` 按目标芯片 AIV 核数调整。

perf 循环接入（每轮主 kernel 前调用）：

```cpp
for (int iter = 0; iter < PERF_LOOP_COUNT; ++iter) {
    heavy_add_kernel<<<HEAVY_BLOCK_NUM, nullptr, stream>>>(cacheFlush, CACHE_FLUSH_ELEM_COUNT, cacheFlushBlockLen);
    aclrtSynchronizeStream(stream);          // 确保 flush 完成（热度不再残留）
    auto t0 = std::chrono::steady_clock::now();
    MainKernel<<<...>>>(...);
    aclrtSynchronizeStream(stream);
    // 计时统计（剔除首轮）
}
```

> 验收：msprof 结果中 heavy_add_kernel 记录数 == PERF_LOOP_COUNT（每轮 1 次）；缺此记录即 flush 未接线。

---

## 5. 性能采集工程模板（AG ST）

`all_gather_quant_matmul` ST 工程提供了更完善的性能采集模板，推荐新算子参考（均为仓内实测文件）：

| # | 文件 | 用途 |
|:---|:---|:---|
| 1 | `apace/tests/st/all_gather_quant_matmul/run.sh` | ST 驱动脚本，支持 cases.csv 行选择、`--perf` msprof 采集、`KERNEL_TIMEOUT` 超时覆盖 |
| 2 | `apace/tests/st/all_gather_quant_matmul/scripts/parse_prof.py` | 解析 msprof `op_summary_*.csv`：latency 为 warmup-skip（WARMUP_SKIP=3）+ outlier 过滤（>1.2×min）后的每卡 average；cube_utilization 取 median |
| 3 | `parse_prof.py --check-latest-threshold` | CI 门禁模式，末行输出 latency 数字（供 grep 提取），exit 2 表示超阈值 |
| 4 | `KERNEL_SUBSTR` 适配 | KERNEL_SUBSTR 需替换为新算子入口名（parse_prof.py 硬编码 AllGatherQuantMatmulKernel） |

---

## 6. 投产级性能验证门槛（红线 R15）

**精度全 PASS ≠ 可投产**。性能验证必须满足以下矩阵才算完成验收（对应 SKILL.md R15）：

| # | 要求 | 说明 |
|:---|:---|:---|
| 1 | **真实 shape 矩阵** | 覆盖目标业务的真实大 shape（K/N 达数千量级），禁止只用 toy shape（K 仅数百量级）下结论——toy shape 下 CUBE 占比极低，SCALAR/同步开销主导，性能结论无代表性 |
| 2 | **rank 数多档** | 至少覆盖两档 rank 数（如 R=2 与 R=4）；通信并行度、tile 粒度分档的收益随 R 变化显著 |
| 3 | **参考路径对标** | 与可获得的对标路径同 shape 对比归档：① mc2 融合算子路径（如 aclnn 融合算子）；② hccl+mm 分步路径；③ 本算子 ST 直调路径。产出对比表，无对标数据的"性能达标"结论无效 |
| 4 | **隔离归因与归档** | 性能不达标时做 COMPUTE/COMM 隔离测试确认瓶颈侧再优化；msprof 原始数据 + 解析结果（Task Duration 中位数、跨 rank max）归档到 `profiling/` |

> 反面教材：只采集单个 toy shape 的基线、无大 shape、无多 rank 档、无对标路径——既不能暴露架构级缺陷（如单核串行通信），也无法支撑投产决策。

**三路径对标实现方法**：

| 路径 | 实现方式 | 关键注意点 |
|:---|:---|:---|
| ① mc2 融合算子路径 | 调用 CANN 内置 aclnn 融合算子（如 `aclnnMatmulReduceScatterV2`，torch_npu 或 aclnn C API 均可），同 shape 矩阵 + 同 HCCL 建链 | 需确认目标 CANN 版本内置该融合算子且支持相同 dtype/量化格式；无对应融合算子时此路径标注 N/A 并说明 |
| ② hccl+mm 分步路径 | 组合调用单算子（如 `aclnnQuantMatmulV5` 本地 mm + `HcclReduceScatter` 或 `aclnnReduceScatter` 通信），串行执行无重叠 | 分步路径必须同 dtype/同 golden 语义；通信用 HCCL 高阶 API（此路径非直调 kernel，HCCL 可用） |
| ③ 本算子 ST 直调路径 | 本工程 perf 模式（heavy_add_kernel + N 迭代计时） | 与 ①② 同 shape、同 L2 flush、同取值口径（warmup 3 + 去离群平均 × 跨 rank max，见 §6.1 #2） |

**对标产出**：`profiling/comparison/` 下三路径 Task Duration 对比表与加速比；`profiling/ANALYSIS.md` 归档 pipe 级指标（mac/mte2/fixp 逐路径对比）与瓶颈归因。无 ①② 对标数据时 R15 判 FAIL（仅基线采集 = 未达投产门槛）；若因架构约束（如 Win 容量限制 M×N 上限）无法覆盖业务真实大 shape，须在 ANALYSIS.md 显式声明约束边界与适用范围。

> aclnn 对标算子的**调用契约坑**（CCU 引擎在 fork 直调场景不可行、FP8 必须带 scale、scale 形状校验、group 不可手动创建、libhccl 误链旧版本等生产试错记录）见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §4.1——对标是设计期活动，试错成本须在 PLAN 预算。

### 6.1 profiling 目录范式（已验证结构）

```
profiling/
├── st_direct/            ← 本算子 ST 直调路径：raw/{shape}/op_summary_dev{0..R-1}.csv
├── mc2_test/             ← 对标① mc2 融合算子路径（如 aclnnMatmulReduceScatterV2），同 shape 矩阵
├── hccl_mm/              ← 对标② hccl+mm 分步路径，同 shape 矩阵
├── isolation_test/       ← 隔离拆解：raw/COMM_{shape}/ 与 raw/COMPUTE_{shape}/（确认瓶颈侧后再优化）
├── comparison/           ← 三路径对比日志（mc2/ST、hccl/ST 加速比表）
└── ANALYSIS.md           ← 对比分析（含 pipe 级指标：mac/mte2/fixp 逐路径对比 + 瓶颈归因 + mask 收益分析）
```

**采集纪律**：

| # | 项 | 规范 |
|:---|:---|:---|
| 1 | L2 flush | 每轮主 kernel 前执行 > L2 容量的 D2D copy（参考工程示例值 256MB，红线 R②）；perf 模式推荐**进程内内嵌**（cacheFlush buffer + boost kernel + N 迭代计时，可复现性优于外置脚本） |
| 2 | 稳态取值 | **官方脚本口径**（parse_prof.py）：跳过前 3 轮 warmup（`WARMUP_SKIP=3`）→ 剔除 >1.2×min 离群（`OUTLIER_FACTOR=1.2`）→ 每卡取**平均** → Overall = 各卡平均值的**平均**（横幅 "avg of card avgs"）；cube_utilization 取中位数。**skill 投产纪律（区别于官方口径，须分开归档）**：另报**跨 rank max**（木桶效应——最慢 rank 决定整体性能）。⚠️ [`profiling_mc2.md`](../../../shared/profiling_mc2.md) §4.4 的多卡后处理是另一套简化口径（每卡末 5 次平均 + 4 卡 max，无离群过滤）——正式投产验收以本表官方口径 + 跨 rank max 为准，两套口径不得混用 |
| 3 | 采集污染检查 | warm-up/刷流水用 `torch.npu.current_stream().synchronize()`（仅同步当前流），禁止全局 flush 类操作（如大循环 `torch.exp` 会在 profiling trace 引入额外算子污染） |
| 4 | 对标同 shape | 三路径必须同 shape 矩阵、同 L2 flush、同取值口径，否则加速比无效 |
| 5 | 隔离测试形态 | COMM-only（仅通信，mm 输出用预填数据）与 COMPUTE-only（仅 mm+归约，通信跳过）分别采集，MTE2/CUBE 占比对比定位瓶颈侧 |

---

## 后续阅读

| 文档 | 何时读 |
|:---|:---|
| [`operator-anatomy.md`](operator-anatomy.md) | kernel 侧骨架（tiling/Impl/入口） |
| [`communication.md`](../fundamentals/communication.md) | 建链机制（builder 内部步骤、CommContext 字段） |
| [`development-guide.md`](development-guide.md) | 开发流程与验收清单 |
| [`../shared/profiling_mc2.md`](../../../shared/profiling_mc2.md) | msprof 采集与多卡后处理方法论 |
