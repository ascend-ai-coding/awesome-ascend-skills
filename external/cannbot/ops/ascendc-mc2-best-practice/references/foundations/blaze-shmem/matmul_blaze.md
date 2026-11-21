# 计算层（Blaze Matmul）参考

本文档承载 MC2 skill 的"计算子能力"。涵盖：Blaze 模板选型、BlockMmad 接入、Tiling 数据流、与通信层的 buffer 协议。

> Blaze 是 Ascend C 的 CUTLASS 风格模板库，由 `tensor_api/` 提供。**两条路线的 Blaze/tensor_api 来源不同**：
> - **blaze-shmem 路线**：tensor_api 来自 `gitcode.com/cann/asc-devkit` 仓 `feature/tensor_api_from_9.0.0` 分支，Blaze 来自 CANN toolkit 内置 `opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common`（参考工程 CMakeLists 用 `_BLAZE_COMMON_DIR` 指向该目录）
> - **apace 路线**：Blaze **和** tensor_api 均来自 `gitcode.com/cann/ops-tensor` 仓（tensor_api 是 ops-tensor 的 submodule，指向 asc-devkit）
>
> 与 `ascendc-blaze-best-practice` 共享 Blaze 基底，但聚焦 MC2 场景下的差异点。**禁止使用 asc-devkit 的 matmul 高阶 API（`AscendC::Matmul` 等）**——它是官方通算融合（单算子 API 调用）路径的计算接口，不支持 Kernel 直调场景。

---

## 1. MC2 场景下的 Blaze 子集

参考工程用到的 Blaze 组件：

| 层级 | 组件 | 文件 | 作用 |
|------|------|------|------|
| **Kernel** | `Blaze::Gemm::Kernel::QuantMatmulMxKernelSwat`（项目级） | `include/kernel/qbmm_mx_kernel.h` | SWAT 量化 Matmul kernel 包装，遍历所有 rank 累加 L0C |
| **Block** | `Blaze::Gemm::Block::BlockMmad` | `blaze/gemm/block/block_mmad_qbmm_mx.h`（toolkit 内） | 单 Block 的 MMAD 计算（L0A×L0B→L0C→GM） |
| **Block Scheduler** | `Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3` | `include/block/quant_matmul_mx_block_scheduler_swat.h` | 多 Block 间任务切分 |
| **Dispatch Policy** | `Blaze::Gemm::MatmulWithScaleMx` | `blaze/gemm/policy/dispatch_policy.h`（toolkit 内） | 流水策略（含 scale 处理） |
| **Tile** | `Blaze::Gemm::Tile::*` | `include/tile/*.h` | L1→L0 搬运、Scale pad |
| **Layout/Tensor** | `AscendC::Te::*` | `tensor_api/`（SHMEM: asc-devkit clone; apace: ops-tensor submodule） | Tensor / Layout 抽象 |

**Agent 开发原则**：`include/block/`、`include/tile/`、`include/policy/` 下的文件 **`[REUSE]`**，常规 MC2 算子不需要改。需要改的是：
- `include/kernel/qbmm_mx_kernel.h`：Scale 处理、A/B 来源切换；
- `include/kernel/all_to_all_matmul_impl.h`：通算流水编排；
- `include/tiling/quant_matmul_mx_tiling_swat.h`：Tiling 字段。

---

## 2. 从 Blaze 到 MC2 的桥接

参考工程在 `qbmm_mx_kernel.h` 定义 `QuantMatmulMxKernelSwat`，把标准 Blaze `BlockMmad` 接入 MC2 通算流水。AIC 在通信 buffer 上遍历所有 rank，把各 rank 的部分和在 L0C 上累加，最后一次 `mmadOp_` 触发 fixpipe 输出 GM。

```cpp
// qbmm_mx_kernel.h ProcessSingleBatch 核心逻辑（简化伪码）
for (uint64_t rank = 0; rank < rankSize; rank++) {
  auto actualMPos = rank * oriM + mPos;            // local A 按 oriM 分段
  auto actualCommMPos = rank * headMSize + mPos;   // 通信 buffer 按 headMSize 分段

  // 默认从通信 buffer 读 A；rank == rankId 时改从本卡 GM 读
  auto gmBlockA = gmA.Slice(actualCommMPos, ...);
  auto gmBlockScaleA = gmScaleA.Slice(actualMPos, ...);
  if (rank == rankId) {
    gmBlockA = gmALocal.Slice(actualMPos, ...);    // 本卡 GM
    gmBlockScaleA = gmScaleALocal.Slice(actualMPos, ...);
  }

  // B / ScaleB 始终从本卡 GM 读，按 rank 切 K 轴段
  auto gmBlockB = gmB.Slice(rank * K + nPos, ...);

  // L0C 上累加：remoteRankCnt=0 时 L0C reset；最后一次触发 fixpipe
  mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC,
          singleShape, remoteRankCnt);
  remoteRankCnt++;
}
```

### 2.1 为什么 rank == rankId 时从本卡 GM 读？

`AllToAllComm::PutToAllRanks` 中每个 Block 只 Put 给 `remoteRank != rankId` 的对端——本 rank 不 Put 给自己。所以通信 buffer 中本 rank 的段是**未填充**的，AIC 算到 `rank == rankId` 时必须切换到本卡 GM（`localAGmAddr_`），否则会读到未初始化数据。

### 2.2 remoteRankCnt 的语义

`mmadOp_` 的最后一个参数 `remoteRankCnt` 控制 L0C 累加位置：
- `= 0`：L0C reset，开始新的累加序列；
- `> 0`：在已有结果上累加；
- `== splitKNum - 1`（这里 splitKNum = rankSize）：触发 fixpipe，把 L0C 累加结果写回 GM。

参考工程在 `SetupParams` 中固定 `splitKNum = rankSize`，保证 `for rank` 循环最后一次正好触发 fixpipe。

---

## 3. Layout 与 Tensor 构造

参考工程在 `qbmm_mx_kernel.h::ProcessSingleBatch` 构造 Blaze Tensor 句柄：

```cpp
// Layout 构造
auto layoutA = MakeLayoutA{}(rankSize * Te::Get<MNK_M>(problemShape),
                              Te::Get<MNK_K>(problemShape));
auto layoutB = MakeLayoutB{}(rankSize * Te::Get<MNK_K>(problemShape),
                              Te::Get<MNK_N>(problemShape));
auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(problemShape),
                              Te::Get<MNK_N>(problemShape));

// Tensor 句柄
auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aGmAddr_), layoutA);
auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), layoutB);
auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);
```

### MC2 场景的 layout 维度

注意 layout 维度包含 `rankSize *`，因为 A 和 B 矩阵在 MC2 中是"按 rank 切分后逻辑拼起来"的：

- **A 矩阵**：每卡持有自己的 M 段，但 SHMEM buffer 收齐所有 rank 的 M 段后，逻辑上是 `rankSize * headMSize` 行的矩阵；
- **B 矩阵**：每卡持有自己的 K 段（kPerRank），逻辑上是 `rankSize * kPerRank = K` 行的矩阵；
- **C 矩阵**：本卡只输出自己 M 段的结果，维度是 `headMSize * N`（不含 rankSize）。

Slice 时按 rank 切：

```cpp
// 取远程 rank 的 A 段
auto gmBlockA = gmA.Slice(
  Te::MakeCoord(rank * Get<MNK_M>(problemShape) + mPos, kPos),
  Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(problemShape)));
```

---

## 4. Tiling 数据流

参考工程的 Tiling 链路：

```
Host: QuantMatmulTilingSwat::GetTilingData()
    ↓ 填充 QuantMatmulTilingData (baseM/baseN/baseK/...)
Device: AllToAllQuantMatmulImpl::SetupParams()
    ↓ 转成 Blaze Params (BlockMmadParams + L1Params + ...)
Device: QuantMatmulMxKernelSwat::Run()
    ↓ 调用 BlockMmad
Device: BlockMmad::operator() 做 MMAD
```

### Tiling 字段速查（`include/tiling/quant_matmul_tiling_data.h`）

| 字段 | 含义 |
|------|------|
| `m, n, k` | 问题 shape（注意 k 是单卡的 kPerRank） |
| `baseM, baseN, baseK` | 单 block 的 tile shape |
| `mTailTile, nTailTile` | 尾块切分 |
| `mBaseTailSplitCnt, nBaseTailSplitCnt` | 尾块分裂策略 |
| `mTailMain, nTailMain` | 尾块主区大小 |
| `usedCoreNum` | 实际使用的 AIC 核数 |
| `dbL0c` | L0C DoubleBuffer 深度（1 或 2） |
| `scaleKL1` | Scale 在 L1 的复用深度 |
| `stepK` | K 轴 step |
| `nBufferNum` | L1 buffer 数 |

### MC2 专属 Tiling 字段（`include/tiling/all_to_all_matmul_tiling_data.h`）

```cpp
struct AllToAllCommTilingData {
    uint32_t tileCnt;      // M 轴切分块数（headMSize * tileCnt = M）
    uint32_t bufferSize;   // 通信流水深度（典型 4）
};
```

参考工程 host 侧的 Tiling 计算（`src/all_to_all_matmul.cpp`）：

```cpp
uint32_t headMSize = 512;  // 参考工程默认值，非最优——详见 pipeline_tuning.md
uint32_t tileCnt = (m - tailMSize) / headMSize;
tilingData.commTilingData.tileCnt = tileCnt;
tilingData.commTilingData.bufferSize = 4;
// 每块 matmul 的 Blaze tiling 由 GetTilingData 根据 headMSize 自动推导
tilingEngine.GetTilingData(headMSize, n, ka, false, true, tilingData.tileQbmmTilingData);
```

> `headMSize=512` 只是参考工程经验起点，**实际最优 `tileCnt`（即 `headMSize = M/tileCnt`）以 `msprof op` 实测为准**。精度调试阶段建议先用 `tileCnt=1`（`headMSize=m`）做串行基线，性能调优阶段再扫描 `tileCnt` 找最优——详见 [`pipeline_tuning.md`](../../shared/pipeline_tuning.md)。

### Tiling 算法

`include/tiling/quant_matmul_mx_tiling_swat.h` 实现了 SWAT tiling 算法（SWAT 为项目内部命名，CANN 库中无此术语的官方定义。算法行为是滑动窗口 + N 轴 zigzag 的 tile 调度）：

1. `CalcBasicBlock()`：从 256 出发，对齐到 CUBE_BLOCK（16）和 L1 对齐粒度；
2. `OptimizeEdgeBasicBlock()`：合并 K 对齐时的尾块；
3. `CalcTailBasicBlock()`：尾块按 M/N 双向切分；
4. `CalcPathSpecificL1()`：搜索 L1 深度（A/B 对称起步，必要时打破）。

新算子一般不改这些算法，只调 host 侧的 `headMSize`（等价于调 `tileCnt = M/headMSize`）等参数。

---

## 5. DispatchPolicy 选择

参考工程用 `MatmulWithScaleMx`（带 MX 量化 scale 的 dispatch policy）：

```cpp
// all_to_all_matmul_impl.h using 声明段
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>;
```

- `NONE_FULL_LOAD_MODE`：A/B 不全载 L1（与 `ascendc-blaze-best-practice` 的模式选择一致）；
- 第二个模板参数 `false` 是 `ATOMIC_ADD`（是否启用输出 atomic add，默认关闭）。

可选的 DispatchPolicy（详见 `ascendc-blaze-best-practice` skill `references/scenarios/index.md`）：
- `MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>`：通用多 block SWAT；
- `MatmulMultiBlockPolicy<A_FULL_LOAD_MODE>`：A 全载（N≫M 时用）；
- `MatmulWithScaleMx<...>`：MX 量化 matmul（参考工程用）。

MC2 算子若非量化场景，可改用 `MatmulMultiBlockBasic`。

---

## 6. Scale（量化系数）处理

参考工程是 MX FP8 量化 matmul，Scale 处理是关键。两条路径：

### 6.1 Scale 的 SHMEM Put

Scale 不参与 M 轴流水，AIV 启动时一次性 Put（`all_to_all_matmul_impl.h::AllToAllProcess` 开头）：

```cpp
allToAllComm_.PutScaleToAllRanks(0, axisM_);  // offset=0, 全 M 行
```

### 6.2 Scale 的 L0B 加载

参考工程在 `include/tile/copy_scale_l1_to_l0b.h` 处理 Scale 从 L1 到 L0B 的搬运。`qbmm_mx_kernel.h` 的 `SetScaleL2Cache`（`ProcessSingleBatch` 内）控制 Scale 的 L2 cache 行为（数据对齐时禁用 cache 走 stream，不对齐时走 normal）。

### 6.3 非 MX 场景

若新算子不做量化（纯 BF16/FP16 matmul），可以：
- 去掉 `allToAllComm_.PutScaleToAllRanks`；
- 把 `BlockMmad` 模板从量化版（`block_mmad_qbmm_mx.h`）换成普通版（`block_mmad.h`）；
- Tiling 字段去掉 scaleKL1、stepK 等。

---

## 7. 排错速查

| 现象 | 可能原因 | 排查方向 |
|------|---------|---------|
| 编译报 `BlockMmad` 模板参数错误 | DispatchPolicy 与 BlockScheduler 的 `*_LOAD_MODE` 不一致 | 检查 `NONE_FULL_LOAD_MODE` 是否两侧都用 |
| 精度错（局部对，整体差） | `remoteRankCnt` 没从 0 起算 / rank==rankId 时未切换到本卡 GM | 核对 ProcessSingleBatch 中的 if (rank == rankId) 分支 |
| L0C 累加结果不对 | `splitKNum` 与实际 rank 遍历数不一致 | 检查 SetupParams 中 `splitKNum = rankSize_` 是否正确 |
| fixpipe 时崩溃 | C 地址偏移错（mOffset 算错） | 打印 `cGm_ + mOffset * axisN_ * sizeof(CType)` 与预期对比 |
| 性能不达标（cube ratio 低） | L1 深度过浅，K 轴反复加载 | 调大 `scaleKL1` 或换 `MatmulMultiBlockBasic<A_FULL_LOAD_MODE>` |
| blaze 头文件找不到 | `target_include_directories` 漏了 `_BLAZE_COMMON_DIR` | 参考 `CMakeLists.txt` 的 `target_include_directories` 段 |

---

## 8. 与 ascendc-blaze-best-practice 的关系

`ascendc-blaze-best-practice` skill 是 Blaze 单算子（无跨卡通信）的完整指南。复用其 Blaze 基底，但在以下方面不同：

| 维度 | ascendc-blaze-best-practice | MC2 场景 |
|------|-----------------------------|----------------|
| 通信 | 无 | SHMEM/UDMA 跨卡 |
| 数据来源 | 全部本卡 GM | rank==rankId 走本卡 GM，其他 rank 走 SHMEM buffer |
| Tiling | 单卡 L1/L0 容量 | + SHMEM 空间预算 |
| Scale | 可选 | 量化场景必需（参考工程是 MX FP8） |

新算子设计时，**先读 `ascendc-blaze-best-practice` 选 Blaze 模板，再将模板接入 MC2 通算流水**。

---

## 9. 后续阅读

| 想了解 | 读 |
|--------|---|
| SHMEM/UDMA 通信层 | `comm_shmem.md` |
| MC2 整体架构 | `mc2_architecture.md` |
| Blaze 单算子细节（模板选型、Tiling 算法） | `ascendc-blaze-best-practice` skill `references/scenarios/index.md` |
| 参考工程改造食谱 | `codebase_map.md` |
