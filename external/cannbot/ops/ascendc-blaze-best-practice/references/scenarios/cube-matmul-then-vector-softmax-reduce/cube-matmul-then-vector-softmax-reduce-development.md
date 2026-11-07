# Cube MatMul → Vector Softmax/Reduce 开发指导

本文在同目录 [设计指导](cube-matmul-then-vector-softmax-reduce-design.md) 已冻结、`implementation_route=blaze_custom` 且 `selected_scenario=cube-matmul-then-vector-softmax-reduce` 后用于编译项目 PLAN。Step 4 不重新决定归约公式、归约域、workspace 布局或同步协议。

## 1. PLAN 输入和阅读

PLAN 必须绑定 workspace 布局、producer/consumer 集、归约公式（softmax 或 reduce）、splitM 合同和基础 MatMul ABI。必读：

- [Online Softmax/Reduce Epilogue 设计专题](online-softmax-reduce-epilogue-design.md)
- [CV Sync 与两阶段 Entry 编排专题](cv-sync-and-two-phase-entry.md)
- [Reduce 适配指导](reduce-adaptation-guide.md)（reduce 变体必读）
- [SplitM 专题](../elementwise-broadcast-epilogue-fusion/splitm-contract-and-debugging.md)（DESIGN 激活 splitM 时）
- Investigation 指定的具备 L0C2UB 能力的 BlockMmad、GemmUniversal、同步、Tiling 来源
- [Tiling 方法](../../kernel-design/tiling-selection.md)、[Launcher 方法](../../launcher/launcher-development.md)
- [同步方法](../../fundamentals/blaze-sync-patterns.md) 和当前 CANN 实际头文件

## 2. 有序动作

### 2.1 核对 Base MatMul

核对 DESIGN/Investigation 与当前 Blaze 源码版本的抽象一致性：

- BlockMmad 的 L0C2UB 输出路径使用 `CopyL0C2UB` + splitM trait（如 `block_mmad_matmul_fixpipe_opti.h`、`block_mmad_qbmm_mx.h` 等）
- 对应 kernel 的 `operator()` 调用路径（如 `kernel_matmul_fixpipe_opti.h`）：`epilogueOp.Init` → `blockMmad.Init` → `MatmulProcess`
- `block_scheduler_matmul_basic.h` 的 `GetBlockShape`/`GetBlockCoord` 接口
- 确认 kernel 结构特征与 [cv-sync §0](cv-sync-and-two-phase-entry.md) 匹配（5 个维度逐一核对，不匹配时按适配方向调整）

### 2.2 选择执行拓扑

- `__mix__(1, 2)`：1 AIC + 2 AIV per block
- `splitM=1`：PerTileEpilogue 按 sub-block 分配行
- `ubDB=1`：首版不启用 UB ping-pong（slot 恒为 0）
- 先闭合 CV sync 生命周期（构造/析构 + tile loop），再做性能选择

### 2.3 复制并适配资产

1. 复制 `online_softmax_per_tile_epilogue.h` 到项目 `blaze_custom/epilogue/`（reduce 变体复制后按 [Reduce 适配指导](reduce-adaptation-guide.md) 改写）
2. 复制 `online_softmax_cross_core_epilogue.h` 到项目 `blaze_custom/epilogue/`（reduce 变体同理）
3. 按 [cv1_v2 适配指导](cv1-v2-adaptation-for-softmax-reduce.md) 从 `group_matmul_kernel_cv1_v2.h` 适配生成项目 `blaze_custom/kernel/matmul_softmax_kernel.h`（在 `operator()` 开头添加 `InitWorkspaceGlobal` + `SyncAll`）。适配后不得在项目内保留原始 `group_matmul_kernel_cv1_v2.h`：两个特化模板参数列表相同，同处一个编译单元会触发歧义错误（见适配指导 §7）
4. 复制 `blaze_custom/utils/common_utils.h` 和 `integral_constant.h`（如项目无已有副本）
5. 确认 CV sync 常量与 Blaze 库 BlockMmad 一致（以 Blaze 库源码为准）
6. 根据 DESIGN 冻结的 dtype 实例化模板参数（如 `bf16` 输入 + `float` 输出）

### 2.4 编写 Kernel Entry

1. 在 `op_kernel/matmul_softmax_kernel.h` 中定义类型别名：
   - `DispatchPolicy = MatmulMultiBlockFixpipeOpti<ND_ALIG_1V2_FIXPIPE, 0>`  // 示例为普通 matmul；MX 量化等变种使用对应 DispatchPolicy（如 MatmulWithScaleMx）
   - `BlockScheduler = BlockSchedulerMatmulBasic<ProblemShape>`
   - `BlockMmad = BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>`
   - `PerTileEpilogue = OnlineSoftmaxPerTileEpilogue<OutT, OutT, false>`
   - `CrossCoreEpilogue = OnlineSoftmaxCrossCoreEpilogue<OutT>`
   - `EpiloguePipeline = AscendC::Std::tuple<PerTileEpilogue, CrossCoreEpilogue>`
   - `KernelImpl = GemmUniversal<ProblemShape, BlockMmad, EpiloguePipeline, BlockScheduler>`
2. 计算 workspace 布局（softmax 变体：onlineMax/onlineSum/mHistory/expWorkspace 的 offset；reduce 变体：partialResult 的 offset，详见设计指导 Section 4.2）
3. 构造 `KernelImpl::Params`（嵌套 `cv1Params` + `epilogueV2Params`）
4. 调用 `KernelImpl kernel; kernel(kParams);`（两阶段编排由适配后的特化内部完成）
5. V1 epilogue 析构需 `initialized_` guard（idle 核未调 `Init` → 跳过 `CleanUpSyncFlag`）

### 2.5 编写 Tiling 和 Launcher

1. 复用 `blaze_matmul_tiling.h` 的 `MatmulTilingSwat` 引擎
2. 扩展 tiling data POD：追加 `cubeCoreNum`、`vecCoreNum`、`ubDB` 字段
3. Host 侧 workspace 大小计算使用设计指导 Section 4 的布局公式
4. Launcher 按 `usedCoreNum` 启动 `__mix__` grid

### 2.6 生成 Golden

Golden 分两步：先按所选 MatMul 变种计算 logits，再对 logits 的最后一维执行归约。MatMul 部分的输入构造和计算由具体变种决定（普通 matmul / MX 量化 / grouped matmul 等），归约部分按变体选择：

- softmax 变体：`golden = torch.softmax(logits.to(float), dim=-1)` → fp32
- reduce 变体：`golden = torch.reduce(logits.to(float), dim=-1)` → fp32（reduce 为可按行合并的归约操作，如 max、sum、min 等）
- logits 为 MatMul 的 FP 输出，归约沿 N 维（完整逻辑行）
- 数据存为二进制文件，host 侧读取

## 3. 验证和交付

### 3.1 精度验证

| 维度        | 覆盖                                                 |
| ----------- | ---------------------------------------------------- |
| N-tile      | 单 N-tile（N=baseN）、多 N-tile（N=baseN*2, *4, *8） |
| 多核        | 单核（totalTiles=1）、多核（totalTiles=cubeCoreNum） |
| M 奇偶      | M 为偶数、M 为奇数                                   |
| 非对齐      | K/N 非 16 对齐、M 非 baseM 对齐                      |
| tail        | N-tail（N%baseN≠0）、M-tail（M%baseM≠0）           |
| 重复 launch | 相同输入连续运行 3 次                                |
| 大 case     | M≥1024, K≥4096, N≥6144                            |

精度门禁：`max_abs_error < 2e-3` 或 `max_rel_error < 0.01`。

每个 case 记录实际 tiling（baseM/baseN/baseK/cubeCoreNum）和实际误差。

### 3.2 交付

- 编译通过（`cmake --build`）
- 精度全部 PASS
- 清理诊断后 clean build 并重跑 Full
- 交付分别声明正确性和性能状态
- 项目内不含原始 `group_matmul_kernel_cv1_v2.h`（已适配重命名为 `matmul_softmax_kernel.h`，见 §2.3 第 3 条）
