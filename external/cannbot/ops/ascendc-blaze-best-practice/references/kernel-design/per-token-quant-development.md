# Per-token Quant 组件开发指导

本页是含 per-token quant 的完整 Scenario 共用的组件级事实源：读取 FP
workspace，为每条逻辑输出行计算唯一 scale，写量化输出和行 scale。它不执行
Scenario 路由，也不表示一个可独立选择的纯 Vector Scenario。

本页只描述 per-token quant 的完整逻辑行量化域、静态实现路线、阶段交接、
参考资产和验证矩阵；事实源、证据和 custom 授权边界由唯一命中的场景 DESIGN
冻结。

## 1. 组件契约

### 1.1 设计主线

- 输入是已由完整 Scenario 的上游阶段生产完成的 FP workspace。
- 输出按完整逻辑行生成唯一 per-token scale 和量化值。
- 本组件不拥有 Cube MatMul、GLU、elementwise、dequant 或 Scenario 路由。
- tile/chunk-local quant、量化域未确认或 scale 规则未确认均不满足组件合同。

完整需求必须先在 [Scenario 索引](../scenarios/index.md)唯一命中一个场景，再由该
Scenario 引用本组件；不得把本页与其他叶子自由拼装成未登记路线。

### 1.2 接口与数据流：冻结量化域

设输入 workspace 的逻辑 shape 为 `[R,Q]`。每行只产生一个 scale，归约域必须覆盖完整逻辑 `Q`：

```text
rowAbsMax[m] = reduce_max(abs(workspace[m, 0:Q]))
yScale[m]    = ScaleRule(rowAbsMax[m])
y[m, q]      = Quantize(workspace[m, q], yScale[m])
```

空 Tensor 与全零数值行是两个不同合同：

- 空 Tensor 的逻辑行数 `R=0`，不存在需要归约或量化的行，输出为
  `y[0,Q]` 和 `yScale[0]`。
- 全零数值行的 `R>0`，只是某一完整逻辑行的元素全部为零；该行仍按冻结的
  `ScaleRule`、`scaleMin` 和量化范围生成输出。

`ScaleRule`、全零数值行行为、clamp、舍入模式和饱和范围由算子契约决定，并
逐项写入 `DESIGN.md` 和 Python Golden。当前资产显式支持两种编译期 scale
规则：

```text
ClampBeforeDiv: yScale = max(rowAbsMax, scaleMin) / quantMax
ClampAfterDiv:  yScale = max(rowAbsMax / quantMax, scaleMin)
```

目标算子必须按 Golden 选择
`ScaleClampMode::BEFORE_DIV` 或 `ScaleClampMode::AFTER_DIV`，不能仅传入一个
数值后假设两种运算顺序等价。`scaleMin`、`quantMin` 和 `quantMax` 通过两个
quant Epilogue 的 `Params` 传入；默认范围为 `[-127,127]`，目标算子可按合同
显式选择 `[-128,127]`。Host 校验 `scaleMin` 和量化范围后，将同一组值传给
device。当前资产只接受 `[-127,127]` 和 `[-128,127]` 两组饱和范围，不能接受
任意上下界后仍使用 `quantMax` 作为 scale 分母。RNE 和 NaN/Inf 行为均按用户
合同实现。

接口同时携带：

- 逻辑宽度 `Q`：scale 的归约域
- 物理宽度或 `workspacePitch`：相邻行 GM 地址步长
- tile/chunk 宽度：一次搬入 UB 的片段

Per-token quant 本身即表示每条逻辑行只有一个全局 scale，不在组件名中重复加入
`FullRow`。这不表示整行一定驻留 UB。普通 MatMul 的单纯 quant 通常
`Q = N`；只有上游 GLU 契约明确输出宽度减半时才可使用 `Q = N / 2`。

### 1.3 成立条件/门禁

- `ScaleRule`、全零数值行行为、clamp、舍入、饱和范围和 NaN/Inf 策略必须
  由合同与 Golden 冻结。
- 若算子合同允许 `R=0`，DESIGN 必须声明处理路径；不得把合法空 Tensor 直接
  传给只证明支持正维度的 tiler 或 kernel。
- Host 必须校验并序列化同一组量化参数；device 不得使用另一套常量。
- scale 归约域是完整逻辑 `Q`；物理 pitch 和一次 UB chunk 均不能改变它。
- workspace producer、阶段同步、Tensor Slice、Host selector、Tiling 和
  Params 必须按目标版本重新绑定。

### 1.4 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 把 tile/chunk 宽度当作 scale 归约域 | 搬运粒度被误当成逻辑 Tensor 宽度 | 对完整 `[0,Q)` 生成唯一行 scale |
| 因普通 MatMul 的 `N` 为偶数就使用 `Q=N/2` | 宽度减半只来自明确的 GLU 语义 | 普通 quant 使用完整 `N`，GLU 输出才按合同取 `Q` |
| 在可复用资产中硬编码某个算子的 epsilon 或饱和下界 | 量化合同被实现默认值替代 | 编译期选择 scale 顺序，通过 Params 传入 clamp 值和饱和范围 |
| 将 clamp-before-div 与 clamp-after-div 当成同一公式 | 零行、subnormal 和舍入路径可能不同 | 按 Golden 显式选择 `ScaleClampMode` |

## 2. 两种静态实现

### 2.1 设计主线

#### `BlockEpiloguePertokenQuantSinglePass<ScaleMode>`

参考实现：
`assets/blaze_custom/epilogue/block_epilogue_pertoken_quant_single_pass.h`。

适用于一条完整逻辑输入行、量化输出 staging、归约临时空间和 scale 能同时满足 UB 预算的场景：

1. 每行只执行一次 GM→UB。
2. 在一个连续 Vector scope 内完成全行 absmax、scale 和 INT8 quant。
3. 写 `y[m, 0:Q]` 与 `yScale[m]`。

UB 门限由静态资源公式证明，并覆盖边界前、等于和后一档。

#### `BlockEpiloguePertokenQuantTwoPassChunked<ScaleMode>`

参考实现：
`assets/blaze_custom/epilogue/block_epilogue_pertoken_quant_two_pass_chunked.h`。

适用于完整逻辑行不能放入 UB 的任意宽度场景：

1. 第一遍按 chunk 读取整行，累计唯一 `rowAbsMax`。
2. 由完整行 absmax 计算唯一 scale。
3. 第二遍重新按 chunk 读取，使用同一 scale 完成 quant 和写回。

两遍都处理最后一个 chunk 的 tail；第一遍只累计整行 absmax。`N=400000`
等超宽场景选择该路径。

### 2.2 接口与数据流：Host 选择

Host 分别实例化 single-pass 和 two-pass kernel，并根据完整行 UB 容量选择
静态实例。交付证据包含实际命中的实例。

当前 GroupMatmul 两阶段组合参考实现为
`assets/blaze_custom/kernel/group_matmul_kernel_cv1_v2.h`。它不新增或改排
`GemmUniversal` 模板参数，而是把两个 Epilogue 类型绑定到既有
`BlockEpilogue_` 位置：

```cpp
using SinglePipeline = AscendC::Std::tuple<V1, SinglePassQuant>;
using ChunkedPipeline = AscendC::Std::tuple<V1, TwoPassChunkedQuant>;

using SingleKernel = GemmUniversal<ProblemShape, BlockMmad, SinglePipeline, BlockScheduler>;
using ChunkedKernel = GemmUniversal<ProblemShape, BlockMmad, ChunkedPipeline, BlockScheduler>;
```

这里的类型级 `tuple` 就是 V1/V2 的绑定，不创建运行时 tuple，也不把具体公式
写进新的算子专用 Kernel 类名。两阶段 specialization 复用已经选定并完成 final
drain 的 `GemmUniversal<ProblemShape, BlockMmad, V1, BlockScheduler>`，只新增全核
交接、完整行 AIV 重分配和 `V2(realM)`。因此 V1 可以是 GLU、unary elementwise、
dequant 或其他已经闭合的 C+V 后处理；本文件不拥有其公式和控制流。

`C+V1 -> SyncAll -> V2` 的阶段顺序参照 ops-transformer INT8 输入 GMM SwiGLU
per-token quant 实现 `gmm/common/cgmct/kernel/kernel_gmm_swiglu_pertoken_quant.h`；
参考的是阶段骨架，不复制其 SwiGLU、group traversal 或 `Q=N/2` 语义。公式、
problem traversal、workspace 宽度和 final drain 仍由被组合的 C+V1 Kernel
负责。该 C+V1 specialization 必须排除 `IsCv1V2EpiloguePipeline<BlockEpilogue>`，
把 tuple specialization 唯一留给本组合层，避免两个 partial specialization
同时匹配。

Host 侧复用
`assets/op_tiling/matmul/blaze_group_matmul_pertoken_quant_selector.h`。
其中 `MakeUbLayout()` 是 Host selector 和两个 device Epilogue 共同使用的唯一
UB 公式；selector 本身不写入序列化 TilingData。在 Bisheng 联合编译 `.asc`
时，device helper 使用 `__aicore__ inline`，Host/ASC Host helper 使用
`inline constexpr`；`__ASC_NPU_HOST__` 负责开放 `Selection/Select` 等
Host-only 类型。普通 Host C++ 不需要额外的项目宏，纯 device include 不进入
Host-only 分支；不得让 Host 分支条件移除 device helper 的 `__aicore__` 限定。

Host 调用 `Select(logicalQ, availableUbBytes, scaleMin)` 完成路径选择和
`scaleMin` 有限非负校验，再将同一个值写入目标工程 TilingData/Params。对于本
算子的冻结 Golden，Host/Launcher 必须显式传入该 Golden 要求的 `scaleMin`；
本 Skill 不规定具体数值，通用 Epilogue 和 selector 不得写死 `1e-12`、
`float32_tiny` 或其他 epsilon。DESIGN/PLAN 必须同时记录 `scaleMin`、
clamp-before/after-div、scale denominator、`quantMin` 和 `quantMax`，并由
Host 与 device 共同消费同一组参数。
“实例化 Kernel”指目标工程在 `op_kernel/<op>_kernel.h/.cpp` 中分别声明上述
两个静态类型，并各提供一个 `__mix__` entry；Host selector 返回静态 variant，
Launcher 启动对应 entry。本 Skill 提供可复用 Kernel/Epilogue 资产和 Launcher
开发合同，但不预置具体算子的 entry 名、Host 调用或 Launcher。

RegBase VF tail 的 `UpdateMask` 引用更新语义以
`/ascendc-regbase-best-practice` 为唯一事实源。本页只要求实现和审查时核对该规则，
不复制其 API 说明。

### 2.3 成立条件/门禁

- single-pass 的完整输入行、量化输出 staging、归约临时空间和 scale 必须同时
  满足 UB 预算。
- single-pass 的容量阈值必须由实际 UB/resource 公式和 selector 证明；任何具体
  `Q/N` 上限只属于目标 DESIGN/PLAN，不能从一个已通过的数字外推为硬件上限。
  没有宽度 selector 或 two-pass/chunked 路径时，Host 应 fail closed，而不是宣称
  任意宽度已支持。
- two-pass 的两遍都覆盖完整逻辑行并处理最后一个 chunk tail。
- 每条逻辑行的 `yScale` 是独立的 GM 标量输出。若 Tensor slice 或 Copy
  specialization 不能证明保留父 Tensor 的行偏移，必须使用已证明的 GM 基址加
  逻辑行偏移；不能把 `scaleRows` 的局部坐标当成物理地址证据。
- 对 `y` 和 `yScale`，对齐后的 UB pitch 只用于内部 staging；写回必须使用逻辑
  `[0,Q)` 的有效字节数和已证明的 GM row stride。不能让 aligned width 泄漏到
  下一行，也不能用直接 scalar GM store 代替已证明的 UB-to-GM completion path。
- 当前 single-pass 和 two-pass 资产在 256B 对齐的 UB 标量区写一个 `float` 时，
  使用 `DataCopyUnAlign<..., POST_MODE_UPDATE>(dst, data, unalign, 1)` 紧接
  `DataCopyUnAlignPost(dst, unalign, 0)`；这是本 Asset 登记的已批准正例，保持
  原序列。它只证明当前 UB 地址形态、元素数、post 参数和流水位置，不能外推到
  缺少 `DataCopyUnAlignPost`、改变地址形态或未经验证的其他序列。
- two-pass 必须在第一遍覆盖完整行并完成 scale 后，才用该 scale 执行第二遍。
  当前正例允许先发起 `yScale` 写回，再写各个 `y` chunk；最终完成条件是最后一次
  MTE3 写回 drain 以及 Kernel/stream 完成。任何并发消费者都不得把 `yScale`
  的可见性当作 `y` 已完成的同步 token。
- 循环中的反向 wait 只能等待已经由前一轮 producer 发出的事件；首轮没有
  producer 时必须预发事件或显式跳过该 wait，尾轮必须完成 drain。
- selector 与两个 Epilogue 共用唯一 UB layout 公式，selector 不修改序列化
  TilingData。
- Host 选择静态入口，不修改既有 Tiling ABI，也不在 device kernel 中隐藏
  未证明的资源分支。
- 两阶段类型必须占用既有 `BlockEpilogue_` 参数
  `AscendC::Std::tuple<V1,V2>`；不得增加 `GemmUniversal` 模板参数，也不得另造
  `GroupMatmulKernelDequantSwiGluPertokenQuant` 等算子专用 Kernel 类。

### 2.4 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 用一个已通过 shape 宣称 single-pass 任意宽度 | 没有证明 UB 容量上界 | 使用资源公式并覆盖门限前/等于/后 |
| two-pass 第一遍为每个 chunk 生成 scale | 局部 absmax 破坏完整逻辑行语义 | 第一遍只归约，整行结束后生成唯一 scale |
| 直接把 `scaleRows` 局部坐标作为 GM 标量地址 | Tensor slice 的逻辑坐标不一定保留 Copy specialization 的物理偏移 | 用父 GM 基址加逻辑行偏移，并用设备正例验证 |
| 省略 `DataCopyUnAlignPost`、改变当前 scalar store 的地址形态/post 参数，或把未验证序列当成正例 | `DataCopyUnAlign` 的 post 状态和流水合同不再与已批准资产一致 | 保留当前 `DataCopyUnAlign + DataCopyUnAlignPost` 完整序列；其他形态重新取得同设备证据 |
| 把已可见的 `yScale` 当作 `y` 完成标志 | two-pass 正例允许 scale 写回先于各 y chunk，两个输出的可见顺序不是阶段完成协议 | 等待最终 MTE3 drain 和 Kernel/stream 完成，或使用 DESIGN 冻结的独立完成协议 |
| 在首个 two-pass chunk 无 producer 时等待反向事件 | 首轮 event 尚未产生，设备可能 hang | 首轮预发 event 或按 `col > 0` 等已有 producer，尾轮 drain |
| 通过修改 TilingData 让 device 动态选择路线 | 静态资源入口和序列化 ABI 被混合 | Host selector 选择独立静态 entry |
| Host 分支宏改变 shared helper 的 device 声明 | Host-only selector 与 device helper 的编译上下文被混在一起 | 用 `__ASC_NPU_HOST__` 只开放 Host-only 类型；device helper 使用 `__aicore__ inline`，Host helper 使用 `inline constexpr` |
| 依赖联合编译中未验证的 Host 宏 | Host/device 变体可能看到不同代码 | 对普通 C++、`.asc` Host 和 Device entry 分别编译验证 |
| 为组合公式新建算子专用 Kernel 或增加 `GemmUniversal` 模板参数 | 类型职责和公共模板签名随算子名漂移 | 用既有 `BlockEpilogue_` 承载 `AscendC::Std::tuple<V1,V2>`，由目标 entry 静态实例化 |

## 3. 与上游阶段连接

### 3.1 设计主线

上游可以是：

- 普通 MatMul：若 Fixpipe 能直接写 GM workspace，优先直接写，不增加 identity TileEpilogue。
- dequant、GLU 或其他 Vector 后处理：由其 Epilogue 写 FP workspace。

### 3.2 接口与数据流

行分工属于 Kernel。当前参考骨架与上述 ops-transformer 路线一致，使用 AIV 的
`GetBlockIdx()` 作为逻辑 rank，以 `GetBlockNum() * GetTaskRation()` 作为消费者
总数，计算连续 `[rowStart, rowEnd)`，
再从带真实 row pitch 的父 Tensor 分别切出 `workspaceRows[rowCount,Q]`、
`yRows[rowCount,Q]` 和 `scaleRows[1,rowCount]`。两个 quant Epilogue 的调用形式为：

```cpp
quantEpilogue(workspaceRows, yRows, scaleRows, rowCount);
```

Epilogue 使用 slice-relative row。inactive AIV 遵守 Kernel 选择的消费者同步
协议，随后可传入 `rowCount=0`。

组合 entry 从已有 Problem/Tiling/workspace 合同一次性派生 `realM` 和
`workspaceRowPitch`，同时组装 C+V1 Params 与两阶段 Kernel Params；这两个字段是
entry 内的运行时派生 view，不增加序列化 TilingData 字段。V2 的唯一逻辑宽度来自
`epilogueV2Params.n`。通用组合层不得假定 GLU、双分支或 `Q=N/2`。

### 3.3 成立条件/门禁

量化阶段开始前必须满足：

1. 上游所有 workspace 写入已经完成。
2. 当前 C/V tile 协议已完成 final drain，没有仍可覆盖 workspace 的生产者。
3. 每个 quant consumer 都能证明其读取前已观察到所有相关 producer 完成。
4. 完成交接后按逻辑行重新分配 AIV；量化不能沿用只覆盖局部列 tile 的 ownership。

当前参考实现先让选定的 C+V1 Kernel 返回并完成最终 drain，再由所有 MIX 参与者执行
`AscendC::SyncAll<true, GROUP_MATMUL_CV1_V2_SYNC_ALL_CONFIG>()`；模板参数
`true` 将同步 effect domain 限定为 AIV，随后只有 AIV 进入 V2。文件级 config
的 trigger/wait 都显式为 `PIPE_ALL`；本文记号 `CV1 + V2` 中的 `+` 指这个全核
阶段交接。它不等于本核
`PipeBarrier<PIPE_ALL>()`：后者不能证明其他 AIV 的 workspace 已完成。V1 的
本地 MTE3 final drain 必须先闭合，所有同步参与者再到达 `SyncAll()`，之后才允许
inactive AIV 退出或由 active AIV 运行 V2。Launcher 还必须满足目标 CANN 版本
对硬同步 Kernel 的调度模式、逻辑 block 数和物理核数约束。

`DESIGN.md` 冻结每条逻辑行的 producer 集、consumer 集、workspace 完成条件、
交接机制和行重分配前的依赖。具体协议按
[Blaze 同步模式](../fundamentals/blaze-sync-patterns.md) 选择和验证，本页不
复制同步 API 规则。

若两个阶段复用 UB，资源预算按生命周期计算为 `max(upstreamUbBytes, quantUbBytes)`；只有生命周期重叠时才相加。任何 early return 都不能导致部分 block 跳过 Kernel 已选择的交接或消费者同步协议。

含本组件的 Scenario 必须在 DESIGN 中冻结 producer/consumer 拓扑；正确性等价的
候选可以包括单一 `__mix__` 入口、同一 stream 上的独立 AIC producer 与 AIV
consumer，或已证明的更细粒度完成协议：

```text
producer entry (AIC or MIX)
    -> workspace completion dependency
    -> consumer entry (AIV or MIX) owns complete logical rows
    -> Vector quant writes y + yScale
```

若使用 MIX，Kernel 必须按设备可见的真实 AIV 参与者集合和映射切分逻辑行；不能
仅凭 `GetBlockIdx()`、task ratio 或命名约定推断稠密 owner。若映射无法证明，
可以采用同一 ACL stream 的独立 producer/consumer entry，或冻结明确的完成信号
协议。最终方案必须先满足依赖正确性，再在正确性等价候选中用同设备性能数据
选择，不能把 `SyncAll`、MIX 或 split entry 预先写成永远最优。

当目标 MIX binding 已由源码/编译 witness 证明 AIC 与 AIV 使用相同的物理任务
排列时，AIV 用 `GetBlockIdx() / GetTaskRation()` 恢复与 AIC 相同的逻辑 tile
索引；这只完成 AIC tile 的归一化，不代表 Vector rank 唯一。需要唯一 Vector
rank 时，再结合当前 entry 已证明的 `GetSubBlockIdx()` 语义。`GetBlockNum()`
的总量含义不得从命名或另一类 entry 推断，必须由当前源码或最小 probe 证明并
记录在 DESIGN/PLAN 中。

### 3.4 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| quant 沿用上游局部列 tile ownership | 单个 consumer 看不到完整行 | 交接完成后按逻辑行重新分配 AIV |
| Epilogue 再次读取物理 block index 或叠加 `rowStart` | Kernel 和 Epilogue 重复分工 | Kernel 切片，Epilogue 使用 slice-relative row |
| inactive AIV early return 跳过交接 | 参与者集合与同步合同不一致 | 先完成协议，再以 `rowCount=0` 调用或退出 |
| 用 `PipeBarrier<PIPE_ALL>()` 表示 `CV1 + V2` | 该 barrier 只排空本核流水，不能建立所有 V1 producer 到 V2 consumer 的跨核可见性 | V1 local drain 后由已证明的 AIV 参与者执行显式 `{PIPE_ALL,PIPE_ALL}` config 的 `SyncAll`，再重分配完整行 |
| 直接相加两个不重叠阶段的 UB 预算 | 忽略了同步隔离的生命周期 | 不重叠时取最大值，重叠时才相加 |
| 未冻结 MIX binding 就照搬 `GetBlockIdx()`/task ratio 作为行 owner | 其他 binding 的物理 index 可能稀疏或包含非量化参与者，scale owner 会冲突 | 当前参考只在已对齐 ops-transformer INT8 GMM 路线、且 Launcher 合同证明 raw AIV index 连续时使用该映射；其他路线重新冻结参与者和映射 |
| 将同 stream 的 producer/consumer split 视为天然正确或天然错误 | stream 顺序只解决阶段先后，不自动证明行 ownership 或跨 kernel 可见性 | 先证明 workspace 完成和完整行 owner，再用设备性能选择 |

## 4. 接口和命名

### 4.1 设计主线与接口

两个 quant Epilogue 的 `Params` 只保存逻辑 `Q`、chunk 和量化契约字段。
FP workspace、`y` 与 `yScale` 由 Kernel 以已经分配好的 Tensor slice 传给
`operator()`；父 Tensor 的 layout 保存物理 stride。MatMul accumulator、
dequant scale、GLU 分支和全局多核拓扑由其所属组件持有。

作为当前 `C+V1 + V2` Kernel 的 V2 时，Epilogue 还公开
`InputType`、`OutputType`、`AuxOutputType`，并保持
`operator()(workspaceRows, outputRows, auxiliaryRows, rowCount)` 接口；当前两个
quant 资产的辅助输出即逐行 `yScale`。V1 必须已有
`operator()(Cv1Kernel::Params, __gm__ V2::InputType* workspace)` 的 C+V Kernel
接口并在返回前完成 local final drain；组合 entry 从与 V1 相同的
Problem/Tiling/workspace 事实源派生 `realM` 和 `workspaceRowPitch`，不得新增第二份
序列化 ABI 事实源。

推荐组合边界：

```text
V1 = BlockEpilogueDequantSwiGlu
     writes FP32 workspace [M, Q]

V2 = BlockEpiloguePertokenQuantSinglePass<ScaleMode>
     or BlockEpiloguePertokenQuantTwoPassChunked<ScaleMode>
     reads the same workspace and writes y + yScale

Kernel BlockEpilogue = AscendC::Std::tuple<V1, V2>
```

### 4.2 成立条件/门禁

- quant Epilogue Params 只保存逻辑 `Q`、chunk 和量化合同字段。
- workspace、`y` 与 `yScale` 由 Kernel 以已分配 Tensor slice 传入。
- 父 Tensor layout 保存真实物理 stride。
- V2 的三个 element type alias、`Params::n` 和四参数 slice 调用必须与 Kernel
  一致；`realM` 和 `workspaceRowPitch` 必须由 entry 从已有合同派生并写入组合
  Params，不能形成第二份序列化 ABI。

### 4.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| quant Epilogue 接收 accumulator、GLU 分支或全局多核拓扑 | 单一组件承担了上游和 Kernel 职责 | 保持 slice-only 调用边界 |
| 单纯 quant 派生出带 GLU/dequant 名称的类 | 名称错误暗示公式和依赖 | 直接复用 per-token quant Epilogue |
| 通用 C+V1+V2 Kernel 内实现 GLU group traversal、双分支 view 或 `Q=N/2` | 组合层被某个公式和路线锁死 | 让既有 C+V1 Kernel 拥有公式与 traversal；组合层只负责交接、完整行重分配和 V2 |
| C+V1 specialization 未排除 Epilogue tuple | 它与通用 tuple specialization 可能同时匹配 | 用 `IsCv1V2EpiloguePipeline` 将 tuple 唯一保留给组合层 |

## 5. 验证和性能门禁

### 5.1 场景验证矩阵

精度矩阵至少覆盖：

- single-pass 容量边界的前一档、等于门限和后一档
- two-pass 大 `Q`、跨多个 chunk 和尾列
- base、M tail 和空 expert
- 合同允许时的空 Tensor `R=0`
- `R>0` 的全零数值行、tiny、大正负值，以及契约要求的 NaN/Inf
- scale、RNE、clamp 和 INT8 saturation 的边界值
- absmax 刻意放在第二个及以后 VF slice，防止 tail 进度错误跳过后续 slice
- 相同输入重复运行至少 10 次
- Host selector 证据，确认每个 case 启动预期静态 kernel

定向用例必须证明其目标性质在本次输入中实际成立，不能只依赖用例名称或生成
意图。Generator 或 manifest 应记录可复算的目标谓词、命中数量和关键位置，例如
absmax 所在 VF slice、`normalized` 到 half-integer 的距离、发生 saturation 的
元素数，或 selector 容量边界两侧的实际 `Q`。测试入口必须断言这些条件；零命中、
命中错误区域或启动错误静态 entry 都应直接失败。仅有最终输出比较 PASS 不能证明
该定向路径已覆盖。

### 5.1.1 完整行归约 API 与结果物化

`rowAbsMax` 的归约算子、归约结果的寄存器/lane 语义、UB/GM 物化路径和最终
`yScale` 读取必须作为一条证据链闭合。不能因为 API 名称包含 `ReduceMax`，或
因为某个寄存器值可被写回，就推断它已经代表完整逻辑行；归约覆盖范围和结果
物化必须由当前 SDK 源码、同版本最小 device probe 或已证明的同设备正例确认。

DESIGN/PLAN 至少记录归约域、active mask/tail、结果所在 lane/register、写回
API 的元素数和 post 状态，以及消费者读取前的完成依赖。定向用例必须把最大值
放在首个 VF slice 之外，并在 manifest 中记录实际列和命中数量；验证顺序为
`workspace -> yScale -> y`。归约或物化事实为 `unknown` 时保持 `blocking`，
不得修改 Golden、阈值或把首 lane 当作整行结果。若项目在已冻结的有限支持域内
采用标量归约，只能作为项目本地恢复方案并记录其支持边界，不能登记为通用资产
默认实现。

宽范围随机输入负责端到端验证；精确二进制小数输入只用于隔离 quant 路径，
不能用后者的 PASS 替代前者。RNE 在 `k + 0.5` 处不连续，上游 workspace 或
device 除法的少量 ULP 差异都可能让最终 INT8 相差 1；相同的 `yScale` 也不能
证明发生 mismatch 的 workspace 元素相同，因为该行 absmax 可能来自其他列。

发生 RNE 边界 mismatch 时，必须保留失败输入并执行以下分阶段诊断：

1. 导出并比较 device FP workspace、device `yScale` 和 Python Golden 中间值。
2. 使用 device workspace 与 device `yScale` 在 Python 中按合同重新量化。
3. 若重算结果与 device `y` 一致，继续定位 MatMul/workspace 生产路径；若与
   Golden `y` 一致但与 device `y` 不同，定位 device division/RNE/cast 路径；
   若 `yScale` 已不同，先定位 absmax/scale 路径。
4. 在 workspace dump 前，MMAD 累加顺序、除法或转换差异都只能标为假设。
5. 位级输出合同未满足时保持未闭环；不得缩窄输入分布或只保留精确输入来宣称
   精度 PASS。

若验收合同允许按 half-tie 分类有界 mismatch，Golden generator 必须从生成
Golden `y` 的同一个内存 `normalized = activated / scale` tensor 同次落盘
normalized 资产，Golden `y` 也直接由该 tensor 执行冻结的 round/clamp/cast
得到。Verifier 必须读取该同源资产判断 half-tie，不能从另一次加载或分别舍入的
workspace/scale 重建它。使用 device workspace 与 device scale 的重量化是独立
诊断层，不能替代同源 Golden normalized。具体 mismatch 比例、half 距离和 scale
阈值仍属于目标算子验收合同，不写入可复用资产默认值。

本参考实现采用 finite-only 输入前置条件。NaN/Inf 负向用例由输入可见的测试入口
在 launch 前拒绝；生产 Kernel 不增加全量非有限值扫描。

性能比较必须同轮、同 shape、同输入、同 Tiling 和同精度门限。报告 single-pass 与 two-pass 的实际 GM→UB pass 数；没有 matched baseline 时不得宣称领先。优化 single-pass 不能删除 two-pass，因为后者承担任意宽度的能力闭包。

### 5.2 成立条件/门禁

- Golden 使用冻结的 scale、舍入、clamp 和 saturation 顺序。
- 每个 case 记录实际命中的静态 entry，并覆盖对应容量边界。
- 分别记录空 Tensor 和全零数值行的实际处理路径；不能用其中一个代替另一个。
- 每个定向 case 断言目标谓词、命中数量和关键位置；目标未命中时不得计为覆盖或
  精度 PASS。
- RNE mismatch 未完成 workspace、scale 和 quant 分阶段归因时保持未闭环。
- finite-only 前置条件必须由输入可见的入口执行。

### 5.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 只用精确二进制小数输入替代随机端到端验证 | 规避了上游 ULP 对 RNE 边界的影响 | 同时保留宽范围随机与隔离输入 |
| 定向用例只比较最终输出，不断言目标性质已命中 | 输入可能退化为普通 case，PASS 不能证明目标路径 | 在 Generator/manifest 中记录可复算谓词、命中数量和关键位置，并由测试入口强制断言 |
| `yScale` 相同就判定 mismatch 元素的 workspace 相同 | absmax 可能来自该行其他列 | 导出并比较对应 workspace 中间值 |
| 通过缩窄输入分布消除 mismatch | 位级合同没有被修复 | 保留失败输入并按分阶段诊断归因 |
| 优化 single-pass 后删除 two-pass | 任意宽度能力被移除 | 保留 two-pass 能力闭包并分别报告性能 |
