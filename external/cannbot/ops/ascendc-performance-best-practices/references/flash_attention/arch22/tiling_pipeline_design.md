# arch22 FlashAttention Tiling、buffer 与同步设计

> 适用架构：arch22（DAV_2201 / Atlas A2/A3）。本文以 FIA arch22 实现为事实源，不用 A5 FA 的 tile、事件或共享 buffer 拓扑补位。

## 1. 优化目标

在 QK、online softmax、PV 和 rescale-O 的整条链上同时保证：AIC 填核、L1/L0/UB 预算闭合、workspace 不放大到反噬前后算子、事件同步不串行化流水。

## 2. 架构概览

```text
Host Tiling: M/S2 tile、核数、Split-KV、buffer 与 workspace 联合求解
  -> AIC: GM -> L1 -> L0 -> QK/PV
  -> AIV: online softmax -> rescale-O -> 输出
  -> 可选 FlashDecode: 局部结果 -> workspace -> 跨核归并
```

| 环节 | 主要耦合 | 需要核对的现象 |
|---|---|---|
| Host Tiling | tile、核数、中间区与 workspace | 大 tile 减少循环但增加空间 |
| 多核 / Split-KV | 任务均衡与局部结果归并 | 尾核空转或归并超过并行收益 |
| L1/L0/UB | 多级 buffer 深度与地址区间 | 容量超限或复用覆盖 |
| 搬运与同步 | DMA 尾块、队列和 HardEvent | 搬运越界或流水被等待串行化 |

## 3. 关键参数 / 修改点

| 参数或手段 | 调整方向 | 必须同时复核 |
|---|---|---|
| `mBaseSize` / `sInnerSize` | 联合搜索 M 与 S2 tile | QK/PV 中间区、workspace 和 feature 约束 |
| `used AIC/AIV` / block dim | 沿 B、N2、S1G/M 分配任务 | 逻辑任务、尾核和 kernel mode |
| Split-KV | 用更多核处理同一 query | 局部写回、跨核同步和最终归并 |
| L1/L0/UB buffer 深度 | 增加可重叠阶段 | 地址、容量、事件 ID 与生命周期 |
| `DataCopy` / `DataCopyPad` | 合并或修正搬运 | 单位、page 边界、尾块和 padding |

### Host Tiling 联合求解

arch22 FIA 不是只选一个 `sInnerSize`。至少联合求解：

```text
mBaseSize
sInnerSize / sInnerSizeAlign
sInnerLoopTimes / tail
used AIC/AIV and block dim
QK/PV intermediate sizes
normal workspace
FlashDecode workspace (when splitKV)
```

`sInnerSize` 增大可以减少循环，但会同时放大 MM1/Vec1 中间结果和 workspace；GQA/MLA、paged cache、mask 压缩、TND/NTD layout 还有独立约束。选型必须把 workspace 公式与 tile 一起报告，不能只说“tile 越大越快”。

### 多核与负载均衡

优先沿 B、N2、S1G/M 维建立独立任务。当 `B*N2` 不能整除核数时，arch22 MLA tiling 会用最小公倍数推导可均分的 `mBaseSize`，同时设置最小 M tile 门限；若无法整分，才回到 S2/Split-KV 路径。

报告至少包含：

- 逻辑任务数与目标 AIC 数；
- 每核最大/最小 QK、PV tile 数；
- 尾核是否仍执行初始化或同步；
- `CalcTschBlockDim` 的 AIC/AIV 口径是否与 kernel mode 一致。

### Split-KV / FlashDecode

Split-KV 把同一 query 的 KV 序列分段到多核，局部输出写 workspace，再归并 `accumOut`、`lseMax`、`lseSum`。开启前比较：

```text
saved_critical_path_from_more_cores
    > local_partial_write + cross_core_sync + final_combine
```

workspace 由核数、`mBaseSize`、对齐 head dim 和局部 QK/PV 结果共同决定。启用/扩大 Split-KV 时，必须复算正常中间结果与归并区，保留 workspace 上限门禁。

### L1/L0/UB buffer

arch22 MLA 源码展示了目标专用的多级规划：

- Q/P 侧 L1 4-buffer；
- K/V 侧 L1 3-buffer；
- L0A/L0B double buffer；
- L0C 独立输出区；
- Vector 侧 TQue ping-pong 与 softmax 临时 TBuf。

这些 buffer 数量与 arch22 事件 ID 绑定。调整任一深度时要同步重算：容量、偏移、事件 ID、首轮预置、尾轮回收和所有 feature 分支。

### 数据搬运

arch22 使用 `DataCopy`/`DataCopyPad` 处理连续块、跨 stride 和非对齐尾。方案中明确：

- GM 与 UB/L1 stride 的单位；
- blockLen 是字节还是 datablock；
- `blockCount` 上限；
- page block 边界和 S2 尾块是否会跨块；
- `DataCopyPad` 的 padding 是否参与后续 softmax mask。

不得用 arch35 NDDMA 能力假设替换 arch22 的搬运实现。

### 同步

标准队列数据流优先用 TQue 自动同步；手排 L1/L0 ping-pong 使用精确 `SetFlag/WaitFlag<HardEvent>`。`SyncAll`/CrossCore flag 只用于真实跨核归并或 superkernel 契约，不能拿 `PIPE_ALL` 替代局部依赖。

事件审核表必须覆盖：事件类型、ID、生产者、消费者、buffer、首次初始化、循环中复用和退出前闭合。

## 4. 核心计算循环 / 关键流程

```text
for each assigned query tile:
    prepare Q and the current K/V tile in separate buffer lifetimes
    for each S2 tile:
        wait only for the buffer actually consumed by QK/PV
        QK -> online softmax -> PV -> rescale-O
        prefetch the next K/V tile when capacity and events permit
        release the consumed buffer before its next reuse
    write final output, or write partial output for FlashDecode
if Split-KV is enabled:
    combine partial output and LSE after cross-core completion
```

这是依赖顺序示意，不是可直接替换的 arch22 kernel。每条等待和释放都必须映射到目标实现中的队列或 HardEvent；Split-KV 归并还须单独核对同步与 workspace。

## 5. 效果与适用边界

| 候选修改 | 预期观察 | 不应采用的条件 |
|---|---|---|
| tile / 核数重选 | 最长核缩短、AIC 填充改善 | 中间区或 workspace 超预算 |
| Split-KV | 欠并行 case 的关键路径缩短 | 局部写回与归并成本覆盖收益 |
| 多级 buffer / 搬运 | MTE、Cube、Vector 重叠改善 | 地址、事件或尾块不能闭合 |

本页不提供 A2/A3 固定收益数字；目标型号和 case 的精度、timeline、workspace 与逐 case Task Duration 一起决定是否保留修改。

### 验证

- 主/control shape：普通 GQA、MLA、paged cache、mask、TND/NTD、Split-KV 开关各有覆盖。
- 精度：online softmax 和归并顺序变化单独验收。
- 性能：逐 case 比较 AIC 填充率、最长核、MTE2/MTE1/Cube/Vector/Scalar 和 workspace 增量。
- A2 与 A3 结果分开记录；没有 A3 设备结果时标注 `A3_not_run`。

### 静态检视判据

<a id="prc-fa-001"></a>
#### S2/M 切分与 workspace 联合求解（PRC-FA-001）

只增大 `sInnerSize` 以减少循环、却沿用旧 workspace，是典型错误。沿 `sInnerSize/mBaseSize -> MM1/Vec1/MM2/Vec2 中间区 -> normal workspace/FlashDecode workspace` 追值，代入目标 case 字节并核对上限门禁；同时检查 GQA/MLA、paged cache、mask 和 TND/NTD 分支的 tile 约束，以及核数或 Split-KV 开关变化后的重算。公共 tiling API 已完整推导且调用链可证明时不报。

finding 必须列出各 workspace 分量的公式与数值、上限来源和受影响分支，不能只凭“大 tile 占空间”给无公式建议。

<a id="prc-fa-002"></a>
#### 多级 buffer 与事件 ID 共同闭合（PRC-FA-002）

Q/P 多缓冲、KV 多缓冲、L0 双缓冲和 Vector TQue 的深度、容量、事件 ID 必须一起改。检视时绘制 L1/L0/UB 地址区间，列出事件类型、ID、生产/消费流水、首轮预置、复用周期及尾轮回收；dummy/superkernel 分支也须覆盖必要同步。增加第三个 buffer 却继续复用前两个 buffer 的同类型事件 ID，是需证明生命周期冲突的候选。

不同 HardEvent 类型可合法复用同一 ID，代码级互斥也可能排除冲突。finding 必须给出地址表、容量总和、事件拓扑、冲突生命周期和退出路径；只 grep 到相同 `EVENT_ID` 不足以报错。机制证据见 book-910：0067。

### 实现证据

实现依据来自 FIA arch22 的入口分派、nonquant/MLA host tiling，以及 Cube/Vector 的 buffer 与事件编排。使用时在目标工程核对实际分派、模板参数和 feature 分支；源码存在不等于目标 case 已选中或运行时有效。
