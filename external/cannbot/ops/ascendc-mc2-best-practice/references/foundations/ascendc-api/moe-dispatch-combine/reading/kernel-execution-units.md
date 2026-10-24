# 核内执行单元阅读参考

这份文档不是某一版 dispatch/combine 的实现说明，而是给阅读类 agent 用的“核内执行单元识别规则”。目标是回答：看到 AscendC 代码时，怎样基于官方架构定义，再结合代码证据判断 `Scalar`、`MTE`、`AIV`、`AIC` 分别在做什么。

## 官方依据

这部分优先依据昇腾官方文档，而不是经验推测：

- 抽象硬件架构：
	- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0015.html
- 硬件实现与基本架构：
	- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0008.html

官方文档给出的稳定结论有：

- `Scalar`：执行地址计算、循环控制等标量计算工作，并把向量计算、矩阵计算、数据搬运、同步指令发射给对应单元执行。
- `Vector`：负责执行向量运算。
- `Cube`：负责执行矩阵运算。
- `DMA / MTE`：负责数据搬运，包括 Global Memory 和 Local Memory 之间、以及不同层级 Local Memory 之间的数据搬运。
- 在具体硬件实现里：
	- `MTE2` 负责 `GM -> {L1, L0A/B}` 和 `GM -> UB` 这类搬运。
	- `MTE3` 负责 `UB -> GM` 这类搬运。
- 典型向量数据流是：`GM -> UB -> Vector -> UB -> GM`。
- 典型指令流是：`Scalar` 读取指令并把 Vector / Cube / MTE 指令发射到各自队列，不同指令序列可并行执行；跨序列依赖通过同步机制控制。

因此，阅读 agent 在识别执行单元职责时，应先服从这些官方定义，再用代码去定位“当前这段逻辑落在哪个官方定义的单元上”。

## 代码映射依据

在官方定义已经明确之后，再用下面 3 类代码证据把职责落到具体代码段：

1. 原语名字本身

- `DataCopy` / `DataCopyPad` / `DataCopyExtParams`：按照官方 DMA / MTE 定义，优先判为搬运路径。
- `Cast` / `Add` / `Mul` / `Muls` / `Sum` / `ReduceSum` / `CompareScalar` / `Select` / `Exp`：按照官方 Vector 定义，优先判为向量计算路径。
- `SyncFunc<MTE2_V>`（sample 自定义封装，等价于官方 `SetFlag<HardEvent::MTE2_V>(FetchEventID<MTE2_V>()) + WaitFlag<MTE2_V>(id)`）、`SyncFunc<V_S>`、`SyncFunc<S_MTE3>`：事件名直接暴露了"哪个执行单元的结果要交给哪个执行单元"。
- `PipeBarrier<PIPE_V>`、`PipeBarrier<PIPE_MTE3>`、`PipeBarrier<PIPE_ALL>`：说明当前代码正在显式处理同核不同 pipe 的可见性与串接。

2. 数据流方向

- 如果某段代码把数据从 GM / window / 状态区搬到 `LocalTensor` / UB，再靠事件同步交给后续计算，这段首先归到官方定义中的 DMA / `MTE` 搬运路径。
- 如果某段代码在 `LocalTensor` 上做 cast、逐元素算子、mask、reduce，再把结果交给 `Scalar` 读取或交给 `MTE3` 写回，这段首先归到官方定义中的 `Vector` 路径。
- 如果某段代码主要在做分支判断、区间计算、状态轮询、地址偏移、ready 判定、循环推进，而不是做大块 tensor 算子，这段首先归到官方定义中的 `Scalar` 路径。

3. 任务类型和分支

- 如果入口显式写了 `ASCEND_IS_AIV`、`KERNEL_TYPE_AIV_ONLY`，说明主热路径至少是 AIV 可见的；这时不要默认 `AIC` 一定参与。
- 只有当代码里真的出现 cube / matmul / 专门走 `AIC` 的算子或任务类型证据时，才把某段路径归到 `AIC`。
- 如果图里画了 `AIC`，必须能在代码里指出它参与的真实算子或真实路径；否则应明确写“当前主热路径未见 AIC 参与”。

## 各执行单元的官方职责与阅读落点

### Scalar

官方定义：

- 执行地址计算、循环控制等标量计算工作。
- 把向量计算、矩阵计算、数据搬运、同步指令发射给对应单元执行。

阅读时常落在：

- 区间切分
- 地址和偏移计算
- ready / not ready 判断
- while / for 轮询推进
- 条件路径选择
- 读取少量标量结果并决定下一步控制流

常见代码信号：

- `if` / `while` / `for`
- `GetValue()` / `SetValue()` 读取或写少量标量
- 对 `tokenIndex`、`slotIndex`、`begin/end`、`count` 的控制逻辑

### MTE2 / MTE3

官方定义：

- `MTE` 属于搬运单元。
- 在具体硬件实现里，`MTE2` 负责 `GM -> UB` 等搬入路径，`MTE3` 负责 `UB -> GM` 等搬出路径。

阅读时常落在：

- GM / window / 状态区 与 UB 之间的数据搬运
- 把向量结果写回 GM / window / 状态区

常见代码信号：

- `DataCopy` / `DataCopyPad`
- `DataCopyExtParams` / `DataCopyPadExtParams`
- `SetFlag<HardEvent::MTE2_V>` / `WaitFlag<HardEvent::MTE2_V>`（或 sample 中的封装 `SyncFunc<MTE2_V>`）、`SetFlag<S_MTE2>` 等：跨 pipe 事件同步

阅读时的映射规则：

- `GM -> UB` 优先记到 `MTE2`
- `UB -> GM` 优先记到 `MTE3`
- 如果代码只写了 `DataCopyPad`，但上下文是“从 GM 读入 LocalTensor”，就按 `MTE2` 画；如果上下文是“把 LocalTensor 写回 GM”，就按 `MTE3` 画。

### AIV / Vector

官方定义：

- `Vector` 负责执行向量运算。
- 在具体硬件实现里，Vector 的源数据和目标数据都要求放在 `Unified Buffer` 中。
- 典型向量数据流是：`GM -> UB -> Vector -> UB -> GM`。

阅读时常落在：

- cast
- add / mul / reduce
- mask 处理
- 向量比较
- dequant / quant 中的向量部分
- token 载荷整理和张量级中间结果生成

常见代码信号：

- `Cast`
- `Add` / `Mul` / `Muls`
- `Sum` / `ReduceSum`
- `CompareScalar` / `Select`
- `Exp` / `Mins` / `GatherMask`
- `PipeBarrier<PIPE_V>`

阅读时的映射规则：

- 只要是在 `LocalTensor` 上做批量元素级或向量级运算，先归到 `AIV`。
- 如果一段逻辑包含“先搬入，再在 UB 上一连串算，再把结果写回”，中间那一串通常就是 `AIV` 主体。

### AIC

官方定义：

- `Cube` 负责执行矩阵运算。
- 在 AI Core 分离模式下，`AIC` 是一组 `Cube Core + Vector Core` 组合中的 `Cube Core`。

阅读时常落在：

- cube / matmul / 矩阵类融合计算

常见代码信号：

- 明确的 cube / matmul 类算子
- 明确的 `AIC` 任务类型或专门路径

阅读时的映射规则：

- 没看到明确证据，就不要自动把“复杂计算”归到 `AIC`。
- 许多 MoE dispatch/combine 实现虽然逻辑复杂，但主热路径仍可能只是 `MTE + AIV + Scalar`。

## 怎样从同步原语反推职责边界

同步原语是最稳定的证据之一：

- `SetFlag<MTE2_V>` + `WaitFlag<MTE2_V>`（sample 中封装为 `SyncFunc<MTE2_V>`）：说明"搬入已经完成，接下来向量计算可以消费"。
- `SetFlag<V_S>` + `WaitFlag<V_S>`：说明"向量结果已经可被标量读取或用于控制判断"。
- `SetFlag<S_MTE3>` + `WaitFlag<S_MTE3>`：说明"标量已经完成地址/控制决策，接下来可以写回"。
- `PipeBarrier<PIPE_V>`：说明同核向量阶段之间有真实数据依赖。
- `PipeBarrier<PIPE_ALL>`：说明当前阶段切换涉及多条 pipe 的整体可见性。

这些判断同样不是拍脑袋得出的，而是因为官方文档明确说明了：

- `Scalar` 会把不同类型指令发射到独立的分类序列。
- 不同指令序列之间可以并行执行。
- 跨序列依赖通过同步机制控制。
- `PipeBarrier` 用于序列内部约束执行顺序，`SetFlag/WaitFlag` 用于序列之间建立同步关系。

所以，阅读 agent 不应该只说“这里有同步”，而应该进一步写出：

- 这是哪两个执行单元之间的交接点
- 为什么这个交接点存在
- 它是否切断了原本可重叠的流水

## 给阅读 agent 的落地规则

当 agent 画核内 stream 图时，按下面顺序判断：

1. 先找 `DataCopy*`，标出 `MTE2/MTE3` 主路径。
2. 再找 `Cast/Add/Mul/Sum/Reduce/Mask`，标出 `AIV` 主路径。
3. 再找 `if/while/GetValue/SetValue` 和等待判定，标出 `Scalar` 控制路径。
4. 最后再问一句：代码里有没有明确证据表明 `AIC` 参与；没有就写“未见 AIC 参与”。

## 给文档作者的硬约束

- 不要把执行单元职责写成“我觉得像什么”；先回到官方架构定义，再落到代码。
- 不要因为图里想凑完整，就默认把 `AIC` 画进去。
- 不要把“函数名阶段”直接等同于“执行单元职责”。一个函数里常常同时跨 `Scalar`、`MTE`、`AIV`。
- 不要只写“这里是计算阶段”；要明确是 `AIV` 还是 `AIC`，依据是什么。
- 不要只写“这里有同步”；要明确是 `MTE2 -> V`、`V -> S`，还是 `S -> MTE3`。