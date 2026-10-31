# MC2 通信与计算

<适用>
语言: 不限
侧别: All
领域: true
触发: hccl_, AllGather, AllReduce, ReduceScatter, AlltoAll, SyncAll, SetFlag, WaitFlag, PipeBarrier, SyncFunc, expert, expertIds, dispatch, combine, moeExpertNum, quant, BasicQuantMode, PERTENSOR, PERCHANNEL, AlltoAllV, sendCounts, InitV2, Finalize, SetCcTiling, CCU, AICPU, mc2, MC2, Mc2, Hccl, HCCL, AllToAll, Dispatch, Combine, all_gather_matmul, matmul_reduce_scatter, moe_distribute, groupEp, groupTp, GetHcclContext
关联适用: 任务输入或概要已用调用、注册或构建关系确认 MC2 归属，附关联文件、证据位置和来源
默认启用: true
排除场景: 与 MC2 无调用、注册或构建关系的独立计算或通信实现；仅出现同名变量或 API 而没有归属证据的代码。归属未确认时注明缺少的依据；用户说明保留来源。
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

领域说明和主题导航见 [MC2 总览](mc2-specific.md)。各条款的适用条件与检视方法可独立读取。

## 快速索引

按条款标题定位正文，读取到下一个同级或更高级标题之前，跳过代码块中的注释。每组容量见文件头，派发沿用 [负载均衡](../core/review-load-balance.md)。

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---|---|---|---|---|
| MC2-3.1 | 跨核读取和缓冲复用必须等待前序操作完成 | 通信同步 | 高 | [适用: Kernel] |
| MC2-3.2 | 流水事件必须与读写依赖正确配对 | 通信同步 | 高 | [适用: Kernel] |
| MC2-3.4 | 同一通信域内的 rank 必须按协议参与通信 | 通信同步 | 高 | [适用: All] |
| MC2-3.5 | AlltoAllV 各 rank 间的收发量必须配对，访问范围不能越界 | 通信同步 | 高 | [适用: All] |
| MC2-3.6 | HCCL 调用必须符合目标版本的接口要求 | API使用 | 高 | [适用: All] |
| MC2-3.7 | 通信结果必须在完成后读取，通信对象必须在使用结束后销毁 | 资源管理 | 高 | [适用: All] |
| MC2-4.1 | AllGatherMatmul 输出必须与各 rank 的数据位置对应 | 数据搬运 | 高 | [适用: All] |
| MC2-4.2 | MatmulReduceScatter 不能遗漏 rank 的计算结果或改变运算顺序 | 精度保护 | 高 | [适用: All] |
| MC2-4.3 | 通信计算重叠优化必须验证结果正确性和性能收益 | 性能优化 | 中 | [适用: Kernel] |
| MC2-4.4 | 性能统计必须避免重复计数，并正确换算单位 | 性能优化 | 中 | [适用: Kernel] |
| MC2-5.1 | 专家索引必须使用正确的编号，且不能超出本地数组范围 | 内存安全 | 高 | [适用: All] |
| MC2-5.2 | Combine 必须按 Dispatch 的映射和权重还原 token | 数据搬运 | 高 | [适用: All] |
| MC2-6.1 | 量化数据与 scale 的类型和分组必须对应 | 类型安全 | 高 | [适用: All] |
| MC2-6.3 | 量化公式、计算精度和类型转换必须符合模型要求 | 精度保护 | 高 | [适用: All] |

## 详细规范

## MC2-3.1: 跨核读取和缓冲复用必须等待前序操作完成
<a id="mc2-3-1"></a>
<a id="mc2-3-8"></a>
<a id="mc2-3-3"></a>

MC2-3.1: 条例正文

严重级别：高

适用范围：[适用: Kernel]；用于已确认属于 MC2 的代码，涉及通信缓冲、核间屏障及参与核、GM ready/epoch、跨核可见性或跨轮复用。

问题描述：通信开始前，所有生产核必须写完 sendGM，并使数据对通信操作可见。先写完 payload，再发布本轮 ready；所有消费者使用完缓冲区后，才能覆盖其中的数据。

### 错误示例

本例中，各参与核执行相同轮数，每个 rank 只由 isCommCore 指定的核发起通信。示例假设 SyncAll 能保证参与核完成前序读写并使数据可见，且 `AllGather<true>` 返回时通信已经完成。使用该写法前，必须根据目标版本的实际重载核实这两个条件。

```cpp
for (int round = 0; round < roundNum; ++round) {
    ComputeRound(sendGM, round);           // 各核写自己的分块
    if (isCommCore)
        hccl_.AllGather<true>(sendGM, recvGM, count, dataType, rankCnt, tileCnt);
    ConsumeRound(recvGM, round);           // 错：其他核可能尚未写完/通信未完成
}
```

### 正确示例

```cpp
for (int round = 0; round < roundNum; ++round) {
    ComputeRound(sendGM, round);
    SyncAll();                            // 本例目标保证所有分块已写完并可见
    if (isCommCore)
        hccl_.AllGather<true>(sendGM, recvGM, count, dataType, rankCnt, tileCnt);
    SyncAll();                            // 其他核等待阻塞通信结束
    ConsumeRound(recvGM, round);
    SyncAll();                            // 所有消费者结束后再进入下一轮
}
```

### 检视方法

先用目标 CANN 版本和设备的文档及实际封装，确认屏障参与范围、内存可见性和通信完成条件。已有预研覆盖时可复用；缺少这些契约时列出待核实项，不凭 API 名称判断同步有效。

- 展开实际执行路径中的 GM 访问和同步封装，核对 payload、ready 的地址、epoch、参与核，以及数据何时可见。用两轮不同的数据并延迟生产者，检查数据发布、读取和缓冲复用的顺序；同时检查空工作量、尾块和提前退出路径。发生失败时，按通信域的协议通知其他参与者。
- 屏障总数相同不代表各核停在同一阶段。volatile、CPU acquire/release 或原子 flag 本身也不能保证设备上的 payload 可见。已有有效同步时无需重复添加；部分核参与的同步和阻塞语义都按实际 API 核实。
- 调度要求见 [TOPK 第 10、13、14 条](ascendc-topk.md)。superKernel 的限制只适用于该模式。

#### 条例结束

## MC2-3.2: 流水事件必须与读写依赖正确配对
<a id="mc2-3-2"></a>

MC2-3.2: 条例正文

严重级别：高

适用范围：[适用: Kernel]；用于已确认属于 MC2 的代码，涉及 HardEvent、SetFlag、WaitFlag 或 PipeBarrier。

问题描述：流水事件的方向、类型、ID 和 epoch 必须与实际读写依赖一致。同步封装函数必须在当前执行路径中被调用。

### 错误示例

本例中的两个事件 ID 都已合法分配，检查的是同一核内 MTE2→S 事件的配对。

```cpp
SetFlag<HardEvent::MTE2_S>(EVENT_ID7);
WaitFlag<HardEvent::MTE2_S>(EVENT_ID8);      // 错：等待了另一事件
// 同一执行路径若把 WaitFlag(7) 放到 SetFlag(7) 前，也可能一直等待。
```

### 正确示例

```cpp
SetFlag<HardEvent::MTE2_S>(EVENT_ID7);       // 对应搬入操作之后
WaitFlag<HardEvent::MTE2_S>(EVENT_ID7);      // 读取搬入结果之前
// 事件类型、ID、轮次一致；上一轮消费完成后才能复用 EVENT_ID7。
```

### 检视方法

按目标 CANN 版本和设备核对事件类型、发布与等待的实际重载，优先复用对应 API 预研；未覆盖时查官方文档和本地头文件。

- 从每个 Wait 找到对应的有效 Set，检查首轮、循环和退出路径中的事件发布、复用及释放。Add、ReduceSum 等操作访问重叠区域时，要保证先写后读，并在读完后再覆盖数据。
- 循环外预先发布的事件，以及队列或框架提供的顺序保证，都可能构成有效同步。同一核内的事件不能证明其他 rank 的通信已完成。
- 连续超过 3 个屏障只能作为性能调查线索，删除前要核对尾块和跨轮依赖，并进行测量。找到函数定义，或在局部搜索中没有结果，都不能直接证明调用是否配对。

#### 条例结束

## MC2-3.4: 同一通信域内的 rank 必须按协议参与通信
<a id="mc2-3-4"></a>
<a id="mc2-5-3"></a>

MC2-3.4: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 EP/TP 成员、rank 分支、空输入或通信轮次。

问题描述：同一通信域内的参与 rank，必须在同一轮按相同顺序调用集合通信；各 rank 的参数必须满足该原语的配对要求。

### 错误示例

本例为定长 AllGather，签名与单位按 [MC2-3.6](#mc2-3-6) 核实。

```cpp
// 同一通信域、同一轮定长 AllGather。
auto sendCount = rankId == 0 ? count : count / 2; // 错：两 rank 数量不同
hccl_.AllGather<true>(sendGM, recvGM, sendCount, dataType, rankCnt, tileCnt);
```

### 正确示例

```cpp
// 本例各 rank 均有 count=8 个 FP16 元素，且接收容量足够。
hccl_.AllGather<true>(sendGM, recvGM, count, dataType, rankCnt, tileCnt);
// 各 rank 数据内容可以不同，调用顺序与本原语要求的参数须一致。
```

### 检视方法

按目标 CANN 版本、后端和实际通信重载确认参与者及参数配对要求，优先使用对应预研，未覆盖时查官方文档和本地头文件。

- 逐 rank 核对 dtype、count、reduceOp、成员表，以及通信域的 size 和 rank 编号。检查条件分支、空输入和失败返回是否改变了调用顺序。变长通信和 root 相关参数按原语要求配对；AlltoAllV 见 [MC2-3.5](#mc2-3-5)。
- 用 EP≠TP 和最大支持的通信域规模检查循环范围及偏移，设备号不能直接当作域内 rank。乘法、容量和整数窄化见 [红线第 3、4 条](ascendc-red-line.md)；32 位类型能够表示全部合法值时，可以保留。

#### 条例结束

## MC2-3.5: AlltoAllV 各 rank 间的收发量必须配对，访问范围不能越界
<a id="mc2-3-5"></a>

MC2-3.5: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 counts、displacements、dtype 或路由分布。

问题描述：对于每一对发送、接收 rank，AlltoAllV 的发送量必须与对端接收量匹配。各端访问范围不能越界，接收区间不能有接口禁止的重叠。

### 错误示例

以下示例检查两个 rank 的参数。assert 只用于验证示例结果，不能代替生产代码在 release 中的参数校验。

```cpp
// 下标：[本 rank][peer]，count 均为相同 dtype 的元素数。
int sendCounts[2][2] = {{1, 2}, {3, 1}};
int recvCounts[2][2] = {{1, 2}, {3, 1}};    // 错：直接按行复制
// 0→1：sendCounts[0][1]=2，recvCounts[1][0]=3。
```

### 正确示例

```cpp
int sendCounts[2][2] = {{1, 2}, {3, 1}};
int recvCounts[2][2] = {{1, 3}, {2, 1}};
for (int src = 0; src < 2; ++src)
    for (int dst = 0; dst < 2; ++dst)
        assert(sendCounts[src][dst] == recvCounts[dst][src]);
// rank0: recvOffsets={0,1}, capacity=4；rank1: {0,2}, capacity=3。
// 每段检查 offset<=capacity && count<=capacity-offset，且无非法重叠。
```

### 检视方法

先从目标 CANN 版本的实际 AlltoAllV 接口确认 count、displacement 单位及允许的类型转换。使用对应预研或官方文档与本地头文件，不能套用其他重载的签名。

统一数据编码及 count、displacement 的单位，用非对称 counts、传输量为零的 rank 对，以及处于边界的 displacement，逐对检查收发量、访问区间和通信轮次，并确认完成后才使用结果。接口允许表示转换时，可比较转换后的等价字节量。本地发送总量可以与接收总量不同；全局总量相等，也不能证明每对 rank 都已正确配对。

#### 条例结束

## MC2-3.6: HCCL 调用必须符合目标版本的接口要求
<a id="mc2-3-6"></a>

MC2-3.6: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 HCCL 调用、重载、dtype、reduceOp 或 count。

问题描述：HCCL API 的签名和使用限制以目标 CANN 版本为准，核对时必须确认实际调用的重载。

### 错误示例

示例前提：已核实目标重载的 count 单位为元素，buffer 仅包含 8 个 half。SendElements 是展示单位的封装，不代表固定 SDK 原型。

```cpp
uint64_t elements = 8;
SendElements(buffer, elements * sizeof(half), fp16Type); // 错：把字节数当元素数
```

### 正确示例

```cpp
SendElements(buffer, elements, fp16Type); // count=8，与物理元素数一致
// 其余实参仍须逐一对照目标重载：顺序、类型、reduceOp、容量和完成语义。
```

### 检视方法

确认目标 CANN 版本、设备、后端和实际调用的重载，再复用覆盖相同目标的 API 预研。未覆盖的接口用 `/ascendc-docs-search` 查询目标版本文档，并结合本地头文件和变更说明，核对参数顺序、类型、数量、单位、dtype、reduceOp、生命周期及 Host 配置。目标版本不明或材料不足时列出待核实项，不能用最新文档或历史静态表认定代码错误。

#### 条例结束

## MC2-3.7: 通信结果必须在完成后读取，通信对象必须在使用结束后销毁
<a id="mc2-3-7"></a>

MC2-3.7: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及初始化、配置、Commit、Wait 或 Finalize。

问题描述：按实际 API 重载的阻塞或非阻塞语义，检查通信提交、等待完成、结果读取和对象销毁的顺序。

### 错误示例

以下示例只展示成功路径。返回类型及 `<true>`、`<false>` 的含义，必须先按 [MC2-3.6](#mc2-3-6) 核实。

```cpp
// Init 和 SetCcTiling 已成功；本例 <false> 为非阻塞。
auto handle = hccl_.AllGather<false>(sendGM, recvGM, count, dataType, rankCnt, tileCnt);
UseReceived(recvGM);                       // 错：尚未提交/等待完成
hccl_.Finalize();
```

### 正确示例

```cpp
auto handle = hccl_.AllGather<false>(sendGM, recvGM, count, dataType, rankCnt, tileCnt);
hccl_.Commit(handle);
hccl_.Wait(handle);
UseReceived(recvGM);                       // 确认完成后才能读取
hccl_.Finalize();                          // 所有引用该对象的任务都已结束
```

### 检视方法

先按目标 CANN 版本、设备和实际重载核实 Init、Commit、Wait、Finalize 的配对与完成语义。复用匹配的 API 预研；未覆盖时查官方文档和本地头文件。`<true>`、`<false>` 不能一律解释为是否需要 Commit+Wait。

沿通信域 handle 检查首次使用、复用及已有的失败、取消或超时路径。Finalize 可以由外层 owner 负责；提交失败或超时不代表任务一定未执行，释放输入和销毁对象前须确认最后一次使用结束。后端传输上限见 [MC2-7.3](mc2-resources-errors.md#mc2-7-3)，资源所有权见 [MC2-7.4](mc2-resources-errors.md#mc2-7-4)。

#### 条例结束

## MC2-4.1: AllGatherMatmul 输出必须与各 rank 的数据位置对应
<a id="mc2-4-1"></a>

MC2-4.1: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 AllGatherMatmul 的聚合轴、布局或写回。

问题描述：AllGatherMatmul 必须按接口规定的轴聚合数据，并保持各 rank 的数据与输出位置之间的对应关系。payload、scale、gatherOut 和主输出要分别检查。

### 错误示例

以下 Python 示例演示矩阵行与输出位置的对应关系。各 rank 的 W 相同，每行 [a,b] 与 W 相乘的结果为 [a+3b,2a+4b]。实际设备支持的输入范围和 stride 仍要按目标实现核实。

```python
# 两rank各一行，按rank沿M拼接；W=[[1,2],[3,4]]，无转置/bias/量化。
gathered = [[1, 2], [3, 4]]
w = [[1, 2], [3, 4]]
product = [[sum(row[k] * w[k][j] for k in range(2))
            for j in range(2)] for row in gathered]
output = [None, None]
output[0] = product[1]                    # rank1先算本地，却写到rank0位置
output[1] = product[0]
assert output == [[15, 22], [7, 10]]       # shape正确，行顺序错误
```

### 正确示例

```python
output = [None, None]
output[1] = product[1]                    # rank1本地计算仍可先执行
output[0] = product[0]
assert output == [[7, 10], [15, 22]]
# G=(R*m,K)，output=(R*m,N)；rank r首行=r*m。
# 行主序无padding时，首字节偏移=r*m*N*sizeof(output_element)。
```

### 检视方法

输出布局、gatherOut 和支持的输入组合以被检提交的算子定义为准；涉及通信或 Matmul API 限制时核对目标 CANN 版本、设备及实际重载。

- 为各 rank 使用不同的非对称数据，核对接收缓冲区、矩阵行和输出位置的对应关系。检查转置后的 M、K、N，stride，本地和远端数据，以及完整分块和尾块。关闭 gatherOut 不能影响主输出的结果或所需容量。
- NZ、padding、转置和非连续输入都按实际布局计算地址，不能用未知的动态维度作为容量。scale 的分组和独立等待要求见 [MC2-6.1](#mc2-6-1)。

#### 条例结束

## MC2-4.2: MatmulReduceScatter 不能遗漏 rank 的计算结果或改变运算顺序
<a id="mc2-4-2"></a>

MC2-4.2: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 MatmulReduceScatter 的归约、bias、scale 或分片。

问题描述：MatmulReduceScatter 的每个输出分片都必须包含所有应参与归约的 rank 的计算结果，并保持局部计算、反量化、归约和分片的规定顺序。

### 错误示例

以下 Python 示例使用两个 rank，M=4、N=1，不使用 bias 或 scale。先对完整的四行做 SUM 归约，再沿 M 等分；其他分片轴或 reduceOp 不能直接套用本例。

```python
# X0=[1,2,3,4]^T，X1=[10,20,30,40]^T；W0=W1=[[1]]。
p0, p1 = [1, 2, 3, 4], [10, 20, 30, 40]
rank0, rank1 = p0[:2], p1[2:]              # 错：提前取本地片段，缺远端贡献
assert rank0 == [1, 2] and rank1 == [30, 40]
```

### 正确示例

```python
reduced = [a + b for a, b in zip(p0, p1)]  # 本例 SUM
rank0, rank1 = reduced[:2], reduced[2:]
assert reduced == [11, 22, 33, 44]
assert rank0 == [11, 22] and rank1 == [33, 44]
# 每rank接收2个元素，全局归约4个元素。
# 设备实现可融合计算/归约，无需显式物化 reduced。
```

### 检视方法

先确认被检提交承诺的归约和输出语义；涉及 ReduceScatter 或 Matmul API 限制时，按目标 CANN 版本、设备及实际重载查证。

- 根据 X、W 的分片方式和 reduceOp，列出每个输出应包含哪些 rank 的计算结果，再核对发送区、接收区、recvCount、容量、尾块和缓冲复用时机。
- 用非零 bias 和不同 scale，将各 rank 的输出切片与全局参考结果比较，检查舍入和激活的影响。各 rank 在归约前分别加 bias，与归约后只加一次 bias 不等价；反量化也不能任意调整顺序。合法的预分片或非均匀分片实现不要求显式执行 SUM 循环。

#### 条例结束

## MC2-4.3: 通信计算重叠优化必须验证结果正确性和性能收益
<a id="mc2-4-3"></a>

MC2-4.3: 条例正文

严重级别：中

适用范围：[适用: Kernel]；用于已确认属于 MC2 的代码，涉及同步调整、重复搬运、双缓冲或流水重叠。

问题描述：将计算与通信重叠，或减少数据搬运后，仍必须满足读写顺序、缓冲复用和容量要求。性能收益需要在相同输入和环境下进行对照测量。

### 错误示例

以下时序由假设的依赖关系推算，并非实测结果。两个独立分块各需要 4 个时间单位计算、3 个时间单位通信。前提是设备支持计算与通信重叠，两个缓冲槽容量充足且不会互相覆盖；通信 1 只需等待计算 1，不需要等待通信 0 完成。

```text
计算0 [0,4] → 通信0并等待 [4,7] → 计算1 [7,11] → 通信1并等待 [11,14]
每块后立即等待，把可独立执行的计算1推迟到7；推演完成时刻14
```

### 正确示例

```text
计算0 [0,4] → 通信0 [4,7]，同时计算1使用另一个槽 [4,8]
计算1完成后通信1 [8,11]；各槽最后消费者完成后才复用
推演完成时刻11；实际测量可能受竞争和启动开销影响，不能承诺固定加速比
```

### 检视方法

- 核对地址、长度、搬运次数、缓存和资源竞争。只有一个缓冲槽，或前后操作存在依赖时，不能直接并行执行；合并消息或增大缓冲区后，要重新检查通信协议及容量。
- 在相同输入和环境下重复测量各阶段及端到端耗时，检查小消息、大消息、空分区和不均匀路由，并核对结果误差、测量噪声及项目阈值。API 调用次数不能直接代表总线流量；没有测量时，只能提出待验证的性能假设。

#### 条例结束

## MC2-4.4: 性能统计必须避免重复计数，并正确换算单位
<a id="mc2-4-4"></a>

MC2-4.4: 条例正文

严重级别：中

适用范围：[适用: Kernel]；用于已确认属于 MC2 的代码，涉及计时、清零、epoch、统计归约或报告单位。

问题描述：性能统计的计数方式和单位必须符合指标定义。计时开关要在首次判断前完成初始化；启用统计时，必须完成采样、单位换算和结果写回。

### 错误示例

本例要求同一 operation、epoch、block 中的每个事件只记录一次，total 初值为 0。采用 sum 还是 max，要另按指标定义判断。

```cpp
// 同一 epoch/block 的 ready 在两次轮询中都为 true，elapsed=7。
if (ready) total += elapsed;               // poll1: total=7
if (ready) total += elapsed;               // poll2: total=14，重复计数
```

### 正确示例

```cpp
// 本例仅一个记录者。
if (ready && !recorded[epoch][block]) {
    total += elapsed;                     // 本事件只计7
    recorded[epoch][block] = true;
}
// 多记录者须原子化检查和标记；进入新 epoch 使用独立状态。
```

### 检视方法

沿 MC2 各阶段的 operation、epoch 和 block 核对采样、归约和清零位置，按指标定义选择 sum 或 max，检查同一事件被轮询多次时是否重复计数。各阶段结果换算为同一单位后再汇总；依赖设备时钟时核实目标设备的计时单位。

多个执行者更新同一统计项时，检查现有同步是否覆盖检查与标记。只在实际共享更新的路径要求同步；单记录者无需增加原子操作。未运行测量时只报告已查明的统计逻辑问题。

#### 条例结束

## MC2-5.1: 专家索引必须使用正确的编号，且不能超出本地数组范围
<a id="mc2-5-1"></a>

MC2-5.1: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及 expertId、映射、丢弃哨兵或专家容量。

问题描述：读取 expertIds 前要检查访问范围；取得专家 ID 后，还要检查它所属的 rank、全局编号到本地编号的映射，以及目标数组的实际容量。

### 错误示例

```cpp
// localExperts=2，globalExpertId=7，localBase=6
localBuffer[globalExpertId] = value; // 写下标7，容量只有2
```

### 正确示例

本例的本地专家连续分配在 `[localBase, localBase + localExperts)`；非连续分配应按实际映射取得本地编号。

```cpp
if (globalExpertId < localBase) return InvalidArgument;
auto localId = globalExpertId - localBase;
if (localId >= localExperts) return InvalidArgument;
localBuffer[localId] = value; // 7-6=1
```

### 检视方法

1. 从任务输入或概要中的 MC2 关联依据，定位专家 ID 的来源、所属通信域和 owner rank。按实际分配方式核对全局到本地编号的转换；只有连续分配才能减去起始编号，非连续分配需检查映射表或等价算法。
2. 检查读取 expertIds 的 tokenIdx，以及映射后访问本地数组的下标和实际容量。上游校验若覆盖同一个值，且使用前没有改值，可直接作为有效保护，无需下游再加一次检查。
3. 按接口允许的值检查空路由、合法边界、越界值和丢弃哨兵。允许 -1 时，该分支应跳过数组访问，不能取模变成有效索引。专家总数合法不能证明每个 ID 都合法。

映射、容量或上游检查材料不足时列出待核实项，不仅凭变量名或固定容量报错。编号语义以目标算子定义为准；涉及接口版本差异时，使用目标版本的预研或官方资料和本地实现核对。

#### 条例结束

## MC2-5.2: Combine 必须按 Dispatch 的映射和权重还原 token
<a id="mc2-5-2"></a>

MC2-5.2: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及重排、inverse、路由权重、共享专家或空分片。

问题描述：Combine 必须使用同一次 Dispatch 的 token、expert、rank 映射和路由权重，正确还原各条路由的结果。两端专家参数的定义、默认值和实际传值必须一致。

### 错误示例

本例每个 token 只有一条路由，routingWeight=1；token 下标和 inverse 中的值都已通过边界检查。

```cpp
// Dispatch 已将 [A,B] 排为 [B,A]；inverse={1,0}。
for (size_t token = 0; token < 2; ++token)
    output[token] = expertOut[token];       // 错：[f(B),f(A)]
```

### 正确示例

```cpp
for (size_t token = 0; token < 2; ++token)
    output[token] = expertOut[inverse[token]]; // 正确：[f(A),f(B)]
```

### 检视方法

先检查 token 下标是否超出 inverse 的容量，再检查 inverse 中的值是否超出重排结果的行数。随后逐条路由核对 token 还原和 routingWeight，检查 EP≠TP、重复路由、共享专家及空分片。合法的内部重排可以保留，不能仅凭变量命名不同就判定数值错误。

#### 条例结束

## MC2-6.1: 量化数据与 scale 的类型和分组必须对应
<a id="mc2-6-1"></a>

MC2-6.1: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及量化 dtype、scale 分组、分片起点或通信。

问题描述：量化 payload 和 scale 的编码、分组必须对应；scale 的数量和索引必须覆盖当前分片涉及的所有全局分组。

### 错误示例

groupSize=4、shardStart=3、localCount=3 时，只传 ceil(localCount/groupSize)=1 个 scale，或按 localIndex/groupSize 选组，都会遗漏第二个全局分组。

### 正确示例

全局索引 [3,4,5] 对应 [group0,group1,group1]，传入两个 scale，本地映射为 [0,1,1]。

### 检视方法

量化编码和 scale 布局以目标算子定义为准；涉及通信 dtype 或打包格式的接口限制时，使用目标 CANN 版本的预研、官方文档和实际头文件核对。

- 将 payload 和 scale 放在一起核对，检查各自的 dtype、全局起点、分组范围、count、offset、通信顺序及完成事件。MX 等打包格式按实际编码判断。
- 检查 quantMode 和 scale 在读取、赋值、比较、打印时的类型是否正确；接口允许的 bool 到整数编码或其他表示转换可以保留。日志参数的类型提升、格式符和 ABI 要求见 [安全编码第 8.2 条](cpp-secure.md)。

#### 条例结束

## MC2-6.3: 量化公式、计算精度和类型转换必须符合模型要求
<a id="mc2-6-3"></a>
<a id="mc2-6-4"></a>

MC2-6.3: 条例正文

严重级别：高

适用范围：[适用: All]；用于已确认属于 MC2 的代码，涉及模型常量、预计算、scale 运算或量化 Cast。

问题描述：Host 预计算、Kernel 累加、scale 运算和 Cast 都必须符合模型公式，保留规定的单位、中间精度，以及舍入和饱和转换语义。

### 错误示例

以下 Host 示例中，x 是 [-130,130] 内的整数，已按目标规则完成舍入；模型要求后续转换采用饱和语义。

```cpp
// 目标算法要求 int8 饱和，实现却只保留低8位。
uint8_t low = static_cast<uint8_t>(x);      // 转换按模256取值
int32_t wrapped = low < 128 ? low : int32_t(low) - 256;
int8_t q = static_cast<int8_t>(wrapped);    // -130→126，130→-126，错误
```

### 正确示例

```cpp
// Host 数值模型：x 为有限、已按目标舍入处理且 int32_t 可表示的整数值。
int32_t clipped = x < -128 ? -128 : (x > 127 ? 127 : x);
int8_t q = static_cast<int8_t>(clipped);    // [-130,0,130]→[-128,0,127]
```

### 检视方法

- 用正负极值、舍入和饱和转换的边界值，以及多个量化分组，对比参考结果和允许误差。检查是否过早窄化导致回绕，以及计算位置或常量是否被改错。区分 1/sqrt(headDim) 与 1/headDim 时，要使用 headDim 不等于 1 的输入。
- 模型允许的其他转换语义可以保留。Cast 次数或全零输入不能证明精度正确；数学 API 和 Cast 的约束按目标版本文档核实；特殊值处理见 [TOPK-6](ascendc-topk.md)。不要求 Kernel 使用 std::sqrt。

#### 条例结束
