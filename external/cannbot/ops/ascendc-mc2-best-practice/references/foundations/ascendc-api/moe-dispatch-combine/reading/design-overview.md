# MoE Dispatch/Combine Design Overview

**本篇算子设计介绍基于<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>**

## 1. 背景与总体设计方案

### 1.1 MoE架构的通信瓶颈与挑战

#### 1.1.1 MoE架构概述
在大规模模型训练与推理领域，混合专家（Mixture of Experts，MoE）架构凭借其动态专家激活机制所带来的计算稀疏性优势，已成为支撑千亿参数级模型的核心技术方案。该架构通过**分发（Dispatch）** 与**组合（Combine）** 两大关键操作，实现了输入数据的动态分配与多专家输出的高效整合，在维持海量参数规模的同时保障了计算效率。

然而，随着专家并行（Expert Parallelism，EP）规模的持续扩展，专家节点间频繁的数据交互所引发的高额通信开销，已逐渐成为制约大模型推理性能的关键瓶颈。

#### 1.1.2 传统通信方案的局限性

**AllToAllV 通信的效率缺陷**
在动态专家选择机制下，每个Token被分发的目标专家呈现离散分布特征，导致：
- **数据分发不均**：不同专家接收的Token数量存在显著差异，不得不依赖低效的AllToAllV通信；
- **元数据同步开销**：获取收发信息需调用前置AllGather算子收集路由表，并在Host侧完成同步，引入额外通信开销与Stream同步延迟。

**小数据包与Host Bound问题**
在推理场景中，Token数据量通常较小，引发双重挑战：
- **算子下发延迟**：传统Host驱动通信需构造子图并进行调度，其下发时延随EP规模线性增长；
- **RDMA同步开销**：RDMA通信的前后同步过程引入额外的RTT时延。


### 1.2 创新解决方案设计

#### 1.2.1 通算融合算子架构
基于上述瓶颈分析，开发了**MoeDistributeDispatch**与**MoeDistributeCombine**两个通算融合算子。

在DeepSeekV3模型的MoE架构中，采用动态路由机制，每个Token动态选择topK个专家进行处理。其中：
- **Dispatch操作**承担核心调度功能，基于Token与专家的路由对应关系表，采用分布式计算策略：首先将各专家节点需处理的Token数量计算任务下沉至对应设备执行，随后通过AllToAllV通信完成Token的跨设备传输，同时预计算Combine阶段所需参数；
- **Combine操作**负责整合各专家输出的计算结果，执行加权求和，并通过逆向的AllToAllV通信将处理后的Token数据恢复至原始位置，完成整个分布式专家计算的协同与整合。

#### 1.2.2 技术优势
Dispatch/Combine操作本质上是计算与通信的紧密结合。通算融合算子相较于传统的AllToAllV通信实现了以下突破：
- 将路由计算等Host侧逻辑下沉至Device侧，彻底消除Host与Device间的同步开销；
- 实现Combine操作中部分计算与AllToAllV通信的流水并行，有效掩盖计算与通信耗时。

### 1.3 基于AIV+MTE融合架构的超节点内全互联方案

#### 1.3.1 架构概述
我们基于AIV+MTE融合架构构建了超节点内全互联（Fullmesh）方案，充分发挥了昇腾硬件NPU的计算与通信能力。

#### 1.3.2 处理流程

**预处理阶段（AIV）**
  简化路由与数据组织，获取每个Token的路由信息，按照数据量分核后直接进行发送，与AICPU直驱通信方案相比，有如下变化：
- __无需重排：__ 取消原有的专家索引重排步骤。直接通过MTE的通信路径实现数据的高效分发。
- __直接映射目标Rank：__ 利用MTE的地址映射能力，直接将Token数据与目标Rank的内存地址关联，无需手动汇聚数据。
- __共享内存管理：__ AIV将Token数据直接写入共享内存的预分配区域，通过MTE的地址映射快速定位到目标Rank地址。

**通信驱动（MTE）**
- AIV将需要发送的数据与状态位与目标rank的地址，数据长度等信息写入MTE共享内存的控制区，触发MTE直接执行数据传输。
- 在Device侧MTE自主完成通信，无需Host侧构造子图与调度任务，消除Host侧时延。
- MTE通过更轻量的传输协议代替传统RDMA，减少通信延迟。

**通信等待与后处理（AIV）**
- 在通信环节，AIV轮询状态区求和，确保每个核负责的Token数据全部接收完成。
- AIV将共享内存中的数据按照专家汇总搬出，为后续FFN层的计算提供数据准备。

#### 1.3.3 技术价值
这一系列优化形成了完整的低延迟处理闭环，实现了：
- **通信计算融合**：将通信准备与计算任务深度融合。
- **设备侧自治**：减少Host侧干预，提升处理效率。
- **全流程优化**：从数据预处理到后处理的端到端性能提升。

## 2. MoeDistributeDispatch实现方案

### 2.1 概述

MoeDistributeDispatch算子的实现方案构建了一个完整的数据分发处理流水线。该方案通过四个核心阶段的紧密协作，实现了从Token路由计算到跨设备分发的全流程优化：

1. **Token处理与发送阶段**：按照Token发送量分核处理Token，基于目标专家Id直接获取目标地址进行发送
2. **Status处理与发送阶段**：按照总专家数分核处理状态位，向对端卡发送状态位
3. **状态接收同步阶段**：按数据来源分专家接收状态位
4. **发送后处理阶段**：对接收数据进行结构化重组与计算，生成下游计算所需的元数据信息

### 2.2 Token处理与发送

#### 2.2.1 设计背景

Token处理涉及三元组计算，每个token都有自己的一份三元组信息，围绕expertIds输入矩阵展开，该矩阵维度为BS×K，其中expertIds(i,j)表示第i个Token被分配给第j个专家的索引。Combine算子需要Dispatch算子提供assistInfoForCombine输出，该矩阵为A×128维度，其中前A×3位有效值，A表示本卡需要分发的最大Token数量，即某张卡，经过Dispatch处理后，最多接收到所有卡给它的Token数量为A。

Token发送涉及发送给共享专家与发送给MOE专家，按照Token数据量进行分核处理，增加并发性与网络吞吐量。

#### 2.2.2 实现方案

##### 三元组信息设计

每个token后附加三元组信息 [epRankId, tokenIndex, topKIndex] 其含义如下：

- **epRankId**：该token来自哪张卡
- **tokenIndex**：该token是原始BS中的第几个Token
- **topKIndex**：该token是原始TopK中的第几个值

##### 索引计算流程

对于MoE专家，遍历expertIds矩阵，对于每个元素expertIds(i,j)：

1. 目标专家索引 = expertIds(i,j)
2. 目标rank索引 = expertIds(i,j) // localExpertNum + sharedExpertRankNum
3. 专家在目标rank上的局部索引 = expertIds(i,j) % localExpertNum
4. Token在目标rank目标专家所属区域内的索引= curExpertCnt，表示在此token前该专家已经收到多少token 

下面主要介绍AIV如何加速计算curExpertCnt


###### i. AIV分核并行处理
- **核分配**：将BS个token均匀分配到多个AIV计算核
- **独立计算**：每个核处理分配的token子集，计算专家匹配和偏移

###### ii. 专家ID匹配与计数

输入: expertIds矩阵(BS×K), 目标专家ID

处理:

- **向量比较:** 批量比较token专家ID与目标专家ID

- **统计匹配:** 统计匹配的token数量

- **有效计数:** 得到发送给目标专家的实际token数

输出: 每个专家的接收token数量

###### iii. 偏移地址计算
基于"同一专家token连续存储"的特性：

**偏移计算原理**：
- 每个专家维护独立的处理计数器
- 当前token偏移 = 该专家已处理的token数量
- 原子操作保证多核并发下的计数准确性

##### 关键优化点

###### 向量化指令优化：
- 使用`Duplicate`批量复制目标专家ID
- `Sub`指令并行计算专家ID差值
- `ReduceSum`快速统计匹配数量

###### 内存访问优化：
- 连续访问专家ID矩阵，提高缓存命中率
- 计数器使用局部存储，减少全局内存访问

###### 并行度最大化：
- token级别并行：各AIV核独立处理不同token
- 专家级别并行：不同专家计数可并行更新
- 计算搬运重叠：MTE搬运与下一批计算并行

对于共享专家，token需要发送给所有的共享专家，对于多卡一专家情况，只发给其中的一张卡：
1. 目标rank索引 = epRankId % rankNumPerSharedExpert + toSharedExpertIndex * rankNumPerSharedExpert

通过上述索引值，直接获取本卡要发的token在目的卡的偏移地址，便于MTE搬运。

##### Token分核发送机制

对于发送数据次数进行均分分核，
假设有BS个Token需要发送，对于sharedExpertNum个共享专家，则需要发送次数 sendCnt = BS * sharedExpertNum 次；对于MoE专家，topK = k ，则需要发送 sendCnt = BS * k 次。

则aivNum个AIV核，每个核分到 sendCnt // aivNum 个数据进行发送，对于余数 remain = sendCnt % aivNum，则由前remain个AIV核进行发送。

##### 数据区窗口分配策略

对于每一块rank，其windows区的数据区先按rank分配，再按专家分配，再按BS分配（按上限预留，每个rank上最多每个专家接收BS个token），因此每张卡的数据区需要占用内存为epWorldSize * moeExpertNumPerRank * BS * dataLen字节。

| Rank | Expert | BS | 内存 | 说明 |
|:----:|:------:|:--:|:----:|:----|
| **R₁** | **E₁** | **B₁** | **L** | Rank 1, Expert 1, Token 1 |
| R₁ | E₁ | B₂ | L | Rank 1, Expert 1, Token 2 |
| R₁ | E₁ | B₃ | L | Rank 1, Expert 1, Token 3 |
| R₁ | E₁ | ... | ... | ... |
| R₁ | E₁ | B₍BS₎ | L | Rank 1, Expert 1, Token BS |
| R₁ | **E₂** | **B₁** | **L** | Rank 1, Expert 2, Token 1 |
| R₁ | E₂ | B₂ | L | Rank 1, Expert 2, Token 2 |
| R₁ | E₂ | ... | ... | ... |
| R₁ | E₂ | B₍BS₎ | L | Rank 1, Expert 2, Token BS |
| R₁ | **E₃** | **B₁** | **L** | Rank 1, Expert 3, Token 1 |
| R₁ | E₃ | ... | ... | ... |
| R₁ | **E₍ₘₒₑEₓₚₑᵣₜNᵤₘₚₑᵣRₐₙₖ₎** | **B₍BS₎** | **L** | Rank 1, Expert E, Token BS |
| **R₂** | **E₁** | **B₁** | **L** | Rank 2, Expert 1, Token 1 |
| R₂ | E₁ | B₂ | L | Rank 2, Expert 1, Token 2 |
| ... | ... | ... | ... | ... |
| **R₍epWorldSize₎** | **E₍moeExpertNumPerRank₎** | **B₍BS₎** | **L** | Rank R, Expert E, Token BS |

**总计: R × E × B × L = epWorldSize × moeExpertNumPerRank × BS × dataLen 字节**

### 2.3 Status处理与发送

#### 2.3.1 设计背景

Status处理涉及TokenSendExpertCnt计算，即当前卡给某个专家发送的token数量是多少，该TokenSendExpertCnt会和状态位一起发送，代表某个专家来自当前发送卡的数据发送完毕。

Status处理按照专家数量 = sharedExpertRankNum + moeExpertNum 进行分核处理，增加并发性与网络吞吐量。这里使用sharedExpertRankNum是因为需要给所有共享专家卡都发送状态位。

##### 索引计算流程

对于MoE专家，遍历expertIds矩阵，对于每个元素expertIds(i,j)：

1. 目标专家索引 = expertIds(i,j)
2. 目标rank索引 = expertIds(i,j) // localExpertNum + sharedExpertRankNum
3. 专家在目标rank上的局部索引 = expertIds(i,j) % localExpertNum

对于共享专家，token需要发送给所有的共享专家，对于多卡一专家情况，只发给其中的一张卡：
1. 目标rank索引 = epRankId % rankNumPerSharedExpert + toSharedExpertIndex * rankNumPerSharedExpert

通过上述索引值，直接获取本卡要发的token在目的卡的偏移地址，便于MTE搬运。
#### 2.3.2 实现方案

##### 状态区窗口分配策略

对于每一块rank，其windows区的状态区分配策略：

1. **共享专家卡**：每个rank分配2个状态位（flag + TokenSendExpertCnt），共epWorldSize × 2位
2. **MoE专家卡**：先按moeExpertNumPerRank分配，再按epWorldSize分配，每个分配2个状态位

**状态位定义**：
- 第1位：flag（标志位）
- 第2位：TokenSendExpertCnt（发送token数量计数）

**共享专家卡状态分配表**：
| Rank | Expert | 状态位1(FLAG1) | 状态位2(TokenSendExpertCnt) |
|:----:|:------:|:-------------:|:---------------------------:|
| R₁ | 共享专家 | Flag | TokenNum |
| R₂ | 共享专家 | Flag | TokenNum |
| ... | ... | ... | ... |
| R₍epWorldSize₎ | 共享专家 | Flag | TokenNum |

**MoE专家卡状态分配表**：
| Rank | Expert | 状态位1(FLAG1) | 状态位2(TokenSendExpertCnt) |
|:----:|:------:|:-------------:|:---------------------------:|
| E₁ | R₁ | Flag | TokenNum |
| E₁ | R₂ | Flag | TokenNum |
| E₁ | ... | ... | ... |
| E₁ | R₍epWorldSize₎ | Flag | TokenNum |
| E₂ | R₁ | Flag | TokenNum |
| E₂ | R₂ | Flag | TokenNum |
| ... | ... | ... | ... |
| E₍moeExpertNumPerRank₎ | R₍epWorldSize₎ | Flag | TokenNum |

### 2.4 状态接收同步阶段

##### 接收同步机制

采用分核循环等待策略：

1. **专家分配**：将所有epWorldSize * moeExpertNumPerRank平均分配给每个核（对共享专家卡，此处moeExpertNumPerRank即为1），每个核需处理recStatusNumPerCore个状态
2. **循环**：轮询FLAG1值的算数和，直到刷新为recStatusNumPerCore，确认Token接收完成
3. **数据处理**：对Token数量FLAG2求和，得到每个核均分到的recStatusNumPerCore接收到的所有Token数，获取每个核处理Token的偏移

### 2.5 发送后处理

#### 2.5.1 设计背景

数据接收完成后，需要进行以下处理：

1. **数据重排**：依据元数据将实际数据内容重新排列，确保同一专家的Token在GM内存中保持顺序连续，并将其三元组信息进行对应重排列
2. **元数据计算**：生成epRecvCount和expertTokenNum两个关键输出

#### 2.5.2 实现方案

假设系统包含epWorldSize张卡，每张卡部署moeExpertNumPerRank个专家，则可以根据的状态window区得到tokenNums矩阵维度为 epWorldSize × moeExpertNumPerRank 作为epRecvCount矩阵，记录每个locaMoE获取了来自各卡多少Token数。
expertTokenNum输出为本地每个专家的Token数量前缀和数组，即epRecvCount的最后一行按照rank维度进行sum计算。

处理流程框架：

1. **数据加载**：从状态窗口区加载tokenNums矩阵到local tensor
2. **累加和计算**：对tokenNums矩阵进行列向累加，生成epRecvCount累加和矩阵
3. **专家总数提取**：从累加和矩阵最后一行直接提取各专家接收总数

**累加和优化：**

- expertTokenNum计算：直接提取累加和矩阵最后一行，无需跨rank求和
- 寻址优化：累加和矩阵直接提供token输出缓冲区的偏移地址

#### 2.5.3 算子间数据同步机制

##### 设计背景

在分布式专家并行架构中，由于dispatch和combine算子执行特性不同，存在以下时序冲突风险：

- **dispatch有全卡同步**：dispatch算子内部存在全卡同步点，保证所有rank的dispatch进度一致
- **combine无全卡同步**：combine算子内部无同步点，各rank执行进度可能不一致
- **算子串行执行**：同一卡上dispatch和combine串行执行，但不同卡间可能存在重叠

@startuml
title 算子时序冲突示意图

participant "Rank A" as RankA
participant "Rank B" as RankB
participant "Rank C" as RankC
participant "Win区" as Win

== 初始状态 ==
note over RankA, RankC: dispatch1开始，有全卡同步

== Rank A (慢卡) ==
RankA -> Win: dispatch1写入数据
note over RankA: dispatch1内部全卡同步中

== Rank B (中速卡) ==
RankB -> Win: dispatch1快速完成
RankB -> Win: combine1写入数据
RankB -> Win: combine1本地处理完成

== Rank C (快卡) ==
RankC -> Win: dispatch1快速完成
RankC -> Win: combine1写入数据并快速完成
RankC -> Win: 开始dispatch2

== 时序冲突点 ==

group 关键冲突：dispatch2与dispatch1重叠
  RankC -> Win: dispatch2写入数据到缓冲区
  RankA -> Win: dispatch1仍在处理中（使用同一缓冲区）
end

group 缓冲区踩踏
  Win --> RankA: dispatch1数据被dispatch2覆盖
  note right Win: 数据完整性破坏
end

group 状态位混乱
  Win --> RankA: dispatch1状态位被dispatch2更新
  note right Win: 死锁风险
end

@enduml

如上图所示，时序冲突的核心原因在于：

1. **dispatch有全卡同步**：所有rank的dispatch开始时间一致，但结束时间因处理速度而异
2. **combine无全卡同步**：快卡完成combine后可直接进入下一轮dispatch
3. **缓冲区复用不安全**：快卡的dispatch2可能复用慢卡dispatch1仍在使用的缓冲区

具体场景：
- RankA（慢卡）仍在dispatch1处理中
- RankC（快卡）已完成combine1，进入dispatch2
- 由于dispatch2与dispatch1可能使用同一缓冲区，导致快卡写入覆盖慢卡数据

##### 实现方案

采用基于全卡同步的缓冲区轮转机制，核心设计如下：

**四缓冲区分区架构**
- 将win区划分为dispatch-0、dispatch-1、combine-0、combine-1四个独立存储块
- 每个存储块包含独立的数据区和状态区，避免算子间干扰
- dispatch和combine各自维护独立的bufferChosen标志位

<table>
    <tr>
        <th colspan="6" style="text-align: center;">windows状态区</th>
    </tr>
    <tr>
        <th colspan="2">0区</th>
        <th colspan="2">1区</th>
        <td colspan="1" rowspan="2">dispatch0/1区标识</td>
        <td colspan="1" rowspan="2">combine0/1区标识</td>
    </tr>
    <tr>
        <td>dispatch状态区</td>
        <td>combine状态区</td>
        <td>dispatch状态区</td>
        <td>combine状态区</td>
    </tr>

</table>

**缓冲区选择逻辑**
- dispatch算子：
  - 读取dispatch_bufferChosen标志位
  - 标志位为0：使用dispatch-0区
  - 标志位为1：使用dispatch-1区
  - 执行完成时翻转标志位：dispatch_bufferChosen = dispatch_bufferChosen ^ 1
- combine算子：
  - 读取combine_bufferChosen标志位
  - 标志位为0：使用combine-0区
  - 标志位为1：使用combine-1区
  - 执行完成时翻转标志位：combine_bufferChosen = combine_bufferChosen ^ 1

**同步保证机制**
1. **算子内同步**：dispatch内部有全卡同步点，保证同一轮次内所有rank使用相同缓冲区
2. **算子间隔离**：dispatch和combine使用独立缓冲区，避免串行执行时相互干扰
3. **轮次间隔离**：Double Buffer交替使用，避免快卡第N+1轮数据覆盖慢卡第N轮数据
4. **进度一致性**：由于dispatch内部全卡同步，所有rank的dispatch进度基本一致，减少combine等待时间

**缓冲区复用安全规则**
- dispatch第N轮使用缓冲区X，完成后切换至缓冲区Y
- combine第N轮使用缓冲区M，完成后切换至缓冲区N
- 当快卡执行到第N+2轮时，慢卡的第N轮必定已完成（因dispatch有全卡同步）
- 第N+2轮可安全复用第N轮的缓冲区，形成稳定的轮转周期

该方案通过算子隔离和双缓冲轮转，在保证数据一致性的同时最大化并行效率，避免了算子间数据竞争和不同轮次间的数据踩踏问题。


# 三、MoeDistributeCombine实现方案

## 3.1 概述

MoeDistributeCombine算子负责将分布式专家计算的结果进行整合与还原，实现从多专家输出到原始输入格式的转换。

## 3.2 Token处理与发送

### 3.2.1 设计背景

对于发送数据次数进行均分分核，所需要的发送数据次数由之前Dispatch的输出epRecvCount给出，对应为Combine的输入epSendCounts，每个token需要返还的路由由Dispatch的输出assistInfoForCombine给出。
假设有sendCnt个Token次需要发送，每个核分到 sendCnt // aivNum 个数据进行发送，对于余数 remain = sendCnt % aivNum，则由前remain个AIV核进行发送。

### 3.2.2 实现方案

在前缀和形式的epSendCounts矩阵中，每个元素(i, j)表示专家i发送到rank j的累计Token数量（从rank 0到rank j）。以下以卡0发送至卡1的数据流程为例，详细说明处理过程。

#### 前缀和矩阵示例
假设卡0的sendCounts矩阵（前缀和形式）如下表所示：

| 专家索引 | rank 0 | rank 1 | rank 2 | rank 3 |
|----------|--------|--------|--------|--------|
| 0        | 1      | 1      | 2      | 3      |
| 1        | 3      | 8      | 10     | 12     |

- **专家0**：返还至rank 1的Token数量（即dispatch阶段接收的数量）= sendCounts(0,1) - sendCounts(0,0) = 1 - 1 = 0
- **专家1**：返还至rank 1的Token数量（即dispatch阶段接收的数量）= sendCounts(1,1) - sendCounts(1,0) = 8 - 3 = 5

则可以得到，combine阶段，所有需要返还发送的sendCnt数为最后一个值12，其中有0+5个token返还给卡1。

#### 数据位置索引路由
在expandX缓冲区中，Token按顺序存储，其顺序与assistInfoForCombine中的信息一一对应。

每个token对应的三元组信息 [epRankId, tokenIndex, topKIndex] 其含义如下：

- **epRankId**：该token来自哪张卡
- **tokenIndex**：该token是原始BS中的第几个Token
- **topKIndex**：该token是原始TopK中的第几个值，对于共享专家，即为第几个共享专家

根据三元组信息计算GM地址，Token发送阶段通过MTE搬运，直接将Token返还给来源卡，并将其写入目标卡对应tokenIndex、topKIndex的空间中。

**Combine数据分配表**：

**BS行 × (K+sharedExpertNum)列矩阵**

### MoE专家区 (K列)
| 行/列 | Col₁ | Col₂ | ... | Col₍K₎ |
|:-----:|:----:|:----:|:---:|:------:|
| Token₁ | data | data | ... | data |
| Token₂ | *here* | data | ... | data |
| ... | ... | ... | ... | ... |
| Token₍BS₎ | data | data | ... | data |

### 共享专家区 (sharedExpertNum列)
| 行/列 | Col₍K+1₎ | Col₍K+2₎ | ... | Col₍K+SN₎ |
|:-----:|:--------:|:--------:|:---:|:---------:|
| Token₁ | data | data | ... | data |
| Token₂ | data | data | ... | data |
| ... | ... | ... | ... | ... |
| Token₍BS₎ | data | data | ... | data |

**行数：BS行（每个token一行）**
**列数：K + sharedExpertNum列（K列MoE + SN列sharedExpertNum）**

若某个Token的三元组信息为[0,1,0]则表示它来自0卡的第1个token，且其topK下标为0，因此需要填入0卡combine数据window区的 *here* 里（见上表格）。对于共享专家卡，需要将其返还给来源rank，填入对应的共享专家区域。

## 3.3 Status发送与等待

### 3.3.1 设计背景

在每一个token次发送任务下发后，会紧接着发起目标位置的状态位发送，状态区域分布与数据区完全一致。

### 3.3.2 实现方案

**Combine状态分配表**：

**BS行 × (K+sharedExpertNum)列矩阵**

### MoE专家区 (K列)
| 行/列 | Col₁ | Col₂ | ... | Col₍K₎ |
|:-----:|:----:|:----:|:---:|:------:|
| Token₁ | status | status | ... | status |
| Token₂ | *here* | status | ... | status |
| ... | ... | ... | ... | ... |
| Token₍BS₎ | status | status | ... | status |

### 共享专家区 (sharedExpertNum列)
| 行/列 | Col₍K+1₎ | Col₍K+2₎ | ... | Col₍K+SN₎ |
|:-----:|:--------:|:--------:|:---:|:---------:|
| Token₁ | status | status | ... | status |
| Token₂ | status | status | ... | status |
| ... | ... | ... | ... | ... |
| Token₍BS₎ | status | status | ... | status |

**行数：BS行（每个token一行）**
**列数：K + sharedExpertNum列（K列MoE + SN列sharedExpertNum）**

在发送完某个三元组信息为[0,1,0]的Token数据后，因此需要同步发送状态填入0卡combine状态window区的 *here* 里（见上表格），表示此token到达标识。

**同步机制设计**：
- 接收端采用分核循环等待策略：
  - 将BS个Token平均分配给各计算核
  - 每个核负责若干token的接收状态查询
  - 轮询Flag值的算数和直至刷新为K + sharedExpertNum，确认数据接收完成

## 3.4 加权求和（Sum）

### 3.4.1 设计背景

在MoE架构中，每个Token被分发至K个专家与S个共享专家进行处理，Combine阶段需要：

- **数据整合**：将K倍于原始输入的专家输出Token整合还原
- **加权计算**：基于各专家权重系数进行加权求和
- **格式还原**：恢复至Dispatch输入的原始数据格式

由于topK和BS信息在数据区排布中为天然的下标信息，可以省去索引路由的步骤，直接进行计算，大大减少token元数据计算耗时。
### 3.4.2 实现方案

**分核并行计算：**
- **核分配**：BS个token平均分配给多个AIV计算核
- **核内计算**：每个核处理连续的token批次
- **核间独立**：各核独立计算，无数据依赖

**计算流程：**
1. **权重加载**：各核加载负责token对应的权重数据
2. **向量化计算**：核内并行计算多个token的加权和
3. **结果写回**：各核将计算结果写入对应GM地址

**核内伪代码：**
```cpp
// 每个AIV核独立执行
int tokens_per_core = BS / num_aiv;
int start = aiv_id * tokens_per_core;
int end = start + tokens_per_core;

for (int t = start; t < end; t++) {
    float sum = 0.0f;
    // MoE专家加权
    for (int m = 0; m < K; m++) {
        sum += combine_data[t][m] * weights[t][m];
    }
    // 共享专家加权
    for (int s = 0; s < sharedExpertNum; s++) {
        sum += combine_data[t][K + s] * shared_weights[t][s];
    }
    output[t] = sum;
}
```
**加权求和公式：**
$$
\text{Output}_i = \sum_{j=1}^{K} w_{ij} \cdot \text{MoE}_{ij} + \sum_{k=1}^{S} w'_{ik} \cdot \text{Shared}_{ik}
$$

其中：
- $i$ 表示第 $i$ 个token
- $K$ 表示MoE专家数量
- $S$ 表示共享专家数量（sharedExpertNum）
- $w_{ij}$ 表示第 $i$ 个token在第 $j$ 个MoE专家上的权重
- $w'_{ik}$ 表示第 $i$ 个token在第 $k$ 个共享专家上的权重
- $\text{MoE}_{ij}$ 表示第 $i$ 个token经过第 $j$ 个MoE专家处理后的输出
- $\text{Shared}_{ik}$ 表示第 $i$ 个token经过第 $k$ 个共享专家处理后的输出

