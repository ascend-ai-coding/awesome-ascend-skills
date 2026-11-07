# MC2 领域代码检视规范

## 适用与分工
<a id="mc2-routing"></a>
<a id="mc2-conventions"></a>

用于检视 MC2 通算融合及相关接口、Tiling、Kernel、图融合、构建和测试。根据注册、调用或构建关系确认代码是否属于 MC2，不能只看目录名或同名 API。与 MC2 没有这些关系的独立计算或通信实现不适用。

helper 或绑定代码中没有触发关键词时，使用任务输入中已有的关联依据，或概要现有的调用关系图、跨文件防御摘要和跨文件关联，确认它与 MC2 的关系。依据须包含关联文件、关系和证据位置。用户说明保留来源及确认状态，不能写成执行者已读取代码核实。

计划设计按主题文件的“关联适用”声明匹配这些依据，再按侧别和实际内容筛选条款，并将依据随计划传给条款执行者，无需增加概要字段。关键词只用于定位候选代码；归属尚未确认时，暂不启用 MC2 条款，注明待核实及缺少的材料。

选规所需的归属条件写在各主题文件的适用声明中，执行所需的前提写在对应条款内。文件、PR 和快速检视均可按现有流程读取单条章节，总览用于导航和补充说明。

只在任务允许的文件和概要范围内收集证据。全量检视检查当前实现，PR 检视检查变更影响，快速检视根据用户的问题选择规则。Host 包括 Tiling；Python、构建脚本和测试代码根据它们影响的实现选择规则。对于非 MC2 使用方，只检查 [MC2-9.8](#mc2-9-8) 所述公共修改带来的影响。

通用检查使用下表中的规则；MC2 条款进一步检查参数如何传递、通信执行顺序和资源生命周期。条款中的交叉引用只用于查阅，不会自动触发任务派发或报告合并。

| 通用检查 | 已有规范 |
|---|---|
| 越界、除零、溢出、未初始化、数据竞争 | [红线](ascendc-red-line.md) |
| 指针、字符串、资源、ABI、日志 | [安全编码](cpp-secure.md) |
| 返回值、属性类型、输入校验、核间同步与调度 | [TOPK](ascendc-topk.md) |
| DataCopy 与 Kernel API、废弃头文件和库 | [Kernel 规范](ascendc-kernel-rules.md)、[日落 API](ascendc-sunset-api.md) |
| 注册与模板语法、SIMT 接口 | [Host 规范](ascendc-host-rules.md)、[Kernel 规范](ascendc-kernel-rules.md)、[SIMT 规范](ascendc-simt.md) |
| 命名、常量、注释代码 | [代码风格](cpp-style.md) |

先确认目标 CANN 版本、设备、实际调用的重载和构建条件。优先使用已有 API 预研结果；未覆盖的接口用 `/ascendc-docs-search` 查询，并结合本地头文件核实。示例中的模板参数、传输上限和版本阈值，只适用于示例说明的前提。

判定和报告格式沿用 [方法论](../core/methodology.md)。上游校验必须检查风险操作实际使用的值，且该值在使用前没有改变；已有有效校验时，无需在下游重复添加。缺少材料或其他检视组提供的依赖信息时，注明待核实。搜索不到结果不能证明实现不存在，没有运行验证也不能写成验证通过。修复时要保留合法默认值、接口支持范围、精度要求和通信副作用。

## 术语
<a id="mc2-terms"></a>

| 术语 | 含义 |
|---|---|
| MC2 / MC² | Matrix Computation & Communication，通算融合，将集合通信与计算融合到一个算子中。 |
| 通算融合 | 通信与计算融合（集合通信 + Cube/Vector 计算），实现流水线并行。 |
| Cube / AIC | 矩阵计算单元（AI Core Cube），执行 Matmul/GroupedMatmul 等。 |
| Vector / AIV | 向量计算单元（AI Core Vector），执行 Add/Mul/Cast/Reduce 等向量操作。 |
| UB | Unified Buffer，Vector 核本地存储，AIV 侧数据搬入/搬出的中间缓冲区。 |
| V核 | 即 AIV（Vector 核）。 |
| SyncAll | 核间同步接口；参与核范围按实际重载和目标设备核对。 |
| SyncFunc | 流水事件同步封装，如 `SyncFunc<HardEvent::S_MTE3>()`，具体语义以所用实现为准。 |
| SetFlag / WaitFlag | 核内流水事件同步接口。SetFlag 设置事件，WaitFlag 等待事件；配对的事件类型和 EVENT_ID 必须一致。 |
| HardEvent | 硬件同步事件类型，标识源流水与目标流水，用于 SetFlag/WaitFlag，如 `S_MTE3`（Scalar→MTE3）、`MTE2_V`（MTE2→Vector）。 |
| PipeBarrier | 核内流水屏障，按 pipe 参数等待所选流水的前序读写完成，如 `PipeBarrier<PIPE_V>()`。 |
| Pipe | Ascend C 流水线，包括 PIPE_S（标量）、PIPE_V（向量）、PIPE_M（Cube）、PIPE_MTE2/MTE3（数据搬运）等。 |
| TP | Tensor Parallelism，张量并行，沿张量维度切分计算。 |
| EP | Expert Parallelism，专家并行，将 MoE 专家分配到不同参与者上。 |
| SP | Sequence Parallelism，序列并行，沿序列维度切分数据。 |
| 大EP | 大规模跨节点专家并行，dispatch-combine 多轮通信与计算模式。 |
| CCU | Collective Communication Unit，集合通信处理器，用于执行集合通信任务。 |
| AICPU | AI CPU，在通信场景中可承担通信算法编排和任务提交。 |
| HCCL | Huawei Collective Communication Library，昇腾集合通信库。 |
| HCOMM | HCCL 通信基础库，提供控制面和数据面接口。 |
| PTA | PyTorch Adapter，PyTorch 的昇腾适配层。 |
| 集合通信原语 | AllGather、AllReduce、ReduceScatter、AllToAll/AlltoAllV、Broadcast 等标准通信操作。 |
| Tiling | 算子参数配置流程，将计算参数和通信参数打包传递给 Kernel。 |
| Mc2InitTiling | 由 GetTiling 返回的通信任务初始化配置。 |
| Mc2CcTiling | 由 GetTiling 返回的各通信域中每个通信任务的参数配置。 |
| EnQue / DeQue | TPipe 队列的入队与出队操作，提供队列管理和相应的流水同步。 |
| Double Buffer | 双缓冲流水线模式，交替使用两个 buffer 实现搬入、计算与搬出重叠。 |
| 通信域与 rank | 通信域确定一组通信参与者；rank 是参与者在该通信域内的编号。读取 rank 或通信域大小时，要确认使用的是哪个通信域，不能混用 EP 和 TP 的值。 |
| device ID 与 rank | device ID 标识设备，rank 标识通信域内的参与者；两者不能直接互换，须核对实际映射。 |
| 全局与本地 expert ID | 全局 expert ID 标识完整专家集合中的专家，本地 expert ID 用于索引当前参与者上的专家数组。两者按实际专家分布转换，只有连续分配时才能用减去起始编号的方式转换。 |

## 按主题阅读

| 内容 | 条款 | 文件 |
|---|---|---|
| 原型、传参、推导、Tiling、尾块与 SIMT | MC2-1、MC2-2 | [接口与 Tiling](mc2-interface-tiling.md) |
| 同步、HCCL、Matmul、MoE、量化与性能 | MC2-3 至 MC2-6 | [通信与计算](mc2-communication-compute.md) |
| workspace、异步资源、请求状态及失败恢复 | MC2-7、MC2-8 | [资源与错误处理](mc2-resources-errors.md) |
| 平台、图融合、构建、迁移、示例与测试 | MC2-9、MC2-10 | [工程集成与测试](mc2-integration-tests.md) |

各文件的语言、侧别、启用条件、容量和条款数见 [MC2 主题文件表](README.md#mc2-主题文件)。按主题文件的适用声明和快速索引选择条款，再阅读对应的正反例与检视方法。本页不参与条款派发。

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---|---|---|---|---|
| <a id="mc2-1-1"></a><a id="mc2-1-3"></a>MC2-1.1 | [原型、属性读取和接口传参必须一致](mc2-interface-tiling.md#mc2-1-1) | API使用 | 高 | [适用: Host] |
| <a id="mc2-1-2"></a><a id="mc2-1-4"></a>MC2-1.2 | [参数校验和修改必须在实际调用入口生效](mc2-interface-tiling.md#mc2-1-2) | API使用 | 高 | [适用: Host] |
| <a id="mc2-1-5"></a><a id="mc2-1-7"></a><a id="mc2-7-9"></a>MC2-1.5 | [tensor 的类型、布局和传入地址必须与实际存储一致](mc2-interface-tiling.md#mc2-1-5) | 类型安全 | 高 | [适用: All] |
| <a id="mc2-1-6"></a>MC2-1.6 | [输出 shape 和 dtype 的推导必须覆盖所有支持的输入](mc2-interface-tiling.md#mc2-1-6) | API使用 | 高 | [适用: Host] |
| <a id="mc2-2-1"></a><a id="mc2-6-2"></a>MC2-2.1 | [每种支持模式都必须能选中对应的 Kernel 实现](mc2-interface-tiling.md#mc2-2-1) | Tiling设计 | 高 | [适用: All] |
| <a id="mc2-2-2"></a><a id="mc2-8-5"></a>MC2-2.2 | [Tiling 和任务参数必须完整传到执行端](mc2-interface-tiling.md#mc2-2-2) | Tiling设计 | 高 | [适用: All] |
| <a id="mc2-2-3"></a><a id="mc2-2-4"></a>MC2-2.3 | [分块和尾块处理不能遗漏数据或越界](mc2-interface-tiling.md#mc2-2-3) | Tiling设计 | 高 | [适用: All] |
| <a id="mc2-2-5"></a>MC2-2.5 | [SIMT 边界检查不能让必需的同步参与线程提前退出](mc2-interface-tiling.md#mc2-2-5) | 并发安全 | 高 | [适用: Kernel] |
| <a id="mc2-3-1"></a><a id="mc2-3-8"></a><a id="mc2-3-3"></a>MC2-3.1 | [跨核读取和缓冲复用必须等待前序操作完成](mc2-communication-compute.md#mc2-3-1) | 通信同步 | 高 | [适用: Kernel] |
| <a id="mc2-3-2"></a>MC2-3.2 | [流水事件必须与读写依赖正确配对](mc2-communication-compute.md#mc2-3-2) | 通信同步 | 高 | [适用: Kernel] |
| <a id="mc2-3-4"></a><a id="mc2-5-3"></a>MC2-3.4 | [同一通信域内的 rank 必须按协议参与通信](mc2-communication-compute.md#mc2-3-4) | 通信同步 | 高 | [适用: All] |
| <a id="mc2-3-5"></a>MC2-3.5 | [AlltoAllV 各 rank 间的收发量必须配对，访问范围不能越界](mc2-communication-compute.md#mc2-3-5) | 通信同步 | 高 | [适用: All] |
| <a id="mc2-3-6"></a>MC2-3.6 | [HCCL 调用必须符合目标版本的接口要求](mc2-communication-compute.md#mc2-3-6) | API使用 | 高 | [适用: All] |
| <a id="mc2-3-7"></a>MC2-3.7 | [通信结果必须在完成后读取，通信对象必须在使用结束后销毁](mc2-communication-compute.md#mc2-3-7) | 资源管理 | 高 | [适用: All] |
| <a id="mc2-4-1"></a>MC2-4.1 | [AllGatherMatmul 输出必须与各 rank 的数据位置对应](mc2-communication-compute.md#mc2-4-1) | 数据搬运 | 高 | [适用: All] |
| <a id="mc2-4-2"></a>MC2-4.2 | [MatmulReduceScatter 不能遗漏 rank 的计算结果或改变运算顺序](mc2-communication-compute.md#mc2-4-2) | 精度保护 | 高 | [适用: All] |
| <a id="mc2-4-3"></a>MC2-4.3 | [通信计算重叠优化必须验证结果正确性和性能收益](mc2-communication-compute.md#mc2-4-3) | 性能优化 | 中 | [适用: Kernel] |
| <a id="mc2-4-4"></a>MC2-4.4 | [性能统计必须避免重复计数，并正确换算单位](mc2-communication-compute.md#mc2-4-4) | 性能优化 | 中 | [适用: Kernel] |
| <a id="mc2-5-1"></a>MC2-5.1 | [专家索引必须使用正确的编号，且不能超出本地数组范围](mc2-communication-compute.md#mc2-5-1) | 内存安全 | 高 | [适用: All] |
| <a id="mc2-5-2"></a>MC2-5.2 | [Combine 必须按 Dispatch 的映射和权重还原 token](mc2-communication-compute.md#mc2-5-2) | 数据搬运 | 高 | [适用: All] |
| <a id="mc2-6-1"></a>MC2-6.1 | [量化数据与 scale 的类型和分组必须对应](mc2-communication-compute.md#mc2-6-1) | 类型安全 | 高 | [适用: All] |
| <a id="mc2-6-3"></a><a id="mc2-6-4"></a>MC2-6.3 | [量化公式、计算精度和类型转换必须符合模型要求](mc2-communication-compute.md#mc2-6-3) | 精度保护 | 高 | [适用: All] |
| <a id="mc2-7-1"></a><a id="mc2-7-2"></a>MC2-7.1 | [workspace 必须分配足够空间，资源用量不能超过上限](mc2-resources-errors.md#mc2-7-1) | 内存安全 | 高 | [适用: All] |
| <a id="mc2-7-3"></a>MC2-7.3 | [拆分后的每次通信都必须满足后端的传输上限](mc2-resources-errors.md#mc2-7-3) | 资源管理 | 高 | [适用: Host] |
| <a id="mc2-7-4"></a><a id="mc2-7-5"></a>MC2-7.4 | [异步资源必须由 owner 持有，直到最后一次使用完成](mc2-resources-errors.md#mc2-7-4) | 资源管理 | 高 | [适用: All] |
| <a id="mc2-7-6"></a><a id="mc2-7-10"></a><a id="mc2-8-6"></a>MC2-7.6 | [每次请求的查询和执行必须使用同一份配置](mc2-resources-errors.md#mc2-7-6) | 并发安全 | 高 | [适用: All] |
| <a id="mc2-7-7"></a><a id="mc2-7-8"></a>MC2-7.7 | [容量、地址和分块计算不能溢出或除零](mc2-resources-errors.md#mc2-7-7) | 数值安全 | 高 | [适用: All] |
| <a id="mc2-8-1"></a><a id="mc2-8-4"></a>MC2-8.1 | [调用失败时必须返回错误，并准确记录原因](mc2-resources-errors.md#mc2-8-1) | API使用 | 高 | [适用: All] |
| <a id="mc2-8-2"></a><a id="mc2-8-3"></a>MC2-8.2 | [替换错误处理宏不能改变判断逻辑，release 中的校验必须有效](mc2-resources-errors.md#mc2-8-2) | 输入验证 | 高 | [适用: All] |
| <a id="mc2-8-7"></a><a id="mc2-8-8"></a>MC2-8.7 | [部分提交失败后，必须保留任务状态并正确处理重试和 fallback](mc2-resources-errors.md#mc2-8-7) | 资源管理 | 高 | [适用: All] |
| <a id="mc2-9-1"></a>MC2-9.1 | [平台配置、注册和构建产物必须对应同一目标平台](mc2-integration-tests.md#mc2-9-1) | 编译配置 | 高 | [适用: All] |
| <a id="mc2-9-2"></a>MC2-9.2 | [跨代际依赖必须符合当前项目的明确限制](mc2-integration-tests.md#mc2-9-2) | 编译配置 | 高 | [适用: All] |
| <a id="mc2-9-4"></a><a id="mc2-9-5"></a>MC2-9.4 | [图融合必须满足实现的支持条件，并保留输出和外部依赖](mc2-integration-tests.md#mc2-9-4) | 精度保护 | 高 | [适用: Host] |
| <a id="mc2-9-6"></a>MC2-9.6 | [图变换失败后必须恢复有效图，或禁止继续使用该图](mc2-integration-tests.md#mc2-9-6) | 资源管理 | 高 | [适用: Host] |
| <a id="mc2-9-7"></a><a id="mc2-10-1"></a><a id="mc2-9-3"></a>MC2-9.7 | [构建产物和发布包必须包含所需实现、依赖和注册信息](mc2-integration-tests.md#mc2-9-7) | 编译配置 | 高 | [适用: All] |
| <a id="mc2-9-8"></a>MC2-9.8 | [修改公共组件时必须检查受影响的调用者](mc2-integration-tests.md#mc2-9-8) | API使用 | 高 | [适用: All] |
| <a id="mc2-10-2"></a>MC2-10.2 | [接口迁移必须实现当前需求，并保留调用者依赖的行为](mc2-integration-tests.md#mc2-10-2) | API使用 | 中 | [适用: All] |
| <a id="mc2-10-3"></a><a id="mc2-10-6"></a>MC2-10.3 | [示例必须具备运行依赖，并启动全部通信 rank](mc2-integration-tests.md#mc2-10-3) | API使用 | 中 | [适用: All] |
| <a id="mc2-10-4"></a>MC2-10.4 | [测试 mock 不能替代被测逻辑，全局状态必须正确恢复](mc2-integration-tests.md#mc2-10-4) | 资源管理 | 高 | [适用: All] |
| <a id="mc2-10-5"></a>MC2-10.5 | [测试必须能发现目标错误，并验证合法边界仍能通过](mc2-integration-tests.md#mc2-10-5) | 输入验证 | 高 | [适用: All] |
| <a id="mc2-10-7"></a>MC2-10.7 | [worker 的失败和异常必须传到调用者及启动器](mc2-integration-tests.md#mc2-10-7) | API使用 | 高 | [适用: All] |

## 参考资料
<a id="mc2-sources"></a>

以下固定 revision 用于定位资料；检视时核对待审源码的实际版本。

| 内容 | 来源与适用口径 |
|---|---|
| AllGatherMatmulV2 算法与量化表示 | [接口文档](https://gitcode.com/cann/ops-transformer/blob/be415abc6e045dc5eb3eca5a4da557b8d4262a5a/mc2/all_gather_matmul_v2/docs/aclnnAllGatherMatmulV2.md)；MC2-4.1 的小矩阵为数学教学模型 |
| MatmulReduceScatterV2 算法顺序 | [接口文档](https://gitcode.com/cann/ops-transformer/blob/be415abc6e045dc5eb3eca5a4da557b8d4262a5a/mc2/matmul_reduce_scatter_v2/docs/aclnnMatmulReduceScatterV2.md)；MC2-4.2 的示例假设 SUM 和沿 M 分片，不泛化支持范围 |
| rank 输出偏移与 scale 独立等待 | [perblock Kernel](https://gitcode.com/cann/ops-transformer/blob/be415abc6e045dc5eb3eca5a4da557b8d4262a5a/mc2/all_gather_matmul_v2/op_kernel/arch35/all_gather_quant_bmm_perblock.h)；实际分支及设备适用性另查当前构建 |
| HCCL / HCOMM 库说明 | [官方说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/api_ref/header_and_lib.md)，CANN 社区版 9.1.0；目标版本不同须重新查询 |
| CCU 通信模型 | [官方说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/commopdev/docs/zh/comm_op_dev_guide/prog_models_concepts/comm_model.md)，CANN 社区版 9.1.0；传输限制按目标后端和接口核对 |
| AICPU 通信引擎 | [官方说明](https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclopdev_000005.html)，CANN 商用版 8.5.0；不据引擎名称推定无传输限制 |
| PTA 名称 | [PyTorch 模型迁移和训练指南](https://www.hiascend.com/doc_center/source/zh/canncommercial/700/modeldevpt/ptmigr/CANN%207.0.0%20PyTorch%E6%A8%A1%E5%9E%8B%E8%BF%81%E7%A7%BB%E5%92%8C%E8%AE%AD%E7%BB%83%E6%8C%87%E5%8D%97%2001.pdf)，CANN 7.0.0，Memory View 说明中的 PyTorch Adapter |
| Mc2InitTiling / Mc2CcTiling | [TilingData 结构体](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/API/ascendcopapi/atlasascendc_api_07_10048.html)，CANN 社区版 8.2.RC1.alpha003；配置通过 GetTiling 获取，字段按目标版本核对 |
| 核间同步 | [SyncAll 接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md)，CANN 社区版 9.1.0；参与核范围和同步方式按实际重载及设备核对 |
| 核内流水事件同步 | [官方接口说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html)，CANN 社区版 9.0.0；事件编号和模式限制按目标版本核对 |
| 流水屏障 | [官方接口说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0271.html)，CANN 社区版 9.0.0；不据名称推定跨核或跨 rank 完成 |

## 旧编号迁移

旧版引用按检查内容迁移到主题文件。以下映射覆盖 HCCL 协同规则使用的旧编号，旧编号不再用于条款派发。

| 旧编号 | 原检查内容 | 当前条款 |
|---|---|---|
| MC2-02 | 流水事件配对 | [MC2-3.2](mc2-communication-compute.md#mc2-3-2) |
| MC2-04 | 集合通信参与顺序和参数一致性 | [MC2-3.4](mc2-communication-compute.md#mc2-3-4) |
| MC2-19 | AlltoAllV 跨 rank 收发量配对 | [MC2-3.5](mc2-communication-compute.md#mc2-3-5) |
| MC2-18 | HCCL 参数与通信生命周期 | [MC2-3.6](mc2-communication-compute.md#mc2-3-6)、[MC2-3.7](mc2-communication-compute.md#mc2-3-7) |
| MC2-13 | CCU 单次传输上限 | [MC2-7.3](mc2-resources-errors.md#mc2-7-3) |
