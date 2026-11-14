# apace 路线失败导航（Failure Navigation）

> 按**现象**定位排查方向。每行给出首个失败层分类与应查阅的文档锚点。
> 本表只做路由：根因细节、修复手法与红线定义以目标文档为准，不在此重复。

## 首个失败层分类

| 分类 | 含义 | 典型信号 |
|:---|:---|:---|
| 编译 | 编译/链接阶段失败，未进入设备执行 | 编译报错、API 符号缺失、模板实例化失败 |
| ABI | host/kernel 接口与二进制契约不一致 | 入口变体错配、dtype 分发错误、参数布局漂移、host 前置校验缺失导致非法配置下发 |
| 通信时序 | 通信与计算的先后/配对时序错误 | 读到未就绪数据、GET/PUT 可见性错乱、框架原语语义错配 |
| 同步 | CrossCore flag / 事件 / barrier 配对或计数错误 | 死锁（507015/507014）、flagId 溢出、删同步后偶发错误 |
| 设备精度 | 设备侧输出与 golden 不符 | matched_ratio 不达标、误差不收敛、部分输出为零 |
| 性能 | 精度已 PASS，性能不达标或数据不可信 | cube_utilization 低、归约耗时高、perf 数据不可复现 |
| 通信建链 | HCCL/URMA window 资源建链阶段失败 | Win 分配失败、rank 建链超时、CommContext 与引擎不匹配 |

## 失败导航表

| 现象 | 首个失败层 | 去哪里查 |
|:---|:---|:---|
| 死锁（aclError:507015） | 同步 | [`../review-checklist.md`](../review-checklist.md) R1/R5/R12；[`../fundamentals/architecture.md`](../fundamentals/architecture.md) §10 ①（禁 schedmode）；[`../fundamentals/communication.md`](../fundamentals/communication.md) 陷阱 #9（UB 静态区被 TPipe 覆盖）；[`../fundamentals/fusion.md`](../fundamentals/fusion.md) §5（MTE 未排空路径鉴别） |
| localMatmul=1 MTE 异常 | 同步 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §5.4（RunLocalMatmul 与 RunMatmul 之间补 `PipeBarrier<PIPE_ALL>()`） |
| 精度不达标 | 设备精度 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §5（AllToAll 语义）+ §6.2.6（归约 float32 累加）；[`../workflow_integration.md`](../workflow_integration.md) Step 6（精度验收流程）；[`../review-checklist.md`](../review-checklist.md) 常见 FAIL 原因 |
| golden 全错、误差不收敛 | 设备精度 | [`../workflow_integration.md`](../workflow_integration.md) Step 2 §golden 语义（先核对每卡输入/输出与切分轴，再怀疑 kernel）；[`../operator-design/host-and-testing.md`](../operator-design/host-and-testing.md)（gen_data 工程侧）；先修 golden 再怀疑 kernel |
| T=1 全 PASS、T>1 精度错 | 设备精度 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §3.3/§6.2.3（多 tile 布局与 flag 逐轮计数配对）；按 tile/轮次定位超差分布 |
| 编译失败 / API 符号缺失 | 编译 | [`../operator-design/development-guide.md`](../operator-design/development-guide.md)（工程搭建与依赖）；核对 CANN 内置 apace 事实源版本 |
| Blaze matmul 排错 | 编译 | [`../fundamentals/compute.md`](../fundamentals/compute.md) §8 排错速查；[`../review-checklist.md`](../review-checklist.md) R7（禁 `AscendC::Matmul` 高阶 API） |
| 通信时序错误 | 通信时序 | [`../fundamentals/communication.md`](../fundamentals/communication.md) 陷阱表；[`../fundamentals/fusion.md`](../fundamentals/fusion.md) §3（GET/PUT flag 编排不变量） |
| flagId 计数器溢出（硬件异常中断） | 同步 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §3.3/§6.2.3（flagId ∈ [0,15] 硬件数量上限 + Set/Wait 严格配对规避；计数深度按 R9 口径不设数值拒绝）；[`../review-checklist.md`](../review-checklist.md) R9 |
| tail tile 非对齐精度问题 | 设备精度 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §6.2.7（尾块策略 A：padding 对齐 + realFragmentSize 限读）；[`optimization-playbook.md`](optimization-playbook.md) §4 |
| 归约性能差 | 性能 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §6.2.6（多行批量归约 + 2D DataCopyPad）；[`optimization-playbook.md`](optimization-playbook.md) §2 |
| 死锁（aclError:507014，归约路径） | 同步 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §6.2.6 纪律 1 + [`../scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.3 事件模板（四类 HardEvent 同迭代配对 + 残留事件消费）；[`../review-checklist.md`](../review-checklist.md) R19 |
| E5M2 变体精度全错 | ABI | [`../review-checklist.md`](../review-checklist.md) R3；[`../operator-design/development-guide.md`](../operator-design/development-guide.md) §3.5（4 变体入口 + host 运行期 dtype dispatch）；[`../operator-design/operator-anatomy.md`](../operator-design/operator-anatomy.md) |
| N>redUbN 时部分输出为零 | 设备精度 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §6.2.6 纪律 3（2D DataCopyPad strided 隐式上限，redUbM ≤ 32 或 1D 退化） |
| 归约结果系统性错误 | 设备精度 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §6.2.6 纪律 2（禁 in-place BF16→FP32 Cast，独立 srcFP32 双缓冲）；[`../review-checklist.md`](../review-checklist.md) R18 |
| perf 数据不可复现 / MTE2 带宽虚高 | 性能 | [`../operator-design/host-and-testing.md`](../operator-design/host-and-testing.md) §4（L2 flush 实接线模板）；[`../review-checklist.md`](../review-checklist.md) R20 |
| 大 shape 无法运行 / 间歇 FAIL | ABI | [`../operator-design/development-guide.md`](../operator-design/development-guide.md) §3.5（host 前置校验：正确性类整除/对齐/核数/Win 容量；单轮 PUT 大小按 R14 风险提示复核，非强制拒绝；R ≤ 32 为 fragment 数上限，R×T 无硬约束）；[`../fundamentals/communication.md`](../fundamentals/communication.md) 陷阱 #12/#13 |
| cube_utilization 低 | 性能 | [`optimization-playbook.md`](optimization-playbook.md) §2（CUBE bound 判据与手法）；[`../fundamentals/fusion.md`](../fundamentals/fusion.md) §6.2.2（R×T 子调用 SCALAR 主 bound 特征） |
| GET 模式 4+rank 不稳定 | 通信时序 | [`../fundamentals/fusion.md`](../fundamentals/fusion.md) §3.4/§6.2（GET 环形回压契约与 4+rank 数据可见性风险，计算在前场景优先 PUT）；[`../fundamentals/communication.md`](../fundamentals/communication.md) GET 契约 |
| 仅非对齐 shape 精度 FAIL | 设备精度 | [`optimization-playbook.md`](optimization-playbook.md) §4（tail 路径排查模式）；[`../fundamentals/fusion.md`](../fundamentals/fusion.md) §6.2.7 |
| 精度误差随 K 放大 | 设备精度 | [`optimization-playbook.md`](optimization-playbook.md) §4（累加位宽不足，先高精度中间累加保底） |
| 重构后某维度性能退化 | 性能 | [`optimization-playbook.md`](optimization-playbook.md) §4/§2（数同步次数，批量摊薄） |
| 复用框架通信原语大面积错数据 | 通信时序 | [`optimization-playbook.md`](optimization-playbook.md) §4/§3（work partition 地址偏移语义核对，targetRank vs sourceRank） |
| 删减 SyncAll 后 NaN/偶发错误 | 同步 | [`optimization-playbook.md`](optimization-playbook.md) §4（同步是正确性保障不可裁剪）；[`../fundamentals/fusion.md`](../fundamentals/fusion.md) §3 |
| 性能优化无从下手/无收益 | 性能 | [`optimization-playbook.md`](optimization-playbook.md) §1（阶段路径，勿跳级）/§2（现象→手法速查）/§5（实验纪律） |
| 静态分析判定"不可行"但有争议 | 性能 | [`optimization-playbook.md`](optimization-playbook.md) §5（实证复核原则：最小实验复核，"原假设 vs 实测结果"对照归档） |
