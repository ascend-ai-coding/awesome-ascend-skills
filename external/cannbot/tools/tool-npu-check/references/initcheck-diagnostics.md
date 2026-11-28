# Initcheck 重点诊断

在完成报告来源和可信度核对后，遇到未初始化读取、寄存器依赖或初始化分析不完整时读取本页。
初始化结论描述已知写入及其同步可见性，不能替代地址检查或数值验证。

## SIMD UB 的实际字节诊断

支持普通 SIMD 与 SIMD VF 的 UB 初始化检查，不支持 SIMT VF。
`Uninitialized UB memory read` 与部分未初始化报告按实际读取字节解释；
`UB address space: [0x0,0x40000)` 是完整 UB 状态表范围，不是动态区、逻辑 buffer 或当前读取长度。

**预留区、跨分区、SIMD 禁止区中的实际访问也跟踪初始化。** memcheck 报 UB 跨区或缺少布局，不抑制实际 UB 写入
及合法 GM 目标写入。布局未知本身不令 initcheck 不完整；不能把 memcheck 的不完整状态机械复制给 initcheck。
真实访问或同步缺失仍需保留分析限制，物理范围外或不可表示的记录也不能用于更新字节状态。

例如，A 未初始化，搬到 B 时可报告读取 A 未初始化；B 的实际写入仍建立初始化证据，后续有正确可见性保证的读取 B
可以不再报错。这不传播未初始化污点，也不证明 B 的数值正确。stride 空洞未写，不因其他字节写过而初始化。
不同逻辑存储核、设备及 launch 的 UB 状态不能互相补证，完整规则见 [UB 检查](ub-checks.md)。

## 内存与寄存器诊断

| 典型输出 | 含义 | 必看证据与误读边界 |
| --- | --- | --- |
| `Uninitialized ... memory read` | 本次读取范围没有已知且可见的初始化写入 | 核对空间、访问起点、读取范围、所属核/分配、launch 与写入来源；malloc 成功不等于内容已初始化 |
| `Partially uninitialized ... memory read` | 本次实际读取的字节中，只有一部分具有初始化证据 | 区分 `Read size`、`uninitialized`、`first uninitialized byte`。`initialized bytes` 是所属分配或状态表的初始化统计，UB 按完整地址空间解释；不必等于本次读取总长度减缺失长度 |
| `Register ... was used before initialization` | 报告中的消费指令需要的持久化寄存器，使用前没有有效设置 | 读取寄存器名、consumer 指令、源码或 PC、核和流水。显式设置为 0 也属于设置；只检查当前模式/有效轴实际需要的寄存器，不能要求所有寄存器无条件设置 |

例如，`Read size: 128 bytes; uninitialized: 64 bytes` 表示本次读取缺少 64B 初始化证据，
不是越界 64B，也不表示整个 allocation 只初始化了 64B。
中间空洞或 stride 跳过的字节必须按真实读取集合判断，不能仅用首尾地址包住的总区间代替。

检查“代码里已经写过”时，应核对写入是否覆盖本次读取、是否属于同一分配/核与有效状态，以及同步关系是否证明
写入对消费指令可见。后面的设置、另一个核的私有片上写入，或无关 allocation 的初始化，不能替当前读取补证。

## 分析不完整不等于已确认未初始化

| 原因或迹象 | 如何解释与继续核对 |
| --- | --- |
| `unordered_initialization` | 无法证明候选写入先于读取；核对实际读写位置、同步方向和范围，不能直接说没有发生写入或已经通过 |
| `memory_coverage_unknown` 等访问覆盖缺口 | 实际访问范围或相关观察不完整；保留具体原因，不能据零条确定错误宣称初始化完整 |
| `unit_flag_initialization_unsupported` | 该报告所属版本不能完整处理相应 unitFlag 路径，不是内存一定未初始化；核对实际安装包，不能据此推断其他版本的支持情况 |

同流水或跨流水“已经有同步”不代表全部字节已初始化；同步只建立受其范围保护的顺序，不产生数据写入。
日志可能有局部明确诊断同时存在分析缺口，应分别交付，不把全部内容一律视为无效或完整通过。

## 资料与用例依据

- [GM 初始化用例](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/initcheck/gm_uninitialized/README.md)：`middle_hole`、`offset_read`、`strided_read/strided_holes` 和独立分配对照。
- [寄存器依赖用例](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/memcheck/memory_access/check_memory_access_end_to_end.sh)：缺少 `nddma_loop0_stride` 的消费指令及诊断。
- [L0A/L0B 用例](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/initcheck/l0ab_memory/README.md)、[unitFlag 版本回归](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/initcheck/unitflag/README.md)：顺序/范围不完整与版本差异。

UB 初始化规则见 [UB 检查](ub-checks.md#simd-ub-的-initcheck-独立规则)。结合实际访问字节、同步可见性和报告上下文解读诊断。
