# Racecheck 重点诊断

在完成报告来源和可信度核对后，遇到访问竞争或竞争分析不完整时读取本页。
结论限定为同一次下发内的受支持访问，不由一次数值正确推断没有竞争。

## 先核对冲突访问对

| 典型诊断或场景 | 含义 | 必看证据与误读边界 |
| --- | --- | --- |
| `RAW` / `WAR` race | 读写访问在重叠字节上缺少必要顺序 | 同时核对 current/previous 两方的读写类型、空间、区间、核、block、流水及 launch，至少一方写；报告提供源码或 PC 时一并核对，标题不能代替访问对证据 |
| `WAW` race | 两次写在重叠字节上缺少必要顺序 | 确认两方确为写且访问同一存储对象；不能把不同核的私有片上地址仅因偏移相同视为共享对象 |
| `ATOMIC_SEMANTIC` race | 两次不兼容的原子写在重叠范围缺少必要顺序 | 既有 GM 等受支持原子路径中，相同 ADD/MIN/MAX 模式的原子写之间兼容；不同模式仍需核对同步，不能概括为“用了原子就安全”或外推未知 UB 原子语义 |
| 原子写与普通访问冲突 | 原子性不替代原子写与普通读写之间的必要顺序 | 原子写与普通写可报 WAW，与普通读可报 RAW/WAR；不能将它们全部解释成 ATOMIC_SEMANTIC |
| 使用 unitFlag、局部 barrier 或跨核同步后仍报冲突 | 所用同步未必保护这对访问及当前轮次 | 核对同步覆盖的空间、参与核、流水、范围与轮次。例如 L0C 的 unitFlag 交接不能直接证明 FIX 的 GM 写已完成；组内同步也不能自动保护跨组共享 GM 访问 |

报告中 `overlap [begin, end), N bytes` 表示冲突的重叠范围，N 不是任一完整访问的长度。
对已有版本的 RAW/WAR 方向差异，保留原始标题，先核对两方访问和重叠证据；不要仅凭 current/previous
标签推断源码发射先后或硬件实际完成顺序，也不要擅自将错误空间、错误访问对解释成已知方向差异。

同址多轮复用时，仅有核、流水、范围等描述可能无法唯一定位某次动态指令或某一轮。
缺少可区分身份时只能列出可疑访问对，不声称已经精确归因最后一轮。
同一范围仅报告首次确认的竞争，错误条数也不能解释为所有动态冲突次数或独立根因数。
发生在两次访问之后的 barrier 可以等待已有访问完成，但不能倒补为这两次访问之间的顺序。

## UB 与 SIMD VF 的潜在竞争

普通 SIMD 和 SIMD VF 的 UB 潜在 RAW/WAR/WAW **使用 WARNING 报告**。
未精确建模寄存器依赖的完成顺序，可能误报；交付为“已建模同步下的潜在冲突”，不能仅凭标题认定确定硬件冲突。
不扩展到 SIMT、语义未知的 UB 原子操作或跨 launch 竞争；既有 ADD/MIN/MAX 兼容规则不能无条件套到 UB 原子访问。

核对双方 `executor`（执行核）与 `memoryOwner`（实际存储核）。只比较同一物理 UB 的实际重叠字节，
不同私有 UB 偏移相同不代表同一存储；跨核访问应按明确的 owner 归属判断，不能用 executor 代替缺失 owner。

VF 队列串行不代表 VF 与异步搬运有序，也不使 VF 内所有 vector 访问按展开序号串行完成。
VF 的共同 Entry/Completion 只提供边界，内部访问需要相应局部 membar 等同步建立先后；VF scalar 的保序规则也不能
代替 vector 的局部屏障。核对局部屏障的 src/dst 访问类别和外层 SET/WAIT 是否保护这对访问与当前复用轮次。
VF 展开完成回调表示解析结束，不等于设备执行已完成；回调顺序、ID 数值或展开序号不能当作完成顺序。

| 场景 | 判读 |
| --- | --- |
| `MASK_NOT_REFRESHED`，访问记录仍标为 `EXACT` | mask 未知时按全 mask 保守展开；warning 本身不使分析 incomplete，EXACT 不否定 mask 假设。依赖 mask 的访问范围可能被扩大，应核对实际 mask 后再判断潜在冲突 |
| 缺少 PUSH_PB、PB 超限、展开失败或访问/屏障/owner 缺失 | 相关分析不完整，不能解释为该 VF 没有访问或没有竞争；保留具体失败证据 |
| completion 的 access/event/raw 计数不相等 | 合法零字节记录过滤可造成差异，不能只因不相等判丢失；按各计数语义及片段完整性核对，成功空 VF 与展开失败分开 |

地址模式与范围见 [UB 检查](ub-checks.md)。缓冲区复用还须等待上一轮读取完成，正向生产到消费的同步不自动证明复用安全。

## 分析受限与竞争诊断分开

| 原因或典型文本 | 如何解释与继续核对 |
| --- | --- |
| `atomic_configuration_unknown` | 原子配置无法识别，相关分析覆盖不完整；既不是确认发生竞争，也不是确认原子写安全 |
| `incomplete_memory_observation`、`sync_coverage_unknown` | 访问或同步证据不完整；核对丢失、跳过及具体原因，不用 `errors=0` 代替完整性结论 |
| `host_memory_allocation_failed` | 检查器分析所需的 Host 内存申请失败，不等于被测算子的 GM 分配失败或发生竞争 |
| `capacity_or_count_overflow`、`invalid_or_incomplete_analysis`，并提示 launch analysis stopped | 分析器容量、计数或输入问题导致该 launch 分析停止、结果不完整；核对失败阶段及明细，不能与 synccheck 的设备同步 `COUNTER_OVERFLOW` 混为一谈 |

分析停止时，按实际报告说明哪些结果被丢弃、哪些局部诊断仍有充分证据；不从未输出竞争诊断推导通过，
也不把分析器失败当成设备死锁。遇到未知原因保留原文与版本，不强行套用上述类别。

## 资料与用例依据

- [访问对与空间对照](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/racecheck/buffer_hazards/README.md)：RAW/WAR/WAW、片上核隔离及报告方向限制。
- [典型竞争及同步范围用例](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/racecheck/typical/README.md)：unitFlag、跨组同步、轮次和访问描述不能唯一归因等场景。
- [原子与历史访问用例](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/racecheck_v2/README.md)及[使用手册](https://gitcode.com/cann/asc-tools/blob/master/docs/zh/02_npu_check.md)：既有原子与历史访问语义。
