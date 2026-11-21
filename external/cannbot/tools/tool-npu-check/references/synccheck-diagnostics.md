# Synccheck 重点诊断

在完成报告来源和可信度核对后，遇到配对、模式、计数或参与者诊断时读取本页。
以下跨核诊断延续既有同步检查语义；本技能的最新能力范围见能力范围参考，实际字段仍以目标版本输出为准。

## 核内配对

| 典型输出或场景 | 含义 | 下一步核对与误读边界 |
| --- | --- | --- |
| 重复 `SET_FLAG` / `GET_BUF`，`duplicate_opens` | 同一配对对象的前一次设置或获取尚未消费，又发生了设置或获取 | 核对实际执行路径、流水、event/buffer ID 和次数；源码中两侧总数相等，也不能排除次序违规 |
| `WAIT_FLAG` / `RLS_BUF` 无匹配前置操作，`unmatched_closes` | 实际等待或释放没有可匹配的设置或获取 | 核对 trigger、related/expected point、pair kind/key；源码里出现 SET/GET 不等于本次参数及执行记录匹配 |
| `SET_FLAG` / `GET_BUF` 未消费，`unconsumed_opens` | 结算时仍有未等待的设置或未释放的获取 | 区分未执行相应分支、次数不等和配对参数不同；不能只凭报错标题指定源码修复 |

## 跨核模式、计数与参与者

| 标识或典型输出 | 含义 | 必看证据与结论限制 |
| --- | --- | --- |
| `FLAG_MODE_SWITCH_BEFORE_COMPLETION` | 同一 participant、flag 的旧模式状态未完成，又尝试切换模式 | 核对 `participant`、`flagId`、`activeMode/newMode`、`pendingSetRounds` 和 `remainingCount`。旧状态完成要求 SET 聚合已结束且剩余计数已消费，不能只看其中一个为 0 |
| `COUNTER_OVERFLOW` | 完成的 SET 聚合试图使某个目标核、某个 flagId 的计数超出支持上限 | 同一次 launch 内按目标 participant 与有效 flagId 分别计数，不跨核、跨 flag 相加。核对 mode 及 `currentCount/attemptedCount/maximum`；15 合法，尝试增加到 16 才报错。SET 调用次数不等于当前未消费计数，详见下方示例 |
| `Synchronization error detected`，带 `expectedSet/actualSet/missingSet` | 某个 SET 聚合轮次缺少应到参与者 | 核对 mode、flag、group、round、direction、target 与参与集合。完全没有回调的核或组也可能属于 expectedSet；这不是“WAIT 参与者未齐”，不能仅据 missingSet 确认某源码分支漏执行 |
| `remaining counter token` | 已产生的目标计数在结算时仍未消费 | 核对 `CROSS_CORE_TOKEN`、participant、flag、mode、tokens 和 launch。与 SET 聚合未完成不同，报告条数也不等于剩余 token 数 |
| `wait without sync token` | 结算后，该 WAIT 仍没有可消费的同步通知 | 核对 WAIT 位置、规范化 pair key、参与者、group、flag 及模式/指令类别。不同组、AIC/AIV bank 或不兼容 WAIT 可能不匹配；不能仅按数字 flag 相同认定对象相同 |
| `Synchronization error detected`，原因涉及不支持的 mode/topology 或参数组合 | 同步参数、核角色或拓扑组合无效或不受支持 | 读取报告给出的模式条件和 observed topology；拓扑缺失时说明上下文不足，不一律解释成缺 SET，也不直接建议加 barrier |

### 为什么计数上限需要核对 participant 和 flagId

**15 是同一次 launch 内单个目标参与核、单个有效 flagId 的计数上限。** `participant` 确定目标核，
`flagId` 确定该目标核上的计数器；不同 flagId 或不同目标核的计数分别保存，不能把全部 SET 次数相加。
这里关注收到同步通知的目标核，而不只是发出 SET 的源核；使用报告中的规范化身份和有效 flagId 核对。

例如，在模式及参数合法、每轮 SET 聚合均已完成、期间没有 WAIT 消费的条件下：

| 同步对象 | 当前计数 | 是否超过 15 |
| --- | --- | --- |
| `participant={group0/AIC}, flagId=1` | 10 | 否 |
| `participant={group0/AIC}, flagId=2` | 6 | 否 |

两个计数器分别是 10 和 6，不会合并成 16，因此不会触发 `COUNTER_OVERFLOW`。
这只说明没有计数溢出；若结算时仍未消费，可能另外报告 `remaining counter token`。

SET 需要先按模式完成参与者聚合，才会产生相应计数；匹配的 WAIT 会消费计数。
因此，当前未消费计数不同于源码中或整个运行期间的 SET 总次数。
同一计数器即使累计产生超过 15 个通知，只要及时消费、从未尝试让计数从 15 增加到 16，也不会因此溢出。
`mode` 用于核对聚合及 WAIT 匹配规则，不能把切换模式当作另开一个独立计数器。

解释 `COUNTER_OVERFLOW` 或计数上限时，**主动给出一个最小数值对照例子，无需用户另行要求举例**：例如同一目标核的
`flagId=1` 当前计数为 10、`flagId=2` 当前计数为 6，两个计数器分别计算，不因合计 16 触发溢出；
再说明匹配的 WAIT 会消费计数，所以 SET 总次数超过 15 不等于当前计数器溢出。示例数字可按需选择。例子用于解释计数归属，
不能替代对实际报告的 `participant`、`flagId`、`mode`、launch 和消费记录的核对。

### 模式切换的两个容易误读的状态

```text
FLAG_MODE_SWITCH_BEFORE_COMPLETION ... activeMode=2 newMode=0 pendingSetRounds=1 remainingCount=0
FLAG_MODE_SWITCH_BEFORE_COMPLETION ... activeMode=2 newMode=0 pendingSetRounds=0 remainingCount=1
```

第一条表示 SET 聚合尚未结束，计数为 0 也不能切换；第二条表示聚合已结束，但计数尚未消费，同样不能切换。
限制针对同一 participant 与 flag 的状态，不意味着不同参与者不能使用相同数字的 flag。
应先核对旧模式的聚合与消费过程，再核对新模式触发位置；不能只建议“补一个 WAIT”而忽略聚合缺失。

## 关联诊断与执行状态

- 相同数字 flag 不保证是同一同步对象。优先使用报告规范化的 key，结合 group、参与核、bank、模式/指令类别及 launch；不凭裸编号猜匹配关系。
- 多核 trace 的分片或回调到达顺序不等于设备同步操作的真实先后，不因日志中 WAIT 先出现就认定源码先 WAIT。
- 模式切换、参与者缺失和无 token 可能属于同一串异常演化。保留各条证据，结合上下文核对关联，不能把错误数当作独立根因数。
- 同步违规与死锁分开说明。`0 deadlocks` 不否定已有配对或次序错误；外部超时但未形成完整报告时，仍按检查未完成处理。

## 资料与用例依据

- [跨核用例说明](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/synccheck/cross_core/README_cross_core.md)与[逐例断言](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/synccheck/cross_core/cross_core_verify_cases.py)：`cross_core_mode_switch_pending_round`、`cross_core_mode_switch_remaining_token`、`cross_core_mode_reuse`、`cross_core_counter_overflow_boundary`，以及参与者、身份与拓扑反例。
- [核内重复设置用例](https://gitcode.com/heqinglun/ascendc_smoke/blob/49796e76c96f27729bafef60735db6fdab6c74c8/npu_tools/npu_check/examples/synccheck/flag_set_set_wait_wait/verify.py)：两侧总数相等仍可能次序违规。
- [使用手册](https://gitcode.com/cann/asc-tools/blob/master/docs/zh/02_npu_check.md)：同步能力及模式适用条件。

用例包含预期、回放及可能阻塞的反例，不能视为所有设备测试均已通过，也不因参考多 launch 隔离用例而扩展运行配置。
