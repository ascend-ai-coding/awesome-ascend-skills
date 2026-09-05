# Kernel层 Fused Kernel 设计与开发

本文维护本场景的 Kernel层 CV 契约。Step 3 用它冻结 concrete Kernel 的编排增量，随后将所需动作编译进 PLAN；Step 4 只有在 PLAN action 绑定本文时才读取。本文不定义所有 Fused Kernel 的通用 ABI。

## 1. Concrete 入口和 adapter

以当前 Investigation 的真实 Kernel witness 为唯一来源，逐项记录：

```text
mix_entry / modifier / specialization
aic_role / aiv_role / actual_ratio
adapter_signature_and_parameter_units
block_shape / gm_offset / split_or_sub_inputs
params_and_tilingdata_fields
```

如果存在 `Init`、`GetTensor(slot)`、tile `operator` 或等价 adapter，必须从调用点抄录返回单位、地址推导和生命周期；不能从示例 Asset、相似 Kernel 或名称推测五参 ABI。单位必须明确是元素、字节、LocalTensor 还是逻辑索引。

## 2. CV 生命周期

DESIGN/PLAN 必须覆盖当前 concrete 控制流：

1. AIC->AIV 的 producer/consumer、C-ready flag、pipe 和通知点；
2. AIV->AIC 的 V-done、pipe、slot 和覆盖前等待；
3. 首轮初始化、循环、slot reuse、empty task、partial/final drain；
4. 每 tile 的 BlockShape、GM offset、split/sub 参数和 Epilogue 首消费者；
5. cross-core wait 到本地 Vector/MTE/其他首消费者 pipe 的交接；
6. localRows=0 或空 AIV 时的 release 行为。

历史 `source_observed` 只说明源码控制流；`device_verified` 必须绑定同一 Investigation、目标 Blaze 组装方案、编译和验证记录。固定 ratio、slot 数、bridge 或 flag 不能升级为场景默认。

## 3. Slot 与地址合同

对每个 DESIGN 声明的 slot 方案写出：

```text
slot_count
slot_index_lifecycle
slot_base_unit
GetTensor(slot) return unit
slot_capacity
C_range
staging_ranges
output_range
reuse_wait
```

AIC 写入 C 和 AIV 读取 C 必须由同一 slot 起点和单位推导；staging 不能跨 slot。若 SplitM 激活，必须分别说明本地 UB C 是否已经按 sub 写入，以及 GM operand/output 是否需要全局 sub offset，避免二次偏移。

## 4. 同步验证与扩展

同步候选只有在以下证据齐全时才进入正式 DESIGN：

- 源码中存在对应 wait/set/pipe 控制流；
- 负向版本只移除该同步变量，能稳定复现同一责任域错误；
- 正向版本恢复该变量并完成相关边界及 Full 重复回归；
- 该结论绑定具体 Blaze 组装方案，而不是相似 Kernel。

只在 DESIGN 明确授权时复制 Kernel 到项目 custom 目录，使用独立符号/namespace，并保持 Mmad、调度、Params、输出和未涉及的同步不变量。不能通过加 `__mix__`、增加 event 或复制相邻实现临时制造 Fused Kernel。

## 5. PLAN action 清单

场景 development guide 编译 PLAN 时，按 DESIGN 实例化：

1. concrete Kernel/Wrapper 来源和只读范围；
2. 官方复用或 custom 副本、首次修改点和类型接线；
3. adapter/Params/TilingData/GM ABI 接线；
4. AIC/AIV、slot、flag/pipe、bridge、empty/final/reuse action；
5. 结构检查、构建、最早失败域诊断、单变量负向/正向和最终清理回归。

每项初始 action 必须有 source refs、前置、预期输出、checkpoint 和 rollback。本文不补充 DESIGN 未冻结的参数或语义；Step 4 可在项目根内补充实现文件并记录实际变更。
