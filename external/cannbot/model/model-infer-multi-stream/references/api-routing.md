# 多流与控核 API 路由

本文件用于把“执行路径 / 问题类型”映射到上游文档和推荐 API。

## 官方文档入口（已落到本 skill `resources/`）

| 路径 | 先读文档 | 再读文档 | 适用场景 |
| --- | --- | --- | --- |
| Ascend IR / GE 图模式 | [`resources/ascend_ir_multi_stream.md`](../resources/ascend_ir_multi_stream.md) | [`resources/ascend_ir_limit_cores.md`](../resources/ascend_ir_limit_cores.md) | 图内多流表达、`torchair.scope.npu_stream_switch`、`torchair.scope.npu_wait_tensor`、GE 图模式控核 |
| npugraph_ex / aclgraph | [`resources/npugraph_ex_multi_stream.md`](../resources/npugraph_ex_multi_stream.md) | [`resources/npugraph_ex_limit_cores.md`](../resources/npugraph_ex_limit_cores.md) | `torch.npu.Stream` / `torch.npu.Event` / `record_stream`、Stream 级控核 |

## 路径要点

### Ascend IR / GE 图模式

- 多流主要面向 Cube 资源未完全使用的场景（官方表述为「Ascend IR 图内资源并发」）；若 Cube 已吃满，不要默认开启多流，可能因调度开销劣化原计算性能。
- 仅适用于 GE 图模式场景（`torchair.CompilerConfig()` + `torchair.get_npu_backend()`）。
- 优先接口是 `torchair.scope.npu_stream_switch(stream_tag, stream_priority=0, enable_inner_parallel=True)` 和 `torchair.scope.npu_wait_tensor(self, dependency)`。
- 控核分两层：算子级 `torchair.scope.limit_core_num`，全局级 `config.ge_config.aicore_num = "${aicore}|${vector}"`；**算子级优先级高于全局级**。
- 静态 shape 下不要和 `enable_single_stream` 混用，也不要直接在 SuperKernel 内手搓多流（如需在 SuperKernel 内分流，使用 stream-fusion 编译选项）。
- 动态 shape 默认单流；如果依赖 `ENABLE_DYNAMIC_SHAPE_MULTI_STREAM=1` 开启多流，脚本内显式多流表达优先级更高。
- 配置结果可通过 graph dump（`config.debug.graph_dump.type="txt"`）在 `attr` 的 `_op_aicore_num` / `_op_vectorcore_num` 中确认。

### npugraph_ex / aclgraph

- 多流主要面向 aclgraph 间资源并发；官方路径围绕 `torch.npu.Stream`、`torch.npu.stream`、`torch.npu.Event`、`tensor.record_stream`。
- 切流 scope：`with torch.npu.stream(stream)`，其中 `stream = torch.npu.Stream()`。
- 时序控制：`event = torch.npu.Event()` + `event.record()`（默认在 current stream 上记录）+ `event.wait(stream)`（指定流上等待）。
- `record_stream` 只在短生命周期 tensor 会被其他流继续使用时才需要补；权重等长生命周期对象一般不需要。
- 控核是 Stream 级，接口为 `torch.npu.npugraph_ex.scope.limit_core_num(op_aicore_num, op_vectorcore_num)`。
- 仅 Ascend C 算子支持控核；非 Ascend C 算子（包括非 AI Vector 控核的通信类算子）暂不支持，micro-batch 多流场景如果夹杂不支持控核的算子，收益可能下降，严重时可能卡死。
- CANN ≤ 8.5.0 时，静态 kernel 编译与控核同时开启的情况下，**优先保留控核功能**，静态 kernel 编译失效。
- 配置结果优先通过 Ascend PyTorch Profiler 结果中的 `kernel_details.csv`（`Block Num` / `Mix Block Num` 列）确认核使用情况。

## 先判执行路径

| 当前场景 | 推荐 API 风格 | 首选 API | 先读文档 |
| --- | --- | --- | --- |
| eager / patch 改造 | 显式流对象 | `torch.npu.Stream()`、`Stream.record_event()`、`Stream.wait_event()`、`Stream.wait_stream()`、`tensor.record_stream()` | 先看仓库案例；如要对齐显式 stream / event 语义，参考 [`resources/npugraph_ex_multi_stream.md`](../resources/npugraph_ex_multi_stream.md) 中关于 Eager 模式的说明 |
| `ge_graph` / TorchAir 图内多流 | 图内 scope | `torchair.scope.npu_stream_switch`、`torchair.scope.npu_wait_tensor` | 先看 [`resources/ascend_ir_multi_stream.md`](../resources/ascend_ir_multi_stream.md)；需要控核时再看 [`resources/ascend_ir_limit_cores.md`](../resources/ascend_ir_limit_cores.md) |
| `npugraph_ex` / aclgraph | 显式 stream + Event | `torch.npu.Stream()`、`torch.npu.stream()`、`torch.npu.Event()` + `Event.record()` / `Event.wait(stream)`、`tensor.record_stream()` | 先看 [`resources/npugraph_ex_multi_stream.md`](../resources/npugraph_ex_multi_stream.md)；需要控核时再看 [`resources/npugraph_ex_limit_cores.md`](../resources/npugraph_ex_limit_cores.md) |

## 再判问题类型

| 问题类型 | 推荐 API | 什么时候用 | 注意事项 | 先读文档 |
| --- | --- | --- | --- | --- |
| 需要把一段计算切到副流 | `torch.npu.stream(stream)`（eager / npugraph_ex）或 `torchair.scope.npu_stream_switch`（GE 图模式） | 已确认两段路径没有直接 `data` 依赖，只在后面汇合 | 先明确汇合点，再决定是补 `Event` + `Stream.wait_stream`、`Event.wait(stream)` 还是 `npu_wait_tensor` | 对应路径的 `multi_stream` 文档 |
| 需要显式控制跨流时序 | GE 图模式优先 `torchair.scope.npu_wait_tensor`；显式 stream 路径优先 `Event.record()` / `Event.wait(stream)`；已有 tagged event 风格时沿用 `npu_record_tagged_stream` / `npu_tagged_event_wait`（tagged event 命名空间：GE 路径用 `torchair.ops.*`，npugraph_ex 路径用 `torch.npu.npugraph_ex.scope.*`；注意 GE 路径的切流 / `npu_wait_tensor` 才在 `torchair.scope.*`） | 两条流之间存在控制依赖，但后继不直接吃前驱输出 tensor | 不要为了“统一风格”强行把已有 tagged event 代码改写成另一套语义 | 对应路径的 `multi_stream` 文档 |
| 需要延长 tensor 生命周期 | `tensor.record_stream(other_stream)` | 短生命周期 tensor 会在别的流继续使用 | 主要看 aclgraph / eager / capture 阶段；权重等长生命周期对象一般不需要 | [`resources/npugraph_ex_multi_stream.md`](../resources/npugraph_ex_multi_stream.md) |
| overlap 已成立但一条流明显拖尾 | `limit_core_num`（GE 路径用 `torchair.scope.limit_core_num`；npugraph_ex 路径用 `torch.npu.npugraph_ex.scope.limit_core_num`） | 已看到两条流资源争抢，或一条流长期占满 Core | GE 是算子级 + 全局级（算子级优先），npugraph_ex 是 Stream 级且仅对 Ascend C 算子生效，不要混着理解 | 对应路径的 `limit_cores` 文档 |
| 需要进一步查看或设置 stream 资源限制 | `torch_npu.get_stream_limit` / `torch_npu.set_stream_limit` | 已进入控核或 stream 资源调优阶段 | 这不是第一手多流 API，通常在资源调优阶段再用 | 本文件“官方文档入口” + 本 skill 案例 |
| 需要扩大计算窗口，掩盖权重搬运 | `torch_npu.npu_prefetch` | overlap 正确，但仍有访存或带宽空洞可被前序轻算子掩盖 | 只在前序算子不明显抢带宽时使用；常和多流 + 控核联动 | 本 skill 案例 |

## 推荐决策顺序

1. 先确定当前是 eager / patch 还是 graph / TorchAir。
2. 先选一套主 API 路径，不要混着写。
3. 先把依赖和同步做对，再确认是否真的有 overlap。
4. 只有在 overlap 正确但拖尾明显时，才进入控核、stream limit、预取调优。

## 常见误区

- 不要在 eager 路径里照搬 TorchAir 的 tagged event 风格。
- 不要把 `limit_core_num` 当成默认步骤；它只解决资源分配问题，不解决依赖错误。
- 不要用 `npu_prefetch` 掩盖一个本来就不该并行的链路；先证明链路没有错误依赖。
- 不要在 aclgraph / eager 路径里省略 `record_stream()` 的生命周期判断；只切流不管内存同样会出错。
- `npu_tagged_event_record` / `npu_tagged_event_wait` 这类 tagged event 同步原语，GE 路径在 `torchair.ops.*`、npugraph_ex 路径在 `torch.npu.npugraph_ex.scope.*`；优先跟随仓库现有案例代码（如 `examples/moe-shared-expert-dual-stream.md` 的 `tng.ops.*`，`tng` 即 `torchair`），不要脱离上下文自己猜语义。
