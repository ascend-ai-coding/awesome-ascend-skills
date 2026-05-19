# 多流与控核 API 路由

本文件用于把“执行路径 / 问题类型”映射到上游文档和推荐 API。

## 上游文档入口

| 路径 | 先读文档 | 再读文档 | 适用场景 |
| --- | --- | --- | --- |
| Ascend IR / GE 图模式 | `cann-recipes-infer/docs/zh/ascend_ir/features/advanced/multi_stream.md` | `cann-recipes-infer/docs/zh/ascend_ir/features/advanced/limit_cores.md` | 图内多流表达、`npu_stream_switch`、`npu_wait_tensor`、GE 图模式控核 |
| npugraph_ex / aclgraph | `cann-recipes-infer/docs/zh/npugraph_ex/advanced/multi_stream.md` | `cann-recipes-infer/docs/zh/npugraph_ex/advanced/limit_cores.md` | `torch.npu.Stream` / `Event` / `record_stream`、Stream 级控核 |

## 路径要点

### Ascend IR / GE 图模式

- 多流主要面向 Cube 资源未完全使用的场景；若 Cube 已吃满，不要默认开启多流。
- 仅适用于 GE 图模式场景。
- 优先接口是 `npu_stream_switch` 和 `npu_wait_tensor`。
- 控核分两层：`torchair.scope.limit_core_num` 是算子级，`config.ge_config.aicore_num` 是全局 session 级；算子级优先级更高。
- 静态 shape 下不要和 `enable_single_stream` 混用，也不要直接在 SuperKernel 内手搓多流。
- 动态 shape 默认单流；如果依赖 `ENABLE_DYNAMIC_SHAPE_MULTI_STREAM=1` 开启多流，脚本内显式多流表达优先级更高。

### npugraph_ex / aclgraph

- 多流主要面向 aclgraph 间资源并发；官方路径围绕 `torch.npu.Stream`、`torch.npu.stream`、`torch.npu.Event`、`tensor.record_stream`。
- `record_stream` 只在短生命周期 tensor 会被其他流继续使用时才需要补；权重等长生命周期对象一般不需要。
- 控核是 Stream 级，接口为 `torch.npu.npugraph_ex.scope.limit_core_num`。
- 仅 Ascend C 算子支持控核；micro-batch 多流场景如果夹杂不支持控核的算子，收益可能下降，严重时可能卡死。
- 配置结果优先通过 profiler 结果中的 `kernel_details.csv` 查看核使用情况。

## 先判执行路径

| 当前场景 | 推荐 API 风格 | 首选 API | 先读文档 |
| --- | --- | --- | --- |
| eager / patch 改造 | 显式流对象 | `torch.npu.Stream()`、`record_event()`、`wait_event()`、`wait_stream()`、`record_stream()` | 先看仓库案例；如果要对齐显式 stream 语义，参考 `cann-recipes-infer/docs/zh/npugraph_ex/advanced/multi_stream.md` |
| `ge_graph` / TorchAir 图内多流 | 图内 scope | `npu_stream_switch`、`npu_wait_tensor` | 先看 `cann-recipes-infer/docs/zh/ascend_ir/features/advanced/multi_stream.md`；需要控核时再看 `cann-recipes-infer/docs/zh/ascend_ir/features/advanced/limit_cores.md` |
| `npugraph_ex` / aclgraph | 显式 stream + Event | `torch.npu.Stream()`、`torch.npu.stream()`、`torch.npu.Event()`、`record_stream()` | 先看 `cann-recipes-infer/docs/zh/npugraph_ex/advanced/multi_stream.md`；需要控核时再看 `cann-recipes-infer/docs/zh/npugraph_ex/advanced/limit_cores.md` |

## 再判问题类型

| 问题类型 | 推荐 API | 什么时候用 | 注意事项 | 先读文档 |
| --- | --- | --- | --- | --- |
| 需要把一段计算切到副流 | `torch.npu.Stream()` 或 `npu_stream_switch` | 已确认两段路径没有直接 `data` 依赖，只在后面汇合 | 先明确汇合点，再决定是补 `Event`、`wait_stream` 还是 `npu_wait_tensor` | 对应路径的 `multi_stream.md` |
| 需要显式控制跨流时序 | Ascend IR 路径优先 `npu_wait_tensor`；显式 stream 路径优先 `record_event()` / `wait_event()`；已有 tagged event 风格时沿用 `npu_record_tagged_stream` / `npu_tagged_event_wait` | 两条流之间存在控制依赖，但后继不直接吃前驱输出 tensor | 不要为了“统一风格”强行把已有 tagged event 代码改写成另一套语义 | 对应路径的 `multi_stream.md` |
| 需要延长 tensor 生命周期 | `record_stream()` | 短生命周期 tensor 会在别的流继续使用 | 主要看 aclgraph / eager / capture 阶段；权重等长生命周期对象一般不需要 | `cann-recipes-infer/docs/zh/npugraph_ex/advanced/multi_stream.md` |
| overlap 已成立但一条流明显拖尾 | `limit_core_num` | 已看到两条流资源争抢，或一条流长期占满 Core | Ascend IR 是算子级 / 全局级，npugraph_ex 是 Stream 级，不要混着理解 | 对应路径的 `limit_cores.md` |
| 需要进一步查看或设置 stream 资源限制 | `torch_npu.get_stream_limit` / `torch_npu.set_stream_limit` | 已进入控核或 stream 资源调优阶段 | 这不是第一手多流 API，通常在资源调优阶段再用 | 本文件中的“上游文档入口” + 本 skill 案例 |
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
- `npu_tagged_event_record` 这类高级同步原语优先跟随仓库现有案例代码，不要脱离上下文自己猜语义。
