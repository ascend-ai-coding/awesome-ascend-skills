# NPU 多流优化案例手册

本文收录仓库内与 NPU 多流优化相关的案例，按“**每次优化算一个案例**”整理，而不是按模型统计。目录中的每个 Markdown 文件只讲一个案例，聚焦优化动机、执行编排、关键代码片段、适用场景和复用价值。

## 收录原则

- 文档中明确描述了多流、双流、stream overlap 或多流并行。
- 代码中存在显式多流编排，例如 `npu_stream_switch`、`torch.npu.Stream()`、`wait_event`、`record_event`、`wait_stream`。
- 相同优化方法在不同模型中复用时，合并到同一类案例中统一整理。

## 分类导航

### MoE 多流

- [案例：MoE 共享专家双流并行](./moe-shared-expert-dual-stream.md)
- [案例：HunyuanImage-3.0 MoE 多流变体](./hunyuanimage-moe-multi-stream.md)
- [案例：Qwen3-Next Patch 形态的 MoE 双流](./qwen3-next-moe-dual-stream-patch.md)

### Attention / Prolog 多流

- [案例：Indexer Prolog 多流并行](./indexer-prolog-multi-stream.md)

### KVCache Offload 异步流

- [案例：KVCache Offload 异步搬运流](./kvcache-offload-async-stream.md)

### Prefill 双流流水

- [案例：Prefill Micro-Batch 双流流水](./prefill-microbatch-dual-stream.md)

### 多流 + 控核 / 分离式编排

- [案例：LongCat-Flash 多流与控核联动](./longcat-flash-multi-stream-limit-core.md)
- [案例：LongCat-Flash AFD 通信计算 overlap](./longcat-flash-afd-overlap.md)

## 快速选型表

| 优化模式 | 先看案例 | 再看源码 | 什么时候优先选 | 典型风险 |
| --- | --- | --- | --- | --- |
| MoE shared expert 双流 | [`moe-shared-expert-dual-stream.md`](./moe-shared-expert-dual-stream.md) | [`cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py`](../../../../models/deepseek-v3.2-exp/models/modeling_deepseek.py), [`cann-recipes-infer/models/glm-5/models/modeling_glm.py`](../../../../models/glm-5/models/modeling_glm.py) | 路由专家和共享专家结果在后面汇合，且 decode shape 稳定 | 同步点放错会导致 merge 读到未完成结果；共享专家过重时 overlap 不一定有收益 |
| Indexer / Prolog 多流 | [`indexer-prolog-multi-stream.md`](./indexer-prolog-multi-stream.md) | [`cann-recipes-infer/models/glm-5/models/indexer.py`](../../../../models/glm-5/models/indexer.py) | Attention 前处理链路可拆成两段或多段，等待在后面汇合 | 这类优化常常是前处理子链 overlap，不是完整大模块并行，边界容易拆错 |
| KVCache offload 异步流 | [`kvcache-offload-async-stream.md`](./kvcache-offload-async-stream.md) | [`cann-recipes-infer/models/deepseek-v3.2-exp/models/offload_cache.py`](../../../../models/deepseek-v3.2-exp/models/offload_cache.py), [`cann-recipes-infer/models/glm-5/models/offload_cache.py`](../../../../models/glm-5/models/offload_cache.py) | 设备内存紧张，需要把搬运从主计算流剥离 | 主流和搬运流的状态一致性最重要；异步搬运可能掩盖不了 H2D/D2H 带宽瓶颈 |
| Prefill micro-batch 双流 + event | [`prefill-microbatch-dual-stream.md`](./prefill-microbatch-dual-stream.md) | [`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`](../../../../models/deepseek_r1/models/modeling_deepseek.py) | prefill 同时有明显计算和通信，且切 micro-batch 后 shape 线性度还可以 | 最容易引入 host bound、shape 劣化和事件编排错误 |
| 多流 + 控核 | [`longcat-flash-multi-stream-limit-core.md`](./longcat-flash-multi-stream-limit-core.md) | [`cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py`](../../../../models/longcat-flash/models/modeling_longcat_flash.py) | 已有 overlap，但一条流明显拖尾或资源被另一条流吃满 | 控核值不是通用常量；多流、控核、预取和图模式往往是耦合设计 |
| AFD 通信/计算 overlap | [`longcat-flash-afd-overlap.md`](./longcat-flash-afd-overlap.md) | 以案例文档为主 | 分离式部署或通信链路成了主要瓶颈 | 重点不在本地双算子并行，而在通信和本地计算 overlap；等待链容易拖尾 |
| Patch 形态多流 | [`qwen3-next-moe-dual-stream-patch.md`](./qwen3-next-moe-dual-stream-patch.md) | [`cann-recipes-infer/models/qwen3-next/patches/stage1/0003-feat-moe-multi-stream.patch`](../../../../models/qwen3-next/patches/stage1/0003-feat-moe-multi-stream.patch) | 优化不能直接落到模型仓，需要嵌进 runtime 或 patch | patch 代码更依赖现有生命周期，`wait_stream()` 顺序错了会直接破坏运行时逻辑 |
| 多模态变体 | [`hunyuanimage-moe-multi-stream.md`](./hunyuanimage-moe-multi-stream.md) | [`cann-recipes-infer/models/hunyuan-image-3.0/adaptor_patches/hunyuan.py`](../../../../models/hunyuan-image-3.0/adaptor_patches/hunyuan.py) | 共享流在模块初始化阶段就作为能力注入，而不是在前向里临时加 scope | 多流能力进入模块构造期后，不能只复制一段前向代码；初始化和分布式上下文要一起看 |

## 使用顺序

1. 先确定当前优化属于哪一类模式。
2. 先读对应案例文档，确认模块边界、依赖关系和同步方式。
3. 再读代表代码或补丁，确认实际 API 风格和 enable 开关设计。
4. 如果一个实现同时落在多类模式里，先选主模式，再把其他能力当补充手段。

## 案例清单

| 案例名称 | 优化方法 | 代表模型 | 概述 |
| --- | --- | --- | --- |
| MoE 共享专家双流并行 | 共享专家与路由专家并行 | DeepSeek-V3.2-Exp / DeepSeek-R1 | 把共享专家放到副流，与路由专家路径重叠，减少 decode 可见耗时。 |
| Indexer Prolog 多流并行 | Attention 前处理多流 | DeepSeek-V3.2-Exp / GLM-5 | 将 Q 路径与权重投影路径拆到不同流，缩短 Indexer 前处理串行段。 |
| KVCache Offload 异步搬运流 | Cache 异步卸载/回迁 | DeepSeek-V3.2-Exp / GLM-5 | 用独立流完成 KVCache 搬运，降低主计算流阻塞。 |
| Prefill Micro-Batch 双流流水 | 计算通信双流掩盖 | DeepSeek-R1 | 两个 micro-batch 分别跑在两条流上，并通过 event 编排 dispatch/combine。 |
| LongCat-Flash 多流与控核联动 | 多流 + limit_core_num | LongCat-Flash | 把 Attention 路径和 MoE 路径拆流后再分核，减少拖尾。 |
| LongCat-Flash AFD 通信计算 overlap | 分离部署下的双流编排 | LongCat-Flash | 通过 Stream0/Stream1 overlap Send/Recv 与主计算，隐藏分离式通信耗时。 |
| HunyuanImage-3.0 MoE 多流变体 | Shared MLP 独立流 | HunyuanImage-3.0 | 用独立 NPU stream 承载共享 MLP，形成多模态模型里的 MoE 双流变体。 |
| Qwen3-Next Patch 形态的 MoE 双流 | Patch 级 NPU 双流改造 | Qwen3-Next | 在 SGLang patch 中引入 shared expert stream，与 DeePEP 路由过程并行。 |

## 常见误用

- 不要把 `MoE shared expert 双流` 和 `Prefill micro-batch 双流` 当成同一种流水，它们的同步粒度完全不同。
- 不要看到有两条流就默认需要 `limit_core_num`，控核只在资源争抢和拖尾明显时再引入。
- 不要把 `KVCache offload` 这类搬运流当成计算流优化，它优先关注的是状态一致性和带宽掩盖。
- 不要直接拼接多个案例的代码片段，必须先按当前执行模式选一套主路径。
