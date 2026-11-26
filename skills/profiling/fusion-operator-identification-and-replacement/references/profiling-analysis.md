# Profiling 数据分析方法与字段解释

> 本文是 SKILL.md 第 3、4 步的展开：如何解析昇腾 Profiling 数据、如何做热点与瓶颈分析、如何用证据链识别融合算子。
>
> **字段名与文件名随 CANN / 工具版本变化，一切以用户实际文件的表头与目录结构为准；本文的字段表只是常见版本的参考。**

## 目录

1. [数据来源与常见产物](#1-数据来源与常见产物)
2. [环境与版本自查](#2-环境与版本自查)
3. [op_summary / kernel_details 常见字段](#3-op_summary--kernel_details-常见字段)
4. [数据清洗与聚合方法](#4-数据清洗与聚合方法)
5. [热点与瓶颈分类](#5-热点与瓶颈分类)
6. [融合算子识别的证据链](#6-融合算子识别的证据链)
7. [常见陷阱](#7-常见陷阱)

---

## 1. 数据来源与常见产物

| 采集方式 | 说明 | 典型产物 |
|---|---|---|
| msprof（CANN 命令行采集） | `msprof --application=... --output=...` 等方式采集训练/推理过程 | `PROF_*` 目录，内含 op_summary CSV、kernel_details.csv、trace_view.json、communication 相关 CSV/JSON 等（具体文件名以版本为准） |
| torch_npu profiler | PyTorch Profiler 的 NPU 集成，可带 stack / shape / memory 选项 | Chrome trace JSON（时间线），可用 TensorBoard 等查看 |
| MindStudio / MindStudio Insight | GUI 采集与分析，底层为同一套 Profiling 数据 | 可导出的各类汇总 CSV |

拿到数据先做**盘点**，结果写进报告第 2 节：

- 有哪些文件，覆盖哪些 device / stream，时间范围多大
- 是否包含 shape / dtype / stack / memory 信息
- 数据是否完整（有无截断、掉采样、多 step 混在一起）
- LLM 推理场景尽量区分 **prefill 与 decode** 阶段（特征差异大，混在一起会互相稀释热点）

大文件处理：op_summary 可能极大，**用脚本（python/awk）先做聚合**，只把聚合结果与代表性片段带入分析，不要把整个文件读进上下文。

## 2. 环境与版本自查

命令路径可能因安装方式不同而不同，以实际环境为准（详见 `ascend-operator-reference.md`）：

```bash
npu-smi info                                                # NPU 型号、驱动/固件版本、健康状态
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg    # CANN 版本（常见安装路径）
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
pip list | grep -Ei "vllm|transformers|mindie|atb|accelerate"   # 推理框架与加速库版本
```

环境实证（机器上直接查到的版本）优先于用户口头描述，两者不一致时在报告中指出。

## 3. op_summary / kernel_details 常见字段

> 下表是常见版本的参考列；真实分析以实际表头为准，缺列就如实说明"本数据不含 X，因此不能做 Y"。

| 字段 | 解释 | 用途 |
|---|---|---|
| Op Name / Name | 算子实例名（可能含调用层级信息） | 关联调用点；同名会多次出现 |
| Op Type / Type | 算子类型（kernel 名） | **聚合的主力键** |
| Task Type | 执行单元：AI Core（含 Cube / Vector）、AI CPU、混合等 | 区分 cube / vector / aicpu 负载 |
| Stream ID | 所在流 | 多流分析、通信/计算重叠判定 |
| Start Time | 起始时间戳 | 时间线还原、序列模式识别、间隙分析 |
| Duration | 耗时（**注意单位**，常见 us，也可能是 ns） | 热点统计、带宽反推 |
| Input / Output Shapes | 输入输出形状 | 访存量计算、融合证据（E3） |
| Input / Output Data Types | 数据类型（FP16 / BF16 / INT8 / FP32…） | 访存字节数、量化证据 |
| Block Dim | 使用的 AI Core 核数 | 并行度证据 |
| PMU/利用率类指标 | 部分版本附带（如 cycles、带宽/队列计数） | ⚠ 以实际导出为准；有则用于瓶颈佐证 |

torch_npu profiler 的 trace 中，NPU kernel 事件与 host 侧 aten 算子事件并存——**区分 host 耗时与 device 耗时，热点分析以 device 为准**；host 侧信息用于判断下发开销。

## 4. 数据清洗与聚合方法

按以下顺序产出统计表（这是报告第 4 节的素材）：

1. **按 Op Type 聚合**：调用次数、总耗时、平均、最大、占总时间百分比。排序得到第一张热点表。
2. **按时间线排序**（有 start time 时）：还原执行顺序，识别算子序列模式——例如 `Mul → Rsqrt → Mul → Mul` 的 RMSNorm 手写展开、`Sigmoid → Mul → Mul` 的 SwiGLU 展开。**只有聚合表、没有时间顺序时，序列模式无法识别**，这是数据完整性的关键缺口。
3. **按 stream 分组**：计算流 / 通信流 / 拷贝流的时间区间是否重叠（区间求交）；不重叠的部分才是真瓶颈。
4. **间隙分析**：相邻 kernel 之间的空隙总量与分布。空隙大 → 调度/下发瓶颈或 host 等待。
5. **阶段切分**：按 step 或时间段把 prefill / decode 分开统计。

## 5. 热点与瓶颈分类

对聚合结果按下表归类，**每类都要写出证据**（哪些算子、占比多少、依据是什么），写进报告第 4 节：

| 瓶颈类型 | 判定证据（典型组合） | 对融合策略的含义 |
|---|---|---|
| 计算瓶颈 | MatMul / BatchMatMul 类占总耗时的主体；大 shape GEMM 单 kernel 耗时长；耗时与理论 FLOPs 匹配（有 PMU 数据时利用率高） | 融合收益有限，优先看量化、tiling、更优 GEMM 实现 |
| 内存访问瓶颈 | elementwise / reduce / copy 类（Add、Mul、Cast、TransData、Reduce*）占比高；算术强度低；耗时 ≈ 搬运字节数 ÷ 带宽 | **融合的主战场**：减少中间张量读写 |
| 通信瓶颈 | HCCL 类算子（AllReduce / AllGather / ReduceScatter / Send / Recv 等）占比高；计算流存在等待通信的空隙 | 通信+计算融合（成本高）或并行策略调整 |
| 调度/下发开销 | kernel 数量极多且单 kernel 耗时很小（个位~几十微秒量级）；时间线空隙显著；host 事件密集 | 算子合并（减少 launch 次数）、图模式 |
| 其他 | H2D/D2H memcpy、aclrtMalloc、AI CPU 算子、锁页内存 | 逐项单独分析 |

注意：decode 阶段小 batch 下，"内存访问瓶颈"与"调度开销"常常并存——融合的收益既来自访存减少，也来自 launch 次数减少，估算时两部分都要算（见 `performance-estimation.md`）。

## 6. 融合算子识别的证据链

四类算子（单算子 / 算子库融合 / 图编译器自动融合 / 自定义融合）的判定证据，按强度排序：

| 编号 | 证据 | 强度 |
|---|---|---|
| E1 | kernel 名称与已知融合算子匹配（对照 `npu-fused-op-catalog.md`、CANN 算子库或项目源码） | 弱 |
| E2 | **序列证据**：模型源码/计算图中是 N 个算子，Profiling 时间线只出现 1 个 kernel，且输入输出 shape 与语义吻合 | 强 |
| E3 | 输入输出数量 / shape 与任何单一基础算子的语义都不符（如单 kernel 多输入多输出） | 中 |
| E4 | **框架配置证据**：torch.compile / torchair / MindIE 图模式开关、torch_npu 适配层替换日志 | 中 |
| E5 | **图 dump / 编译融合日志**（GE 导出图、onnx/air dump、编译日志中的融合记录） | 强 |

置信度分级（写进报告第 3 节）：

- **确认**：有 E2 或 E5
- **高置信**：E1 + E3，或 E1 + E4
- **待核实**：仅有 E1 —— 必须标注 ⚠，写明核实途径（查模型源码 / 图 dump / 图模式开关对比实验）

典型模式提示（**只是线索，不是结论**）：

- 时间线中 QKV 投影后紧跟单个类 attention kernel、无独立 Softmax/缩放/Mask 序列 → 可能在用 FA 类融合算子，按 E1+E3 初判，待源码确认
- 出现 `Mul → Rsqrt → Mul → Mul`（或 `Pow → ReduceMean → Rsqrt → Mul` 等）RMSNorm 展开序列 → **该处未融合**，本身就是候选
- 出现 `Sigmoid → Mul → Mul`（SwiGLU）或 `GELU → Mul` 展开 → 未融合，候选；先查 catalog 中 `npu_swiglu` / `npu_gelu_mul` 等现成算子
- `TransData` / `Transpose` 频繁出现 → 布局转换开销；能否消除取决于上游能否直接产出目标布局（需要图信息）
- Attention 路径出现 `BatchMatMul → Add(mask) → Softmax → BatchMatMul` 长序列 → 未用 FA 类算子，候选方向是改用现成 FA 接口而非自研

## 7. 常见陷阱

1. **耗时 ≠ 可节省耗时**：被融合算子的耗时里只有"访存部分 + launch 部分"可能被省掉，详见 `performance-estimation.md`。
2. **TransData 不一定是浪费**：它可能是 GEMM 所需布局（如 NZ/5HD）的必要转换；判断能否消除需要知道上下游算子的布局需求。
3. **AI CPU 算子**：Task Type 显示跑在 AI CPU 上的算子，其优化路径与 AI Core 融合不同，不要混入同一类分析。
4. **多流重叠**：两个 stream 上时间重叠的算子耗时不能简单相加，重叠段只计一次；被融合的算子若本来与通信重叠，节省量要打折。
5. **单位与采样**：duration 单位（us/ns）先与时间线总长交叉验证；profiling 本身有开销，绝对值可能偏大，融合收益可能被高估。
6. **无 shape 时**：无法计算访存量，性能预测只能停在定性层面——明确告知用户需要补什么数据（shape、dtype、start time、环境版本）。
