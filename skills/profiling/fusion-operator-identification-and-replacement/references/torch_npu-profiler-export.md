# torch_npu profiler 导出数据解析指南

> 本文补充 `profiling-analysis.md`:当用户交付的是 **torch_npu profiler(`torch_npu.profiler` / MindStudio Insight)导出目录**时的文件关系、字段口径、已知陷阱与环境证据提取方法。字段名以用户实际文件表头为准;本文描述的是一种经实测验证的常见形态。
>
> **经实测验证的形态来源**:Ascend 910B 单卡、torch_npu profiler 导出(9 个文件:5 CSV + 1 JSON + 2 SQLite + 1 空标记文件 `analyse.done`)。

## 目录

1. [文件清单与相互关系](#1-文件清单与相互关系)
2. [统计口径:以哪个文件为准](#2-统计口径以哪个文件为准)
3. [已知解析陷阱](#3-已知解析陷阱)
4. [从数据自身提取环境证据](#4-从数据自身提取环境证据)
5. [时间线重建与步结构推断](#5-时间线重建与步结构推断)
6. [host 与 kernel 的对应(E2 证据的落地路径)](#6-host-与-kernel-的对应e2-证据的落地路径)

---

## 1. 文件清单与相互关系

| 文件 | 层级 | 内容 | 用途 |
|---|---|---|---|
| `op_statistic.csv` | device 聚合 | 按 **OP Type(kernel 类型)** 聚合:Count / Total / Min / Avg / Max Time(us) / Ratio(%) | 热点排行的现成口径;**应与 kernel_details 复算交叉验证后再引用** |
| `kernel_details.csv` | device 明细 | 每 kernel 一行:Type、Stream ID、Start Time(us)、Duration(us)、Input/Output Shapes 与 Data Types、PMU 计数(aic_mac_ratio、aiv_vec_ratio、cube_utilization 等) | **最权威的 device 证据源**:热点复算、时间线、序列识别、shape/dtype、瓶颈佐证 |
| `operator_details.csv` | host 明细 | 每 aten 算子调用一行:Name(`aten::…` 等)、Host Self/Total Duration、Device Self/Total Duration | host 开销与下发证据;**无时间戳列,不能做时间线**;行数可达数十万 |
| `api_statistic.csv` | host 聚合 | 按 acl API(如 `Add_Tiling`)聚合的 Time(us)/Count | 下发/tiling 开销佐证 |
| `task_time.csv` | device 任务 | 每 task:kernel_name/type、stream_id、task_start/stop(us);含 MEMCPY_ASYNC、PROFILING_ENABLE 等非计算任务 | 任务级时间线(与 kernel_details 互补) |
| `step_trace_time.csv` | 汇总 | Computing / Communication(Not Overlapped) / Overlapped / Free / Stage / Bubble | **窗口级时间构成**;见 §3 的 Step 为空与 Free 语义陷阱 |
| `trace_view.json` | 全量时间线 | Chrome trace 事件数组(cpu_op、kernel 等,`ts`/`dur` 微秒) | host↔kernel 对应(§6);文件可达数百 MB,须脚本处理 |
| `analysis.db` | 汇总 | StepTraceTime 表(同 step_trace_time.csv) | 冗余 |
| `ascend_pytorch_profiler_*.db` | 源数据库 | META_DATA、NPU_INFO、TASK、PYTORCH_API、CANN_API、SESSION_TIME_INFO 等 20+ 表 | **环境证据(§4)与 host↔kernel 对应(§6)的原始来源** |

**双计警告**:同一 kernel 执行会同时出现在 kernel_details.csv(device 侧)与 operator_details.csv 的 Device Duration 列(映射到宿主算子)。**热点统计只用 device 侧文件**;operator_details 的 Host Total Duration 含子调用,求和会重复计算,只做 host 开销分析用。

## 2. 统计口径:以哪个文件为准

1. **热点排行**:以 kernel_details.csv 按 **Type 列**聚合(不要用 Name 列——Name 含实例信息,如 `aclnnXxx_…AiCore_…`);与 op_statistic.csv 对比,两者应当一致(实测逐项一致);不一致时以自己从明细复算的为准并解释差异。
2. **占比分母 = 全部 kernel Duration 之和**(实测 op_statistic 的 Ratio 即此口径)。注意:
   - 多流重叠时,该总和可能超过 wall time,占比只反映"kernel 时间构成",不是"占 wall time 的比例"——两个口径都要报,写清是哪个;
   - 该总和与 step_trace_time 的 Computing 可能有小差异(实测差 0.11%),引用时注明;
   - **采集窗口时长 ≠ 计算时间**:Free(空闲)可能远大于 Computing。真实利用率 = ΣDuration ÷ 窗口时长。实测案例中 busy 仅占窗口 3.4%——若不报这个数,会把"kernel 时间构成"误当"负载画像"。
3. **单位**:Duration/Time 列为微秒(us);trace_view.json 的 `ts`/`dur` 也是微秒但 **ts 是字符串**;Start Time 是 epoch 微秒(绝对时刻),做时间线用差值。
4. **聚合键**:kernel_details 的 Type 列是 kernel 类型;operator_details 的 Name 列是 aten 算子名。两者粒度不同,不可直接互相验证"算子数"。

## 3. 已知解析陷阱(实测踩过)

| 陷阱 | 现象 | 处理 |
|---|---|---|
| 尾随制表符 | kernel_details/task_time 部分字段值尾部带 `\t`,直接 `float()` 报错 | 解析时对每个字段 `strip().strip('\t')` |
| epoch 微秒 | Start Time 为 ~1.79e15 量级 | 做差值用;不要当作"耗时" |
| ts 为字符串 | trace_view.json 的 `"ts": "1790…"` | 转 float |
| Step 列为空 | step_trace_time.csv 可能整行无步号 | 不能据此数步数;用 §5 方法从 kernel 明细推断 |
| Free >> Computing | 窗口大部分空闲是常态(如推理服务等请求、profil er 采集窗口含空闲) | 如实报告 busy 占比,不要把 Stage/Free 当异常,也不要把 Stage 当 step 时长 |
| Model ID = 4294967295 | 即 0xFFFFFFFF(无效值标记) | 忽略该列 |
| 无效 PMU 值 | 部分 kernel 的 PMU 列为 0.0 或 N/A | 用 PMU 前先过滤有效行 |
| **列存在但全空** | operator_details.csv 有 Call Stack 列但实测 0/565,496 行填充 | **判断采集配置必须查列内容,不能只看列名/schema**:with_stack 是否开启以 trace_view.json 是否含 `python_function` 事件为准 |
| **同名统计不同源** | db 的 TASK 表行数(实测 71,156)≠ task_time.csv 行数(实测 66,676) | 报告中每个引用数字**注明确切来源文件/表**;盘点表里某文件的行数要真的数那个文件 |

## 4. 从数据自身提取环境证据

环境信息缺失时,**先查数据再问用户**(顺序不可反)——下列证据就在 `ascend_pytorch_profiler_*.db` 里(用 `sqlite3` 只读打开,参考 `scripts/inspect_profiler_db.py`):

| 表 | 提供的证据 | 实测示例 |
|---|---|---|
| `NPU_INFO` | **NPU 型号**(id, name) | `Ascend910B` |
| `RANK_DEVICE_MAP` | rank ↔ device 映射(卡数线索) | 单卡:rank 0 → device 7 |
| `META_DATA` | 环境变量(如 `PYTORCH_NPU_ALLOC_CONF`)、trace id、db schema 版本 | `expandable_segments:True` |
| `SESSION_TIME_INFO` | 采集会话起止(ns) | 换算窗口时长 |
| `AICORE_FREQ` | AICore 频率曲线 | 佐证降频问题 |

提取不到的(CANN 版本、torch/torch_npu 版本、模型身份、框架配置)才进入"待补充信息"。**注意:NPU_INFO 给出型号 ≠ 给出版本;版本兼容性结论仍按 `ascend-operator-reference.md` §4 核实。**

## 5. 时间线重建与步结构推断

有 Start Time 时(kernel_details.csv 或 task_time.csv),按下列顺序产出(脚本:`scripts/timeline_analysis.py`):

1. **按流分组**后各自按 Start Time 排序(跨流不排序);主计算流通常是 kernel 数最多的流。
2. **busy/间隙**:每流 ΣDuration、区间并集、间隙分布(中位数/均值/总量)。**间隙总量与 busy 的比值是瓶颈分类的关键证据**——间隙占主导时,瓶颈是调度/下发(见 `profiling-analysis.md` §5),此时融合收益的主要来源是 launch 次数减少,纯带宽模型会严重失真。
3. **跨流重叠**:两流区间求交。重叠可忽略时,各流耗时可直接相加;重叠显著时,以并集为 wall 口径。
4. **步结构推断**:找一个"每步恰一次"的特征 kernel(如实测中 LM head 投影是唯一 dur>100us 的 MatMulV2,个数=步数;或用每步固定次数的算子数 ÷ 每层次数反推)。得到步数后再算每步 kernel 数与每步 busy。**不要假设采集窗口内步数均匀**。注意:选择不同的标记 kernel 会让"每步 kernel 数"略有出入(实测:按 LM head 切步得 362/步,按每步唯一的调度标记 kernel 切步得 369/步,差值 ≈ 归属边界不同的采样尾部小 kernel)。**报告步结构时必须写明用什么标记切步**,数字才可复现。
5. **prefill/decode 区分**:按特征算子的 shape 变化切分(实测:开头一批 batch>1 的 FA 后全是 batch=1 → 1 次 prefill + N 次 decode)。**报告"典型 shape"必须给实例分布**——按 Input Shapes 字符串分组计数取众数;首行/首步的 shape 可能是 prefill 或热身,不代表典型值(实测:FA 4480 次中 4452 次 q=[1,16,128]、仅首步 28 次为 [14,16,128];若拿首步 shape 当典型会得出"每步 14 token"的错误负载画像)。

## 6. host 与 kernel 的对应(E2 证据的落地路径)

E2(源码/宿主侧 N 个算子 → 时间线上 1 个 kernel)在本格式下的可执行路径:

1. **trace_view.json**(首选):`cpu_op`(host aten 调用)与 kernel 事件并存,按 `ts`/`dur` 的时间包含关系 + pid/tid 归属建立对应;host 调用链嵌套(aten::linear → aten::mm)可还原调用层级。
2. **db 表**:`PYTORCH_API`(host,含 inputShapes/callchainId)→ `CANN_API`(acl 层)→ `TASK`(device,含 streamId),按 startNs/endNs 时间对应;`connectionId` 可辅助关联。
3. **operator_details.csv 只有聚合列没有时间戳**——它不能用于 E2,只用于 host 开销分析(常见误用,注意)。

> 若时间包含关系出现"多个 host 调用落在同一 kernel 执行窗口内",这是融合的**强线索**(宿主展开 N 个算子、device 只见 1 个 kernel),但要先排除"窗口恰好重叠的独立调用"——用输入输出 shape 与调用嵌套确认。

## 7. 与其他参考文件的关系

- 瓶颈分类、证据链 E1–E5、置信度分级:见 `profiling-analysis.md`(本文不重复)。
- 序列模式 → 候选判定的后续流程:见 `fusion-rules.md`。
- 本文件描述的形态若与用户实际表头不符,以实际表头为准并在报告第 2 节说明差异。
