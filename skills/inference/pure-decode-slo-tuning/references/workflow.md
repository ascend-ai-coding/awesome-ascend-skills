# 纯 D 寻优执行流程

本流程从 A5 DeepSeek-V4.1-Flash-1017 的 10 ms 实验提炼；数值只是案例，不自动成为新任务的默认参数。**数据集、模型和权重由用户指定**；不要根据旧实验或共享盘上已有文件自行替用户选择。先完整记录用户对这些输入及 SLO、输出和请求数的要求，任何一次口径变更都废弃不兼容的旧排名，从基线重新运行所需阶段。

## 1. 实验契约与环境

- 保存原始启动脚本字节和 SHA256，另做变体，禁止覆盖证据。记录机器/卡数、镜像名称与 ID、安装包版本、模型和量化、共享盘、容器归属、已有 NPU 进程、可用空间。
- 从用户指定的数据集位置读取文件，核验 SHA256、行数、实际 input-token 分布；文件名中的“20k–40k”不证明每条都在范围内。确认用户指定的模型名称与权重位置对应且权重可读取；不能擅自换用同名缓存或其他版本。确定原顺序还是固定随机顺序，并将取数脚本、seed 和每轮前 N 条规则固化。两种顺序不可混排。
- 固化完整 benchmark 命令。A5 本轮要求 `--custom-output-len 700 --ignore-eos --disable-shuffle --temperature 0.6`，每条实际输出 700 token，`request-rate=inf`；下次以用户合同为准。`--custom-output-len` 与其他 `output-len` 选项不可只凭名称当作等价。
- 写清 SLO：例如 `ITL P50 / 固定真实 AL < 10 ms`、请求 100% 成功、`--gpu-memory-utilization ≤ 0.95`。阈值与比较符号、分位数、超时都不能含糊；TTFT 若不设 SLO，只是观测项。
- 检查固定 KV/RPC 端口与宿主机监听及临时端口区间，启动失败先查端口占用。只操作拥有或被明确授权的容器。

## 2. 真实接受率与假 KV 口径

1. 对每个要比较的 speculative K / draft sampling，用真实 Prefill+Decode 连接测试同一模型、镜像、输入分布和输出政策；这一步不注入 synthetic rejection。先 canary，再按声明样本数运行。记录完成/失败、input/output token、draft 数、accepted token、逐位置接受率、AL。
2. 用完整精度计算 `AL_K = 1 + rates[0] + … + rates[K-1]`。例如 5-token greedy 的 AL 不能直接当 4-token probabilistic 的分母。若用户指定全量 AL，则必须完成全量请求；仅跑 64 条就只称“64 条样本 AL”。
3. 假 KV 阶段在当前镜像确认 `DecodeBenchConnector`、`kv_role`、fill 模式、synthetic rejection、调度开关的实现和限制；不要因配置被解析就认为特性生效。A5 案例使用 `kv_both`、`fill_mean=fill_std=0`，且 `recompute_scheduler_enable=false`，这些是版本相关配置。
4. 启动全部 API rank 和代理，逐个查询 `/v1/models`，验证服务模型名；短请求检查 fake KV 实现接受率接近注入 rates。假 KV TTFT 不代表真实 Prefill 或 KV 网络耗时。
5. 计算 `TPOT Px = ITL Px / 真实 AL_K`（若用户采用此定义），报告 P50/P90/P99 和原始 ITL；绝不取假 KV 当前随机 AL 作每轮不同的分母。保存 benchmark 原生 TPOT 时明确其不同算法。

## 3. 原始基线和合法切分

从原始脚本的 MNS、MBT、HCCL、投机、显存参数开始；只改 DP×TP，检查卡数乘积、模型/核的 TP 限制和 Engram/EP 约束。每个切分冷启动、健康检查、8-rank 或等价逐 rank canary。对启动失败保存最早的内核/形状/OOM/端口错误与恢复结果。不可把 A5 某 TP 失败推广成全部硬件不可用。

## 4. MNS/MBT 组合与容量

对可运行切分生成预定 MNS×MBT 列表，遵守当前调度器的 token 下限；`MBT ≥ MNS×(1+K)` 仅在本镜像/任务约定适用时作为下限，不能要求一律取等号。先固定 HCCL 等其他参数：

1. 每组从低 cc 做少量请求 canary，粗爬坡覆盖低、中、高并发，找到 QPS 增长和 TPOT 触 SLO 附近。
2. 在边界细化，至少记录合规点及相邻较高点；若噪声或非单调，再测邻近点。每个点使用同一数据顺序；筛选样本按用户约定约 `3–4×cc`，不要每个点都跑 1000 条。
3. 每组记录最高**合规 QPS**点，不自动取最大 cc。实际 `Running`/`Waiting` 用 8 个 rank 聚合，观察代理分流和排队。
4. 如果某 MNS/MBT 组合因 `Get WinSize failed` 等明确缓冲问题无法启动，可按证据用更高 HCCL buffer 做兼容性复测，并在表中写清原缓冲值失败和救回值；这不等同于在并发爬坡同时优化 HCCL。其他 HCCL 候选放参数阶段。
5. 选出 MNS/MBT 阶段合规 QPS 最高的组合，未完成独立爬坡的组合不能参与排名。

## 5. 其他参数和投机配置

- 以胜出 MNS/MBT 为基线，一次改一个参数族。覆盖当前镜像适用的显存比例、block size、异步/同步调度、图/eager、HCCL buffer/算法、prefix cache、SWA bounded replay、共享 expert、权重/KV NZ 等。参数可用性需查运行镜像的源码/配置及实际启动日志；不适用则写证据。
- 会改变容量的参数重新做低并发 canary、粗/细爬坡和边界；单点对照只说明该 cc 的表现。启动 OOM、请求 OOM、超时和不支持配置分别标记，不能写 0 QPS。
- 投机 K 或采样法变化，先真实接受率，再对应 synthetic rates，最后独立爬坡。无投机对照可测试，但用 AL=1。输出长度固定并忽略 EOS 时，通过 `total_output_tokens = 完成请求数 × 预设输出长度` 核验。
- 两个开关一起加入最终脚本时，应另测组合；从单开关小增益不能推断叠加收益。未经重新爬坡的组合只能声明“已测 cc 合规”，不能继承旧脚本的 cc 边界。

## 6. 最终确认和恢复

按用户约定的口径，只对已选最终配置/cc 执行一次完整数据集确认。检查 `completed == 预期行数`、失败/超时、总输出 token、数据集哈希、QPS、输出 token/s、ITL、固定 AL 折算 TPOT、Running/Waiting 和各 rank 监控抓取错误。若最终全量不达标，不能回填短测为最终结果；回到边界或参数阶段。

单列无 SLO 限制下的成功吞吐峰值及其 TPOT。它说明极限，不代表可用容量。若用户要求服务恢复，重启并验证所有 rank 与代理；若要求停容器，先收集结果、后停本任务容器，验证 NPU 进程释放。停止不等于删除。

## 案例中的关键教训

- A5 的 10 ms 轮次曾因继承 30 ms 参数和遗漏 MNS/MBT 逐组爬坡而需要重来；只继承**方法顺序**，不继承旧结果或旧参数。
- 把 HCCL 通信兼容性救援与常规参数扫描区分，避免无意间把 HCCL 作为 MNS/MBT 爬坡的共同变量。
- `--ignore-eos`、温度和数据顺序变更会改变工作负载，旧排名作废。
- NZ2 与 KV NZ 单开关短测未见可辨认收益；组合追加后做了独立 1000 条确认，但未重新验证相邻 cc。报告必须区分这些证据等级。
