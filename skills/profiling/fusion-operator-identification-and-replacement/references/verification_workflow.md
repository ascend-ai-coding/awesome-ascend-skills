# 阶段二：验证工作流

本文件是 SKILL.md 阶段二的展开：逐个验证替换的完整流程、逐 step 数据记录、HTML 报告生成和 REPORT_DATA 字段说明。

## 进入条件

只有在用户输入包含**可以实际运行模型的脚本、命令或等价拉起方式**时，才进入完整验证流程。可运行入口至少要能复现 baseline 和替换后任务，通常包括：

- 训练/推理脚本或启动命令。
- 必要参数、配置文件、环境变量。
- 数据集、权重、tokenizer、checkpoint 等路径，或可获取方式。

缺少整网入口时记录整网 A/B 的具体阻塞。在已有设备、代表规格和参考实现上继续获准的局部对照，分别报告局部与整网证据，并继续验证其他具备条件的候选。

## 核心原则

- 代码可以一次改完，但**评测必须一个一个来**。
- 每次只验证一个替换，和未修改的 baseline 对比。
- **按业务契约验收**：只有适用的正确性门槛失败才跳过该项计时。布局或 dtype 导致接口无法执行时记录兼容性结果；精度一栏填“未取得输出”。随后执行授权队列的下一项。
- 长任务，需要 TaskList + 过程记录，方便中断后续接。

## 随机采样的正确性与性能

验证前记录目标分布或任务指标、代表参数和边界规格、种子、样本量及判定阈值。保留形状、dtype、原地更新、有限值及取值范围检查；若模型依赖生成器状态推进，同步核对该副作用。需要旧实现精确回放时另测逐样本一致性。

例如指数采样可对多个种子生成的样本检查经验 CDF 与目标指数分布的偏差，并以非均匀概率词表检验最终 token 频率。报告样本量、偏差和阈值，考虑样本相关性与多重比较。未观察到显著偏差只支持所测规格，不代替整网质量验证。若仅逐样本回放不同而概率语义达到业务门槛，继续在主规格预热、同步、交替计时，报告中位数与波动，再按局部结果决定整网 A/B。

## 单个确定性训练替换的流程示例

```
改这一个替换的代码
    │
    ▼
跑精度（50+ 迭代，对比 baseline loss）
    │
    ├── loss diff > 1% → 标注「不通过」，结束（不测性能）
    │
    └── loss diff ≤ 1%（或早期迭代就明显发散，提前结束）
            │
            └── 看每个 step 耗时是否有稳定下降
    │
    ▼
逐 step 数据写入 JSONL → 生成 HTML 报告 → 给出结论
```

## 精度评测

- 跑 50+ 个迭代。
- 修改前后 loss diff **在 1% 以内**算通过。
- **早停**：如果比较早的迭代就有明显精度 loss 差异，提前结束，没必要跑完浪费时间。
- 精度不满足 → 直接标注不通过，不进性能评测。没收益的替换不值得再花时间。

## 性能评测

看端到端每个 step 耗时是否有稳定下降（不是噪声波动）。端到端耗时有明显下降 = 有收益。


### （可选）所有候选验证完后统一做一次 profiling

当所有候选替换都逐个验证完毕、确认精度和性能都有收益后，可以统一做一次 profiling 采集，目的：

1. **确认算子已替换**：profiling 的 `kernel_details.csv` 里，替换前的小算子序列应消失，替换后的融合算子应出现。
2. **耗时对比**：替换前那段 device 算子序列（原始几个算子的总耗时）vs 替换后融合算子耗时。
3. **调用次数**：融合后调用次数应减少（多次 launch 合并为一次）。

注意，profiling 不要采集太久，1-2个step即可，够看就行。

## 对比口径

- 性能对比的 baseline 是**未做任何替换的原始版本**，不是"上一次替换后的版本"。
- 每个替换都独立对比 baseline，不在一个替换上叠加另一个替换的对比。这样每个替换的收益归因清晰。
- 精度也是每次和 baseline 对比，不累积。

---

## 逐 step 数据记录（JSONL）

验证期间，从训练/推理日志中提取每个 step 的关键指标，写入 JSONL 文件。

### 目录结构

```
fusion_verify/
├── baseline.jsonl           # baseline 逐 step 数据
├── rmsnorm.jsonl            # 候选 rmsnorm 逐 step 数据
├── rope.jsonl               # 候选 rope 逐 step 数据
└── ...
```

### JSONL 行格式

```json
{"step": 0, "loss": 12.598, "step_time": 2.927}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `step` | int | 是 | 迭代序号，从 0 开始 |
| `loss` | float | 是 | 该 step 的训练 loss |
| `step_time` | float | 是 | 该 step 的端到端耗时，**单位秒** |
| `phase` | str | 否 | `wait`/`warmup`/`active`/`tail`，用于区分不同阶段的步耗时特征；日志里有就填 |


### 数据来源

从训练日志中提取 stdout 日志、TensorBoard 事件、或训练脚本输出的 JSON/metrics 文件。不要编造数据；如果日志中没有逐 step 记录，只能记录能拿到的 step，其余标注为缺失。

---

## HTML 报告生成（模板填充）

每个候选验证完（精度通过或失败）后，**复制模板、替换 REPORT_DATA** 生成 HTML 报告。不通过脚本生成 HTML。

### 步骤

1. **合并 JSONL**：

```bash
python3 scripts/merge_jsonl.py fusion_verify/baseline.jsonl fusion_verify/rmsnorm.jsonl
# stdout → metrics JSON 数组，直接粘贴进模板
# stderr → thresholds（max_rel_diff、first_fail_step）
```

2. **复制模板**到工作目录，命名为 `validation_<candidate>.html`：

```bash
cp assets/validation_report_template.html validation_rmsnorm.html
```

3. **替换 REPORT_DATA**：把模板里的 `const REPORT_DATA = {...}` 整块替换为真实数据。需要填的字段：

```javascript
const REPORT_DATA = {
    basic_info: {
        task_name: "DeepSeek V4 Tiny Train",     // 模型/任务名
        candidate_name: "RMSNorm -> npu_rms_norm", // 候选名
        report_title: "",                         // 可选：覆盖自动生成的 HTML 标题
        baseline_status: "Completed",             // baseline 运行状态
        fused_status: "Completed",                // fused 运行状态
        final_conclusion: "Pass",                 // Pass / Fail / Skip / Blocked
        work_dir: "/path/to/work",                // 工作目录
        hardware: "Ascend NPU (Atlas A2)",        // 硬件
        world_size: 8,                            // world size
        rank_count: 8                             // rank 数
    },
    commands: {
        baseline_cmd: "torchrun ... train.py",    // 真实执行的 baseline 命令
        fused_cmd: "torchrun ... train.py --use_fused_rmsnorm", // 真实执行的 fused 命令
        script_path: "tmp/run_train.py",          // 启动脚本
        data_path: "/data/corpus",               // 数据路径
        ckpt_path: "/checkpoints/baseline"       // checkpoint 路径
    },
    diffs: [
        {
            file_path: "src/model.py",             // 修改的文件
            hunks: [
                {
                    old_line: 50,                  // 原始起始行号
                    new_line: 50,                  // 修改后起始行号
                    old_code: "...",               // 修改前代码（来自 git diff / 文件快照）
                    new_code: "..."                // 修改后代码
                }
            ]
        }
    ],
    metrics: [                                    // ← 从 merge_jsonl.py 输出粘贴
        { step: 0, phase: "wait", base_loss: 12.59, fused_loss: 12.59, base_time: 2.92, fused_time: 2.74 },
        // ...
    ],
    thresholds: {
        max_rel_diff: 0.01,                       // 精度阈值（小数，0.01 = 1%）
        first_fail_step: null                      // 首次偏差 step，精度通过则 null（从 merge_jsonl.py stderr 获取）
    }
};
```

> **命令必须来自实际执行记录**（shell history、日志头部、启动脚本），不能凭空编造。代码修改前后内容必须来自实际文件、git diff 或保存的快照。

4. **保存**，用浏览器打开查看。

模板使用 ECharts（CDN），内置完整的渲染和交互逻辑：
- Loss / Step time 双曲线对比（hover 显示 abs diff / rel diff / speedup）
- 对数/线性坐标切换
- 精度失败时 markPoint 标注首次偏差 step，性能区域显示遮罩"性能验证跳过：精度未通过"
- 逐 step 表格（搜索、hover 联动图表、dataZoom 缩放）

> 模板使用 ECharts（CDN: `cdn.jsdelivr.net`），查看报告的机器需能访问该 CDN。

---

## REPORT_DATA 字段说明

### basic_info（基本信息）

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_name` | string | 模型或任务名称 |
| `candidate_name` | string | 候选替换名称 |
| `report_title` | string | 可选。非空时覆盖标题；为空时 HTML 自动生成“模型/任务名 \| 候选替换名称 \| 算子融合替换验证报告” |
| `baseline_status` | string | baseline 运行状态（Completed / 未运行） |
| `fused_status` | string | fused 运行状态 |
| `final_conclusion` | string | `Pass` / `Fail` / `Skip` / `Blocked`（决定 badge 颜色和性能遮罩是否触发） |
| `work_dir` | string | 工作目录 |
| `hardware` | string | 硬件信息 |
| `world_size` | int | world size |
| `rank_count` | int | rank 数 |

### commands（运行命令）

| 字段 | 类型 | 说明 |
|------|------|------|
| `baseline_cmd` | string | **真实执行的** baseline 命令 |
| `fused_cmd` | string | **真实执行的** fused 命令 |
| `script_path` | string | 启动脚本路径 |
| `data_path` | string | 数据路径 |
| `ckpt_path` | string | checkpoint 路径 |

> 命令必须来自实际执行记录，不能编造。没有真实命令时填 `"信息缺失"`。

### diffs（代码修改对比）

数组，每个元素代表一个文件的修改：

| 字段 | 类型 | 说明 |
|------|------|------|
| `file_path` | string | 文件路径 |
| `hunks` | array | 修改块列表 |

每个 hunk：

| 字段 | 类型 | 说明 |
|------|------|------|
| `old_line` | int | 原始代码起始行号 |
| `new_line` | int | 修改后代码起始行号 |
| `old_code` | string | 修改前代码原文（`\n` 分隔多行） |
| `new_code` | string | 修改后代码原文 |

> 代码内容必须来自实际 git diff / 文件快照，不能用伪代码代替。

### metrics（逐 step 数据）

数组，每条对应一个 step。用 `scripts/merge_jsonl.py` 从 JSONL 合并得到：

| 字段 | 类型 | 说明 |
|------|------|------|
| `step` | int | 迭代序号 |
| `phase` | string | 可选。`wait`/`warmup`/`active`/`tail`；缺失时表格显示空 |
| `base_loss` | float | baseline 该 step 的 loss |
| `fused_loss` | float | fused 该 step 的 loss |
| `base_time` | float | baseline 该 step 耗时（秒） |
| `fused_time` | float | fused 该 step 耗时（秒） |

> 以下字段由模板 JS 自动计算，agent 不需要填：`abs_diff`、`rel_diff`、`time_delta`、`speedup`、`is_precision_fail`。

### thresholds（精度阈值）

| 字段 | 类型 | 说明 |
|------|------|------|
| `max_rel_diff` | float | 精度阈值（小数，0.01 = 1%） |
| `first_fail_step` | int / null | 首次明显偏差 step；精度通过则 `null` |

> 从 `merge_jsonl.py` 的 stderr 输出获取。

---

## 数据真实性要求

- **metrics**：必须来自 `merge_jsonl.py` 合并的 JSONL 数据，不能用模拟数据。
- **commands**：必须来自实际执行记录，不能编造。
- **diffs**：必须来自实际文件/git diff，不能用伪代码。
- 如果某些数据缺失，在对应字段填 `"信息缺失"` 或空值，不要用假数据填充。

---

## 长任务过程记录

阶段二是长耗时过程，需要重复拉起任务。必须有：

- **TaskList**：跟踪每个候选替换的验证状态（待测 / 精度验证中 / 精度不通过 / 性能验证中 / 已结论）。
- **结论记录**：每个替换的最终结论（是否替换、是否有收益、收益量级、失败原因）写进本地 markdown，方便中断后续接和最终汇总。

示例记录格式：

```markdown
## 替换验证记录

### 候选 1：RMSNorm → npu_rms_norm
- 状态：✅ 通过
- 精度：50 迭代 loss diff 0.3%，通过
- 性能：step 耗时降 8%，profiling 确认算子已替换，调用次数不变
- 结论：替换，有收益
- 报告：validation_rmsnorm.html

### 候选 2：Attention → npu_fusion_attention
- 状态：❌ 精度不通过
- 精度：第 5 迭代 loss diff 12%，早停
- 原因：causal mask 构建有误（排查中）
- 结论：暂不替换，修复 mask 后重测
- 报告：精度失败，性能跳过
```

## 何时全部结束

- 所有候选替换都验证完，或
- 用户决定停在某几个替换上，或
- 剩余替换预期收益太低不值得继续测。

## 阶段二报告的「逐项判定」表

### 性能回退归因

先用预热后的多轮配对测量确认回退超过波动。每个回退编号、每种回退表现不同的规格，都要解释原本希望减少什么（如启动次数、中间张量写回），实际省掉了什么，以及哪条执行路径增加了耗时。如果局部加速而整网收益落在波动内，单列整网结果及波动范围。

1. **补采证据。** 若只有耗时比，在同卡、同输入、同执行模式下分别运行 baseline 和替换实现。关闭 profiler 测主机提交时间及轮末同步后的总时间；另用 CPU/NPU profiler 给两条路径标记区间，按调用次数归一化，核对 kernel 名称、次数、设备时长及中间张量。图编译候选结合对应生成代码检查哪些操作实际合并、哪些仍分开调用。
2. **定位原因。** 核数减少时，对比新核与原整条链的设备耗时；核数不变时，查明预期消除的操作或读写是否仍存在。设备时间近似持平而总时间增长时，补测主机提交路径，必要时用函数采样定位包装函数、launcher 或运行时等待。主机提交可与设备执行重叠；保留各自的测量口径，不将两列相加或相减推算纯 CPU 时间。
3. **用表交付。** 在下表按「原 → 新」填写实测值和单位。「为什么变慢」栏用一句话串起预期收益、实际路径与耗时增量，例如“原想边除边取最大值；trace 显示仍写回商张量并独立运行 argmax，主要耗时增量出现在提交路径”。注明样本数、波动、计时和采集是否分轮进行，并给出对应日志或 trace 路径。只定位到设备核时，明确该核内部瓶颈仍待定位。无法补采时写明缺失证据和阻塞原因。

|编号及规格|预期节省|实际 kernel（名称、次数）|设备累计耗时|主机提交耗时|同步总耗时|为什么变慢及证据边界|
|---|---|---|---|---|---|---|

### 汇总表格式

训练和推理的最终报告使用同一张汇总表，每个阶段一编号占一行。逐项日志与训练 HTML 可作为明细。表头如下：

| 编号 | 原算子或计算 | 目标融合算子或实际替换实现 | 支持情况 | 不支持原因 | 精度对比 | 性能对比（原 → 新） | 结论 |
|---|---|---|---|---|---|---|---|

- **替换对象**：写出原调用或序列和目标接口全名。自定义实现写注册名；图编译写后端及编译边界内的算子；消除重复计算写实际代码变化。若仅做签名筛查，列出查过的目标接口。
- **支持情况与原因**：区分已上卡、已测整网、当前规格不支持、环境受阻、未获授权。不支持时注明设备、dtype、布局等具体冲突，以及结论来自签名核查还是设备实调；有报错时写关键诊断。已支持项的原因栏填“—”。
- **精度对比**：在相应规格下给出双方指标或输出差值与判定，例如各输出最大绝对差、token 匹配数、loss 偏差。随机采样分别写分布检验与固定种子回放的结果。精度门槛失败时保留实测差值，性能栏注明“未测：精度门槛失败”。
- **性能对比**：按相同负载列出原实现 → 替换实现的实测值和单位。区分算子耗时与整网延迟或吞吐，并注明代表规格、统计口径及波动；共用口径可写在表前。未测整网时只报告已取得的局部数据。
- **缺测与判定**：接口在当前规格无法执行时，精度和性能均填“—”，结论写“排除当前接口/规格”。环境或授权受限时标“待验证/范围外”，避免误判接口不支持。其余行给出推荐替换、保留原实现或暂缓的结论，以该行测过的实现与负载为限。
