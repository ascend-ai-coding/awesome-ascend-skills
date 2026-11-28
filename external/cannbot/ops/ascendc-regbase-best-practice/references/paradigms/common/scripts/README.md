# 存活节点计算脚本

> 配套两层 VF 设计流程（[../vf-split-fusion-rules.md](../vf-split-fusion-rules.md) 段级编排 + 各范式段内融合规则）的定量工具。
> 背景：逻辑存活节点（L）已从设计流程移除（无下游机器消费点，原 LLM 手写 trace 耗时约 9%），其防呆价值由本目录脚本以秒级计算保留。buffer 槽位映射同理——由 LLM 手推改为 allocate.py 程序化生成（调度顺序、pinned、分支结构等语义约束仍由 LLM 决策）。
>
> **allocate.py 的定位**：输入是融合方案定稿后的 **tensor 级物理流**（steps 顺序 = 调度决策，pinned / exclusive_branches 为语义决策），脚本按贪心策略**执行**配槽，产出 B0..Bn 复用映射与 B 名物理流（`--out`，直接作为 physical_p.py 输入）。分配只做"执行"不做"决策"；选槽即 WAR 检查（仍有未来消费者的内容不会被判死），正确性由构造保证。
>
> **physical_p.py 的定位**：输入是 allocate.py 产出的 B 名物理流（即已完成的槽位方案），脚本**验证**该方案的 P 峰值/WAR 合规性/下界自检。真实算子的复用链复杂（如 group_norm 13 步、多 buffer 复用），人工 trace 极易错算/漏算 WAR 违规，脚本保证零出错。生成器与验证器分离，physical_p.py 仍是最终裁判。

## liveness.py — L / P_baseline 计算器

| 模式 | 输入 | 输出 | 用途 |
|------|------|------|------|
| L（理论下界） | 计算图 JSON（tensor 级 inputs/outputs，按调度顺序） | L 值 + 三态 trace | P 的交叉校验下界（可选） |
| P_baseline | 同上 + `pinned`（独占分配 tensor） | P_baseline 值 | 不考虑 VF 融合的基线 P（段内融合分析的起点） |

支持 Tuple Reduce 互斥分支（`exclusive_branches`，各分支峰值取 max 不叠加）与跨阶段驻留排除（`excluded`）。

> **跨阶段驻留量输入契约（三脚本统一）**：设计 trace 中的跨阶段量（如驻 cacheBuf 的归约结果）作为"阶段外上下文"标注，**不参与计数、不进入任何一脚本的输入**（liveness dag / allocate tensor 流 / physical_p B 流同口径）；语义性表达读取用 `excluded`（可读不计持有）；cacheBuf / preReduceResultTail 等固定 extras 由范式文档 Buffer 汇总表单独预算，不进脚本输入。详见 vf-split-fusion-rules.md §12。

```bash
python3 liveness.py dag.json                     # L 模式
python3 liveness.py dag.json --pinned preReduceResult   # P_baseline 模式
python3 liveness.py dag.json --json              # 机器可读输出
python3 liveness.py --demo                       # 已知答案自检
```

## allocate.py — 槽位分配器（贪心配槽）

buffer 分配方案中映射的机械生成环节：给定调度顺序（steps 顺序即调度决策）、独占分配 tensor（pinned）与互斥分支声明，程序化产出复用槽位映射（B0..Bn）及 B 名物理流。

| 模式 | 输入 | 输出 | 用途 |
|------|------|------|------|
| 配槽 | tensor 级物理流 JSON（顺序 steps + `pinned` + `exclusive_branches`） | 槽位映射表 + B 名物理流 JSON（`--out`） | 复用映射零手推；选槽即 WAR 检查（构造性保证） |

贪心策略：按 steps 顺序登场，写入 tensor 时选编号最小且当前内容已死的槽复用，无可用槽则新开；本步内不假设 in-place，同步骤多产出各占独立槽（除非前序产出悬空）。直线流下该策略即最少槽位（区间图按登场顺序贪心最优）；互斥分支（Tuple Reduce）各分支共用物理槽，峰值取 max 不叠加。槽可用性按**变体视角**判定（回溯该变体可见的最后一次写入；公共写须在所有变体中安全）——tensor→槽的正确性由分配器构造保证，physical_p 只校验 B 名流，无法发现"读到的已是覆写后的内容"。约束：单赋值（SSA）、GM 输入须先 CopyIn、分支内产出只允许同分支消费、跨阶段驻留量不进本阶段输入、无消费者的产出（悬空 tensor）其槽**不复用、占槽到流程结束**——三脚本统一口径（liveness/physical_p「持有到流程结束」、allocate「恒阻断」），防悬空漏 CopyOut 导致 P 低估；真实输出应补 CopyOut 步骤，报告会对悬空产出提示。

```bash
python3 allocate.py flow.json                        # 打印映射表 + P 峰值（physical_p 语义复算）
python3 allocate.py flow.json --out flow.alloc.json  # 写出 B 名物理流（physical_p.py 输入）
python3 allocate.py --demo                           # 已知答案自检（含文档 §2 真值锚点）
```

## physical_p.py — P 峰值计算器（融合后物理 trace 预计算）

输入为 VF 段编排 + 融合方案确定后的**物理流** JSON（每 step 的 reads/writes 为 buffer 名，含复用与独占分配），输出 P 峰值、三态 trace、WAR 违规检查与**下界自检**（`P < 输入数+输出数` → 强制解释被寄存器链消掉的中间量，未解释清楚不得进入 tiling）。

```bash
python3 physical_p.py flow.json
python3 physical_p.py flow.json --json
python3 physical_p.py --demo
```

## 输入 schema

三个脚本的 step 均按**调度顺序**排列（顺序即调度决策，不同顺序可能得到不同峰值）。

allocate.py（tensor 级；reads/writes 为 tensor 名，单赋值 SSA）：

```json
{
  "inputs": ["x", "gamma"], "outputs": ["y"], "pinned": ["mu"],
  "steps": [
    {"id": "S1", "op": "CopyIn x",            "reads": [],            "writes": ["x"]},
    {"id": "S2", "op": "ReduceMean(axis=-1)", "reads": ["x"],         "writes": ["mu"]}
  ],
  "exclusive_branches": []
}
```

liveness.py：

```json
{
  "steps": [
    {"id": "S2", "op": "Mul", "inputs": ["x", "y"], "outputs": ["t1"]}
  ],
  "pinned": [], "excluded": [], "exclusive_branches": []
}
```

**两套 step 字段名与互转**：liveness.py 用 `inputs`/`outputs`（计算图层），allocate.py / physical_p.py 用 `reads`/`writes`（物理流层）。三步管线中间的转换即逐字段改名（语义一一对应），固定写法：

```python
dag = {"steps": [{"id": t["id"], "op": t["op"],
                  "inputs": t["reads"], "outputs": t["writes"]}
                 for t in alloc_flow["steps"]],
       "pinned": alloc_flow.get("pinned", []),
       "exclusive_branches": alloc_flow.get("exclusive_branches", [])}
```

physical_p.py：

```json
{
  "inputs": ["x", "y"], "outputs": ["out"],
  "pinned": ["B3"],
  "steps": [
    {"id": "S1", "op": "CopyIn", "reads": [], "writes": ["B0"]},
    {"id": "S2", "op": "asc_vf_call", "reads": ["B0", "B1"], "writes": ["B2"]}
  ],
  "exclusive_branches": []
}
```

## 与设计流程的衔接

```
计算图（S1..Sn）
  → liveness.py（P_baseline 预计算，可选 L 交叉校验）
  → VF 段编排（C1-C9）+ 段内融合穷举（三硬规则）
  → 语义约束声明（LLM：调度顺序 + pinned 独占 + 互斥分支结构）
  → allocate.py（贪心配槽 → B0..Bn 复用映射 + B 名物理流）
  → physical_p.py（P 峰值 + WAR 检查 + 下界自检）
  → P 结论表 / Buffer 分类汇总表 → tiling
```

LLM 负责：计算图推导、段编排判定、融合判据、**配槽语义约束声明（调度顺序 / pinned 集合 / 互斥分支）**、对脚本输出的复核与解释（下界自检触发时的解释、复用映射合法性说明写入 DESIGN.md）。脚本负责：机械的槽位映射生成、峰值计算与违规检查。physical_p 校验不通过（WAR / 超 P / 下界未解释）→ 回退调整语义约束（改调度、拆 VF 链降压）→ 重跑 allocate → 重验。

## 从伪码到 JSON：最小填写示例

以 `y = γ · mean(x)`（mean 沿最后一轴归约、γ 广播相乘，即 `y=(x−μ)·γ` 家族中「归约 + 广播乘」的最小形态）为例，走完「计算图 → dag.json → 约束声明 → allocate.py 配槽 → physical_p.py 验证」全程。

**第 1 步：写计算图**（S 编号即段编排决策记录的锚点）：

```
S1  x     = CopyIn(x)                // GM 输入
S2  mu    = ReduceMean(x, axis=-1)   // 主归约：默认高阶 API，触发 C8 天然段边界
S3  gamma = CopyIn(γ)                // GM 输入
S4  y     = Mul(brc(mu), gamma)      // 广播乘：应用段一条 VF 链
```

**第 2 步：dag.json（tensor 级，liveness.py 输入）**。GM 输入也要写成 CopyIn 步骤（脚本要求输入先存活）；`pinned` 留空跑 L 模式，命令行补 `--pinned mu` 得 P_baseline（mu 落独占 cacheBuf 时）：

```json
{
  "steps": [
    {"id": "S1", "op": "CopyIn x",             "inputs": [],              "outputs": ["x"]},
    {"id": "S2", "op": "ReduceMean(axis=-1)",  "inputs": ["x"],           "outputs": ["mu"]},
    {"id": "S3", "op": "CopyIn gamma",         "inputs": [],              "outputs": ["gamma"]},
    {"id": "S4", "op": "Mul(brc(mu), gamma)",  "inputs": ["mu", "gamma"], "outputs": ["y"]}
  ],
  "pinned": [], "excluded": [], "exclusive_branches": []
}
```

```bash
python3 liveness.py example-dag.json                 # L（理论下界）
python3 liveness.py example-dag.json --pinned mu     # P_baseline
```

输出结论行（三态 trace 逐 step 打印，峰值行带 `← 峰值` 标记）：

```
结论：L（理论下界） = 3，峰值步骤 = S4
结论：P_baseline（含 pinned ['mu']） = 3，峰值步骤 = S4
```

**第 3 步：约束声明 + example-alloc.json（tensor 级，allocate.py 输入）**。段编排定型（C8@S2 段边界，S4 为应用段 VF 链）后，LLM 只给三项语义决策：① 调度顺序（S1..S4 原序——顺序即调度决策）；② mu 落独占 cacheBuf（pinned）；③ 无互斥分支。写成 tensor 级物理流（复用纪律由脚本在选槽时强制：前序内容消费完才可重写，本步内不假设 in-place）：

```json
{
  "inputs": ["x", "gamma"],
  "outputs": ["y"],
  "pinned": ["mu"],
  "steps": [
    {"id": "S1", "op": "CopyIn x",             "reads": [],             "writes": ["x"]},
    {"id": "S2", "op": "ReduceMean(高阶API)",  "reads": ["x"],          "writes": ["mu"]},
    {"id": "S3", "op": "CopyIn gamma",         "reads": [],             "writes": ["gamma"]},
    {"id": "S4", "op": "VF: Mul(brc)",         "reads": ["gamma","mu"], "writes": ["y"]}
  ],
  "exclusive_branches": []
}
```

```bash
python3 allocate.py example-alloc.json --out example-flow.json
```

分配器输出映射：B0 先装 x、S2 消费完后由 S3 复用装 γ；B1 独占承载 mu；y 写 B2。`--out` 落盘的 B 名物理流即 example-flow.json（physical_p.py 输入）：

```json
{
  "inputs": ["x", "gamma"],
  "outputs": ["y"],
  "pinned": ["B1"],
  "steps": [
    {"id": "S1", "op": "CopyIn x → B0",             "reads": [],            "writes": ["B0"]},
    {"id": "S2", "op": "ReduceMean(高阶API) → B1",   "reads": ["B0"],        "writes": ["B1"]},
    {"id": "S3", "op": "CopyIn gamma → B0",          "reads": [],            "writes": ["B0"]},
    {"id": "S4", "op": "VF: Mul(brc) → B2",          "reads": ["B0", "B1"],  "writes": ["B2"]}
  ]
}
```

**第 4 步：physical_p.py 峰值验证**：

```bash
python3 physical_p.py example-flow.json
```

输出结论段（P 峰值 / WAR 检查 / 下界自检；B0 复用与独占 B1 可在 S3/S4 执行行看出）：

```
P = 3，峰值步骤 = S4（含 pinned ['B1']）
下界自检：通过（P=3 ≥ 输入2+输出1=3）
```

把上述 JSON 存为 example-dag.json / example-alloc.json 即可按本节命令复算（example-flow.json 即 allocate.py 的落盘产物）；字段名与 liveness.py / allocate.py / physical_p.py 源码 docstring 的输入 schema 一致。
