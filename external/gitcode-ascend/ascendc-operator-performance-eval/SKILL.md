---
name: external-gitcode-ascend-ascendc-operator-performance-eval
description: 在 ascend-kernel 的 csrc/ops/<op>/test 下维护仅含 JSONL 的 profiler 性能用例，使用 torch_npu.profiler（固定
  warmup=5、active=5）采集，汇总 ASCEND_PROFILER_OUTPUT/op_statistic.csv 的 Total Time(us)，输出含
  DType 列的统一 Markdown 对比报告（自定义算子 vs 标杆）。不生成 perf_cases.json 与 *_profiler_results.json。参考实现见
  examples/layer_norm_profiler_reference/。
argument-hint: 可选：算子名、对比 API（自定义算子 torch.ops.npu.* / 标杆 F.* 或 npu_*）、用例是否与精度 JSONL
  对齐。
original-name: ascendc-operator-performance-eval
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# AscendC 算子 torch_npu.profiler 性能评估

## 本技能目录内参考文件

执行本技能时，应优先使用 **本目录** 下材料：

| 文件 / 目录 | 用途 |
|-------------|------|
| **`SKILL.md`**（本文件） | 流程、目录约定、**完整 JSONL 用例规范**、报告结构、**固定 schedule** |
| **`references/REFERENCE_JSON_CASE_FORMAT.md`** | 与下文「性能用例 JSONL 规范」**同文** |
| **`references/REFERENCE_PROFILER_AND_METRICS.md`** | `torch_npu.profiler`、`op_statistic.csv`、`*_ascend_pt` 路径 |
| **`examples/sample_perf_cases.jsonl`** | 最小 LayerNorm 风格 JSONL，可复制改名 |
| **`examples/layer_norm_profiler_reference/`** | **Layer Norm 参考实现**（`layer_norm_profiler_common.py`、`benchmark_layer_norm_torch_npu_profiler.py`、用例 JSONL、说明）；新算子可复制该目录到 `csrc/ops/<op>/test/` 再替换前向与文件名 |

---

## 角色

在 **ascend-kernel** 中，为 `csrc/ops/<算子名>/` 建立可复用的 **profiler 性能用例** 与 **自定义算子 vs 标杆** 的 Markdown 报告流程。采集必须走 **`torch_npu.profiler`**，且 **`warmup` 与 `active` 固定为 5**（见下节）。细节见 **`references/REFERENCE_PROFILER_AND_METRICS.md`**。

**核心原则（两条强制约束）**：

1. **对比报告始终必须呈现**：无论标杆路径是标杆 API 还是小算子拼接，最终报告都必须含自定义算子 vs 标杆的双路径对比表，**禁止使用单路径报告替代**，**标杆必须在NPU上运行**。
2. **用例生成必须先读 `design.md`**：在生成任何 JSONL 用例之前，必须读取算子目录下的 `csrc/ops/<op>/design.md`，从中提取参数约束、典型 shape、支持的 dtype 及关键属性值，用例须覆盖设计文档中描述的所有执行模式。

---

## 用例来源：从 testcase-gen 用例文档加载（MANDATORY）

**在生成或修改任何 JSONL 用例之前**，**MUST** 首先读取 testcase-gen 产出的用例文档：

### Step 0：读取 testcase-gen 用例文档

```
READ csrc/ops/<op>/test/<op>-test-cases.md
```

从中提取：

| 提取项 | 在用例文档中的位置 | 用途 |
|--------|-------------------|------|
| SUPPORTED_DTYPES | §测试配置 | JSONL 用例的 dtype 覆盖范围 |
| TEST_SHAPES | §测试配置 | 小/中/大规模 shape 的选取基准 |
| GENERAL_SHAPES | §测试配置 | 泛化 shape，可补充用于性能场景 |
| NPU 调用方式 | §算子标杆 | 自定义算子的前向调用 |
| CPU 参考实现 | §算子标杆 | 标杆路径的参考实现 |

### testcase-gen 输出 → JSONL 用例转换规则

1. 从 TEST_SHAPES + GENERAL_SHAPES 中选取**代表性 shape**（覆盖小/中/大规模），避免重复
2. 每个 shape 遍历 SUPPORTED_DTYPES 中的全部 dtype
3. 结合 design.md 中的属性值（如 block_size、eps 等）填充 JSONL 的 `inputs` 字段
4. JSONL 用例总数 **≥ 8**
5. 算子标杆中的 NPU 调用方式和 CPU 参考实现用于构建自定义算子路径和标杆路径

> **若 `<op>-test-cases.md` 不存在**：回退为完全从 design.md 自行设计用例（按下方流程），但需在报告中注明"用例为自行设计，非 testcase-gen 产出"。

---

## 用例生成：必须先读 design.md（强制）

**在生成或修改任何 JSONL 用例之前**（无论是否已加载 testcase-gen 用例文档），必须执行以下步骤：

### Step 1：读取设计文档

```
READ csrc/ops/<op>/design.md
```

从中提取：

| 提取项 | 在 design.md 中的位置 | 用途 |
|--------|----------------------|------|
| 支持的数据类型 | §1「支持的数据类型」 | 用例的 dtype 覆盖范围 |
| 参数约束与取值范围 | §1「参数说明」约束条件列 | 属性值的合法范围（如 block_size ≤ 128） |
| 典型 shape / 输入规模 | §2「计算逻辑」/ §3「Tiling 策略」 | 小/中/大规模用例的 shape 基准 |
| 关键属性的模式组合 | §2「伪代码」/ §1「参数说明」 | 需要各自覆盖的执行路径（如 do_transpose=True/False、is_input_split=True/False） |
| 性能关键点 | §6「性能优化」/ §3「Tiling 策略」 | 影响性能的分支（如转置 vs 非转置走不同 DMA 路径） |

### Step 2：用例设计规则

| 规则 | 说明 |
|------|------|
| **覆盖所有执行模式** | design.md 描述了多个执行路径（如转置/非转置、input_split 模式）时，每种模式必须有至少一个用例 |
| **覆盖所有支持的 dtype** | 每种支持的数据类型至少有一组用例，典型中等规模 shape |
| **小/中/大规模 shape 各一组** | 小规模（内核 launch 开销主导）、中规模（典型生产场景）、大规模（访存带宽主导）各需覆盖 |
| **参数值来自约束范围** | 属性值（如 block_size）必须从 design.md 的约束条件中选取，不得随意设定 |
| **整数/索引张量值须语义合法** | win_lengths、offsets 等张量的具体值需满足算子语义（如 offsets 必须是合法的 window 起始偏移） |

### Step 3：验证用例

生成用例后检查：
- [ ] 所有 dtype 均已覆盖
- [ ] 所有执行模式（由 design.md 定义）均有对应用例
- [ ] 参数值（含属性值和整数张量值）在 design.md 约束范围内
- [ ] 包含至少一个「小 shape」用例和一个「大 shape」用例

---

## 参考路径决策树（强制）

性能评估**始终**需要双路径对比（自定义算子 vs 标杆），按以下顺序确定标杆路径：

```
算子是否有标杆等价 API？
  ├─ 是（如 torch.nn.functional.*、torch_npu 内置算子）
  │    └─ 使用标杆 API 作为标杆路径
  └─ 否（无标杆等价接口）
       └─ 必须实现「小算子拼接」标杆路径 ← 本技能的强制要求
            └─ 用设计文档 §「参考实现」或「伪代码」中的 PyTorch 基础算子组合实现
```

### 小算子拼接标杆路径要求

当无标杆等价接口时，**必须**：

1. **从 design.md 读取参考实现**：设计文档通常包含 PyTorch 参考实现（伪代码或 Python 函数），以该实现为基础构建标杆路径。
2. **使用 PyTorch 基础算子组合**：`torch.zeros`、切片赋值、`.permute()`、`torch.cat` 等。不得使用循环+Python 标量赋值（否则 profiler 采集的是 CPU 算子而非 NPU 算子，无法公平对比）；整个标杆实现必须以张量操作为主，可在 NPU 上执行。
3. **在报告中明确标注**：报告头部须写明「无标杆等价接口，标杆路径为小算子拼接」，列明所用的基础算子。
4. **对比表格必须呈现**：不得因「无标杆接口」而退化为单路径报告，必须保留「自定义算子 per-step」「标杆 per-step」「比值」三列。

**NEVER**：以「无标杆等价接口」为由输出单路径报告或跳过对比表。

---

## 固定 Profiler 步数（强制）

| 参数 | 值 | 说明 |
|------|-----|------|
| `warmup` | **5** | 不允许脚本或 CLI 改为其它值 |
| `active` | **5** | 不允许脚本或 CLI 改为其它值 |
| `wait` | 默认 `0` | 可保留 CLI 或常量，按需 |
| `repeat` | 默认 `1` | 简单场景固定为 1；若 `repeat>1`，须在文档中说明 CSV 选取语义 |

每步末尾必须 **`prof.step()`**；循环总步数 = `repeat * (wait + warmup + active)`。

---

## 文件落点（统一在算子 `test/` 子目录）

所有下列产物放在 **`ascend-kernel/csrc/ops/<op>/test/`**：

| 类别 | 命名约定（`<op>` 如 `layer_norm`） |
|------|-------------------------------------|
| 用例 **仅 JSONL** | **`<op>_perf_cases.jsonl`**（一行一个 JSON 对象）；**不维护、不生成** `<op>_perf_cases.json` |
| Markdown 报告 | **`<op>_torch_npu_profiler_report.md`**（唯一结构化结果落盘；**不生成** `<op>_torch_npu_profiler_results.json`） |
| Profiler 导出根目录 | **`test/profiler_trace/`**（或 `--trace-root` 覆盖） |

性能脚本、公共模块与上述文件 **同处 `test/`**。

---

## 性能用例 JSONL 完整规范

以下为用例文件的 **完整** 字段与类型说明；**仅使用 `.jsonl`** 作为用例载体。

### 1. 文件形态

| 形态 | 说明 |
|------|------|
| **JSONL** | 每行 **一个** JSON 对象，行尾换行；空行忽略。扩展名 **`.jsonl`**。 |

**禁止**在本流程中生成或与用例同步维护 `.json` 数组文件。

### 2. 单条用例顶层结构

每个用例对象 **必须** 含键 **`"inputs"`**，值为 **数组**。

**Layer Norm 示例**（算子不同则替换 `build_inputs` 约定，结构仍须含 `inputs` 数组）：

```json
{
  "inputs": [
    { "name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [8, 128] },
    { "name": "normalized_shape", "type": "attr", "required": true, "dtype": "int", "value": [128] },
    { "name": "use_affine", "type": "attr", "required": false, "dtype": "bool", "value": true },
    { "name": "eps", "type": "attr", "required": false, "dtype": "float", "value": 1e-05 }
  ]
}
```

- `inputs` 内各元素的 **`name` 在同一用例内唯一**。
- 其它算子：张量 / `tensor_list` / `attr` / 整数张量 `range` 等规则见 **`references/REFERENCE_JSON_CASE_FORMAT.md`**。

### 3. JSONL 完整示例（两行，Layer Norm）

```json
{"inputs":[{"name":"x","type":"tensor","required":true,"dtype":"float16","shape":[2,128]},{"name":"normalized_shape","type":"attr","required":true,"dtype":"int","value":[128]},{"name":"use_affine","type":"attr","required":false,"dtype":"bool","value":true},{"name":"eps","type":"attr","required":false,"dtype":"float","value":1e-05}]}
{"inputs":[{"name":"x","type":"tensor","required":true,"dtype":"float16","shape":[4,256]},{"name":"normalized_shape","type":"attr","required":true,"dtype":"int","value":[256]},{"name":"use_affine","type":"attr","required":false,"dtype":"bool","value":false},{"name":"eps","type":"attr","required":false,"dtype":"float","value":1e-05}]}
```

更完整的字段说明（`tensor_list`、`int` 张量 `range` 等）见 **`references/REFERENCE_JSON_CASE_FORMAT.md`**。

---

## Profiler 与目录语义（摘要）

- 每次 `with torch_npu.profiler.profile(...)`，在 handler 目录下生成 **以 `_ascend_pt` 为后缀** 的导出目录；CSV 路径 **`…/*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv`**。
- **每个用例、每种实现**（如 `custom` / `baseline`）**一次独立** `with`；子路径建议 `{trace_root}/{op_trace_tag}/{custom|baseline}/case_XXX/`，运行前清空 `case_XXX`。

详见 **`references/REFERENCE_PROFILER_AND_METRICS.md`**。

## 指标（摘要）

1. 对单次 `with` 对应 CSV：**各算子行 Total Time(us) 求和**。
2. **除以** `active×repeat`（`divisor_mode=active_steps` 时）或仅 `active`（`active_only`）。本技能固定 **`active=5`**；若 **`repeat=1`**，则 **`divisor = 5`**。

---

## 性能对比报告（Markdown）必备结构

报告格式严格参照 **`examples/sample_report.md`**，结构如下：

### 1. 标题

```markdown
# 性能评估结果
```

### 2. 对比表（统一单表，强制双路径）

所有用例在**同一张表**中展示，表头固定为 `Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比`。

示例：

```markdown
## 性能对比

| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |
| ---- | ----- | ----- | ------------- | -------- | -------------- |
| 0 | [128, 4096] | float16 | 9.75 | 10.10 | 1.036 |
| 1 | [128, 5120] | float16 | 10.52 | 9.39 | 0.893 |
| 2 | [128, 6144] | float16 | 10.99 | 14.36 | 1.307 |
| 3 | [64, 6400] | float16 | 9.13 | 9.49 | 1.040 |
| 4 | [2, 1024, 4096] | float16 | 57.01 | 84.92 | 1.490 |
| 5 | [2, 1024, 6144] | float16 | 73.80 | 139.56 | 1.891 |
| 6 | [1, 2048, 6400] | float16 | 75.60 | 143.09 | 1.893 |
| 7 | [64, 4096] | float32 | 8.45 | 7.14 | 0.846 |
```

### 3. 全量汇总

使用 `## 全量汇总` 二级标题，内含**键值对表**：

```markdown
## 全量汇总

| 指标 | 值 |
| ---- | -- |
| 用例数 | N |
| 平均 加速比（>1 表示自定义算子更快） | X.XXX |
| 自定义算子更优（比值>1） | M |
| 标杆更优（比值<1） | K |
```

紧接其下用 `### 按数据类型汇总` 三级标题，展示分 dtype 的汇总表：

```markdown
### 按数据类型汇总

| DType | 用例数 | 平均 加速比 | 自定义算子更优 | 标杆更优 |
| ----- | ------ | ------------------- | ------------- | -------- |
| float16 | 7 | 1.364 | 6 | 1 |
| float32 | 1 | 0.846 | 0 | 1 |
```

### 4. 简短分析

使用 `## 简短分析` 二级标题，列出 **≥3 条** 无序列表形式的简短结论，内容涵盖：整体趋势、不同 dtype / shape 规模差异、访存与计算特征等。

```markdown
## 简短分析

- 平均 加速比 大于 1，自定义算子整体略有优势。
- 大 shape 下自定义算子优势更明显，向量路径利用更充分。
- float32 小 shape 场景自定义算子略逊于标杆，可能与 kernel launch 开销占比较高有关。
```

### 其他约定
- **不写**与报告重复的 `*_profiler_results.json`；中间统计仅存在于脚本内存中并写入 Markdown。

---

## 对话内展示结果（MANDATORY）

生成 **`csrc/ops/<op>/test/<op>_torch_npu_profiler_report.md`**（或已存在且本次运行已更新）后，助手在**当前对话**的回复中 **MUST** 同时完成下列事项，**不得**只输出「报告已生成」和路径而不展示数据：

1. **粘贴主要性能内容**（用户无需打开文件即可阅读结论，展示内容与报告结构一致）：
   - **统一对比表**：表头 `Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比`，所有 dtype 在同一张表中展示。case 多时可截断并注明「其余见报告」。
   - **全量汇总**：键值对形式的汇总指标（用例数、平均比值、自定义算子/标杆更优条数）及按数据类型汇总表。
   - **简短分析**：**≥3 条** 无序列表结论，涵盖整体趋势、不同 dtype / shape 规模差异、访存与计算特征等。

**NEVER**：仅回复报告路径；NEVER 用「请自行打开 Markdown」替代在对话中展示核心数字与结论。

---

## 易错点

- `warmup` / `active` 被改为非 5，与技能约定不一致。
- 未使用 **`torch_npu.profiler`** 或 `prof.step()` 与 schedule 不一致。
- **`repeat>1`** 可能多份 `*_ascend_pt` 导出；按 mtime 取 CSV 时需说明语义。
- CSV 表头 BOM / 列名变化时须兼容 **Total Time(us)**。
- 自定义算子未注册时只做标杆路径；对比前须加载自定义库。
- **未读 `<op>-test-cases.md` 就直接自行设计用例**：testcase-gen 已生成统一用例文档，应优先从中提取 shape 和 dtype，避免重复设计。
- **未读 design.md 就直接生成用例**：导致 shape 不符合约束、缺少关键执行模式的覆盖（如漏掉 transpose=True 路径）。
- **以「无标杆等价接口」为由输出单路径报告**：必须实现小算子拼接标杆路径，始终输出双路径对比表。
- **小算子拼接用 Python 循环逐标量赋值**：profiler 采集的是 CPU 逻辑而非 NPU 算子，导致标杆路径耗时失真；标杆实现应以张量操作为主。
- **助手仅输出报告路径、未在当前对话中展示主要表格与汇总结论**（违反「对话内展示结果」）。

---

## 参考实现（`examples/layer_norm_profiler_reference/`）

与 **`ascend-kernel/csrc/ops/layer_norm/test/`** 中 profiler 相关文件保持同构，包含：

- `layer_norm_profiler_common.py`、`benchmark_layer_norm_torch_npu_profiler.py`
- **`layer_norm_perf_cases.jsonl`（仅 JSONL，无 `.json`）**
- `LAYER_NORM_PROFILER_PERF_GUIDE.md`、`README.md`

新算子从该目录 **整体拷贝** 到 `csrc/ops/<op>/test/`，再替换算子名、前向调用、`build_inputs` 与 trace 子目录名。若仓内 `layer_norm/test/` 有更新，应同步回 `examples/layer_norm_profiler_reference/`。

---

## 检查清单（助手自检）

- [ ] 已读取 **`csrc/ops/<op>/test/<op>-test-cases.md`**（若存在），从中提取 SUPPORTED_DTYPES、TEST_SHAPES、GENERAL_SHAPES、算子标杆
- [ ] 已读取 **`csrc/ops/<op>/design.md`**，从中提取 dtype、参数约束、典型 shape、执行模式
- [ ] 用例覆盖 design.md 中描述的**所有执行模式**（如 transpose/非transpose、input_split 模式等）
- [ ] 用例中参数值（属性值与整数张量值）均在 design.md 约束范围内
- [ ] 已确认标杆路径类型（标杆 API 或小算子拼接），并在报告头部明确标注
- [ ] 若无标杆等价接口，已实现**小算子拼接**标杆路径，且标杆实现以张量操作为主（非 Python 标量循环）
- [ ] 已用 **`torch_npu.profiler`**，且 **`warmup=5`、`active=5`** 未被改写
- [ ] 已生成或更新 **`<op>_torch_npu_profiler_report.md`**，格式与 `examples/sample_report.md` 一致：含 DType 列的统一对比表 + 全量汇总 + 按数据类型汇总 + 简短分析
- [ ] **已在当前对话中展示**含 DType 列的统一对比表、全量汇总、按数据类型汇总与 **≥3 条**简短分析结论，不仅附路径
- [ ] 已说明固定步数约定与指标口径
