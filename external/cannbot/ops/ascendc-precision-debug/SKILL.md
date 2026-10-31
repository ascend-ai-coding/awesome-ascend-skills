---
name: external-cannbot-ops-ascendc-precision-debug
description: 'Ascend（昇腾）算子精度问题定位与修复：NPU 输出与 golden 比对超差/不一致时，按"口径确认 → 误差形态 → 最小复现
  → 二分定界 → 模式匹配 → 源码深潜 → 修复回归 → 报告交付"流程定位根因。

  适用于：算子精度/正确性调试、精度问题修复、aclnn/单算子比对失败、误差形态分析、精度超差修复。不适用于：算子编译/构建失败、性能调优、未定界到单个算子的整网推理异常。'
original-name: ascendc-precision-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# Ascend 算子精度调试

## Rules（调试纪律）

1. **口径先行**：未确认 golden 来源、容差（必须外部给定，skill 无默认值）、输入分布前，禁止改代码；拿不到容差只输出误差事实（`references/tolerance.md`）。
2. **形态工具化**：误差形态结论必须来自工具的机读输出（指定验证工具的判定结果，或自写比对脚本的数值输出），禁止凭肉眼观察数据下结论（LLM 对数值数组的形态判断不可靠）；报告中引用其结论必须附机读输出，转述工具不存在的结论视为虚构证据。
3. **记忆优先**：分析前必须先查下方速查表，按命中读取 `patterns/`——未检索不得读源码（历史会话已反复验证：跳过检索必然绕路）。
4. **修完必回归**：修复有效性只由 Step 0 指定的验证工具判定，手写脚本仅作辅助证据（`references/verification.md`）。
5. **Reduce 类前置判断**：涉及 ReduceSum/Mean/Prod 类算子误差，必须先导出 NPU 实际输出，与 fp64 Golden/标杆真值逐位置比对差异分布（mismatch 位置 / sum 量级 / ±1 模式），再假设 kernel bug——禁止仅凭不含 NPU 实际输出的仿真下结论。

## 形态指纹速查表（Step 1 verdict 或直接观察到的现象 → 优先阅读的模式卡）

**两个入口**：主路径是 Step 1 拿到机读判定（误差形态/verdict）后查表；暂无机读判定时，允许用直接观察到的现象（如"批量跑失败、单跑通过"）直查本表——降级路径，事后仍须补齐机读依据。

| verdict / 现象 | 优先读 | 次选 |
| --- | --- | --- |
| 大输入才错 / 出现 inf | patterns/fp16-overflow.md | patterns/numerical-stability.md |
| 仅极端输入系统性失败（常规值全对；subnormal 窗口 / ±FLT_MAX 级输入） | patterns/ftz-subnormal-boundary.md | patterns/numerical-stability.md |
| 出现 nan（golden 无 nan） | patterns/numerical-stability.md | patterns/uninitialized-memory.md |
| 误差集中尾部 / 非对齐 shape 才错 | patterns/tail-block-misaligned.md | patterns/broadcast-misalignment.md |
| 同一输入多次运行结果偶发不一致 | patterns/pipe-sync-missing.md | patterns/double-buffer-stomp.md |
| 结果不可复现、输入全 0 输出非 0 | patterns/uninitialized-memory.md | patterns/multi-core-atomic-order.md |
| 均匀小误差、随 reduce 维增长 | patterns/reduce-accumulation-order.md | patterns/multi-core-atomic-order.md |
| 特定 shape 组合全错 | patterns/broadcast-misalignment.md | patterns/tail-block-misaligned.md |
| aclnn 直调对，单算子或整网 om 错 | patterns/atc-compile-chain-diff.md | — |
| 单独跑 PASS、批量跑 FAIL、结果跨用例漂移 | patterns/executor-cache-scalar-attr.md | patterns/uninitialized-memory.md |
| 仅特定属性符号/数据模式组合暴露 | patterns/api-mode-semantics-misuse.md | patterns/numerical-stability.md |

根因类别四分：**数值精度 / API 误用 / 框架机制 / 硬件行为（非算子 bug）**。模式匹配时先归类再选卡；判为硬件行为的，只声明不修复。

**修复红线**（历史会话反复踩坑的通用纪律，全程有效）：
- 框架机制类：修复必须对准机制本身（如缓存键漏维度就补全键的构成），禁止把关开关、绕路径等规避手段充作最终修复。
- **先实测最小方案**：改枚举/参数常量类修复必须先做并实测；仅当最小方案实测失败，才允许加结构性样板（分块/分 repeat 循环等），且写进注释的机理必须经实验验证——"平台仅支持某模式"类 kernel 注释、单一 impl 目录的断言、**参考实现的既有写法**均不构成免实测依据。结构性样板落地前，报告中必须已有最小方案的实测记录（改动内容+回归结果），无记录直接上样板视为违规。
- **否决须干净**：改动已生效而行为逐位不变时，先怀疑"新代码未在执行"（缓存/加载层拦截），排除前不得改判"假设错误"；多 bug 共存时，被否决的最小方案在其他 bug 修复后须干净复测一次，复测仍失败才允许上结构性样板。
- 剩余失败定性：golden 是 oracle，"NPU 结果更正确"不构成豁免。**宣布"硬件差异"前必须先按 golden 的运算顺序/语义重算，验证确实无法对齐**；豁免结论交用户确认，禁止自行宣布"软件无法修复"。
- **完成线非逐比特一致**（计算过程涉及浮点运算的算子）：修复以指定验证工具的容差判定为完成线，不以逐比特一致为目标——golden 的实现库（SLEEF/libm 等）自身存在 ulp 级噪声，逐比特一致只能靠逐位复刻该库达成，属偏离而非修复；容差/ulp 级残余已证实源于 golden 侧时，记录证据并按"剩余失败定性"交用户确认，禁止以复刻库实现为手段消除差异。
  - **适用范围**：计算链路中存在任何浮点运算步骤（如 int32→fp32 提升、fp32 sqrt/exp/div 等）的算子路径均适用，不限于"浮点 dtype 输入"——整数输入但内部走浮点计算的路径同样适用。
  - **1-ULP 豁免前提**：豁免不是跳过调试的借口。必须先逐元素比对中间值（用 PRINTF/DumpTensor 在运算前后打印 fp32 中间值与 golden 逐元素对比，见 [references/ascendc-dumptensor.md](references/ascendc-dumptensor.md) 与 [references/printf-debug.md](references/printf-debug.md)），确认差异确实是 1-ULP 级别（非逻辑 bug、非精度不足、非中间类型错误），并确认 1-ULP 差异的来源是硬件运算指令的舍入模式差异（查 `references/hardware-precision-api.md` 确认各运算指令的精度模式选项和默认舍入行为）。只有满足以上前提，才允许提示"容差定义可能有误，建议放宽"。
  - **bitwise_equal 合理性检查**：当容差为 `bitwise_equal` 但计算链路中存在浮点运算时（如 int32→fp32 精度丢失 + fp32 sqrt），`bitwise_equal` 从数学上不可能成立。此时应提示"容差定义可能有误，建议改为 max_relative 并给出 rtol/atol"，交用户确认。
- **修复不偏离设计**：存在设计文档（如 DESIGN.md）时，代码修复不得偏离设计——设计指定的配置/路径不得绕过，设计之外的算法/重构不得引入；认为设计本身有误时，以原始需求/spec 为准并交用户确认，禁止自行偏离。

**API 检索门禁（首次使用）**：首次对 `$ASCEND_HOME_PATH` 执行 grep/find 之前，必须先调用一次 `ascendc-docs-search` skill（若环境中有；其覆盖范围即 Ascend C API 文档与参考实现）——历史会话实测，先查语义约束说明曾省去十余分钟边界实验。声明未命中后才允许以 grep 回退（回退路径 `$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/`）；grep 时禁止凭 kernel 注释或单一 impl 目录的断言下"某模式不支持"的结论。

**变体确认**：`dav_*` 变体头文件按 `__NPU_ARCH__` 宏分芯片并存，grep 命中的变体可能不是目标芯片的生效变体。读 `dav_*/` 下的 impl/断言之前，先跑 `scripts/lookup_arch_variant.sh` 确认生效变体（ini 的 SoC_version→NpuArch/AIC_version）；只在生效变体目录下取证，非生效变体的断言不构成证据。DESIGN/上下文出现 `DAV_<N>` 时，读头文件同理只走 `__NPU_ARCH__ == <N>` 分支。

## 核心工作流

### Step 0 口径确认（未完成不得改代码）

- 确认 golden 来源、容差及来源（外部给定，禁止默认值；`references/tolerance.md`）、输入分布；拿不到容差向用户索要。
- 确认指定回归验证工具（名称/版本/调用/判定配置/覆盖范围，`references/verification.md`）；未指定向用户索要。
- golden 生成不规范 → 按 `references/golden-standard.md` 重建。
- 验证工具产生随机数据时，**从基线起固定随机种子**（TTK 工具加 `--seed=42`；自写脚本 `torch.manual_seed(42)`）——否则跨次失败计数差异不可归因。
- 确认工作区是否存在设计/需求文档（文件名不定，常见于 docs/ 下）；存在则记录路径，供 Step 5 对照用。
- 产出物：口径结论（容差值+来源）。**门禁**：口径有误 → 输出结论终止流程。

### Step 1 误差形态（强制工具化）

- 从测试框架导出 NPU 输出和 golden（如 TTK 的 `--dump out,golden`），逐位置计算误差并分析分布规律（误差位置、周期性、与 shape 轴/dtype 的关联），禁止仅凭 mere/mare 标量值下结论。
- 用指定验证工具或自写比对脚本产出机读判定：逐位置误差分布（max_abs/max_rel/err_ratio/first_bad_index/特殊值/verdict）；禁止肉眼读数下结论。
- 比对必须带 Step 0 确认的容差；容差缺失 → 只输出误差事实，不得声称超差/通过，回 Step 0 索要。
- 产出物：误差形态结论（附机读输出）。**门禁**：无机读判定不得进入 Step 2。

### Step 2 最小复现 + 稳定性分流

- 自写脚本缩到最小可复现 shape；全 1/全 0/单点异常优先。
- 稳定性分流：同输入连跑 3~5 次两两比对——结果一致 → 逻辑类（Step 3 继续）；不一致 → 同步/内存类（`references/dump-tools.md`）。
- **跨用例警告**：若观察到"单独跑 PASS、批量跑 FAIL"，禁止缩单用例，改做单跑 vs 批跑对比；命中后先做 `DISABLE_L2_CACHE=1` 全量回归（约 3 分钟证伪缓存污染），再走 patterns/executor-cache-scalar-attr.md。
- 产出物：最小复现 shape + 稳定性分流结论。

### Step 3 dump 二分定界

- 自写脚本从输出往输入逐 stage dump（`references/dump-tools.md`），找到首个发散 stage；禁止直接啃源码。
- 产出物：发散 stage 与输入条件（shape/dtype/attr）。

### Step 4 模式匹配（记忆优先）

- 按 {verdict 形态 × 发散 stage × dtype} 查上方速查表，读取对应 `patterns/` 卡片；高置信命中 → 严格按卡片"验证方法"确认，确认成立才允许修复。
- 查 Ascend C API 语义/参考实现按上方"API 检索门禁"执行。
- 未命中或验证不成立 → 进入 Step 5。

### Step 5 源码深潜（硬门禁）

- **硬门禁**：进入本步前，Step 1 机读判定与 Step 2 结论必须先存在，且已写下具体假设；只读假设相关函数，禁止通读；每读一段对照假设与证据。
- **设计文档对照（若 Step 0 定位到）**：用假设涉及的 API 名检索设计文档中的选型/参数/运算顺序决策，与实现对照——实现与设计的偏差即高优先级假设；设计文档可能有错（以原始需求/spec 为准），但偏差无论谁对谁错都是线索。

### Step 6 修复回归（指定工具是唯一完成依据）

- 指定工具回归：原失败用例 → 最小复现 → 边界集（极值/nan/inf/0/非对齐/多 shape）；仅原用例通过不算完成。
- 手写脚本仅作"辅助证据"，以其宣称修复完成 = 违规；工具未覆盖的边界补手写并标注，同时建议补入工具用例集。
- 根因报告必须包含"**API 语义依据来源**"字段：逐项列出调试中确认的 API 语义/枚举约束及其来源（ascendc-docs-search 检索结论 / grep 证据（文件:行） / 实测结果）——使 ascendc-docs-search 调用与否可审计。

### Step 7 报告交付

- 根因报告交付用户，必填字段：根因类别、复现维度、容差来源、回归验证工具、Agent 误判记录；案例归档与速查表更新由 skill 维护者完成。

## 资源导航

- `scripts/`：lookup_arch_variant.sh（芯片生效变体确认）、batch_precision_check.py（逐 case 精度批检，见下）、error_analysis.py（误差分布分析）、gen_boundary_test.py（边界用例生成）
- `patterns/`：模式卡；规则冲突以 SKILL.md 为准
- `references/`：tolerance.md（容差来源）/ golden-standard.md（golden 规范）/ verification.md（回归工具规约）/ dump-tools.md（dump 方法与稳定性分流）/ aclnn-api-notes.md（ACLNN 缓存机制与宏速查）/ hardware-precision-api.md（硬件精度 API 表）
- `references/`（补充深度参考）：ascendc-dumptensor.md（DumpTensor 插桩 7 步法，含 `ascendc-dumptensor-refs/` API 材料）/ printf-debug.md（PRINTF 调试法）/ simt-precision-debug.md（SIMT 精度调试）/ diagnosis-workflow.md（完整诊断工作流程）/ binary-search-debug.md（二分定界详细指南）/ common-traps.md（常见精度陷阱）/ data-comparison.md（数据对比法）/ tools-reference.md（工具命令参考）/ best-practices.md / case-studies.md（实战案例）/ debug-summary-template.md（调试总结模板）

## 逐 case 精度批检：`scripts/batch_precision_check.py`

要对一批 case 做"逐个比 actual vs golden、给 PASS/FAIL 表"时，用这个脚本，**别现写一次性比对脚本**
（Rule 2 要的"机读输出"就是它；各自现写会让判定口径不一致，结果之间没法横比）。

分工：调用方只负责把每个 case 的实测输出与 golden 各存一个 `.npy`；
比对、容差判定、**覆盖度校验**、表格、退出码由脚本负责。

```bash
python3 {skill_path}/scripts/batch_precision_check.py \
    --dump-dir <dir> --cases cases.csv \
    --rtol 1e-3 --atol 1e-5 \
    --label <方案标识> --out precision_report.md
```

dump 命名：`<dir>/case<ID>.actual.npy` + `case<ID>.golden.npy`（多输出加序号后缀 `.1` `.2`…）。

**容差没有默认值，必须显式传**（符合 Rule 1「容差必须外部给定」），取值走
`/ops-precision-standard` 的决策树。整型/布尔输出忽略容差、要求逐位相等；
NaN/Inf 的位置（含 Inf 符号）必须与 golden 一一对应。

**退出码**：`0` 全 PASS；`1` 有 case 超差 / NaN-Inf 错位 / shape 不符；
`2` **有 case 没有 dump**（根本没跑）。`2` 单独分出来，是因为"只测代表性用例"最常见的伪装
就是报告里全 PASS、但没跑的 case 压根不出现在表里 —— 覆盖度必须机器判。

## 收尾自检

- [ ] 容差标准与指定验证工具已确认并写入报告
- [ ] 误差形态结论已附机读判定输出（验证工具或比对脚本）
- [ ] 回归由指定验证工具执行并通过（非手写脚本），工具与版本已记录
- [ ] 根因报告已包含"API 语义依据来源"字段
