# {operator_name} MC2 通算融合算子开发计划

> `{operator_name}` → 实际算子名称。本文档在开发流程中持续更新。

本模板是 apace 路线算子的开发计划模板，在 DESIGN 定稿后由设计步骤填写，落地为项目内 `operators/<operator_name>/docs/PLAN.md`。PLAN 沿 MC2 开发主线组织：文件清单（REUSE/MODIFY）→ 通信设计基线（Win 区、Flag、通信轮次 T、localMatmul、tileCnt 扫描）→ 有序动作 → 接线 → 分层 checkpoint（编译 → Tiling 数值 → 多卡冒烟 → 精度 → 性能）→ 交付与回退。第 1、3、9、10 章在设计步骤定稿；第 2、4-8 章随开发更新；第 11 章只追加执行记录。

> 仅 `implementation_route=apace_native | apace_custom` 可以生成 PLAN。`unsupported` 只生成阻塞 DESIGN；补充 Investigation 后仍缺关键事实（如 Win 区容量、Flag 编排、T 推导）时不生成 DESIGN/PLAN。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | {operator_name} |
| 通信方向 | PUT / GET / compute-first（来源 DESIGN） |
| 编排形式 | 严格分离 / 时分复用（来源 DESIGN） |
| 数学公式 | y = f(x) + 通信语义描述 |
| 输入 | x1: shape=[...], dtype=...（含每 rank 输入 + **输入分布轴**：按哪轴切/不切） |
| 输出 | y: shape=[...], dtype=...（含每 rank 输出 + **输出分布轴**：按哪轴分布/不切） |
| Rank 规模 | rank_size=N，多卡启动方式（apace 直调 ST 为 fork 多进程 + TCP rootInfo 交换，见 host-and-testing.md §2） |
| 需求类型 | 特定用例 / 通用 |

本章内容与 DESIGN 保持一致，由设计步骤填写；开发中发现需求本身需要变化时，在第 5、11 章记录 `design_issue` 并回到设计步骤处理，不在 PLAN 中直接改写需求。

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `kernel/{op_name}_tiling_data.h` — Tiling 结构体（host/device 契约，含 CommTilingData 与 Win/Flag/T 字段） | ⬜ |
| `kernel/{op_name}_impl.h` — Impl：Init/Run/编排（AIC mm + AIV 通信/归约） | ⬜ |
| `kernel/{op_name}_frag_kernel.h` — FragmentTensor mm 内核（命名遵循 apace 惯例：`qmm_mx_kernel_{rs/ag/a2a}_frag.h`；compute-first 默认；PUT vendor 复用可省） | ⬜ |
| `kernel/reduce_sum_ref.h` — 增量归约（compute-first 专属；手动 UB 批量形态；其他路线可省） | ⬜ |
| `src/kernel_launcher.h` — `__global__` 入口（`KERNEL_TYPE_MIX_AIC_1_1`，dtype 变体） | ⬜ |
| `src/main.cpp` — Tiling + host 前置校验 + fork 多 rank + HCCL 建链 + launch（perf 模式内嵌 L2 flush） | ⬜ |
| `src/root_info_exchanger.h` / `src/utils.h` — TCP rootInfo 交换与公共工具（从参考算子复制，REUSE） | ⬜ |
| `scripts/gen_data.py` + `scripts/verify_result.py` — 数据生成与多 rank 校验 | ⬜ |
| `cases.csv` — 用例矩阵（含 tile 粒度参数列） | ⬜ |
| `run.sh` — 一键脚本（build/gen/run/verify/perf） | ⬜ |
| `CMakeLists.txt` — 独立工程构建（APACE_ROOT 指向 CANN 内置 apace，禁止复制共享层） | ⬜ |

每行标注 `REUSE`（复用已有文件）或 `MODIFY`（新建/修改）。实施阶段更新状态；新增项同时记录第 11 章。每行映射 §9 冻结合同的目标文件清单。`apace_native` 路线官方框架已覆盖全部需求时，kernel 侧可无新建文件（仅工程壳 + 接线），在清单首行显式注明"native 路线无 kernel 新建"。

---

## 3. 设计基线（冻结合同摘要）

| 合同项 | 冻结值 | DESIGN 引用 |
|--------|--------|------------|
| 通信方向与编排形式 | PUT/GET/compute-first；AIC↔AIV 触发关系 | DESIGN §x |
| Rank 规模与启动 | rank_size、rank_id 获取方式、启动命令 | DESIGN §x |
| Win 区分配 | 每 buffer 容量、偏移表（send/recv/workspace/flag 区） | DESIGN §x |
| Flag 编排 | flag 数量、含义、置位/等待时序（AIC→AIV / AIV→AIC） | DESIGN §x |
| 通信轮次 T 推导 | kernel 侧 `commTurn = splitAxisTileCnt + splitAxisTailCnt`（operator-anatomy.md §3）；compute-first 路线 host 派生默认 `T \| mSeg` 无尾块、单尾块 ≤ 头块可直传（多尾块/尾>头走 padding 策略，其他路线按其语义推导），推导过程与取值依据 | DESIGN §x |
| L2 flush 集成 | flush 时机（每轮 / 每 T 轮）、调用点 | DESIGN §x |
| tileCnt 扫描策略 | 固定 / 二分扫描范围、选择判据 | DESIGN §x |
| 精度标准 | 多 rank 精度矩阵对比阈值、非有限值门禁 | DESIGN §x |

本章各项取自已定稿的 DESIGN，开发阶段只读；需要变更时记录 `design_issue` 并回到设计步骤。

---

## 4. 有序开发动作

| 顺序 | 动作 | 前置 | 预期输出 | 验证 |
|------|------|------|---------|------|
| A1 | 工程框架 + CMake + 空 Kernel 编译 | 无 | 可执行文件编译通过 | build 成功 |
| A2 | TilingData + Host Tiling（Win offset/T/flag 计算） | A1 | Tiling 单测/打印正确 | 数值核对 |
| A3 | Kernel 实现（通信+计算协同、Flag 时序、L2 flush） | A2 | 编译通过 | 静态走查 |
| A4 | 冒烟：最小 shape 单轮通信跑通 | A3 | 多卡运行不挂死 | 退出码=0 |
| A5 | 精度：多 rank 精度矩阵全量用例 | A4 | §6 全部通过 | verify 通过 |
| A6 | 性能：tileCnt 扫描 + msprof 采集 | A5 | §7 达标判定 | 性能基线对比 |

顺序与前置关系必须确定；不能只写"参照文档实现"。失败回退见 §9。

---

## 5. 接线合同（Wiring）

| 接线项 | 内容 |
|--------|------|
| CMake target | 可执行文件名、源文件列表、compile flags |
| include 路径 | CANN 内置 apace 头文件路径（APACE_ROOT 直引，禁止复制共享层）、HCCL 头文件路径、kernel/host 共用 tiling 头路径 |
| 链接库 | libhccl / ascendcl / runtime 库及链接顺序 |
| Win 区接线 | 每逻辑 buffer → Win 区物理偏移映射表 |
| Flag 接线 | 每 flag 索引 → 触发方/等待方/时序点映射 |
| Launcher 接线 | rank_id/rank_size 传入路径、device 绑定、IPC/句柄初始化 |

本章只把 DESIGN 已确定的 Win/Flag/通信合同落到具体工程接线，不引入 DESIGN 之外的新设计。

---

## 6. Checkpoints（分层验证）

| checkpoint | 在动作后 | 范围 | 预期结果 | 完成条件 |
|-----------|---------|------|---------|---------|
| C1 编译 | A1 | 构建 | 编译链接通过 | 产物存在 |
| C2 Tiling | A2 | host | offset/T/flag 数值正确 | 与 §3 冻结值一致 |
| C3 冒烟 | A4 | 设备 | 最小用例多卡不挂死、退出码 0 | 日志无 timeout/hang |
| C4 精度 | A5 | 设备 | 多 rank 精度矩阵全部通过 | Max Diff ≤ 阈值 |
| C5 性能 | A6 | 设备 | msprof 数据归档、达标判定 | 满足 DESIGN 性能标准 |

每个上设备的 checkpoint 必须记录执行上下文（rank_size、设备节点、实际命令、返回码），并区分实现失败、启动前风险与环境不可用；只有真实多卡执行证据才能判定该 checkpoint 通过。

---

## 7. 交付、清理与回滚

| 交付件 | 来源动作 | 验收 checkpoint | 清理要求 |
|--------|---------|----------------|---------|
| 可执行文件 + 启动脚本 | A1-A3 | C1/C3 | 保留 |
| 精度报告（多 rank 矩阵） | A5 | C4 | 归档 docs/ |
| 性能数据（msprof + tileCnt 扫描表） | A6 | C5 | 归档 docs/perf/ |

回滚规则：实现层失败（编译/精度/挂死）→ Step 4 内诊断修复并重跑受影响 checkpoint；需改变通信方向、Win/Flag 编排、T 语义、支持域或验收标准 → 回 Step 3；缺 MC2 源码事实 → 回 Step 2；环境/设备不可用 → 记录为环境阻塞，不当作设计问题回退。备选方案不会因失败自动启用。

---

## 8. 实施阶段持续更新区

### 8.1 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|

### 8.2 多 rank 精度矩阵结果

**状态**: ⬜

| 用例 | rank_size | shape | dtype | 各 rank Max Diff | 结果 |
|------|----------|-------|-------|-----------------|------|
| P1 随机数据 | | | | | ⬜ |
| P2 零值 | | | | | ⬜ |
| P3 边界/tail | | | | | ⬜ |

### 8.3 性能结果

**状态**: ⬜ | **数据**: docs/perf/round_NNN/

| tileCnt | Task Duration | 通信占比 | 判定 |
|---------|--------------|---------|------|

实施阶段持续追加，不覆盖历史行；不改变冻结的精度与性能验收标准。

---

## 9. 冻结开发合同

### 9.1 PLAN Metadata 与 DESIGN 绑定

```text
plan_template_provider: /ascendc-mc2-best-practice
plan_template_path: references/foundations/apace/operator-design/plan-template.md
project_root:
operator_name:
operator_root: <project-root>/operators/<operator_name>/
design_path: operators/<operator_name>/docs/DESIGN.md
implementation_route: apace_native | apace_custom
selected_scenario: <仅 apace_custom 填写，场景注册表 scenarios/index.md 的语义命中场景 ID>
communication_direction: PUT | GET | compute-first
orchestration_form:
rank_size:
multi_rank_launch:
cann_version: <Step 1 environment.md 实测登记的 CANN 版本>
apace_source_ref: <APACE_ROOT 实测路径 或 fetch_apace.sh manifest 的 pin commit>
win_area_layout_ref:
flag_orchestration_ref:
rounds_t_derivation_ref:
l2_flush_integration_ref:
tilecnt_scan_strategy_ref:
precision_matrix_contract_ref:
design_frozen_status: ready | blocking
plan_owner: design_stage
execution_owner: implementation_stage
frozen_plan_status: ready | blocking
```

`frozen_plan_status=ready` 表示设计基线章节（1、3、9、10）已定稿，第 2、4-8、11 章仍随开发更新。

### 9.2 目标文件与交付范围

`target_file_manifest` 每项包含：`file_id / target_file / file_role / action_type: create | copy_and_adapt | modify | read_only / source_refs / design_contract_refs / expected_artifact`。所有 `target_file` 以 `operators/<operator_name>/` 开头；apace 官方源码（CANN 内置框架）与参考算子只登记为只读来源，禁止修改。

### 9.3 有序开发动作

`ordered_actions` 每项包含：`action_id / sequence / design_contract_refs / source_refs / target_files / action / prerequisites / expected_output / verification / failure_rollback`。涉及 Win 区 buffer、Flag、Tiling、Kernel entry 或 Launcher 的动作必须引用 §3 对应冻结合同项。

### 9.4 分层验证 Checkpoint

`validation_checkpoints` 每项包含：`checkpoint_id / after_action_ids / test_scope / inputs / expected_result / evidence_to_record / failure_rollback`。覆盖静态合同核对（Win offset/Flag/T 与 TilingData 一致）、构建、冒烟、多 rank 精度、边界/tail、重复运行与最终回归。第一个设备 checkpoint 前必须静态核对 Win 区偏移与 Flag 时序闭合，缺失不允许启动 Kernel。

### 9.5 失败停止与回退

- apace 源码事实被推翻：保留证据回 Step 2；
- DESIGN/PLAN 缺启动实现必需事实（Win 容量、Flag 编排、T 推导、tileCnt 策略）：回 Step 3；
- 多卡挂死/通信超时：先按 troubleshooting  playbook 诊断（Flag 时序、L2 flush、Win 溢出），确认为实现问题则 Step 4 内修复；
- 修复显示必须改变通信方向、编排形式、Win/Flag 语义、支持范围或验收标准：回 Step 3；
- 设备/NPU 资源不可用：经 `qrun` 排队执行，记录为环境阻塞，不当作设计问题回退。

**章节维护分工**：第 1、3、9、10 章由设计步骤定稿，开发阶段不改动；第 2、4-8 章由开发阶段持续更新；第 11 章只追加。设计基线需要变化时，记录 `design_issue` 并回到设计步骤。

---

## 10. 冻结验收合同（Readiness 门禁）

- `implementation_route` 只能为 `apace_native` 或 `apace_custom`；
- 通信方向、编排形式、rank_size、Win 区布局、Flag 编排、T 推导、L2 flush、tileCnt 策略、多 rank 精度矩阵均有冻结值且无 TBD；
- 所有 action 字段完整，依赖顺序无环（编译 → 冒烟 → 精度 → 性能）；
- 每个验证合同有 checkpoint，多 rank 精度矩阵每个用例可追溯 DESIGN 需求；
- 多卡启动命令、Win/Flag 接线、链接库均已在 §5 闭合；
- 所有交付件均可追溯到动作与 checkpoint；无 TBD、无未决分支、无未解除的 blocking。

---

## 11. 执行记录（只追加）

设计步骤只建立空表；开发阶段只追加：

```text
执行记录:
  - 记录编号
    动作或检查点编号
    起止时间
    实际变更文件
    实际输出
    结果: 完成 | 失败 | 环境阻塞
    证据引用
    与计划的偏差
    已执行的回退
    下一步动作或返回步骤
```

任何改变通信方向、编排形式、Win/Flag 语义、T 语义、验收标准或支持域的 deviation 都必须停止；项目根内新增文件、实现调整、tileCnt 实际值、同步实现或配置修复应更新第 2、4-8 章并追加执行记录，不修改第 9、10 章或 DESIGN。
