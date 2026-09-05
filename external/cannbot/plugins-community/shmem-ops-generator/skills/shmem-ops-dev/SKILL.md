---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-dev
description: 基于 SHMEM 的通信算子与通算融合算子端到端开发编排器，串联需求→设计→代码→编译→验证→性能全流程。触发：用户需要端到端开发、验证、评测或优化
  SHMEM 通信/通算融合算子时使用。关键词：基于SHMEM算子开发、端到端开发、算子开发、编排、工作流、通信算子、通算融合。
original-name: shmem-ops-dev
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 算子端到端开发编排

**Skill类型**：流程导向型（Phase 0–7 工作流，含子阶段 5.5 / 6.5，子技能串行编排）

> ## ⚠ Phase 0 启动门禁（用户 @ 指定本 skill 时 — 首读本节）
>
> **触发**：用户 **主动 @ / 引用** 本 skill（路径可为 `community/Op/shmem-ops-dev/SKILL.md`、`.agents/skills/shmem-ops-dev/SKILL.md` 等）时生效（须在对话中引用后加载，不会自动注入）。
>
> **Fresh Session**：重新做 Phase 0；@ 引用时 **MUST** 上溯 `skills_root`。
>
> **@ 指定后的首步（MUST）**：
> 1. Read 本 `SKILL.md`
> 2. Read [`references/askquestion-template.md`](references/askquestion-template.md)（**Catlass 对齐：先探测环境，缺项再问**）
> 3. Read [`references/agent-execution-contract.md`](references/agent-execution-contract.md)
> 4. Read [`references/shmem-repo-resolution.md`](references/shmem-repo-resolution.md)
> 5. 需要 Docker/NPU 时 Read [`references/docker-exec-contract.md`](references/docker-exec-contract.md)
> 6. 执行 Step 0.1 环境探测 → Step 0.2 补齐业务项（表单 / 聊天 / YAML 确认）
>
> **默认**：`custom-ops/<op>/`，CANN 已探测则**不重复问**。用户明确要求 in-tree 或通算融合（`fused_compute_comm`）时按用户意图走对应模板，**禁止**删除通算能力。
>
> 清单：[`intake-checklist.md`](references/intake-checklist.md) · 模板：[`askquestion-template.md`](references/askquestion-template.md)
> **通算融合开发规范（`op_kind=fused_compute_comm` 时 MUST Read）**：[`references/fused-compute-phase-gates.md`](references/fused-compute-phase-gates.md)
**Docker 执行（用户指定容器时全部命令进容器）**：[`references/docker-exec-contract.md`](references/docker-exec-contract.md)

本 skill 编排九个专职子 skill，驱动 SHMEM 通信算子与通算融合算子从需求到可交付状态。

## 核心原则

0. **中文写作**：所有交付的 `.md` 文档（design.md、review-report.md、performance_report.md 等）必须使用中文撰写。仅 API 名称、代码片段、变量名、数学公式等技术术语保留英文原文，其余描述、分析、结论一律使用中文
1. **阶段串行（含条件性子阶段）**：需求确认 → 设计 → 用例生成 → 代码生成 → 编译正确性 → 代码走读 → Torch 接入 → 性能采集 → 达标进入最终交付；未达标且 `performance_auto_optim: true` 进入性能优化（Phase 6.5，进入后跑满 `max_opt_rounds` 默认 5 轮，达标不是提前停止理由；用户明确指定轮次时跑满指定轮次）→ 最终交付，严格顺序执行
2. **子技能执行**：每个阶段 **MUST** 调用对应子 skill，不得自行实现
3. **阶段门控**：前一阶段MUST检查全部通过后才进入下一阶段
4. **设计驱动编码**：代码生成依赖设计文档中的 Canonical DSL 和六类模块设计
5. **先 correctness 再 performance**：功能或正确性未通过时，不做性能优化
6. **性能优化轮次**：默认「未达标 + `performance_auto_optim: true`」才触发 Phase 6.5；进入后 **MUST** 跑满 `max_opt_rounds`（默认 5）轮，达标 **不是** 提前停止理由；用户明确指定轮次时跑满指定轮次。每轮记录假设、改动、正确性复测和性能结果
7. **结果可视化**：关键结果在聊天界面展示，不要仅输出路径。**Phase 6/6.5 MUST 自动输出性能表**（见 perf-chat-output-spec.md），不得等用户追问
8. **目录结构强制**：所有 `.md` 文档归入 `docs/`，所有 `.cpp/.h` 源文件归入 `src/`，测试脚本归入 `scripts/`。不得扁平散放
9. **不静默修改核心库**：只有 gap analysis 明确授权时才允许改 SHMEM 核心能力

## 可用子 Skill 清单

| Skill | 路径 | 职责 |
| --- | --- | --- |
| `shmem-ops-design` | `shmem-ops-design/SKILL.md` | 将需求转化为 design.md（通信算子 / 通算融合，Canonical DSL + 契约） |
| `shmem-ops-testcase-gen` | `shmem-ops-testcase-gen/SKILL.md` | 生成 case matrix、golden/checker、测试脚本 |
| `shmem-ops-code-gen` | `shmem-ops-code-gen/SKILL.md` | 根据 design.md 生成通信算子或通算融合算子代码、CMake、README |
| `shmem-ops-compile-debug` | `shmem-ops-compile-debug/SKILL.md` | 编译、运行、失败分类和调试 |
| `shmem-ops-code-review` | `shmem-ops-code-review/SKILL.md` | 实现与设计一致性走读 |
| `shmem-ops-correctness-eval` | `shmem-ops-correctness-eval/SKILL.md` | 正确性契约验证和报告 |
| `shmem-ops-torch-bind` | `shmem-ops-torch-bind/SKILL.md` | 封装为 PyTorch CustomClass，生成 Python 测试并验证 |
| `shmem-ops-performance-eval` | `shmem-ops-performance-eval/SKILL.md` | 性能采集、baseline 对比、瓶颈分析 |
| `shmem-ops-performance-optim` | `shmem-ops-performance-optim/SKILL.md` | 性能优化迭代（进入后跑满 `max_opt_rounds` 默认 5 轮） |

## 工作流总览

```
Phase 0       Phase 1       Phase 2         Phase 3        Phase 4            Phase 5       Phase 5.5       Phase 6
需求环境确认 ──▶ 设计文档 ──▶ 用例生成 ──▶ 代码生成 ──▶ 正确性验证 ──▶ 代码走读 ──▶ Torch 接入 ──▶ 性能采集
—              design     testcase-gen   code-gen    correctness-eval  code-review  torch-bind    performance-eval
                                                  │
         ┌────────────────────────────────────────┤
         │ 达标 ─────────────────────────────────▶ Phase 7 交付
         │ performance_auto_optim:false ─▶ Phase 7（仅报告，不改 kernel）
         └ 未达标 + performance_auto_optim:true ─▶ Phase 6.5（进入后跑满 max_opt_rounds 默认 5 轮，达标不是提前停止理由；用户指定轮次时跑满指定轮次）
                                                        │
                                                        ├── torch_required:false ──▶ Phase 7 交付
                                                        └── torch_required:true  ──▶ Torch 回归门禁 ──▶ Phase 7 交付

输入: 算子需求 + 环境                            输出: 可交付算子 + Torch 扩展 + 测试 + 性能报告
```

> Phase 6 仅在 `meta.performance_required: true` 时执行；Phase 6.5 在 **未达标** 且 `meta.performance_auto_optim: true` 时由编排器 **自动** 进入（进入后 **MUST** 跑满 `max_opt_rounds` 默认 5 轮，达标 **不是** 提前停止理由；用户明确指定优化轮次时 **MUST** 跑满指定轮次，即使 Round 0 已达标）。

## 反模式清单（NEVER DO THESE）

### Phase 0

- ❌ `ASCEND_HOME_PATH` 已设置仍强制重复无信息的 CANN 问卷（对齐 Catlass：已探测则沿用）
- ❌ standalone（agent-skills）下询问 custom-ops vs in-tree
- ❌ 未完成 Phase 0（环境+业务确认）就 Read 源码 / 编译 / 调子 skill
- ❌ `ASCEND_HOME_PATH` 未设置时静默 `source` 默认路径（未探测到则 **MUST** 询问）
- ❌ 无表单/聊天确认通道时，静默默认 Torch/性能开关
- ❌ 用户已指定 Docker，却在宿主机执行编译/跑数（见 docker-exec-contract.md）
- ❌ 混用 `docker exec` 与宿主机裸命令

### Phase 1

- ❌ 跳过设计阶段直接写代码

### Phase 2

- ❌ 跳过用例生成，直接进入代码生成

### Phase 3

- ❌ 自行实现算子代码，不调用子 skill
- ❌ 用户未明确要求 in-tree 时，将算子创建在 `examples/` 下（**默认 `custom-ops/<op_name>/`**）
- ❌ 暴露裸 `cmake -S ... -B ...` 给用户/Agent（**MUST** 使用封装脚本，见 build-test.md §2.4）
- ❌ custom-ops 新算子调用 `aclshmemi_*` 或使用已 deprecated 的 `aclshmemx_barrier_all_vec`（见 [internal-api-boundary.md](../shmem-ops-code-gen/references/internal-api-boundary.md)）
- ❌ 仅 smoke 通过即进 Phase 5/6（**MUST** `run_case_matrix.py` 全 functional case PASS + `case_matrix_report.md`）
- ❌ `performance_required:true` 却未采集 Phase 6 或未写 `performance_report.md`（端到端对比表）
- ❌ 存在 HCCL 等价原语却不实现 `baseline/`、不采 `[BASELINE_PERF]`、不做 SHMEM vs HCCL 对比
- ❌ 跳过 Phase 5 `review-report.md` 或 Phase 4 `correctness_report.md` 直接交付
- ❌ Phase 6.5 未满 5 轮即进入 Phase 7（达标 **不是** 提前停止理由）
- ❌ `design.md` 声明 `torch_required:true` 却跳过 Phase 5.5 或缺 `aclshmem_torch.so` / `torch_test_*.py`
- ❌ 缺 `correctness_report.md` / `review-report.md` / `delivery_summary.md` 任一项仍声称 Phase 7 完成
- ❌ **通算融合**：MatMul 用 Device 三重循环或未经 design 授权的简陋 Host 路径作为交付默认（见 fused-compute-phase-gates §4）
- ❌ **通算融合**：用 `proxy baseline`（非真实 HCCL/aclnn/CATLASS）声称 Phase 6 已完成

### Phase 4

- ❌ correctness 失败时做性能优化
- ❌ 只手动设置 `LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib`，不 `source ${SHMEM_REPO}/install/set_env.sh`（缺 bootstrap 插件和 driver 库，易出现 aclError / golden 全 FAIL）
- ❌ `scripts/run.sh` 写死 `IPPORT=tcp://127.0.0.1:27010` 和 `SHMEM_UID_SESSION_ID=127.0.0.1:8899` 且不检测端口占用（典型：`address in use` → 输出全 0 假 FAIL）
- ❌ 输出 bin 全 0 时不查 `/root/shmem/log` 就直接改 kernel（应先排查 shmem init / 端口 / 环境）
- ❌ [custom-ops-entrypoints.md](../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §0 + SHMEM build -examples 不带 `-examples` 后执行 `make <op_name>`（CMake 无该 executable target）
- ❌ 反复全量 build -examples 做增量调试（每次清空 `build/` 和 `install/`）

### Phase 5.5

- ❌ `torch_required:true` 时未自动生成全部算子 `torch_test_*.py` 或未 8PE 验证即进入 Phase 6
- ❌ 使用无范围约束的 `pkill -f shmem` 清理进程（**MUST** 用 `pgrep -f 'build/bin/<op_name>'` 精确匹配；对外部残留进程先输出证据请求授权）

### Phase 6

- ❌ 有可用 baseline（HCCL/aclnn）但不接入
- ❌ 编造硬件测试或 profiler 结果
- ❌ 性能脚本跑完后不在聊天自动输出结果，等用户说「输出性能」
- ❌ Phase 6 只报时延不报带宽对比表（**MUST** 输出带宽表+时延表）

### Phase 6.5

- ❌ 将 `block_dim=1` 补齐到设计并发当作性能优化轮次
- ❌ 未经用户明确要求，性能优化超过 5 轮
- ❌ Phase 6.5 中 optim 自行修改代码或 design.md —— MUST 委托对应子 skill（design/code-gen）

### 通用

- ❌ 修改 SHMEM 核心库却无 gap analysis 授权
- ❌ 引用不存在的子 skill
- ❌ 遇编译/perf/baseline 失败自行停止，等用户说「继续」（**MUST** 自主修复重试，见 agent-execution-contract.md §1）

---

## Phase 0：需求与环境确认

**调用 Skill**：—

**目标**：确认算子开发所需的最小信息集，包括开发环境和算子需求

**Fresh Session 调用 AskQuestion 前 MUST Read**：[`references/askquestion-template.md`](references/askquestion-template.md)，**禁止**自行编造表单（尤其 CANN 第三固定选项）。

**详细清单**：`references/intake-checklist.md`（Agent **MUST** 在新会话首读）

### 启动门禁（Hard Gate — 新会话 MUST 遵守）

在 Phase 0 intake **完成前**，Agent **MUST NOT** 调用任何工具，**例外**仅为 Step 0.1 所需的只读环境检测（`echo $ASCEND_HOME_PATH`、`echo $CONDA_DEFAULT_ENV`、`which bisheng` 等），且 **禁止** 在此阶段 Read 源码、Grep 代码库、Shell 编译/运行、或调用任何子 skill。

**Phase 0 执行（对齐 Catlass）**：

1. **Step 0.1 环境**：探测 CANN / Docker / `SHMEM_REPO`（已满足则不问）
2. **Step 0.2 业务**：补齐 Torch / 性能 / 自动优化（消息已写清则 YAML +「确认」；否则聊天/表单）
3. **standalone**：固定 `build_mode=independent_project`（`custom-ops/<op>/`），**不问** in-tree

详见 [`askquestion-template.md`](references/askquestion-template.md)。

### Step 0.1：环境确认（MUST，先探测）

开发环境是所有后续阶段的前置依赖。

#### CANN（对齐 Catlass）

1. 检查 `ASCEND_HOME_PATH`（有 Docker 时在容器内检查），详见 `cann-env-resolution.md`
2. **已设置**：记录 `CANN_SET_ENV`，**无需再问**
3. **未设置**：**MUST** 询问 `set_env.sh` 路径（表单 Other 或聊天粘贴）

```bash
source "${CANN_SET_ENV}"
```

#### Docker / NPU / SHMEM_REPO

- 用户已写容器名 → `docker_container`，命令一律 `docker exec`
- 定位 `SHMEM_REPO`（[shmem-repo-resolution.md](references/shmem-repo-resolution.md)）；失败则询问
- 上板前选空闲卡写入 `device_list`（见 [idle-npu-selection.md](references/idle-npu-selection.md)）

#### 环境确认检查点

- [ ] CANN 已确定
- [ ] `SHMEM_REPO` 可用（或已追问）
- [ ] NPU/Docker 路径明确

### Step 0.2：算子需求与业务开关

最小确认项：`op_name`、功能语义、SoC/dtype（可默认后回显）。

#### 构建模式（standalone 不问）

| 场景 | 行为 |
| --- | --- |
| `deployment_mode=standalone`（agent-skills 默认） | **固定** `independent_project` → `custom-ops/<op_name>/` |
| `in_shmem_repo` 且用户明确要 in-tree | 才可设 `in_tree_example` |

```yaml
meta:
  build_mode: independent_project
  op_root: custom-ops/<op_name>
```

#### Torch / 性能 / 自动优化（缺项再问）

| 项 | 已在用户消息写清 | 未写清 |
| --- | --- | --- |
| Torch | 记录并纳入 YAML 确认 | 聊天/表单询问 |
| 性能 | 同上 | 同上 |
| 自动优化 | 同上；默认「未达标才触发，进入后跑满 5 轮」；用户明确指定轮次时跑满指定轮次 | 同上 |

```yaml
meta:
  torch_required: true|false
  performance_required: true|false
  performance_auto_optim: true|false
```

### MUST检查

- [ ] Step 0.1 环境已完成（探测或询问）
- [ ] standalone 下未询问 in-tree；`build_mode=independent_project`
- [ ] Torch / 性能 / 自动优化已确认（表单、聊天或 YAML「确认」）
- [ ] `op_name` 与功能语义明确
- [ ] 上板前已用 `npu-smi` 选定空闲卡并写入 `device_list`（或已说明无法判定）
- [ ] `phase0_intake` 摘要已输出
- [ ] SHMEM 仓可构建或已给出安装指引（`-soc_type Ascend910B`）

**全部通过 → 进入 Phase 1**

---

## Phase 1：设计文档

**调用 Skill**：`shmem-ops-design`

**通算融合附加门禁**：若 `op_kind=fused_compute_comm`，编排器 **MUST** Read [`references/fused-compute-phase-gates.md`](references/fused-compute-phase-gates.md)，并在 design 中声明 `schedule.compute.compute_path`。

### 执行内容

```
0. 将 phase0_intake 关键字段预填到 docs/design.md 的 meta（若 design.md 尚不存在则创建骨架）：
   - op_name → meta.op_name
   - build_mode → meta.build_mode
   - torch_required → meta.torch_required
   - performance_required → meta.performance_required
   - performance_auto_optim → meta.performance_auto_optim
   - shmem_repo → meta.shmem_repo
   - docker_container → meta.docker_container
   预填后调用 shmem-ops-design 补全其余设计内容，design skill 打开 design.md 即可直接读取已确认值
1. 如果无 design.md（完整），使用 shmem-ops-design 生成
2. 如果已有 design.md，读取并检查质量门禁
3. 设计文档必须包含：Canonical DSL、capability mapping、gap analysis、
   correctness invariants、compile/test/perf contract
```

### MUST检查

- [ ] design.md 存在且非占位
- [ ] 完整 Canonical DSL yaml block（含 schedule、correctness、performance section）
- [ ] DSL `topology` 含 deployment、拓扑类型、链路带宽（来自设计前确认）
- [ ] DSL `meta.build_mode` 已确认（默认 `independent_project`；in-tree 须 Phase 0 用户明确要求）
- [ ] DSL `meta.torch_required` 已确认（**Phase 0 用户确认或明确表态**，不得 Agent 擅自默认 `true`）
- [ ] DSL `meta.performance_required` 已确认（为 `false` 时 Phase 6 跳过）
- [ ] DSL `meta.performance_auto_optim` 已确认（为 `false` 时 Phase 6 未达标 **不得**擅自进入 Phase 6.5）
- [ ] 六类模块完整（lifecycle/memory/transport/sync/compute/scheduler）
- [ ] capability mapping 分类完整
- [ ] correctness invariants 可测试
- [ ] compile/test/perf contract 可执行
- [ ] Design Review Before Handoff（section 5）**12 项**检查已填写且无 FAIL；拓扑/带宽/分核专项检查已给出量化数据
- [ ] 门禁检查结果已显式输出给用户

Phase 1 的 design.md 门禁必须在进入 Phase 2 前完成。门禁结果中有任何 FAIL 或缺失 section，必须先修订 design.md 再继续，不允许带缺陷进入后续阶段。

**全部通过 → 进入 Phase 2**

---

## Phase 2：测试用例生成

**调用 Skill**：`shmem-ops-testcase-gen`

### 执行内容

```
1. 读取 design.md 的 correctness contract 和 invariants
2. 生成 case matrix（smoke/contract/tail/repeats/gap/medium-scale）
3. 生成 gen_data.py、check_result.py、`scripts/run.sh`
4. 生成 golden 文件
```

### MUST检查

- [ ] case matrix 覆盖 XS/S/M/L 四档规模（来自 testcase-gen）
- [ ] 总 case 数 ≥ 20（来自 testcase-gen）
- [ ] gen_data.py 可独立运行
- [ ] check_result.py 返回明确退出码
- [ ] golden 逻辑与 design.md 语义一致（来自 testcase-gen）
- [ ] 包含中等规模 case（集合通信≥256MB）
- [ ] `scripts/run.sh` 内联 `setup_shmem_runtime_env`（来自 testcase-gen）
- [ ] `scripts/run.sh` 通过 `setup_shmem_dynamic_endpoints` 避免固定端口（来自 testcase-gen）
- [ ] `scripts/run.sh` 支持多 PE 启动（来自 testcase-gen）

**全部通过 → 进入 Phase 3**

---

## Phase 3：代码生成

**调用 Skill**：`shmem-ops-code-gen`

### 执行内容

```
1. 读取通过门禁的 design.md
2. 选择模板和参考 example
3. 渐进式代码生成（lifecycle → transport → compute → scheduler）
4. 生成 README.md
5. 调用 shmem-ops-compile-debug 编译 + 2PE smoke（code-gen 步骤 5）：
   - compile-debug 执行构建并诊断；code-gen 根据诊断结果修复代码
   - 循环直到编译通过且 2PE smoke 运行通过
```

> Phase 3 产出**编译通过 + 2PE smoke PASS 的二进制**。全量正确性验证在 Phase 4。

### MUST检查

- [ ] 代码按 design 的 phase/transport/sync/partition 实现
- [ ] 目录结构符合规范：`docs/` 含全部 `.md`、`src/` 含全部 `.cpp/.h`、`scripts/` 含全部测试脚本
- [ ] main.cpp 不含复杂逻辑
- [ ] README.md 覆盖编译、运行、校验入口
- [ ] e2e_latency_us 口径包含输入到对称堆的拷贝（未预拷贝缩水）
- [ ] 编译成功（或记录环境阻塞；完整正确性在 Phase 4 验证）
- [ ] 同时 **MUST** 满足 `shmem-ops-code-gen` 的全部 P0 项（完整清单见 code-gen SKILL.md §MUST检查），P0 任一 FAIL 则禁止进入 Phase 4

**全部通过 → 进入 Phase 4**

---

## Phase 4：正确性验证

**调用 Skill**：`shmem-ops-correctness-eval`

> 编译已在 Phase 3 完成（二进制 + 2PE smoke PASS）。Phase 4 直接执行全量 case matrix。

### 执行内容

```
1. 使用 shmem-ops-correctness-eval 执行全量 case matrix：
   python3 ${SHMEM_REPO}/custom-ops/<op>/scripts/run_case_matrix.py
2. 失败分类：design bug → Phase 1；code bug → Phase 3（code-gen 局部修复 → compile-debug 增量编译 + 2PE smoke → 回到 Phase 4 全量复测，不重走 Phase 3 全流程）；test bug → Phase 2（修正 testcase-gen 产出后重新走 Phase 4）；env → 记录阻塞
```

### MUST检查

- [ ] 编译成功
- [ ] 全部 case 通过（或分类标记未通过原因）
- [ ] `cases.json` 中 **category ≠ performance** 的 case **100% PASS**（或逐项 DEFER 并写明原因）
- [ ] `docs/case_matrix_report.md` 已生成，列出 **每一个** case_id 与 status
- [ ] `docs/correctness_report.md` 已生成，含汇总表 + 全量 case 表
- [ ] 中等规模 case 已验证（或标记未满足）
- [ ] invariants 已验证

**任一项 category ≠ performance 的 case FAIL 或未执行 matrix → MUST NOT 进入 Phase 5**

**全部通过 → 进入 Phase 5**

---

## Phase 5：实现与设计一致性走读

**调用 Skill**：`shmem-ops-code-review`（`mode: interim`）

### 执行内容

```
1. 读取 docs/correctness_report.md（全量 case 结果，code-review Section 2 的数据来源）
2. 对比 design.md 和实现代码
3. 逐项检查：CMake target、block_dim、transport API、tile/chunk/tail、offset、性能路径
4. 生成 docs/review-report.md（interim 模式：Section 1–2 完整，Section 3 待补齐）
```

### MUST检查

- [ ] `interim` 走读全部 PASS（P0 无 FAIL）
- [ ] review-report.md Section 3 标注「待补齐」，无编造性能数据
- [ ] 无固定单核 launch（除非 design 明确要求）
- [ ] 无未接入的高性能 kernel

**全部通过 → 判断 Phase 5.5 是否执行**（见下）

**`torch_required: true` 时**：**MUST 在本 Phase 结束后立即进入 Phase 5.5**，为**本批次全部算子**生成 Torch 绑定与 `torch_test_*.py`，**不得**等待用户再次确认或说「继续」。

---

## Phase 5.5：Torch 接入与验证（条件性）

Torch 开关来自 Phase 0 intake（消息写清 / 聊天 / 表单均可）；**禁止**未确认时静默默认执行。

**是否执行**：读取 `design.md` DSL `meta.torch_required`：

| 值 | 动作 |
| --- | --- |
| `true`（用户已确认需要） | **MUST** 执行本 Phase |
| `false` | **跳过**本 Phase，在交付摘要中记录跳过原因；若 `performance_required: true` 则进入 Phase 6，否则进入 Phase 7 |
| 未设置 / 未确认 | **停止**，回到 Phase 0 Step 0.2 补齐确认 |

设计阶段写入 `meta.torch_required: true|false`。

### 环境预检（`torch_required: true` 时 MUST 在调用 torch-bind 前执行）

1. `python -c "import torch; import torch_npu; print(torch.__version__)"` — 确认可导入
2. `pip show shmem` — 确认 SHMEM whl 已安装（Phase 0 检查过但当前环境可能已切换）
3. 残留进程清理：**MUST** 以精确匹配方式清理本任务启动的进程（如 `pgrep -f 'build/bin/<op_name>'` 取 PID 后 `kill`，或 `pkill -f 'build/bin/<op_name>'`）；**禁止**使用无范围约束的 `pkill -f shmem`（会误杀无关 SHMEM 作业、测试、服务及自身脚本）。对外部残留进程只输出 PID/端口证据并请求用户授权后再清理（见 [agent-execution-contract.md](references/agent-execution-contract.md) §产物检查）

任一失败 → 修复环境后重试，**禁止**跳过预检直接进入 torch-bind。

**调用 Skill**：`shmem-ops-torch-bind`（仅 `torch_required: true` 时）

### 执行内容

```
1. 读取 design.md 接口契约和算子源码
2. 生成 C++ Torch CustomClassHolder 封装代码（torch_bind_<op_name>.cpp）
3. 集成 CMake（链接 kernel target + torch + torch_npu）
4. 生成 Python 多 PE 测试脚本（torch_test_<op_name>.py）
5. 编译 Torch 扩展 .so（custom-ops → `shmem_custom_ops_torch.so` / in-tree → `aclshmem_torch.so`）
6. 运行 2-PE smoke + 8-PE 全量测试
```

### MUST检查

- [ ] `torch_bind_<op_name>.cpp` 包含 CustomClassHolder 子类和 `REGISTER_SHMEM_OPS_CLASS` 注册
- [ ] `compute()` 包含 TORCH_CHECK（dtype、device、shape）
- [ ] dtype dispatch 覆盖 design.md 声明的全部 dtype
- [ ] Manager 类已存在（含 `attr_init` / `finalize` / `malloc_tensor` / `malloc_like` / `free_tensor`）
- [ ] Torch .so 编译成功（产物名取决于 `build_mode`：custom-ops → `shmem_custom_ops_torch.so`，in-tree → `aclshmem_torch.so`）
- [ ] `torch_test_<op_name>.py` 包含 fixed seed gen_data + multi-PE worker + golden 精度验证
- [ ] Python 测试 2-PE smoke 通过（`torch_required: true` 时）
- [ ] Python 测试 8-PE 通过（`torch_required: true` 时）
- [ ] 或 `torch_required: false` 且已记录跳过原因

**全部通过（或已跳过）→ 进入 Phase 6**

---

## Phase 6：性能采集

**是否执行**：读取 `design.md` DSL `meta.performance_required`：

| 值 | 动作 |
| --- | --- |
| `true`（用户已确认需要） | **MUST** 执行本 Phase |
| `false` | **跳过**本 Phase 及 Phase 6.5（无论 `performance_auto_optim`），在交付摘要中标注「未做性能验证」，直接进入 Phase 7 |
| 未设置 / 未确认 | **停止**，回到 Phase 0 Step 0.3 询问用户 |

**调用 Skill**：`shmem-ops-performance-eval`（仅 `performance_required: true` 时）

### 执行内容

```
1. 读取 perf contract
2. 调用 `shmem-ops-code-gen` 为算子添加性能打点代码（`main.cpp` 添加 `--perf` flag + timing loop + `[PERF]` 输出；kernel 添加 `SHMEMI_PROF_START/END` 宏对；生成 `scripts/perf.sh`）
3. 选择 baseline（HCCL/aclnn/metric-only）；AllToAllV 无 aclnn 头文件时回退 HcclAlltoAllV
4. 调用 `shmem-ops-code-gen` 按选定 baseline API 生成 baseline C++ 代码（`baseline/src/<op>_baseline.cpp` + `baseline/CMakeLists.txt` + `baseline/scripts/run_baseline.sh`）；若 `metric_only` 则跳过 baseline 目录；然后调用 `shmem-ops-compile-debug` 编译（算子 + baseline）；最后调用 `shmem-ops-performance-eval` 运行采集 `[BASELINE_PERF]`
5. **分阶段采集**：[perf-workflow.md §1](../shmem-ops-performance-eval/references/perf-workflow.md)（A baseline → B SHMEM → C 离线）；**NEVER 同会话混跑**
6. 瓶颈分析
7. 聊天 **自动输出**：
   - **带宽表 + 时延表**（S 档 + L 档）
   - **禁止**仅贴单行 `[PERF]` 或只报时延
8. 达标判定：有 baseline 时默认 ≥ baseline **80%**；无 baseline 时通信算子带宽利用率 ≥ 20%
```

参考：[baseline-compare-workflow.md](../shmem-ops-performance-eval/references/baseline-compare-workflow.md)、[perf-chat-output-spec.md](../shmem-ops-performance-eval/references/perf-chat-output-spec.md)

### MUST检查

- [ ] baseline 已选择或记录搜索过程
- [ ] baseline C++ 已实现且 `[BASELINE_PERF]` 已采集
- [ ] 性能数据已采集（S 档 + L 档）
- [ ] 聊天已输出 **带宽表 + 时延表**（含 kernel_bus_bandwidth_GBps、比值、目标、达标）
- [ ] 瓶颈已分析
- [ ] `docs/performance_report.md` §3.5 对比表已填写
- [ ] `performance_auto_optim: true` → Phase 6 **未达标**后 **立即** Phase 6.5（进入后跑满 `max_opt_rounds` 默认 5 轮，达标不是提前停止理由；用户明确指定轮次时跑满指定轮次）
- [ ] `performance_auto_optim: false` → 输出对比/差距表后进入 Phase 7，**不**改 kernel

**`performance_auto_optim: true` 且未达标 → 立即进入性能优化（进入后跑满 `max_opt_rounds` 默认 5 轮；用户明确指定轮次时跑满指定轮次）；否则 → Phase 7**

---

## Phase 6.5：性能优化（进入后跑满 `max_opt_rounds` 默认 5 轮）

**前置**：`meta.performance_required: true` **且** `meta.performance_auto_optim: true` **且** Phase 6 基线已完成 **且** Phase 6 未达标。进入后 **MUST** 跑满 `max_opt_rounds`（默认 5）轮（达标 **不是** 提前停止理由）；用户明确指定优化轮次时 **MUST** 跑满指定轮次（即使 Round 0 已达标）。

**调用 Skill**：`shmem-ops-performance-optim`

### 执行内容

```
每轮（Round 1~5，进入后 MUST 跑满 `max_opt_rounds` 默认 5 轮，达标不是提前停止理由；用户明确指定轮次时跑满指定轮次）：
1. 瓶颈定位（OptimStep 1）
2. 基线锁定（OptimStep 2）
3. 输出修改意见（OptimStep 3），分类为设计级或代码级
4. 委托执行链：
   a. 设计级改动 → shmem-ops-design 修订 design.md（跳过 testcase-gen）
   b. 代码级改动 → shmem-ops-code-gen 修改代码
   c. shmem-ops-compile-debug 编译（compile-debug 只诊断不改代码，失败回 code-gen）
   d. shmem-ops-correctness-eval 正确性复测
5. 性能验证（OptimStep 4 → shmem-ops-performance-eval，固定 Round0 平台区 case）
6. 聊天自动输出：本轮 Δ% + 累计总览（perf-chat-output-spec §3）
7. 决策 keep / revert（OptimStep 5）
```

### 停止条件

- 已进入 Phase 6.5：已完成 `max_opt_rounds`（默认 5）轮（达标 **不是** 提前停止理由；用户明确指定轮次时 **MUST** 跑满指定轮次，即使已达标）
- `performance_auto_optim: false`：**跳过** Phase 6.5
- 5 轮后仍未达标 → 输出差距和瓶颈，进入 Phase 7 交付
- 用户明确要求继续优化 → 可超过 5 轮

### 聊天输出（Hard Gate）

- **每轮结束**：Round N 最优 vs R0 / vs 上轮 / vs baseline（kernel_bus_bandwidth_GBps Δ%）
- **5 轮结束**：最终对比表（perf-chat-output-spec §4）
- **NEVER** 只在最后一轮才报性能变化

**完成 → 进入 Torch 回归门禁（如适用）→ Phase 7**

---

### Torch 回归门禁（Phase 6.5 → Phase 7 之间）

**前置**：`meta.torch_required: true` **且** Phase 5.5 已完成 **且** Phase 6.5 已完成（kernel 代码可能已被修改）。

**调用 Skill**：`shmem-ops-torch-bind`

**执行内容**：

1. 重新编译 Torch 扩展（kernel 签名、buffer 大小、block_dim 可能已变化）
2. 运行 `torch_test_<op_name>.py` 8PE 全量测试
3. 若 Torch 回归 FAIL → 回 `shmem-ops-code-gen` 修复 Torch 绑定代码后重新编译测试，直到 PASS

**MUST 检查**：

- [ ] Torch .so 重新编译成功（`shmem_custom_ops_torch.so` / `aclshmem_torch.so`）
- [ ] `torch_test_<op_name>.py` 8PE 回归通过
- [ ] 或 `torch_required: false` 且已记录跳过原因

---

## Phase 7：最终交付

**调用 Skill**：—

### 执行内容

输出最终交付摘要，并调用 `shmem-ops-code-review`（`mode: final`）更新 `docs/review-report.md` 的 Section 3/5/6。

**Phase 7 进入前 Hard Gate**：

1. 文件完整性：`docs/design.md`、`docs/case_matrix_report.md`、`docs/correctness_report.md`、`docs/review-report.md`、`docs/delivery_summary.md` 必须存在
2. `meta.torch_required:true` 时，Torch 产物与测试记录必须存在
3. `meta.performance_required:true` 时，必须有 S/L 真实 baseline 对比；proxy-only 状态一律判定为 blocked，不得完成交付
4. Phase 6.5 开启后必须有 Round1~5 逐轮记录（改动/复测/指标/决策）

交付摘要必须包含：

- 算子名称和 `docs/design.md` 路径
- 目录结构检查（`docs/` 包含全部 `.md`、`src/` 包含全部 `.cpp/.h`、`scripts/` 包含全部测试脚本）
- 修改文件列表
- 编译命令（完整可复现）
- 测试命令（完整可复现）
- 正确性验证结果表
- Torch 接入结果（Torch .so 路径、编译命令、`torch_test_<op_name>.py` 运行结果；若经过 Phase 6.5 则为**优化后回归结果**）
- 性能采集结果：**S 档 + L 档 kernel_bus_bandwidth_GBps 对比**（Phase 6 自动输出）
- 优化轮次摘要（如有 Phase 6.5）：**每轮 kernel_bus_bandwidth_GBps Δ% 累计总览 + 最终对比表**
- 未完成项、未验证项、环境阻塞或风险

不要只输出路径；关键结果必须在回复中摘要展示。

---

## 阶段间数据流

```
Phase 0 输出                Phase 1 输入
  op_name、env 确认   ────▶   需求语义、SoC、dtype

Phase 1 输出                Phase 2 输入
  design.md (完整)    ────▶   correctness contract、invariants

Phase 2 输出                Phase 3 输入           Phase 4 输入
  case matrix         ────────────────────────────▶   case matrix
  gen_data/checker    ────▶   模板参考
                                design.md      ────▶   design.md

Phase 3 输出                Phase 4 输入
  算子代码、CMake     ────▶   compile contract
  README.md                    test contract

Phase 4 输出                Phase 5 输入
  编译通过            ────▶   design.md vs 实现代码
  correctness PASS           docs/correctness_report.md

Phase 5 输出                Phase 5.5 输入
  走读 PASS           ────▶   design.md interface
                               算子源码

Phase 5.5 输出              Phase 6 输入
  Torch .so           ────▶   perf contract
  torch_test PASS              baseline 策略

Phase 6 输出                Phase 6.5/7 输入
  性能报告            ────▶   达标判断
  瓶颈分析                     优化或交付
```

## 状态跟踪表

| Phase | 前置条件 | 调用 Skill | 关键产出物 |
| --- | --- | --- | --- |
| 0 | 无 | — | op_name + 环境确认 |
| 1 | Phase 0 | `shmem-ops-design` | design.md（Canonical DSL + 契约） |
| 2 | Phase 1 | `shmem-ops-testcase-gen` | case matrix + scripts |
| 3 | Phase 2 | `shmem-ops-code-gen` | 算子代码 + README |
| 4 | Phase 3 | `shmem-ops-correctness-eval` | correctness PASS |
| 5 | Phase 4 | `shmem-ops-code-review`（interim） | `docs/review-report.md`（阶段性） |
| 5.5 | Phase 5 + `torch_required: true` | `shmem-ops-torch-bind` | Torch .so + `torch_test_<op_name>.py` |
| 5.5（跳过） | Phase 5 + `torch_required: false` | — | 跳过记录 |
| 6 | Phase 5.5 完成或跳过 + `performance_required: true` | `shmem-ops-performance-eval` | 性能报告 |
| 6（跳过） | `performance_required: false` | — | 跳过记录 |
| 6.5 | `performance_required: true` + `performance_auto_optim: true` + Phase 6 **未达标**（用户明确指定轮次时按指定轮次） | `shmem-ops-performance-optim`（分析）+ `shmem-ops-design` 或 `shmem-ops-code-gen`（改动）+ `shmem-ops-compile-debug` + `shmem-ops-correctness-eval` + `shmem-ops-performance-eval` | 优化轮次记录 |
| Torch 回归 | Phase 6.5 完成 + `torch_required: true` | `shmem-ops-torch-bind` | Torch 回归 PASS（优化后） |
| 7 | Phase 6 达标 或 6.5 完成 或 Torch 回归完成 | — | 最终交付摘要 |

## 错误恢复

### 中断恢复矩阵

| 检测条件 | 判定阶段 | 恢复动作 |
| --- | --- | --- |
| 无 design.md | Phase 1 未完成 | 使用 `shmem-ops-design` 生成 |
| design.md 缺 Canonical DSL | Phase 1 未完成 | 修订设计并重跑门禁 |
| 无测试脚本 | Phase 2 未完成 | 使用 `shmem-ops-testcase-gen` |
| 有 design 无实现代码 | Phase 3 未完成 | 使用 `shmem-ops-code-gen` |
| 代码存在但编译失败 | Phase 4 未完成 | 从编译调试恢复 |
| 编译通过但 correctness 未过 | Phase 4 未完成 | 从正确性调试恢复 |
| correctness 通过但无走读 | Phase 5 未完成 | 执行 `shmem-ops-code-review`（interim） |
| 走读通过但无 Torch 绑定 | Phase 5.5 未完成（且 `torch_required: true`） | 执行 `shmem-ops-torch-bind` |
| `torch_required: false` 已记录跳过 | Phase 5.5 已跳过 | 若 `performance_required: true` 进入 Phase 6，否则 Phase 7 |
| Torch 绑定通过或已跳过但无性能数据 | Phase 6 未完成 | 执行性能采集 |
| 优化中 design.md 已修订但未重新生成代码 | Phase 6.5 委托链中断 | 回 `shmem-ops-code-gen` 根据修订后的 design.md 重新生成 |
| 优化中代码已修改但编译失败 | Phase 6.5 委托链中断 | 回 `shmem-ops-code-gen` 修复后重新编译 |
| 有性能数据但优化未完成 | Phase 6.5 未完成 | 从下一轮优化继续 |
| 优化完成但无 Torch 回归 | Torch 回归未完成（且 `torch_required: true`） | 执行 Torch 回归门禁 |

### 编译/测试失败

由 `shmem-ops-compile-debug` 内部处理，持续修复直到正确性通过或确认为环境阻塞/设计缺陷。
