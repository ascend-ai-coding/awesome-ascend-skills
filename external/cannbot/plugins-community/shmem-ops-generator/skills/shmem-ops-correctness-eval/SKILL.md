---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-correctness-eval
description: 执行基于 SHMEM 的算子正确性契约验证并生成报告。触发：用户需要验证 SHMEM 算子结果、执行 case matrix 或检查精度时使用。关键词：基于SHMEM正确性验证、correctness、测试执行、精度验证。
original-name: shmem-ops-correctness-eval
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 算子正确性验证

**Skill类型**：验证型（执行测试、分类失败、输出报告）

> **中文写作要求**：正确性验证报告必须使用中文撰写。仅 API 名称、代码片段、Case ID 等技术术语保留英文原文。

按 `shmem-ops-testcase-gen` 生成的 case matrix 执行正确性验证。运行和失败定位由 `shmem-ops-compile-debug` 完成，本 skill 负责结果判定和报告。

## 必读资料

| 文件 | 用途 |
| --- | --- |
| [../shmem-ops-testcase-gen/references/correctness.md](../shmem-ops-testcase-gen/references/correctness.md) | golden 构造策略（§4）、invariant→test 映射（§6） |
| [../shmem-ops-testcase-gen/references/precision-standard.md](../shmem-ops-testcase-gen/references/precision-standard.md) | 精度标准——OpTypes 分类、rtol/atol 取值、双统计判定 |
| [references/GUIDE.md](references/GUIDE.md) | 本 skill 参考索引 |
| [../shmem-ops-compile-debug/references/debug.md](../shmem-ops-compile-debug/references/debug.md) | 运行失败、init、全 0 输出定位 |
| [../shmem-ops-dev/references/fused-compute-phase-gates.md](../shmem-ops-dev/references/fused-compute-phase-gates.md) | 通算融合：case matrix 门控（§2） |

## 输入

- case matrix（来自 `shmem-ops-testcase-gen`）
- gen_data.py / check_result.py / `scripts/run.sh`（来自 `shmem-ops-testcase-gen`）
- 已编译的算子二进制（来自 `shmem-ops-compile-debug`）
- design.md 的 correctness invariants

## 工作流

```
步骤 1  执行 case matrix 中的全部 case
步骤 2  收集每个 case 的结果（PASS/FAIL）
步骤 3  失败分类
步骤 4  验证 correctness invariants
步骤 5  输出报告
```

---

## 步骤 1：执行测试

**Hard Gate**：**MUST** 运行 `python3 scripts/run_case_matrix.py`（或 [custom-ops-entrypoints.md §3](../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) 等价封装），**禁止**仅用 `CASE_ID=smoke_* bash scripts/run.sh` 声称 Phase 4 完成。

执行顺序：
1. 2-PE smoke case
2. design 指定的 dtype/shape/PE/engine 组合
3. tail/chunk 边界 case
4. 多轮 repeats
5. gap analysis 验证 case
6. 中等规模 case

---

## 步骤 2：收集结果

收集每个 case 的执行结果（PASS/FAIL），汇总为结果表。

---

## 步骤 3：失败分类

| 类型 | 处理 |
| --- | --- |
| design bug | 回 `shmem-ops-design` |
| code bug | 回 `shmem-ops-code-gen` |
| test bug | 修正 testcase-gen 产出 |
| environment | 记录阻塞（由编排器 `shmem-ops-dev` 记录，不回退到 compile-debug） |

---

## 步骤 4：验证 invariants

必须验证或显式标记未验证：
- 初始化和 PE/team 拓扑
- symmetric allocation 顺序和地址
- put/get 或 collective 语义
- signal/wait、barrier 完成语义
- output visibility
- dtype tolerance
- phase boundary
- tail 和不均衡分片

---

## 输出报告

正确性验证报告必须写入算子目录 `docs/correctness_report.md`。报告**必须列出 case matrix 中的全部 case**，不得省略。此报告将被 `shmem-ops-code-review` 完整引用到走读报告中。

### 汇总

| 指标 | 值 |
| --- | --- |
| 总用例数 | |
| 通过数 | |
| 失败数 | |
| 通过率 | % |
| 阻塞（未执行）数 | |

### 全量 Case 结果

| Case ID | category | scale | n_pes | shape | dtype | Result | MaxAE | MaxRE | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_2pe_fp16 | functional | XS | 2 | ... | fp16 | PASS | 0.0 | 0.0 | |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Invariant 验证

| Invariant | 验证方式 | 结果 | 说明 |
| --- | --- | --- | --- |
| ... | ... | PASS/FAIL/未验证 | |

未验证 invariants 必须标注原因。

---

## MUST检查

- [ ] 全部 case 已执行或标记阻塞原因
- [ ] **全部 case 结果已列出**（不得省略，必须在报告中列出 case matrix 的每一个 case）
- [ ] 包含汇总表（总用例数、通过数、失败数、通过率）
- [ ] 包含中等规模 case（M 档）和大规模 case（L 档），或标记未满足
- [ ] 失败 case 已分类
- [ ] invariants 已验证或标记未验证原因
- [ ] 报告已保存到 `docs/correctness_report.md`

---

## 门控规则（验证通过标准）

进入后续阶段前必须满足以下全部条件：

- **`run_case_matrix.py` 已执行**，且 `docs/case_matrix_report.md` 已生成
- smoke case 通过
- case matrix 中所有 **category ≠ performance** 的 correctness case **100% PASS**（或逐项 DEFER 并写明原因）
- correctness invariants 已验证，或逐项说明未验证原因
- checker 输出包含可复现 case id、命令、输入路径、输出路径和 tolerance
- repeats case 通过，或 design 明确当前实现不复用 signal/state

**门控判定**：
- 仅 smoke 通过、未跑 matrix → **Phase 4 未完成**，**禁止** Phase 5/6/7
- 正确性未全部通过 → 不能进入 Phase 5 走读
- 全部通过 → 可进入 `shmem-ops-code-review`（Phase 5）
- Phase 5 `interim` 走读 PASS → 若 `meta.torch_required: true` 进入 Phase 5.5；否则若 `performance_required: true` 进入 Phase 6，否则 Phase 7
- Phase 6 未达标且 `performance_auto_optim: true` → 编排器 **MUST** 自动 Phase 6.5；`performance_auto_optim: false` → 仅报告，**禁止**擅自优化
- **禁止**在 Phase 5 未完成时直接进入 `shmem-ops-performance-eval`

---

## 反模式（NEVER DO THESE）

- ❌ 只跑 smoke case 就声称正确性通过
- ❌ **未执行 `run_case_matrix.py`** 就生成 `correctness_report.md` 或进入 Phase 5
- ❌ 缺少 `docs/case_matrix_report.md`（须列出每一个 case_id）
- ❌ 精度 FAIL 时不分类直接进入性能阶段
- ❌ invariant 未验证也不标记原因
- ❌ 编造测试结果或省略实际命令输出
- ❌ 正确性报告只列出部分 case（必须列出 case matrix 全部 case 的结果）
- ❌ 不输出汇总表（总用例数、通过数、失败数、通过率）
