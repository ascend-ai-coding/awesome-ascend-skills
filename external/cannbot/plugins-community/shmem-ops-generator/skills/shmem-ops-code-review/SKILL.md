---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-code-review
description: 基于 SHMEM 的算子实现与设计一致性走读，生成 review-report.md。触发：用户需要检查 SHMEM 算子代码与 design.md
  的一致性或进行代码审查时使用。关键词：基于SHMEM代码走读、code review、design review、一致性检查、走读。
original-name: shmem-ops-code-review
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 算子实现与设计一致性走读

**Skill类型**：检查型（对比 design.md 和实现代码，输出走读报告）

> **中文写作要求**：review-report.md 必须使用中文撰写。仅 API 名称、代码片段、检查项英文缩写等技术术语保留英文原文。

本 skill 分两种模式执行，由编排器按 Phase 传入 `mode`：

| 模式 | 触发 Phase | 报告范围 |
| --- | --- | --- |
| `interim`（阶段性） | Phase 5 | Section 1–2 完整；Section 3 填「待补齐」；Section 5 仅判定设计一致性 + 正确性 |
| `final`（终稿） | Phase 7 | 在 interim 基础上补齐 Section 3 性能摘要、Section 5 性能判定、Section 6 全量交付物 |

正确性通过前后都可执行 interim 走读；**Phase 5 走读 PASS 是进入 Phase 5.5（或跳过 Torch）的门禁**。

## 工作流

```
步骤 1  读取 design.md、mode 和实现代码
步骤 2  逐项检查（CMake/block_dim/transport/sync/compute/style）
步骤 3  输出走读结果表
步骤 4  生成或更新 review-report.md（按 mode 决定 Section 3/5/6 深度）
```

---

## 必读资料

| 文件 | 用途 |
| --- | --- |
| [references/code-review-checklist.md](references/code-review-checklist.md) | 分类检查项、错误/正确做法示例 |
| [references/review-report-template.md](references/review-report-template.md) | 走读报告输出模板（必须严格按此模板生成） |
| [../shmem-ops-code-gen/references/code-style.md](../shmem-ops-code-gen/references/code-style.md) | 代码风格规范，走读时必须对照检查 |
| [../shmem-ops-design/references/implementation-boundary.md](../shmem-ops-design/references/implementation-boundary.md) | 实现边界、unified kernel 路径 |
| [../shmem-ops-code-gen/references/code-patterns.md](../shmem-ops-code-gen/references/code-patterns.md) | §6.4 统一实现 vs size 分支 |
| [references/GUIDE.md](references/GUIDE.md) | 本 skill 参考索引 |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | 对照仓内 examples 走读前定位 `SHMEM_REPO` |

---

## 检查项

**MUST** 逐项检查并记录结果。任一项未检查即视为 FAIL，不得跳过。

| 检查项 | 严重级别 | 标准 |
| --- | --- | --- |
| CMake target | P0 | 包含 design 声明的所有 kernel、Host helper、transport 源文件；baseline 源码在 `baseline/src/`，编译 target 在 `baseline/CMakeLists.txt`，未混入算子 `src/` 或根目录 |
| launch block_dim | P0 | 来自 design 或参数化；固定 `<<<1>>>` 不能作为交付（除非 design 明确要求单核） |
| transport API | P0 | design 声明的 API 实际出现在编译路径中 |
| 跨 PE 传输 | P0 | 使用 `aclshmem_*` / `aclshmemx_*`，**禁止** DataCopy 写远端 |
| Reduce 累加路径 | P0 | reduce-scatter / allreduce RS 阶段 **MUST** 使用 `SetAtomicAdd<T>()`（MTE 批量）+ `mte_get_nbi`（atomic-add-pattern.md §12 优先级1）；**禁止**以串行 UB 累加（get→UB→Add→写回）作为交付路径 |
| 性能指标计算 | P0 | 整个计算链路 **MUST** 正确——(1) `logical_payload_bytes` = `elements × sizeof(dtype)`；(2) `algo_bandwidth_GBps(kernel)` = `logical_payload / (kernel_us × 1e-6) / 1e9`（**达标主指标 MUST 基于 kernel_us**）；(3) `bus_factor` 从 `timing-and-metrics-standard.md §4.3` 唯一参照表取值，**禁止**使用 `n_pes-1` 等错误值；(4) `kernel_bus_bandwidth_GBps` = `algo_bandwidth_GBps(kernel) × bus_factor`（**达标对比 MUST 用此字段**）；(5) `e2e_bus_bandwidth_GBps` = `algo_bandwidth_GBps(e2e) × bus_factor`（**仅作参考，NEVER 作达标主指标**）；(6) `bandwidth_utilization_pct` = `kernel_bus_bandwidth_GBps / peak_bandwidth_GBps × 100`。perf 行字段命名不混淆（**禁止**不同名字段输出相同值） |
| 源文件 License 头 | P0 | 所有 `.cpp`、`.h`、`.py`、`.sh` 文件 **MUST** 包含 CANN Open Software License 头；交付不可省略（见 code-style.md §7.2） |
| SHMEM 接口约束 | P0 | SHMEM 调用 **MUST** 只用对外 API（`aclshmem_*` / `aclshmemx_*`）。若出现 `aclshmemi_*`：(1) 有公开等价 API → **MUST** 改为公开 API；(2) 无等价 API → **MUST** 在 design gap analysis 中有登记记录（见 internal-api-boundary.md） |
| signal/wait 配对 | P0 | 每个 signal 有对应 wait，magic/epoch 策略一致 |
| symmetric 分配顺序 | P0 | 所有 PE 的 symmetric malloc 调用顺序和大小完全一致 |
| e2e 计时口径 | P0 | e2e 起点 **MUST** = 用户 input(GM) 未被搬运的时刻，终点 = output(GM) 写入完成且 stream sync。做法 A（MTE/SDMA/UDMA kernel 内放置）下 e2e_us≈kernel_us 正常，代码中 **MUST** 有注释说明引擎选择；做法 B（RDMA 或 Host 侧搬运）下 e2e_us **MUST** > kernel_us。**禁止**将搬运移到循环外缩水；**禁止**为制造差值而在做法 A 上加无意义搬运。详见 [timing-and-metrics-standard.md §1.2](../shmem-ops-performance-eval/references/timing-and-metrics-standard.md) |
| error handling 宏 | P1 | **MUST** 使用 `ACL_CHECK` / `ACL_CHECK_WITH_RET` / `CHECK_SHMEM` 等宏；**禁止**裸写 `if (status != ACL_ERROR_NONE)`（见 code-style.md §1.1） |
| 有状态 API 返回值检查 | P1 | 有返回值的 ACL/SHMEM/RT 调用 **MUST** 检查返回值（如 `aclshmem_put_nbi`、`aclshmem_get_nbi`、`aclshmem_signal_wait_until`、`aclrtMalloc`、`aclshmem_finalize`、`aclrtDestroyStream`、`aclrtResetDevice`、`aclFinalize` 等）。**禁止**静默忽略返回值（见 code-style.md §1.4） |
| void 同步 API 对称性 | P1 | `aclshmem_barrier_all()` 等返回 `void` 的同步 API（Host/Device 公开声明均返回 `void`）**MUST** 检查：所有 PE 对称调用（无分支遗漏）、barrier 后无 data race、finalize 前数据已可见。**禁止**检查不存在的返回值 |
| README.md 语言 | P1 | **MUST** 使用中文撰写（见 shmem-ops-dev 原则#0） |
| README.md 结构 | P1 | **MUST** 遵循 readme-spec.md 完整结构——含算子介绍、环境要求、目录结构、参数说明、编译项目、运行算子、验证结果、性能采集、注意事项等 section |
| 无硬编码路径 | P1 | 脚本/文档/CMake 中 **MUST NOT** 硬编码用户机器路径，**MUST** 使用环境变量或相对路径（见 code-style.md §10.1） |
| SHMEMI_PROF 分 phase | P1 | kernel 中 **MUST** 使用独立 `frame_id` 覆盖每个关键 phase：至少 copy_in / remote_put_get / signal_or_barrier_wait / local_compute / finalize（≥5 frames；见 timing-and-metrics-standard.md §7） |
| Host 侧 show_prof | P1 | stream sync 后 **MUST** 调用 `aclshmemx_get_prof(nullptr, true)` 导出 Device frame 数据（见 timing-and-metrics-standard.md §7） |
| 同步 API 选型 | P1 | phase 同步 **MUST** 优先 `aclshmem_barrier_all()` 或 `aclshmem_barrier(team)`；**禁止**无理由使用 `aclshmemx_barrier_all_vec()` 替代（仅对齐既有 legacy example 时可保留；见 code-style.md §6.2） |
| phase 边界同步 | P1 | 与 design 一致 |
| tile/chunk/tail | P1 | 按 design 支持并发；非对齐 case 不永久降级为单核 |
| main.cpp 边界 | P1 | 不含复杂计算逻辑；复杂 Host 逻辑已拆模块；不含 route/payload/tiling/packing/golden/checker |
| 性能关键路径 | P1 | 无设计未说明的多余 GM scratch、全局 barrier 或串行 phase；reduce 路径无冗余 GM 中转 |
| unified kernel 路径 | P1 | 默认 single path；无 ≥5% profiling 证据不得保留 `small`/`big` 并行路径（见 implementation-boundary、code-patterns §6.4） |
| 代码风格 | P2 | 逐项对照 `code-style.md §12` 完整审查清单（~32项），每项 **MUST** 标记通过/不适用/FAIL；**禁止**笼统声称"已检查" |

---

## 走读报告要求

走读完成后必须在算子目录生成或更新 `docs/review-report.md`，**严格按照** [references/review-report-template.md](references/review-report-template.md) 格式。

### `interim` 模式（Phase 5）

**必须完整填写**：
- Section 1：设计一致性检查
- Section 2：正确性验证结果（全量 case，不得省略）
- Section 4：已知限制与风险（如有）
- Section 5：仅填写「设计一致性」「正确性验证」两项判定；「性能达标」填 `待 Phase 6 补齐`
- Section 6：列出 Phase 5 前已有交付物及状态

**Section 3 填写规则**：所有字段填 `待 Phase 6 性能采集后补齐`，**禁止编造性能数据**。

**总体结论（interim）**：设计一致性 PASS + 正确性 PASS → **可进入 Phase 5.5**；任一 FAIL → 需修复。

### `final` 模式（Phase 7）

在已有 interim 报告基础上**更新**：
- Section 3：引用 `performance_report.md`，摘录 baseline vs final 对比表和达标状态
- Section 5：补齐「性能达标」判定和最终总体结论（可交付 / 需修复）
- Section 6：列出全部交付物（含 Torch、性能报告），标注实际存在状态

若 interim 报告不存在，**MUST** 先补跑 `interim` 再走 `final`。

### 报告自检

**interim 模式**：
1. Section 1、2 是否完整？Section 2 是否列出 case matrix **全部** case？
2. Section 3 是否标注「待补齐」而非编造数据？
3. Section 5 是否未提前判定性能达标？

**final 模式**：
1. Section 3 是否引用真实 `performance_report.md` 并摘录对比表？
2. Section 6 是否列出全部交付物且状态与磁盘一致？

---

## 最终交付物（Phase 7 齐备）

| 交付物 | 说明 |
| --- | --- |
| `docs/design.md` | 设计文档（来自 shmem-ops-design） |
| 算子代码 | `src/` + `CMakeLists.txt`（来自 shmem-ops-code-gen） |
| 测试脚本 | `scripts/gen_data.py` + `check_result.py` + `scripts/run.sh` + `run_case_matrix.py`（来自 shmem-ops-testcase-gen） |
| `docs/review-report.md` | 走读报告终稿（本 skill `final` 模式更新） |
| `docs/correctness_report.md` | 正确性验证报告（来自 shmem-ops-correctness-eval） |
| `docs/case_matrix_report.md` | Case Matrix 执行报告（来自 shmem-ops-testcase-gen） |
| `docs/performance_report.md` | 性能报告（来自 shmem-ops-performance-eval / optim） |
| Torch 产物（若 `meta.torch_required: true`） | `shmem_custom_ops_torch.so`（custom-ops 默认）或 `aclshmem_torch.so`（in-tree）+ `scripts/torch_test_<op>.py` |

**缺少任一适用交付物不得声称算子完成。**

---

## 门控规则

- 任一 P0 检查项 FAIL → 不能进入 Phase 5.5（或 Torch 跳过后的 Phase 6）
- 正确性未全部通过 → 不能进入 Phase 5.5
- `interim` 模式 PASS → 进入 Phase 5.5（或按 `meta.torch_required` 跳过）
- FAIL 时回 `shmem-ops-code-gen` 修正实现，或回 `shmem-ops-design` 修订设计

---

## MUST检查

### interim（Phase 5）

- [ ] 所有检查项已逐项检查并记录，**禁止**跳过任一项
- [ ] P0 项全部 PASS（任一 FAIL → **MUST** 修复后重新走读）
- [ ] 正确性全量 case 结果已列出（不得省略）
- [ ] invariant 验证已列出
- [ ] Section 3 标注「待补齐」，**禁止**编造性能数据
- [ ] review-report.md 已按模板生成

**P0 全部 PASS → 进入 Phase 5.5（或按 `torch_required` 跳过）；FAIL → 回 code-gen 修复或回 design 修订。**

### final（Phase 7）

- [ ] Section 3 已引用 `performance_report.md` 并摘录对比表
- [ ] Section 5 三项判定完整
- [ ] Section 6 交付物清单完整且与磁盘一致
- [ ] baseline 源码在 `baseline/src/` 下、编译 target 在 `baseline/CMakeLists.txt`，未混入算子 `src/` 或根目录

**P0+P1 全部 PASS → 交付完成；任一 FAIL → MUST 修复后重新走读。**

---

## 反模式（NEVER DO THESE）

- ❌ 不读 design.md 直接走读代码
- ❌ 只看代码能否编译通过，不检查与设计的语义一致性
- ❌ 发现 `block_dim=1` 但不标记为 FAIL
- ❌ 发现 `DataCopy` 远端地址但不标记为 FAIL
- ❌ 不对照 `code-style.md §12` 代码审查清单逐项检查
- ❌ 不检查 reduce 累加路径是否符合 `atomic-add-pattern.md §12` 决策表
- ❌ 不验证 `bus_factor` 公式
- ❌ 不检查 License 头是否存在
- ❌ 不检查错误处理是否使用宏
- ❌ 不检查 README 语言和结构
- ❌ 不检查硬编码路径
- ❌ 不检查 perf 输出字段语义（`algo_bandwidth` 是否基于 e2e_us、字段是否重复混淆）
- ❌ 不检查 `SHMEMI_PROF` frame 分包（单 frame 包裹整个 kernel）
- ❌ 不检查 `aclshmemx_get_prof(nullptr, true)` Host 侧是否调用
- ❌ 不检查同步 API 是否优先 `barrier_all()` 而非 `barrier_all_vec()`
- ❌ 不检查算子代码中是否无正当理由调用了 `aclshmemi_*` 内部接口
- ❌ 正确性结果只列出部分 case（必须列出全部 case matrix 中的 case）
- ❌ **interim 模式在 Section 3 编造性能数据**
- ❌ interim 模式因缺少 performance_report 而 FAIL 整个走读
- ❌ 不生成 review-report.md 就声称走读完成
- ❌ Phase 7 未跑 `final` 模式就声称交付完成
- ❌ 发现 e2e 计时口径不符合做法 A/B 规则却不标记为 FAIL（必须标记为 P0 FAIL）
- ❌ review-report.md 不在 `docs/` 子目录下
- ❌ baseline 源码混入算子 `src/` 或散放在算子根目录（必须归入 `baseline/src/`）
- ❌ 交付物清单声明的文件在磁盘上不存在
