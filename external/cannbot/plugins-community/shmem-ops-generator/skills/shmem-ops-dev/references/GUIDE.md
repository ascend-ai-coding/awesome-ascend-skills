# SHMEM 算子开发 Skill 总索引

编排器 `shmem-ops-dev` 及子 skill 的 **入口导航**。Skill 树可与 SHMEM 仓分离部署；读仓内 `docs/`、`examples/` 前 **MUST** [定位 `SHMEM_REPO`](shmem-repo-resolution.md)。

---

## Phase 0 必读（Fresh Session）

| 顺序 | 文件 | 用途 |
| --- | --- | --- |
| 1 | [../SKILL.md](../SKILL.md) | 编排器 Phase 0–7 |
| 2 | [askquestion-template.md](askquestion-template.md) | Phase 0 Intake：先探测缺项再问模板 |
| 3 | [agent-execution-contract.md](agent-execution-contract.md) | 禁止中断、Torch 自动、性能双表、统一实现 |
| 4 | [shmem-repo-resolution.md](shmem-repo-resolution.md) | `SHMEM_REPO` 定位；**§1.1 区分 SHMEM 原生 vs custom-ops 交付物** |
| 5 | [docker-exec-contract.md](docker-exec-contract.md) | 用户指定 Docker 时全部命令进容器 |
| 6 | [intake-checklist.md](intake-checklist.md) | 启动门禁与 `phase0_intake` 记录格式 |

---

## 子 Skill 一览

| Phase | Skill | GUIDE / 主文件 |
| --- | --- | --- |
| 1 | `shmem-ops-design` | [design/references/GUIDE.md](../../shmem-ops-design/references/GUIDE.md) |
| 2 | `shmem-ops-testcase-gen` | [testcase-gen/references/GUIDE.md](../../shmem-ops-testcase-gen/references/GUIDE.md) |
| 3 | `shmem-ops-code-gen` | [code-gen/references/GUIDE.md](../../shmem-ops-code-gen/references/GUIDE.md) |
| 4 | `shmem-ops-compile-debug` | [compile-debug/references/GUIDE.md](../../shmem-ops-compile-debug/references/GUIDE.md) |
| 4 | `shmem-ops-correctness-eval` | [correctness-eval/references/GUIDE.md](../../shmem-ops-correctness-eval/references/GUIDE.md) |
| 5 | `shmem-ops-code-review` | [code-review/references/GUIDE.md](../../shmem-ops-code-review/references/GUIDE.md) |
| 5.5 | `shmem-ops-torch-bind` | [torch-bind/references/GUIDE.md](../../shmem-ops-torch-bind/references/GUIDE.md) |
| 6 | `shmem-ops-performance-eval` | [performance-eval/references/GUIDE.md](../../shmem-ops-performance-eval/references/GUIDE.md) |
| 6.5 | `shmem-ops-performance-optim` | [performance-optim/references/GUIDE.md](../../shmem-ops-performance-optim/references/GUIDE.md) |

---

## 跨阶段共享参考

| 文件 | 用途 |
| --- | --- |
| [../../shmem-ops-code-gen/references/internal-api-boundary.md](../../shmem-ops-code-gen/references/internal-api-boundary.md) | 禁止 internal API、deprecated barrier、quiet 误用 |
| [../../shmem-ops-dev/references/shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) | **SHMEM 原生 vs custom-ops 交付物**（§1.1） |
| [../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) | **custom-ops 生成交付** 编译/运行/matrix/Torch（md 代码段为准） |
| [../../shmem-ops-compile-debug/references/env-setup.snippet.md](../../shmem-ops-compile-debug/references/env-setup.snippet.md) | 生成 `run.sh` 的运行时环境（内联函数） |
| [../../shmem-ops-performance-eval/references/perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md) | **custom-ops 生成** perf 三阶段 |
| [../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md](../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md) | **SHMEM 原生** `docs/`、`include/`、`examples/` 索引 |
| [../../shmem-ops-performance-eval/references/perf-chat-output-spec.md](../../shmem-ops-performance-eval/references/perf-chat-output-spec.md) | Phase 6/6.5 聊天自动输出双表 |
| [../../shmem-ops-code-gen/templates/fused-compute/GUIDE.md](../../shmem-ops-code-gen/templates/fused-compute/GUIDE.md) | 通算融合算子代码模板（Phase 3，`op_kind=fused_compute_comm`） |
| [fused-compute-phase-gates.md](fused-compute-phase-gates.md) | **通算融合开发规范**（Phase 门控、case matrix、计算实现、baseline P1/P2、6.5 五轮） |
| [../../shmem-ops-performance-optim/references/compute-optimization.md](../../shmem-ops-performance-optim/references/compute-optimization.md) | Phase 6.5 Matmul/compute 调优 |
| [../../shmem-ops-performance-eval/references/perf-compare-spec.md](../../shmem-ops-performance-eval/references/perf-compare-spec.md) | Phase 6 同量对比与达标口径 |
| [../../shmem-ops-performance-eval/references/hccl-expansion-mode.md](../../shmem-ops-performance-eval/references/hccl-expansion-mode.md) | HCCL `HCCL_OP_EXPANSION_MODE`：优先 AIV，回退 AI_CPU |
