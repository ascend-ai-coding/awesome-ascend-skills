# Agent 执行契约（MUST 遵守）

本文汇总 SHMEM custom-ops 端到端交付中反复出现的问题与强制行为。Fresh Session 编排器 **MUST** 在 Phase 1 前 Read 本文件。

---

## 1. 禁止自行中断（Hard Gate）

| 禁止 | 必须 |
| --- | --- |
| 遇到编译/perf/baseline 失败后停止，等用户说「继续」 | 自主诊断、修复、重试，直到通过或明确报告不可恢复阻塞 |
| 后台任务 hang 后放弃 | 先 `pkill` 残留进程 → 换策略重跑；HCCL 失败时 `sleep 15~30s` 后单独重采 |
| 只完成部分 Phase 就进入交付摘要 | 按 `meta.torch_required` / `meta.performance_required` / `meta.performance_auto_optim` **跑完全部** 条件性子阶段 |
| Phase 6 未达标后停止，等用户说「继续」 | `performance_auto_optim:true` 时 **MUST 立即** 进入 Phase 6.5（见 §4.5） |
| 优化 kernel 导致 507034/507057 后不回滚验证 | 先恢复可编译+小档 PASS，再逐步启用 SDMA/大改 |

**重试上限**：同一根因最多 3 种不同修复策略；仍失败则在聊天给出错误日志 + 下一步，**不得**静默结束。

---

## 2. Docker 执行环境（Hard Gate）

用户指明 Docker 容器名，或 `phase0_intake.docker_container` 非空时：

1. **MUST** Read [docker-exec-contract.md](docker-exec-contract.md)
2. **所有** build / run / perf / baseline / Torch / `pkill` / `npu-smi` **MUST** `docker exec <container> bash -lc '...'`
3. 上板前 **MUST** 用 `npu-smi info` 选**空闲卡**写入 `phase0_intake.device_list`（有大进程占用的卡默认跳过；详见 [idle-npu-selection.md](idle-npu-selection.md)）
3. **禁止**在宿主机 shell 裸跑上述命令（避免误用宿主机编译器、误杀进程、污染用户环境）

文件编辑可走工作区挂载；**验证**必须在容器内完成。

---

## 3. Torch 接入（Phase 5.5）— 用户选「需要」则全自动

当 `phase0_intake.torch_required: true`：

1. **MUST** 调用 `shmem-ops-torch-bind`，**不得**等用户再次确认
2. **MUST** 为**当前批次全部算子**生成并验证：
   - `custom-ops/torch_binding/` 共享层 + 各算子 `torch_bind_<op>.cpp`
   - 各算子 `scripts/torch_test_<op>.py`
3. **MUST** 8PE 跑通后再进入 Phase 6（至少 smoke 2PE + 目标 8PE）
4. **MUST** 在 Torch 扩展编译（见 [custom-ops-entrypoints.md §4](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md)）后设置 `LD_LIBRARY_PATH` 含 `torch_binding/build/lib` 与各算子 `build/lib`

**禁止**：只完成 2/4 个 `torch_test_*.py` 就标 Phase 5.5 完成。

---

## 4. 性能采集（Phase 6）— 必须输出完整带宽表

仅报时延、不报带宽 → **Phase 6 未完成**。

### 4.1 达标目标（来自 design.md，对话 MUST 展示）

性能达标线 **MUST** 从 `design.md` 的 `performance.baseline_target` 与 `performance.target_cases` 读取。对比方法见 [perf-compare-spec.md](../../shmem-ops-performance-eval/references/perf-compare-spec.md)（**同量对比**）。缺省参考：有 baseline 时 ≥ baseline **80%**，无 baseline 时带宽利用率 ≥ **20%**。

### 4.2 采集命令（分 shell，禁止混跑；有 docker_container 时每步独立 docker exec）

按 [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md) §1 分三阶段采集（A baseline → B SHMEM → C 离线对比）；**NEVER** 同 shell 混跑 HCCL 与 SHMEM。采集参数以 `design.md` `performance.target_cases` 为准。

### 4.3 聊天 MUST 输出（两块，不得省略）

**块 A — 带宽类**（kernel_bus_bandwidth_GBps + 比值 + 目标 + 达标）

**块 B — 时延类**（comm_only/kernel µs + HCCL/SHMEM 比值 + 达标）

另附明细行：`e2e_us`、`kernel_us`、`algo_bw`、`bus_bw`、`payload_bytes`。

**禁止**：只贴 `[PERF]` 一行；禁止 HCCL 失败时省略带宽表整行（应标注「未采到」并 retry）。

### 4.4 性能采集规模要求

- Round 0 **MUST** 覆盖 S 档 + L 档
- 中间优化轮次以 L 档为准
- 最终轮重新采集 S 档 + L 档
- 单点 WARMUP=10, MEASURE=40, ITERS=50

### 4.5 Phase 6 → 6.5 自动分支（Hard Gate）

当 `meta.performance_auto_optim: true`（且 `meta.performance_required: true`）且 Phase 6 基线采集已完成 **且未达标**：

1. **MUST** 在同一会话 **立即** 调用 `shmem-ops-performance-optim` 进入 Phase 6.5
2. 进入后 **MUST** 跑满 `max_opt_rounds`（默认 **5**）轮机制优化（达标 **不是** 提前停止理由）；**用户明确指定优化轮次时 MUST 跑满指定轮次**（即使 Round 0 已达标）
3. **禁止** 以「已达标跳过优化」「是否继续优化」询问用户或提前进入 Phase 7
4. 进入后中途达标 **不是** 停止条件；唯一正常停止条件是完成全部轮次（或用户明确要求停止）

当 `meta.performance_auto_optim: false`：输出差距表 → Phase 7，**不得**改 kernel。

---

## 5. 常见实现陷阱（写入设计/优化时对照）

### 统一实现 vs 大小分支（设计 + 优化）

- **默认一套实现**覆盖小到大消息；UB 内部分块拷贝是该实现的内部细节，**不算**第二条 size 路径
- ❌ 未证明收益就维护 `small_data` / `big_data`、`alltoallv_small` / `alltoallv_large` 等并行流水
- ❌ Phase 6.5 优先「修通 big_data / 恢复 size 分支」而非机制优化
- ✅ **仅当** profiling 证明 size 分支相对统一实现 **kernel_bus_bandwidth_GBps 或时延 ≥5% 收益** 时，才保留分拆；否则删除死代码、合并 single path
- ✅ 因 **symm/GVA 容量上限** 的外层 tile 循环（如 `GVA_BUFF_MAX_SIZE` 分块）视为内存约束，不是 perf 专用大小路径
- ✅ Phase 6.5 **优先机制优化**：流水 overlap、sender/receiver 分核、per-chunk signal、减 barrier、engine 切换；参数扫描为辅

### alltoallv 大消息

- ❌ sendbuf → 本地 symm → 再 put 远端（双拷贝）
- ✅ 远端：`mte_put(sendbuf+displ → remote symm+put_off)`；本端：`sendbuf → symm_off`
- send/recv 半核分组时 **MUST** `aclshmem_barrier_all()` 夹在 put 与 get 之间
- SDMA：先 MTE 路径验证正确性，再启用；`aclshmemx_sdma_quiet` 失败则回退 MTE
- ❌ `alltoallv_small` 单次 `mte_put/get` 传 `cnt` 超过 `UB_DMA_MAX_SIZE`（约 190KB）→ 大档数据损坏或 nan
- ✅ 复用 `CopyChunkMtePutSimple` / `CopyChunkMteGet` 按 UB 分块
- ❌ host 侧 `symm_elems = symm_data_elems + block_num * SYNC_FLAG_INTERVAL` 把 **int32 槽位数**当 **T 元素数**加 → 大档 symm 头越界
- ✅ 头部长度按 `block_num * SYNC_FLAG_INTERVAL * sizeof(int32_t)` 换算为 T 元素后再加

### fp16 测试数据

- ❌ `gen_data` 用 `pe*1000 + arange*0.01` 生成 fp16 输入，大 `base_count` 时数值超过 ~65504 → golden/output 同为 `inf`，误判为 kernel 错误
- ✅ fp16 用有界序列（如 `pe + arange*1e-4`）或比对时允许 golden 同 inf

### reduce_scatter 大消息

- ❌ `reduce_scatter_small_data` Phase1 单次 `mte_put` 超过 UB → 大档 FAIL
- ✅ Phase1 按 `UB_DMA_MAX_SIZE` 分块 put；本 PE 自拷贝用 `reduce_scatter_copy_chunk(..., use_atomic=false)`

### reduce_scatter perf

- ❌ perf 循环每轮 `memset` 输出（虚高 e2e）
- ✅ 仅 warmup 清零；计时含 `barrier_all` 表 collective 完成

### kernel / API

- ❌ custom-ops 新算子调用 `aclshmemi_*`（如 `aclshmemi_barrier_core_soft`）；用 `AscendC::SyncAll` + 公开 barrier/quiet（见 [internal-api-boundary.md](../../shmem-ops-code-gen/references/internal-api-boundary.md)）
- ❌ 新代码使用已 deprecated 的 `aclshmemx_barrier_all_vec`（→ `aclshmem_barrier_all`；**其他** `aclshmemx_*` 扩展接口仍可用）
- ❌ 纯 MTE 路径用 `aclshmem_quiet` 代替 `aclshmemx_mte_quiet`

### shell / 日志 / Docker

- **`docs/` 只读**：不得修改 `${SHMEM_REPO}/docs/` 原文档；读前先定位 `SHMEM_REPO`（[shmem-repo-resolution.md](shmem-repo-resolution.md)）
- 用户指定 Docker 后 **禁止**宿主机裸跑编译/运行/perf（见 [docker-exec-contract.md](docker-exec-contract.md)）
- **`aclshmemx_init_attr` 失败**：先定位 `SHMEM_REPO`（见 [shmem-repo-resolution.md](shmem-repo-resolution.md)），Read `${SHMEM_REPO}/docs/debug/log_debug.md`；索引 [shmem-repo-docs-index.md](../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md)
- SHMEM 日志：`SHMEM_LOG_LEVEL=DEBUG SHMEM_LOG_TO_STDOUT=1`
- 算子 API/行为不确定：先定位 `SHMEM_REPO`，再 Read `${SHMEM_REPO}/install/shmem/include/` 与 `${SHMEM_REPO}/docs/`（见 shmem-repo-docs-index、shmem-repo-resolution）
- HCCL baseline：主机侧 `uint64_t` count/displ；设备侧指针曾导致 segfault

### 产物检查

- 正确性前 `rm -f output_pe*.bin`，避免陈旧文件误判
- `pkill` 时 **禁止** 匹配自身脚本路径中的算子名（用 `pgrep -f 'build/bin/alltoallv'` 等）

---

## 6. 清理后重生成检查单

删除 `custom-ops/{reduce_scatter,alltoallv,allgatherv,torch_binding}` 后重跑：

- [ ] Phase 0 环境探测 + 业务确认 + `phase0_intake` 摘要（standalone 不问 in-tree）
- [ ] 四算子 design → testcase → codegen → 8PE C++ PASS
- [ ] `torch_required:true` → 四算子 `torch_test_*.py` 8PE PASS
- [ ] `performance_required:true` → [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md) 三阶段采集 + 聊天双表
- [ ] `performance_auto_optim:true` 且未达标 → Phase 6.5 Round 1 **已自动启动**并跑满 `max_opt_rounds`（达标不是提前停止理由；用户明确指定轮次时跑满指定轮次；非仅文字建议）

---

## 7. Skill 迭代

完成一轮交付后，若出现本文件未覆盖的新坑，**MUST** 追加 §5 一条并更新 `perf-chat-output-spec.md` 或子 skill，**禁止**只在聊天口头说明。

---

## 8. 通算融合算子规范（`op_kind=fused_compute_comm`）

编排器在 Phase 1 判定为通算融合后 **MUST** Read [fused-compute-phase-gates.md](fused-compute-phase-gates.md)，并遵守其中全部 Hard Gate。核心约束：

| 域 | 规则 |
| --- | --- |
| Phase 顺序 | Phase 0→7 按序；§7 交付物缺一不得 Phase 7 |
| 正确性 | `run_case_matrix.py` 全 functional case PASS；smoke 通过 ≠ Phase 4 完成 |
| 性能采集 | `performance_required:true` → Phase 6；**S+L 双档**；聊天双表（带宽+时延）； |
| 计算实现 | 默认 L2（CATLASS + CoC）；禁止 L0 三重循环；L1 `host_aclnn` 须 design 授权 |
| Baseline | HCCL/aclnn/CATLASS 拼接见[baseline-selection.md](../../shmem-ops-performance-eval/references/baseline-selection.md) |
| Phase 6.5 | `performance_auto_optim:true` **且未达标** → 进入后跑满 `max_opt_rounds`（默认 5）轮；每轮机制改动 + 复测 + Δ% 表；达标不是提前停止理由 |