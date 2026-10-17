# CANNBot Step 1→7 在 MC2 场景下的工作流映射

本文档把父级 `plugins-official/ops-direct-invoke/AGENTS.md` 的 7 步流程具体化到 MC2 通算融合算子场景。CANNBot 主控和 Architect/Developer/Reviewer 三类 Subagent 在每个 Step 进入前都应先读对应小节，明确"本阶段在 MC2 场景下要做什么、门禁是什么、调用哪些 skill"。

> 父级流程定义以 `plugins-official/ops-direct-invoke/AGENTS.md` 为准；本文件只补充 MC2 场景的差异化要点，不重写父级规则。

---

## Step 1：环境检查（门禁）

### 标准动作（父级定义）

按 `ascendc-env-check` skill 运行：
1. `npu-smi info` 查设备列表与状态；
2. 检查 `ASCEND_HOME_PATH` / `CANN_TOOLKIT_HOME` 等环境变量；
3. 运行 `workflows/scripts/verify_environment.sh {operator_name}`，生成 `environment.json`。

### MC2 场景额外校验

| 校验项 | 命令 / 方法 | 失败处理 |
|--------|------------|---------|
| **NPU 架构必须为 dav-3510** | `npu-smi info` 读 `Chip Name` 应为 `Ascend 950` 系列；CMake 阶段再次校验 `NPU_ARCH=dav-3510` | 非 3510 直接终止，告知用户"仅支持 Ascend 950 (dav-3510)" |
| **SHMEM 第三方库可解析** | 检查 `references/foundations/blaze-shmem/all_to_all_matmul/third_party/shmem/CMakeLists.txt` 是否存在；若不存在，`cmake/shmem.cmake` 会自动 `git clone --branch v1.5.0`（gitcode.com/cann/shmem） | 提示用户检查网络访问 gitcode.com |
| **tensor_api（`AscendC::Te::*`）可解析** | 检查 `third_party/tensor_api/include/tensor_api/tensor.h` 是否存在；若不存在，`cmake/tensor_api.cmake` 会自动 `git clone` asc-devkit（与 ascendc-blaze-best-practice skill 共享来源） | 提示用户检查网络访问 gitcode.com |
| **Blaze 头文件可解析** | blaze 头（`blaze/gemm/block/block_mmad*.h`）位于 CANN toolkit 的 `opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common`，由 CMakeLists 的 `_BLAZE_COMMON_DIR` 指向；无需 clone | 提示用户检查 CANN 安装路径是否正确 |
| **多卡环境可用** | `npu-smi info` 至少能看到 `rankNum` 张卡（默认 4 卡） | 提示用户准备多卡环境；单卡只能跑精度模式，性能模式需多卡 |

### 门禁

`environment.json` 的 `validation.all_passed=true` 且上述 MC2 校验全部通过 → 进入 Step 2。任何一项失败禁止进入 Step 2，告知用户原因。

---

## Step 2：设计（Architect）

### 输出物

父级要求双文件：`operators/{op}/docs/DESIGN.md` + `operators/{op}/docs/PLAN.md`。MC2 场景下 DESIGN.md **必须额外包含**以下小节（Reviewer 在 Step 4 会交叉检查）：

#### §三大约束显式确认

```markdown
## 三大约束确认

### 约束 1：通信走 SHMEM
- 选用的 SHMEM 原语：<aclshmemx_udma_put_nbi / aclshmemx_barrier_all_vec / ...>
- 确认无 Hccl:: 高阶 API：✅

### 约束 2：Matmul 走 Blaze
- 选用的 Blaze 模板：<Blaze::Gemm::Block::BlockMmad + Kernel::QuantMatmulMxKernelSwat 等>
- DispatchPolicy：<MatmulWithScaleMx / ...>
- 确认无 asc-devkit Matmul 高阶 API：✅

### 约束 3：流程合规
- 复用基底：references/foundations/blaze-shmem/all_to_all_matmul/
- 计划修改的文件清单：<列出 [MODIFY] 文件>
```

#### §切分策略

- **两阶段 `tileCnt` 策略**（详见 [`pipeline_tuning.md`](../../shared/pipeline_tuning.md)）：
  - Step 2-4 设计/审查阶段：`tileCnt=1`（`headMSize=m`）做串行基线，简化精度/审查调试；
  - Step 6 性能验收阶段：扫描 `tileCnt ∈ {1, 2, 4, 8, 16, 32}` 找最优，`headMSize=512` 只是起点不是最优值。
- M 轴切分：`headMSize = m / tileCnt`，`bufferSize=4`
- SHMEM 空间预算（默认 1 GB，需够装下 `rankSize * bufferSize * bufferBlockSize + scale`）

#### §AIV/AIC 分工图

明确哪些 work 在 AIV（通信），哪些在 AIC（计算），同步点在哪里（`CrossCoreSetFlag<0x2, PIPE_*>`，其中 `0x2` 是 modeId 即 AIV↔AIC 跨核同步模式）。

### Architect 加载顺序

1. 加载 ascendc-mc2-best-practice skill，读 `references/foundations/blaze-shmem/mc2_architecture.md` 建立整体心智模型；
2. 若通信侧不确定用哪个 SHMEM 原语 → 读 `references/foundations/blaze-shmem/comm_shmem.md`；
3. 若计算侧不确定用哪个 Blaze 模板 → 读 `references/foundations/blaze-shmem/matmul_blaze.md`；
4. 输出 DESIGN.md + PLAN.md。

### 门禁

- 双文件齐全；
- DESIGN.md 包含"三大约束确认"小节，且勾选 ✅；
- 切分策略中 `headMSize` 等参数有可解释的依据（不能拍脑袋）。

---

## Step 2.5：设计串讲

### Developer 串讲模式关注点

- 复用的基底文件清单是否合理（不能把 `[REUSE]` 文件错标成 `[MODIFY]`）；
- AIC 是否在通信 buffer 上遍历所有 rank，rank==rankId 是否切换到本卡 GM；
- SHMEM 空间预算是否够（`rankSize * commMSize_ * kPerRank * bufferSize` + scale 段）。

### Architect 串讲回应关注点

- 若 Developer 提出"想用 HCCL 替代 SHMEM" → 直接拒绝（HCCL 高阶 API 禁止清单及理由见 [`comm_shmem.md`](comm_shmem.md) §5）；
- 若 Developer 提出"想用 asc-devkit Matmul 高阶 API" → 直接拒绝（Matmul 必须走 Blaze 模板，见 [`matmul_blaze.md`](matmul_blaze.md)）。

### 收敛

严格 1 轮串讲；分歧写入 WALKTHROUGH.md `## 设计串讲仲裁`。

---

## Step 3：开发（Developer）

### 起手流程

```bash
# 1. 从基底工程复制
cp -r references/foundations/blaze-shmem/all_to_all_matmul operators/{op_name}

# 2. 按 [MODIFY] 标记定点改造
#    - src/{op_name}.cpp：函数名、kernel 调用、参数解析
#    - include/kernel/all_to_all_comm_udma.h：通信原语（若非 AllToAll）
#    - include/kernel/all_to_all_matmul_impl.h：通算融合主类
#    - include/kernel/qbmm_mx_kernel.h：Blaze Kernel 包装（改 Scale/dtype/bias 时动）
#    - include/tiling/*：TilingData 字段
#    - scripts/gen_data.py + verify_result.py：dtype/容差

# 3. 编译
cd operators/{op_name}
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build -j

# 4. 跑精度
bash run.sh  # 默认 m=2048 k=8192 n=3584 rank=4 precision
```

### 开发阶段红线

- **禁止**新增 `Hccl::` 调用（Reviewer 会在 Step 4 grep）；
- **禁止**包含 asc-devkit 的 matmul 头；
- **禁止**为了赶进度跳过 `run.sh` 的精度模式直接跑 perf；
- 改完每个 `[MODIFY]` 文件后立即跑一次 `bash run.sh` 做冒烟，避免最后一次性 debug。

### 门禁

- 编译通过；
- `bash run.sh`（precision 模式）输出 `verify_result.py: PASS`；
- 代码已提交到 `operators/{op_name}/`。

---

## Step 4：审查（Reviewer）

### 必查清单

按 [`review-checklist.md`](review-checklist.md) R1~R7 逐项检查（含验收条件、常见 FAIL 原因与修复方向）。违反任意红线项 = FAIL。

### 门禁

- REVIEW.md 判定 `PASS` 或 `PASS WITH NOTES` → 进入 Step 6；
- REVIEW.md 判定 `FAIL` → 进入 Step 5 修复循环。

---

## Step 5：修复循环

最多 3 轮。每轮：Developer 修复 → Reviewer 复审。MC2 场景下常见的"修不动"问题：

- **跨核同步 flag 错位**：`CrossCoreSetFlag<0x2, PIPE_MTE3>(mLoopIdx)` 与 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx)` 的 idx 必须一致；
- **UDMA Put 顺序错**：`aclshmemx_udma_quiet(remoteRank)` 必须在每次 Put 后调用，否则数据未真正下发；
- **remoteRankCnt 错位**： `splitKNum` 必须等于 `rankSize`，否则 fixpipe 时机不对（最后一次 mmadOp_ 才触发）；

3 轮仍未通过 → 暂停上报用户。

---

## Step 6：性能验收（Developer）

### MC2 专用流程（必须刷 L2 cache）

> 详细步骤见 `profiling_mc2.md`。这里只给摘要。

```bash
PROJ="$(pwd)"

# 1. 算子二进制已具备 perf 模式（run.sh 第 5 参传 perf）
bash run.sh 2048 8192 3584 4 perf

# 2. msprof task-based 采集（无 warm-up；L2 flush 由 perf 主循环内部保证）
msprof --ai-core=on --aic-mode=task-based \
    --output="${PROJ}/docs/perf/round_001" \
    --application="${PROJ}/build/{op_name} 2048 8192 3584 4 perf"

# 3. 多卡后处理：每卡取最后 5 次 main kernel 平均，4 卡取最大
python3 scripts/extract_perf.py "${PROJ}/docs/perf/round_001" AllToAllQuantMatmulKernelE4M3E4M3 5
```

### 关键点

- **两阶段 `tileCnt` 扫描**：进入 Step 6 时算子默认处于 `tileCnt=1` 串行基线（Step 2-4 期间配置）。Step 6 的工作是扫描 `tileCnt ∈ {1, 2, 4, 8, 16, 32}`，每个值跑一次 msprof 采集 + 后处理，对比整体 Task Duration 找最优。完整流程见 [`pipeline_tuning.md`](../../shared/pipeline_tuning.md) §3 阶段 B + §6 决策树；
- **L2 cache flush 必须在主循环每轮触发**：参考工程在 perf 主循环（`src/all_to_all_matmul.cpp` 的 `mode == "perf"` 分支）中每轮先调用 `heavy_add_kernel`（256 MB bf16 全核扫一遍）刷 L2，再跑主 kernel，避免上一轮的热度污染本轮采集——所以 msprof 不需要 `--warm-up`；
- **perf 模式默认跑 10 轮**：msprof task-based 为每轮 main kernel 生成一条 `op_summary_*.csv` 记录；
- **多卡数据提取规则**：每卡取最后 5 次 main kernel 的 Task Duration 求平均 → 4 卡平均值取最大值作为整体性能（多卡并行由最慢卡决定）。

### 门禁

- `docs/perf/round_NNN/` 存在且包含 4 个 `PROF_*` 子目录（每卡一份）；
- `extract_perf.py` 输出每卡 `rank_avg` 与整体 `max` Task Duration；
- 最优 `tileCnt` 与对应整体 Task Duration 已写入 DESIGN.md "性能调优记录"；
- 若整体 Task Duration 与理论耗时差距 >50% → 视为性能不达标，检查通算流水是否真的并行（AIV time vs AIC time）。

---

## Step 7：完成汇报

CANNBot 主控汇总以下信息给用户：

- **最终判定**：PASS / PASS WITH NOTES
- **总分**：Reviewer 100 分制
- **代码路径**：`operators/{op_name}/`
- **性能概要**：Task Duration、主导流水（MC2 通常 AIC cube 主导）、通信隐藏率（AIV 与 AIC 重叠度）
- **关键问题列表**：Step 4~5 期间遗留的 NOTE
