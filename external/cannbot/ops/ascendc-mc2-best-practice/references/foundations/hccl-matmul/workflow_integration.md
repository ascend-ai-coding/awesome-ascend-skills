# CANNBot 工作流在 910B（A2）MC2 场景的具体化

本文档把 hccl-matmul 路线（910B / A2 / dav-2201，HCCL 高阶 API V2 + `AscendC::Matmul`，aclnn 单算子，无 quant）的设计/开发/验收门禁具体化。本路线仅覆盖 A2，不涉及 A3（910_93）。流程编排归属 plugin 层；本文只补充本路线技术要点。

> 蓝本为已验证功能正常的 [`all_gather_matmul/`](all_gather_matmul)。

> 工作流四阶段：**阶段一 需求与设计**（CP1/CP1.5/CP2）→ **阶段二 开发** → **阶段三 验收**（3.1 精度 / CP3 / 3.2 性能 / CP4）→ **阶段四 上库**（CP5）。

## 阶段一：需求与设计

### 1.1 开发准备（环境检查，门禁）

按父级加载 `ascendc-env-check` + `ascendc-docs-search`，填 `operators/{op}/docs/environment.md`（状态行 `✅ 通过` 才继续）。

| 校验项 | 命令/方法 | 失败处理 |
|--------|-----------|----------|
| NPU 架构 = dav-2201（A2） | `ascendcPlatform.GetCurNpuArch()` → `NpuArch::DAV_2201`；`GetSocVersion()` → `ASCEND910B` | 非 2201：本路线不适用（950 集合通信走 blaze-shmem/apace） |
| 910B 子型号（B4 等） | **蓝本不做自动判定**：`x_all_gather_matmul_tiling_a2a3.cpp:53` 硬编码 `SocVersion::SOC910_B` 实例化公式化 tiling。`SOC910_B4` 枚举存在（`formulaic_tiling_datatype.h:46`），仅在 perf 模型的 K 对齐折扣分支被用到（`matmul_performance.cpp:185`，`K_UNALIGN_UTIL_RATIO_910_B4`） | 若目标是 B4 且性能模型偏差明显，需自行在 arch22 子类里按实际子型号传 `SOC910_B4` |
| HCCL/Matmul 头 + 工具链 | CANN 官方仓库 <https://gitcode.com/cann/asc-devkit> 的 `docs/api/SIMD-API/高阶API/HCCL通信类`、`矩阵计算` 可读；`lib/matmul/matmul_intf.h`、`adv_api/hccl/hccl.h` 可解析；CANN（cann-9.1.0）+ bisheng + `ASCEND_CUSTOM_OPP_PATH` 就绪 | 确认 `ASCEND_HOME_PATH`/`ASCEND_OPP_PATH`/`ASCEND_CUSTOM_OPP_PATH` |
| 硬件/仿真 | 真机或 `cannsim`（`ops-simulator`） | 无真机时优先 `ops-simulator` |

### 1.2 需求分析（→ CP1 用户确认 → 1.2.5 spec 生成/评审 → CP1.5）

按父级 spec-to-design 流程产出算子 spec（名称、数学定义、dtype、shape、通信原语）。MC2 场景须明确：通信算子类型（AllGather/AllReduce/ReduceScatter/AlltoAll/BatchWrite）、rankDim、是否 bias、是否 gather_out。

### 1.3 设计（→ 1.3R 方案评审 → CP2）

DESIGN.md + PLAN.md 必含：

#### §三大约束显式确认
```
- 通信：HCCL in-kernel 高阶 API V2（Hccl<HCCL_SERVER_TYPE_AICPU>，A2 服务端仅 AICPU；InitV2+SetCcTilingV2，蓝本已用）；禁 SHMEM/HCOMM/窗口手动 MTE/RAC
- 计算：AscendC::Matmul（Matmul 跑在 AIC/Cube 核，AIV 通知）；禁 Blaze
- 流程：按注册形态工作流推进，不跳过设计/精度/性能验收
- 本期不支持 quant
```

#### §切分策略
- 通信算子类型；algConfig 在 A2 为预留字段，仅 FullMesh。
- M 轴切分：`tileCnt`（指南 `tileNum`）+ `tailNum`∈{0,1}；主块/尾块/本地 rank 各一套 `TCubeTiling`。
- 窗口优化：`SetSkipBufferWindowCopy` 由 `MC2_BUFFER_TYPE` 驱动（value 2 仅 AllReduce/AlltoAll）。

#### §AIV/AIC 分工与 HCCL 下发模式
- Matmul 在 AIC 计算、AIV 通知消费。
- HCCL 与 Matmul 都在 AIC 执行（**AIC-only**，`if ASCEND_IS_AIC { … }` 门控，AIV 不执行主体）；**Finalize 前需跨核同步**：蓝本 `HcclFinalize()` 用 `CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6)`+`CrossCoreWaitFlag(EVENT_ID_6)`（`full_mesh.h:224-225`），非 `SyncAll<true>()` 但效果等同。HCCL V2（`InitV2`+`SetCcTilingV2`，蓝本已用）。详见 [`mc2_architecture.md`](mc2_architecture.md) §3/§4、[`comm_hccl.md`](comm_hccl.md) §2。

### 1.4 测试设计（→ 1.4R 测试设计评审）
- `examples/golden.py`：CPU fp32 参考（蓝本：gather dim0 → matmul → +bias）；`test_golden_cpu.py` 自测。golden 须**逐卡忠实**——每卡各自的输入分片分别参与运算，不要用"单卡输入 × rank_size"糊过去。
- `examples/run_test.sh`：多轮随机 shape（M∈[1,64]、K/N 抽样）、fp16/bf16 与 bias 有/无按轮次奇偶交替（`isGatherOut` 恒为 1，未参与循环）；每轮 `RESULT: CHECK PASSED`。
- 校验范围：**各卡输出不同**的算子（ReduceScatter/AlltoAll 类）必须**逐 rank 比对**。蓝本只读 rank-0 产物（`output` + `gather_out`）是因为 AllGather 后各卡输出相同；照抄到 scatter 类算子会漏掉 rank 块偏移算错的 bug。

#### 1.4.1 精度判据按融合方向选择（不可照抄蓝本）

| 融合方向 | 例 | 误差量级 | 判据 |
|----------|----|----------|------|
| 先通后算（通信在 Matmul 前） | AllGather+MM（蓝本） | ∝ `\|输出\|`——输出只经历一次"对自身"的舍入 | 蓝本判据可用：误差元素比例 `<1e-2`，逐元素 `np.isclose(rtol=atol=5e-3)`（bf16 `1e-2`），非严格 allclose |
| **先算后通**（归约在 Matmul 后） | MM+ReduceScatter、MM+AllReduce | ∝ **加数**量级——部分积先按输出 dtype 落 workspace（那就是通信发送缓冲）再跨卡求和，故 `\|err\| ~ ε·\|partial\|`、`\|partial\| ~ rms(\|y\|)/√R` | **不可**沿用逐元素 `rtol`/`atol`：改用整体相对误差 `‖y_npu−y_fp32‖₂ / ‖y_fp32‖₂ ≤ 4ε`（fp16 `ε=2⁻¹¹`→`1.95e-3`；bf16 `ε=2⁻⁸`→`1.56e-2`），逐 rank 各判一次 |

先算后通为什么会被逐元素判据系统性误判：randn 输入下 `y ~ N(0, σ)`，必然有一小撮元素 `|y| << σ`——它们是若干 `~σ/√R` 量级的加数**相消**出来的，却仍带着完整的 `ε·|partial|` 绝对误差。这些元素上 `rtol·|y|` 项塌缩到 0，只剩一个与数据量级无关的固定 `atol`，必然判 FAIL。**这是判据不适用，不是算子精度不够**；发送缓冲 dtype 必须等于通信 dtype，是先算后通结构的固有属性，改不掉。

实测标定（MM+ReduceScatter，randn 单位输入，`K=12288`、`R=2`、bf16）：`rms(|y|) ≈ √K·√R ≈ 157`，舍入噪声 `σ_err ≈ 0.3·ε·rms(|y|) ≈ 0.18`；按 `|err| > atol + rtol·|y|`（`atol=rtol=0.02`）积分得超差元素比例 2.1%–2.6%，实测 **2.55%**——误差恰好等于 bf16 能达到的极限，无异常。fp16 同法算得约 0.33%，低于 1% 门限但余量仅 3 倍，所以会偶发踩线（现象是"bf16 几乎必挂、fp16 偶尔挂"）。

整体相对误差以全局能量为分母，对相消免疫；纯舍入下约 `0.3ε` 且**与 `rank_size` 无关**（`rel_l2 ≈ 0.3ε` 与 R 无关，故阈值不必按卡数放大），取 `4ε` 留一个数量级余量。它不比逐元素判据弱：`err_ratio<1e-2` 本来就容忍不到 1% 的元素出错，而任何能让 1% 元素出错的 bug（某片/某 rank 块算错）在整体相对误差上是 `O(0.1)`，比 `O(ε)` 大两三个数量级。参考实现见 `matmul_reduce_scatter` 类算子的 `examples/single_server_check_result.py`。

### Architect 加载顺序
1. [`op_architecture.md`](op_architecture.md) → 2. [`mc2_architecture.md`](mc2_architecture.md) → 3. [`comm_hccl.md`](comm_hccl.md) → 4. [`matmul_fusion.md`](matmul_fusion.md) → 5. [`pipeline_tuning.md`](pipeline_tuning.md)（Matmul API 签名回 `ascendc-api-best-practices` 的 `api-matmul.md`）

## 阶段二：开发（CP2 通过后）

### 起手流程
```bash
cp -r references/all_gather_matmul operators/{op_name}
cd operators/{op_name}
# 0) CANN 环境（必须先 source，否则后续 import torch_npu 会找不到 libhccl.so）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 1) 构建算子（.run + libcust_opapi.so；默认 SoC ascend910b，单次烘焙 fp16+bf16）
bash build.sh -n x_{op_name}          # -n 的值必须 == 外层算子目录名（== OpDef 类名 snake_case）
                                      # 产物 build/custom_opp_<arch>.run（仅 -b host 才拷到 output/）
export ASCEND_CUSTOM_OPP_PATH=<vendors/custom_opp 绝对路径>
# 2) PTA（若用户要在 PyTorch 里调）：走 torch-ascendc-op-extension **路线 B**（aclnn 注册）
#    蓝本不提供 torch_ops_extension，不要在本路线内手写 PTA 层。
# 3) 精度测试（多卡 mp.spawn + HCCL；依赖 PTA 已按路线 B 生成）
bash examples/run_test.sh            # RESULT: CHECK PASSED
```
> 构建机制详见 [`op_architecture.md`](op_architecture.md) §6 与 [`codebase_map.md`](codebase_map.md) §5。HCCL context 由 torch_npu 分布式创建（非虚构 host launcher），见 [`comm_hccl.md`](comm_hccl.md) §7。PTA 生成见 `ops/torch-ascendc-op-extension/routes/aclnn-registry.md`。
> ⚠️ 从 references 复制出来的工程与原工程会共用同一个 `VENDOR_NAME`（`custom_opp`）。同一环境里两份工程交替编译安装时，`.run` 容易装成 `vendors/custom_opp/vendors/custom_opp` 嵌套，运行时加载到旧版本——现象是"改了 kernel 却跑出旧结果"。给复制出的工程改一个不同的 `VENDOR_NAME`（顶层 `CMakeLists.txt:10`），或安装 `.run` 时用 `--install-path=` 显式指定，详见 [`codebase_map.md`](codebase_map.md) §5。

### 开发阶段红线
- 通信：只用 `Hccl<HCCL_SERVER_TYPE_AICPU>` 高阶 API **V2**（`InitV2`+`SetCcTilingV2`）；不得引入 `aclshmemx_*`/`hcomm_`/窗口手动 MTE/RAC/A2 不支持的接口（`AlltoAllV`/`Finalize<false>` 等）。
- 计算：只用 `AscendC::Matmul`（或别名 `MatmulImpl`/`MatmulCompute` 封装）；不得引入 Blaze；不调 `SetLocalWorkspace`（A2 不支持）。
- 不得引入 quant 代码。
- HCCL/计算 AIC-only 门控（`if ASCEND_IS_AIC`）；**Finalize 前必跨核同步**（`CrossCoreSetFlag`/`CrossCoreWaitFlag`，非 SyncAll）。

## 阶段三：验收

### 3.1 精度验收（→ CP3）
Reviewer R1–R7（见下）+ `bash examples/run_test.sh` 多轮 `CHECK PASSED`；精度报告归档 `docs/precision/summary.txt`。

判据必须与融合方向匹配（§1.4.1）：先算后通的归约算子若照抄蓝本的逐元素 `rtol`/`atol`，会得出"bf16 精度不达标"的假结论并浪费一轮返工。判据选择与阈值须在 DESIGN.md 里写明理由。

### Reviewer 速查（R1–R7）
```bash
R1: grep -i "ascend910b\|dav-2201" CMakeLists.txt x_{op}/CMakeLists.txt  # → ascend910b（默认）/dav-2201，无 A3/910_93/950
R2: grep -rn "Hccl<HCCL_SERVER_TYPE_AICPU>\|InitV2\|SetCcTilingV2\|hccl_\.\(AllGather\|AllReduce\|AlltoAll\|Commit\|Wait\|Finalize\)" operators/{op}/
R3: grep -rn "lib/matmul/matmul_intf.h\|AscendC::Matmul\|MatmulImpl\|MatmulCompute" operators/{op}/  # 无 Blaze
R4: grep -rn "aclshmem" operators/{op}/                         # 应空
R5: grep -rni "blaze" operators/{op}/                          # 应空
R6: ls docs/                                                    # DESIGN/PLAN/WALKTHROUGH/REVIEW 齐全；environment.md ✅通过
R7: grep -rni "quant\|qbmm_mx\|copy_scale" operators/{op}/     # 应空
# HCCL 模式：grep "ASCEND_IS_AIC" 命中（AIC-only）；grep "CrossCoreSetFlag\|CrossCoreWaitFlag" 命中（Finalize 前同步，非 SyncAll）；
#   V2（grep InitV2/SetCcTilingV2）；V1 Init/SetCcTiling 应为空（已废弃）
```

### 3.2 性能验收（→ CP4）
```bash
# 前置：build.sh -n x_{op_name} 已构建 + ASCEND_CUSTOM_OPP_PATH 已设 + PTA 已由 torch-ascendc-op-extension 路线 B 生成
bash examples/run_test.sh                       # 精度基线
msprof --application="python examples/single_server_run.py" --output=PROF_xxx --aic-metrics=... --task-based
python scripts/extract_perf.py PROF_*/op_summary_*.csv   # 每卡 last N、跨 rank max
```
> 910B 走 HCCL（AICPU），无 SHMEM B-matrix residency 问题，**无需 950 的 L2 flush**；warm-up 仍建议。指南未述 L2 flush，此为本 skill 推理。
- 两阶段 tileCnt 扫描：先 `tileCnt=1` 基线，再扫 {1,2,4,8,16}（A2 一通信域 Prepare≤63）。
- perf 跑 ≥10 轮，多卡取每卡 last 5、跨 rank max；核数 `MARK_CORE_NUM_SOC910B=20`（910B3/B4 20-core 变体）。
- 门禁：cube_ratio 40–70%、通信隐藏率符合 DESIGN.md。

### 常见 FAIL
| 现象 | 根因 | 修复方向 |
|------|------|----------|
| R2 命中 `hcomm_`/`aclshmem` | 误用 SHMEM/HCOMM | 改回 `Hccl<HCCL_SERVER_TYPE_AICPU>` V2 |
| R5 命中 Blaze | 误抄 950 模板 | 换 `AscendC::Matmul` |
| `Wait` 卡死 | Finalize 前缺跨核同步 | 加 `CrossCoreSetFlag/CrossCoreWaitFlag`（参考 `full_mesh.h:224-225`） |
| 误用 `AlltoAllV` | A2 不支持 | 改 `AlltoAll`（等长） |
| R7 命中 quant | 引入量化路径 | 删除 quant 文件 |
| bf16 稳定超差 2%~3%、fp16 偶发（先算后通算子） | 判据不适用：相消元素被固定 `atol` 误判，**非算子问题** | 换整体相对误差 `≤4ε`（§1.4.1）；先看误差是否 ≈ `0.3·ε·rms(\|y\|)`，是则纯舍入 |
| scatter 类算子精度"全过"但上层结果错 | 只校验了 rank-0（照抄蓝本），各卡不同的输出没查 | 逐 rank 比对（§1.4） |

## 阶段四：上库（CP5）

审查通过且精度+性能验收完成后，汇报：最终判定、总分、代码路径、精度概要（各 dtype 达标状态）、性能概要（Task Duration、cube_ratio、通信隐藏率）、关键问题列表。

## 后续阅读

| 想了解 | 读 |
|--------|-----|
| 算子架构与可复用框架 | [`op_architecture.md`](op_architecture.md) |
| 910B MC2 心智模型 | [`mc2_architecture.md`](mc2_architecture.md) |
| HCCL API 与 V2 生命周期 | [`comm_hccl.md`](comm_hccl.md) |
| Matmul 接入（AIC） | [`matmul_fusion.md`](matmul_fusion.md) |
| PTA 接口生成 | `ops/torch-ascendc-op-extension` 路线 B（aclnn 注册），本路线不重做 |
| 性能采集 | [`../../shared/profiling_mc2.md`](../../shared/profiling_mc2.md)（本路线无需 L2 flush） |
| tileCnt 调优 | [`pipeline_tuning.md`](pipeline_tuning.md) |
| 基座工程改造 | [`codebase_map.md`](codebase_map.md) |
