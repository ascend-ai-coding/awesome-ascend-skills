# 模板成熟度与兼容性

先按本表判断代码可用性，再读取算子族文档。`PRODUCTION_REFERENCE` 是特征门控状态：当前 kernel 与 launcher 必须包含本表摘要及链接文档的核心结构；缺失时将当前路径降为 `EXECUTABLE_BASELINE`，历史优化知识只作候选，不继承性能结论。文件存在不代表优化策略已经实现。

结构文档中的“已实现并实测”是知识证据等级，不属于代码模板成熟度，也不决定候选实施优先级；迁入当前实现仍须核对适用条件并重新验证。

| 状态 | 使用规则 |
|---|---|
| `PRODUCTION_REFERENCE` | 真源位于当前仓库；直接引用生产文件，不在 Skill 复制 kernel |
| `VERIFIED` | 在下列版本完成 PTO lowering、设备运行和精度校验；迁入其他版本仍须复验 |
| `EXECUTABLE_BASELINE` | 可执行的正确性起点，没有性能结论，不得直接进入生产 dispatch |
| `PARTIAL` | 只有部分策略结构或退化为 baseline，必须补全并验证 |
| `DESIGN_ONLY` | 只有设计、tiling 元数据或 Python helper，不是 TileLang kernel |

## Agent 展示分类

成熟度描述实现完成度，不单独代表性能排名。Agent 必须同时检查本表“已验证范围或缺口”和对应算子族文档，再使用以下展示分类：

- ✅ **可直接拷贝优化模式**：`PRODUCTION_REFERENCE` 或当前版本 `VERIFIED` 的 TileLang `.py`，并且优化结构已经真实实现。可以复用实现骨架；只有存在适用于当前比较口径的性能证据时，才能称为高性能模板或预期更快。
- ⚠️ **可执行基线或部分实现**：`EXECUTABLE_BASELINE`、`PARTIAL`，或虽可运行但没有对应优化结构/性能证据的实现。只作为正确性起点、实现参考或待补全候选。
- ❌ **仅设计参考**：`DESIGN_ONLY`，或没有当前版本可执行 TileLang kernel 的资料。不得进入直接实施方案。

不得仅因存在完整 `.py` 文件就判为“可直接拷贝优化模式”；也不得仅因代码位于本 Skill 中就宣称性能优于目标仓库当前实现。

## 迁移前历史验证基线

| 项目 | 已验证版本 |
|---|---|
| 历史算子实现快照 | commit `5802406` |
| 历史框架快照 | commit `84855805` |
| TileLang Python 包 | `0.1.12+cuda.git84855805` |
| PTOAS | `0.58`，commit `f8174330` |
| 后端 | `TILELANG_DEFAULT_TARGET=pto`，Ascend NPU |
| 验证日期 | 2026-08-11 |

下表中已修改构建接口的模板需在当前仓库重新验证，历史记录不构成本次迁移后的通过证明。实际运行前执行 `python -c 'import tilelang; print(tilelang.__file__, tilelang.__version__)'`。版本或源码路径不同即视为未验证组合。

## Bundled code 状态

| 路径/策略 | 状态 | 已验证范围或缺口 |
|---|---|---|
| `broadcast/code/broadcast_add_kernel.py` | `EXECUTABLE_BASELINE` | 显式传入已查询核数后需复验；历史覆盖 fp16 `3x129`、`2x257`；UB/fragment 按 256 padding，GM 仅访问 valid |
| `broadcast/code/onedim_add_kernel.py` | `EXECUTABLE_BASELINE` | 与 broadcast add 共用同一实现和 tiler，不是独立优化策略 |
| `reduce/templates/dav310/kernel_utils.py:euclidean_norm` | `EXECUTABLE_BASELINE` | 显式传入已查询核数后需复验；历史覆盖 fp16 `65x65`、`2x129`、`2x257`；按 `next_power_of_2(R)` padding；仅支持 padded R `<=4096` |
| `reduce/.../euclidean_norm_*tail*.py` | `EXECUTABLE_BASELINE` | 调用同一 padded full-load baseline，不代表专用 tail 优化 |
| `reduce/.../euclidean_norm_group_block_split_r.py` | `DESIGN_ONLY` | 没有 verified split-R kernel，`build()` 主动拒绝使用 |
| `reduce/.../softmax_v2_base.py`、`ar_full_load.py` | `EXECUTABLE_BASELINE` | 稳定 fp32 串行 full-load 正确性路径；没有生产性能结论 |
| `reduce/.../softmax_v2_ar_small_r.py` | `PARTIAL` | 当前仍复用串行 full-load，未实现多 row batching |
| 其他 `softmax_v2_*` 策略文件 | `DESIGN_ONLY` | 只有策略元数据或 online merge helper，没有可调度 kernel |
| `scan/templates/dav310/scan_base.py` | `EXECUTABLE_BASELINE` | 显式传入已查询核数后需复验；历史覆盖 bf16→fp32 `2x257`；串行 inner scan，只用于正确性 |
| `cum_streaming_scan.py`、`cum_tile_resident_scan.py` | `EXECUTABLE_BASELINE` | 共用 scan baseline，不代表 lane-parallel 优化 |
| 其他 `cum_*` Python 策略 | `DESIGN_ONLY` | 只有策略元数据、workspace 或依赖对 helper |
| `rope/code/rope_vf_common.py` | `EXECUTABLE_BASELINE` | 显式传入已查询核数后需复验；历史覆盖 bf16 half-split `2x2x128`；连续、编译期 position offset 基线 |
| `conversion/templates/dav3510/transpose_base.py` | `PARTIAL` | 仅提供接口适配与既有 64 对齐约束检查；必须传入当前仓库已验证的 kernel factory，不包含通用转置 kernel |
| 其他 conversion Python 策略 | `PARTIAL` 或 `DESIGN_ONLY` | 多数只做选择/约束，未提供独立 PTO kernel；逐文件核对后使用 |

## 状态升级门禁

把 `PARTIAL` 或 `DESIGN_ONLY` 升为 `VERIFIED` 时同时记录：目标硬件、CANN、TileLang 与当前仓库版本、完整命令、shape/dtype、容差、lowering 状态、首个失败、性能口径和原始结果文件。没有同条件实测数据时只称“实现候选”或“正确性 baseline”。
