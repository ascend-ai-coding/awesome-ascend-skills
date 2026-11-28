---
type: "PyPTO Performance Optimization Card"
title: "消费点 GM→L1 直载：替换经 UB 的 NZ 转换路径"
description: "把「GM→UB→NZ 转换→L1」的搬运，替换为消费数据的 cube 阶段内一次 pl.load GM→L1 直达（MTE2 按分形布局直填 L1），避免通过 UB 做 NZ 转换，并删除为此专设的 vec 搬运阶段；vector 空闲且能提前发射等场景保留 UB 路径更优。"
status: "stable"
tags: ["pypto-pro", "pipe", "cube", "load", "layout"]
item_id: "pipe-05"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: alternative_to
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#thin-row-contractions-compare-arithmetic-and-transport-paths"
      scope: "针对章节中的原始 GM 操作数经 GM→UB→NZ→L1 路线，比较在 Cube 消费阶段直接 GM→L1 加载的替代方案。"
      boundary: "仅关联运输路径讨论，不关联该章节的算术核心选择。本卡要求无需 Vector 加工、串行链无法有效重叠，并核对尾块可读范围；不能替换实际需要 Vector 计算的生产阶段。"
bound_hint: "PIPELINE"
applicability: "多阶段 V/C 串行链中存在仅为把 GM 数据搬入 L1 而设的 vec 暂存段（GM→UB→NZ 转换→L1），数据只被紧邻 cube 阶段消费、无需 vec 侧加工，串行链同步主导（阶段间无重叠），且上游恒写满全部行（尾块整行加载安全）"
target_api_gate: "Ascend 950PR（A5）实测；依赖已核验的 pl.load（GM→Mat/L1 直达 + order 轴映射 + set_validshape 尾块语义）与 Mat/NZ TileType；尾块整行直载必须确认上游写满全部行，否则越界读未定义内存"
---

# 技术卡片 pipe-05：消费点 GM→L1 直载：替换经 UB 的 NZ 转换路径

- **适用 bound**：调度 / 搬运（串行链时延）
- **一句话**：用 cube 侧本就支持的 GM→L1 `pl.load`，替换「GM→UB→NZ 转换→L1」路径——MTE2 硬件按 L1 分形布局直填，**避免通过 UB 做 NZ 转换**，同时删掉为此专设的 vec 搬运阶段。

```
before：GM ──load──> UB ──NZ 转换──> L1    （专用 vec 段 + 一对跨核事件）
after：  GM ──load──────────> L1           （消费数据的 cube 阶段内一步完成）
```

## 何时用（诊断特征）

- **源码**：串行链上存在一个专用 vec 段/阶段，其全部工作是 `pl.load`（GM→UB）→ `pl.move`（UB 内 ND→NZ 转换）→ `pl.insert`（UB→L1）三步搬运；产出只被紧邻的 cube 阶段当作 matmul 操作数消费，无其它消费者。
- **profiler**：任一计算 pipe（cube/vec）占用都不高，kernel 时长由阶段屏障与跨核事件往返决定（串行链时延主导）——每删一个阶段即省一对事件往返。
- **数据条件**：数据无需 vec 侧加工（cast/数学变换），dtype 与布局可由 `pl.load` 的 `order` 轴映射直接表达；上游 kernel 恒写满全部行（含尾块补零），整行加载不越界。

## 何时不适用

- **vector 空闲且搬运能提前发射时，保留 UB 路径更优**：若 vec 侧空闲、且该搬运段可以提前发起（与 cube 阶段正在做的计算重叠），那么 GM 加载与 NZ 转换的延迟被计算隐藏；此时把 GM→L1 加载压进 cube 阶段反而落在关键路径上，可能更慢。本方法在**阶段间无法重叠的严格串行结构**（如 sync_only 流水配置）下才稳定获益。
- 数据需要 vec 侧加工（cast、数学变换、拼接）后才能进入 matmul。
- 尾块行未被上游写满：整行 full load 会越过 tensor 分配边界读到未定义内存（见技术限制），需要 `validshape` 尾块保护的数据应走 UB 路径或精确设置有效行。
- 数据有多个跨核消费者（UB 中转一份、多处分发）。

## 原理

- `pl.load` 支持把 GM 数据**直达** L1（`Mat` 内存空间）tile，`order` 参数完成 Tensor 轴到 tile 轴的映射（升序不转置、反序转置），MTE2 硬件通路直接按 L1 tile 的分形布局（NZ）填充——原来在 UB 里做的那次软件 ND→NZ 转换（`pl.move`）不再需要。
- 搬运路径从两跳（GM→UB→L1）变一跳（GM→L1），省一次 UB 落地与对应搬运指令；更重要的是**串行链少一个阶段**：少一对跨核事件（set/wait）、少一次阶段边界屏障——串行链时延主导的 kernel 里这是主要收益。
- 加载发起点从 vec 侧挪到 cube 侧后，数据到达不再需要 vec→cube 的跨核通知（load 与 matmul 在同一阶段内，由 tile mutex 保证顺序）。

## 怎么改（before / after）

以下为**嵌入片段**：所用 `pl.*` API 均在目标 Ascend 950PR 工具链的已验证 kernel 代码中核验存在该形态；片段是教学级最小示意，嵌入 kernel 需自行管理组地址、mutex 与跨核事件。例题设定：`BT=K_DIM=128`，`w_in` 为上游 kernel 恒写满 128 行的 GM tensor，`l1_w` 为 `Mat/NZ` 操作数 tile。

**before（专用 vec 段三步搬运 + 跨核通知）：**

```python
# 每个 chunk：先 vec 段搬 w，再 cube 段消费
with pl.section_vector():
    t_w = grp_w16.current(); t_w_nz = grp_sbf16nz.current()
    l1_w = l1_w_group.current()
    pl.load(t_w, w_in, [b_idx, head_id, gm_chunk, ro, 0], order=[3, 4])  # GM → UB
    pl.move(t_w_nz, t_w)                                                 # UB 内 ND→NZ 转换
    pl.insert(l1_w, t_w_nz, [ro, 0])                                     # UB → L1
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=E_W)        # 通知 cube 侧
with pl.section_cube():
    _stage_c3b(l1_w_group, ...)   # 消费 l1_w：move 到 L0A/L0B → matmul
```

**after（GM→L1 直载并入消费它的 cube 阶段，专用 vec 段删除）：**

```python
@pl.pipeline.stage
def _stage_c3b(w_in, l1_w_group, l1_sbf_group, grp_l0a, grp_l0b, grp_acc, ...):
    l1_w = l1_w_group.current()
    pl.set_validshape(l1_w, [BT, K_DIM])
    pl.load(l1_w, w_in, [b_idx, head_id, gm_chunk, 0, 0], order=[3, 4])  # GM → L1 直达
    l0a = grp_l0a.current(); l0b = grp_l0b.current(); acc = grp_acc.current()
    pl.move(l0a, l1_w)          # L1 → L0A
    pl.move(l0b, l1_sbf)        # L1 → L0B
    pl.matmul(acc, l0a, l0b)
```

注意：删除专用 vec 段后，原来为这次搬运服务的跨核事件对（如 `E_W` 的 set/wait）应一并清理，避免留下没有消费者的同步边。

## 性能与验证指标

**预期变化指标**：

- kernel 端到端窗口时长（Task Duration）：预期下降，降幅来自「少一个阶段的同步往返 + 少一跳搬运」。
- 生成物/指令侧：专用 vec 搬运段的 load/move/insert 消失，对应跨核事件对减少；GM→L1 的 MTE2 搬运仍然存在（搬运被合并而非取消）。
- 正确性：全量精度回归必须通过，**尾块用例（非整块对齐的序列长度）必须覆盖**——越界读通常在尾块才触发。

**验证方法**：

1. 先确认当前瓶颈是串行链同步（阶段屏障/事件往返）而非计算或带宽——若搬运延迟已被 tile mutex 的异步机制吸收（例如实测双缓冲预取无收益），本方法收益有限；
2. 修改后跑全量正确性（对照 golden，重点尾块），再按同一采集协议对比改动前后性能；
3. 核对生成物：专用搬运段与事件对确实消失。

**已有实验证据**（单算子一次实验的材料，未在其它算子/SoC 复现，不能作为当前算子的收益承诺）：一例 kernel 把 w 的 GM→UB→NZ→L1 搬运并入消费它的 cube 阶段（串行链 5 段→4 段），窗口 697→634µs（约 −9%）；同算子另一 kernel 把两路输入（S_i、vnew）同样直载，prologue 搬运从 4 路减为 2 路，窗口 696→558µs（约 −20%）；同一算子中第三路输入 k 因尾块行未写满、直载越界读出 NaN 而回退 UB 路径。

## 技术限制与风险

- **尾块越界（主要风险）**：不设有效行的整行加载会越过 tensor 分配边界读未定义内存——表现为 NaN 或偶发错误，通常只在尾块用例触发；只有上游恒写满全部行（含补零）的 tensor 才可整行直载。直载前逐个确认每个输入的写入完整性。
- **权衡不要绝对化**：vector 空闲且搬运可提前发射（与 cube 计算重叠）时，UB 中转做 NZ 转换反而能隐藏延迟；严格串行结构下直载才稳定获益（见"何时不适用"）。
- **同步清理**：删除专用段后遗留的无消费者事件、以及合并阶段后的事件配对关系，必须同步核对（事件收支不平衡可能死锁）。
- **容量**：直载使 L1 tile 的写入时机后移（进入 cube 阶段），若该 tile 同时承担其它时序角色，需重新核对地址与 mutex 复用关系。

## 参考资料

- PyPTO 仓库：load API（GM→L1 直达、order 轴映射、validshape 尾块语义）：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`
- PyPTO 仓库：move API（ND/NZ 布局转换、跨内存层级搬运）：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/move.md`
- PyPTO 仓库：matmul API（L1→L0A/L0B→L0C 数据通路）：`docs/zh/api/pro_api/SIMD-API/cube_computation/matmul.md`
- PyPTO 仓库：性能优化通用流程：`docs/zh/guide/programming_guide/pro/debug/performance_optimization.md`
