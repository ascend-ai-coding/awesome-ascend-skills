---
type: "PyPTO Performance Optimization Card"
title: "运行时按占核数选择输出 N-tile 宽度（N256 / N384 同 kernel 分档）"
description: "MTE2 字节受限的 matmul 前端，按运行时的 M-tile 数把每个工作项的输出宽度在 2 个与 3 个 N-tile 之间切换，用 A 复用换更少的 A 重读；份数不足占满网格时保持窄档。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "mte2", "a-reuse", "runtime-dispatch"]
item_id: "cube-12"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#from-capacity-planning-to-performance-candidates"
      scope: "联合比较 N 宽度、A 复用、L0/Acc 占用与逻辑工作项数，避免只减少读字节却损失并行。"
      boundary: "N256/N384 与阈值属于本卡具体实现；目标不提供跨形状通用分界，也不保证运行期分支共享编译期地址。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:performance-configuration"
      scope: "验证按运行时 M tile 数和实际 worker 数选择的配置，并在额外形状检查 selector。"
      boundary: "配置必须按真实 tile/launch 组合比较；单个宽档获胜不证明整个运行期分派正确或更快。"
bound_hint: "MTE2"
applicability: "matmul 为 MTE2 字节受限、A（激活）按 N-tile 数被反复重读；同一 kernel 要覆盖一批不同 batch（M 维 pad 后 tile 数变化）的形状；占核数（block_num）运行时可查。"
target_api_gate: "仅限 Ascend 950PR 或 950DT；用 pl.get_block_num/pl.get_block_idx 的扁平二维网格与 pl.range 运行时边界即可，无需 const_expr 编译分档；分支内各自使用独立的 tile group，两条分支的 L1/L0 地址需按编译期不重叠规划。"
---
# 技术卡片 cube-12：运行时按占核数选择输出 N-tile 宽度（N256 / N384 同 kernel 分档）

- **适用 bound**：MTE2（左矩阵分块重复装载主导，且 A 的重读次数 = 每 M 行的 N-tile 任务数）
- **一句话**：工作项覆盖的输出列数（2×128 或 3×128）按 M-tile 数与占核数运行时分档。

## 何时用（诊断特征）

- 投影/GEMM 的 A 侧（激活，`[Mpad, K]`）在 N 循环里被整块反复重读；A 的 GM 读字节 ≈ `(M-tile 数) × (每行输出 N-tile 数) × A 单行字节`。
- MTE2 或总字节受限（profiler 上 `aic_mte2` 占比高、Cube 等 MTE2）。
- 一次编译要覆盖随 batch 变化的 `Mpad`（`Mdpad/128` 的 tile 数变化），批大小跨越"网格能被填满 / 填不满"的两侧。

## 何时不适用

- **工作项数不足以占满网格时更宽反而更慢**：本卡实测在 `M-tile 数 = 1`（M 只有 128 行）时，N384 比 N256 慢 **+2.1%（(B=4,N=24)）到 +24%（(B=16,N=6)）**——工作项变少、占核不足，更长的链无法被隐藏。（(B=16,N=6) 时 N384 只有 6 项、24 核里 6 核忙。）
- 输出宽度不是 128 的整数倍、或 N-tile 数不能被 3 整除时，宽档需要尾块处理（本卡场景 `nt_q = 3N` 恒为 3 的倍数）。
- 不是字节受限（latency/占核受限）时，加宽只拉长单项延迟，无收益。

## 原理

每个工作项做一次 K 循环：每轮装 1 个左矩阵分块 + 每个输出 N-tile 一个右矩阵分块，做对应次数的 `matmul_acc`。

- **N256（窄档，2 个输出 N-tile）**：左矩阵分块复用 2 次 → A 的 GM 读次数 = 工作项数 = `m_tiles × (nt/2)`。
- **N384（宽档，3 个输出 N-tile）**：A 复用 3 次 → A 的 GM 读次数 = `m_tiles × (nt/3)`，**A 读字节降到 2/3**；B 与 MMAD 总量不变。

A 读用 R=3 代替 R=2，省下 `m_tiles × (nt/6) × A单行字节`。这份节省随 `m_tiles` 线性增长，所以**大 batch 才有意义**；而工作项总数降到 `m_tiles × (nt/3)`，小 `m_tiles` 时会占不满核，反而更慢。

判据（本卡实测择定）：`m_tiles >= 2` **且** `m_tiles × (nt/3) >= block_num` 才走宽档。

## 与相关卡片的区别

- **cube-04（FullLoad 小侧常驻 L1）/ cube-10（N 外 K 内重构）**：二者靠"把小侧操作数常驻 L1"或"重排 N/K 循环"削减重复读，分别面向 MTE2-bound / cube-bound。本卡**不改循环结构、不做常驻**，只改变**每个工作项覆盖的输出列数**（R=2 ↔ R=3）并**按占核数运行时分档**；目标同为少读 A，但机制与判据不同，可与二者叠加（先按本卡定宽度，再按 cube-04/10 处理循环）。
- **cube-11 机制 C / pipe-06**：二者处理"tile 数不整除核数导致末轮空闲核"与"只占用部分核需扁平铺满"——是**把工作铺满核**。本卡的判据 `m_tiles × (nt/3) >= block_num` 方向**相反**：占核不足时**退回窄档**（见「何时不适用」中 +24% 的负例），而不是去铺满。
- **cube-13**：本卡决定"要不要用宽档"，[cube-13](cube-13-l0b-ks-split-multi-panel-resident.md) 是宽档（≥3 个右矩阵分块）在 64KB L0B 上的必要实现，两者配套使用。

## 怎么改（before / after）

结构示意（截取自已上板验证的融合 kernel，省略其余阶段；两条分支运行期互斥，故用各自独立的 tile group，编译期地址不重叠）。

```python
with pl.section_cube():
    nc_c = pl.get_block_num()
    cid_c = pl.get_block_idx()
    n3_q = nt_q // 3
    # 运行时分档：M-tile 数足够、且 N384 工作项能占满网格时用宽档
    if m_tiles >= 2 and m_tiles * n3_q >= nc_c:
        for t in pl.range(cid_c, m_tiles * n3_q, nc_c):      # N384：3 个输出 N-tile
            i = t // n3_q
            j0 = (t % n3_q) * 3
            ac0, ac1, ac2 = c_acc0.current(), c_acc1.current(), c_acc2.current()
            for k in pl.range(0, k_hidden, 1):
                # 1 个左矩阵分块 + 3 个右矩阵分块；实现细节见 cube-13（L0B 同驻）
                ...
            pl.store(qkv, ac0, [i * _TILE, j0 * _TILE])
            pl.store(qkv, ac1, [i * _TILE, (j0 + 1) * _TILE])
            pl.store(qkv, ac2, [i * _TILE, (j0 + 2) * _TILE])
    else:
        n2_q = (nt_q + 1) // 2
        for t in pl.range(cid_c, m_tiles * n2_q, nc_c):      # N256：2 个输出 N-tile
            i = t // n2_q
            j0 = (t % n2_q) * 2
            ac0, ac1 = c_acc0.current(), c_acc1.current()
            for k in pl.range(0, k_hidden, 1):
                ...
            pl.store(qkv, ac0, [i * _TILE, j0 * _TILE])
            pl.store(qkv, ac1, [i * _TILE, (j0 + 1) * _TILE])
```

N384 档多用一个 L0C 累加器（3 个 `[128,128] fp32` = 192 KB ≤ L0C 256 KB）。

## 性能与验证指标

观察 `aic_mte2` 占比、Cube 等 MTE2 的时间占比与 Task Duration；宽档应使 A 的 GM 读字节降到窄档的 2/3，MTE2 字节下降、总时间随大 batch 下降，而 B 与 MMAD 总量不变。`m_tiles=1`（占核不足）时应观察到宽档反而更慢、分档退回窄档。正确性按常规精度回归，运行期两条分支与不同占核数（跨配额）都要单独覆盖，确认输出在容差内与窄档一致。

## 技术限制与风险

- 阈值由实测择定；换硬件/换算子须重测分界。`m_tiles=1` 侧的宽档负收益可高达 +24%（小 N、占核不足）。
- 宽档需要多一个 L0C 累加器；L0B 侧的安置见 cube-13，**不能只靠复用 L0B Buffer**，否则第 3 个 matmul 与前一个串行，收益归零。
- 运行时分支的两条路径都要单独覆盖精度与竞态测试。

## 参考资料

- 历史来源：`RESULTS.md` §3「前端 B 分档」及 A/B 表，原始修订待定位；当前 [实现](https://gitcode.com/cann/pypto-gym/blob/ccd64a44f843d077feab9a81d894a09bda3c5af2/ops/pro/kimi_k3/mega_recurrent_kda/op_kernel/mega_recurrent_kda.py) 仅供核对 N256/N384 分支。
- 相关卡片：[cube-07](cube-07-weight-nz-prepack.md)（权重 NZ 预打包，同为 MTE2 优化）、[cube-13](cube-13-l0b-ks-split-multi-panel-resident.md)（宽档的 L0B 同驻实现）、[pipe-06](../pipe/pipe-06-flat-2d-work-grid.md)（扁平工作网格）。
