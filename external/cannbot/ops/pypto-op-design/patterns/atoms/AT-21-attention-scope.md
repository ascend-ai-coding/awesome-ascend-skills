---
type: pattern/atom
title: Attention 分阶段子图合图（sg_set_scope）
description: 对「QK^T（Cube）→ softmax（Vec）→ PV（Cube）→ 状态更新（Vec）」的 IFA / Flash Attention 结构，在 Cube 与 Vec 交替的边界上用 `pypto.set_pass_options(sg_set_scope=...)` 手动划定子图边界，把同类的连续 Vec 链合入同一子图，减少跨子图的 GM 落地与调度气泡。
tags:
- subgraph-boundary
flow_pattern:
- C
- V
examples:
- IFA
- PageAttn
- FA 类（QK^T→softmax→PV 结构）
---

## AT-21: Attention 分阶段子图合图（sg_set_scope）

**描述**: 对「QK^T（Cube）→ softmax（Vec）→ PV（Cube）→ 状态更新（Vec）」的 IFA / Flash Attention 结构，在 Cube 与 Vec 交替的边界上用 `pypto.set_pass_options(sg_set_scope=...)` 手动划定子图边界，把同类的连续 Vec 链合入同一子图，减少跨子图的 GM 落地与调度气泡。

**CV 排布**: C/V 交替边界的**子图边界控制**（不改变计算流本身，仅声明合图范围）。

**输入**:
- 一个含 Cube/Vec 交替阶段的 attention kernel 计算图（如 SK-01 的 C1-V1-C2 循环体）

**输出**:
- 阶段化的子图边界配置（`sg_set_scope` 数值范本）

**计算流**（910 平台 IFA 实测范本）:
```python
for s2_idx in pypto.loop(s2_loop, ...):
    # (A) QK^T：Cube，scope 外（默认 -1）
    sij_full = pypto.matmul(qi, kj, ...)

    # (B) softmax vec 链：sg_set_scope=2（mul/amax/sub/exp/sum/cast 合为一子图）
    pypto.set_pass_options(sg_set_scope=2)
    sij = pypto.mul(sij_full, scale)
    tilda_mij = pypto.amax(sij, dim=-1, keepdim=True)
    tilda_pij = pypto.exp(pypto.sub(sij, tilda_mij))
    tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    pij_bf16 = pypto.cast(tilda_pij, pypto.DT_BF16)
    pypto.set_pass_options(sg_set_scope=-1)   # 切回默认

    # (C) pij @ vj：Cube，scope 外
    oi_local = pypto.matmul(pij_bf16, vj, ...)

    # (D) 状态更新（online softmax 合并）：sg_set_scope=1（maximum/exp/mul/add 合为一子图）
    pypto.set_pass_options(sg_set_scope=1)
    ...maximum / exp / mul / add ...
    pypto.set_pass_options(sg_set_scope=-1)   # 切回默认

    # (E) 末块归一化：默认 scope
    if pypto.is_loop_end(s2_idx):
        oi_final = pypto.div(o_upd, l_upd, precision_type=pypto.PrecisionType.INTRINSIC)
```

**约束（必须遵守）**:
- Cube 与 Vec 操作**不得在同一 scope**（同一 scope 混置报 `F41007 OP_SCOPE_ERROR`）
- `sg_set_scope=-1` 用于**结束当前 scope 并切回默认**；不同阶段用不同正整数 ID
- scope 仅对**有直接数据依赖的 Vec 操作链**生效；不跨 `pypto.loop` 边界
- 用于已有正确实现的子图划分实验；调整后复核精度并测量性能。

**实测收益**: 910 平台 IFA decode（b16/s8192）执行时间 287→113 us（与 ready_on_host_tensors 等配置合计）



---
