---
type: "PyPTO Performance Optimization Card"
title: "整块分支重设动态 valid_shape 为静态 shape 常量"
description: "在可静态证明 valid 恒等于 shape 的分支，用 pypto.view 把携带数据依赖 valid_shape 的输入重建为全常量视图，解除 pass 对动态 valid_shape 的优化约束"
status: "stable"
tags: ["pypto", "shape", "valid-shape", "pass"]
item_id: "shape-01"
bound_hint: "mixed"
applicability: "动态边界切出的视图 shape 为常量而 valid_shape 是数据依赖标量，流入大量使用 concat/cat 的公共计算体，且存在可静态证明 valid==shape 的分支（如非 loop-end 整块分支）"
target_api_gate: "pypto（非 pypto_pro）前端 pypto.view 的 shape/offsets/valid_shape 参数；设备支持以算子 README 为准；具体受限 pass 未指明，采用前须在当前工具链复现退化并复核"
---

# 技术卡片 shape-01：整块分支重设动态 valid_shape 为静态 shape 常量

- **适用 bound**：前端 shape 元数据 / pass 优化机会（间接作用于 compute 与访存）
- **一句话**：非末次迭代的 chunk 输入恒为整块（actual_l == _BT），却仍带着数据依赖的
  valid_shape 进入公共计算体；整块分支先用 `pypto.view(x, x.shape, [0, 0],
  valid_shape=x.shape)` 重建为全常量视图，让 pass 恢复优化。

## 何时用（诊断特征）

- 视图 `shape` 是常量，`valid_shape` 含 SymbolicScalar（如由 device `cu` 张量得到的
  `L` 推导 `actual_l = (L - s_idx).min(_BT)`），编译期不可定值。
- 视图流入 if/else 共享的计算体，且其中大量 `pypto.concat`/cat。
- 循环结构上存在整块分支（非末次迭代 `actual_l == _BT` 恒成立），当前仍把动态
  valid_shape 传下去。生成物层面的具体退化为待验证诊断项。

## 何时不适用

- 尾块或任何 `valid < shape` 的路径：会把 padding 当有效数据，直接算错。
- valid_shape 需向下游传播以限制写出的场景（如 `assemble` 只写 `actual_l` 行）。
- 无法静态证明 `valid == shape`，或 pass 已能处理动态 valid_shape 时，重设冗余或禁止。

## 原理

**事实**：来源算子 `chunk_kda_varlen_kernel` 用 `pypto.loop(0, L, _BT)` 步进，每轮
reshape 出的 `q2/k2/v2/g2/b2` 带动态 `valid_shape=[actual_l, ...]`；尾块分支
`is_loop_end` 判定并 `fillpad` 清零。循环结构保证非末次迭代 `L - s_idx >= _BT`，
故 `actual_l == _BT`——整块分支重设不改变数据语义。`pypto.view` 是逻辑视图，
不新增数据搬运。

**机制（作者陈述，未独立复核）**：提交注释原话 "`cat` is heavily used in
`_chunk_compute`, pass has some optimize constraints when valid shape is dynamic;
set it same as shape(constants) make pass happy"。所涉 pass 与被恢复的优化未指明。

## 怎么改（before / after）

**嵌入片段（节选）**：`pypto.view` 参数已对照 pypto 仓库 `python/pypto/operation.py`
核验；取自 pypto-gym 提交 `30fcbeb` 的 `chunk_kda_impl.py`。上下文：pypto
`@pypto.frontend.jit` varlen kernel，bf16 输入内部 fp32，K==V==128、_BT==128；
省略号为本卡无关的计算体。

```python
# before：else（非末次迭代）分支直接把动态 valid 视图传入 _chunk_compute
if pypto.is_loop_end(s_idx):
    ...  # 尾块：fillpad 清零 padding，保留动态 valid
else:
    # after：整块分支重设为 valid==shape 的全常量视图（actual_l==_BT 恒成立）
    q2 = pypto.view(q2, q2.shape, [0, 0], valid_shape=q2.shape)
    k2 = pypto.view(k2, k2.shape, [0, 0], valid_shape=k2.shape)
    v2 = pypto.view(v2, v2.shape, [0, 0], valid_shape=v2.shape)
    g2 = pypto.view(g2, g2.shape, [0, 0], valid_shape=g2.shape)
    b2 = pypto.view(b2, b2.shape, [0, 0], valid_shape=b2.shape)
    oc, s_new = _chunk_compute(q2, k2, v2, g2, b2, s_carry, ...)
```

复用前提：`x.shape` 此时须为常量列表；`offsets=[0, 0]` 相对自身视图取整块。

## 性能与验证指标

- 来源提交只有优化意图，无前后性能数据——**不得引用收益结论**。
- 补证：对整块占主导 case（`cu_seqlens=None`、`T % 128 == 0`）做前后 msprof 对比；
  对比 build 产物定位所涉 pass 后回填本卡。
- 正确性回归：`tests/ops/ling_3_0_flash/chunk_kda/test_chunk_kda.py`（容差 o 1e-2、
  S 1e-3）；提交后在 NPU 的全量运行记录未见，采用前应执行。

## 技术限制与风险

- 语义前提须逐分支证明（循环步进 + `is_loop_end` 划分），不是运行时巧合；尾块误用属
  正确性错误。
- 机制未定位：当前版本 pass 若已处理动态 valid_shape，收益可能消失；重设也可能改变
  两分支的 kernel 特化数量，需生成物对比确认。
- 每次 view 是轻量逻辑节点，但输入多时前端 IR 节点随之增加。

## 参考资料

- 合入提交：pypto-gym `30fcbeb`（`chunk_kda_impl.py`，含作者注释与 before/after 现场）。
- API：pypto 仓库 `python/pypto/operation.py` 的 `view()`。
- 算子背景：`chunk_kda/README.md`（varlen 语义、设备支持与精度容差）。
