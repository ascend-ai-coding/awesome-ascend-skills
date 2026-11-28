---
type: "PyPTO Performance Optimization Card"
title: "热循环查表与守卫的 load-store 减载"
description: "SCALARLDST 饱和时，把热循环内链式查表（getval(meta)→getval(lut, idx)）拆为准常量循环外捕获加常量选择，把 lazy-init 守卫移出热循环一次完成。"
status: "stable"
tags: ["pypto-pro", "scalar", "load-store", "lut"]
item_id: "scalar-02"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-outer-reduce.md#metadata-and-invariant-inputs"
      scope: "区分可提前捕获的小型不变查表与必须逐块读取的运行时索引，减少重复 getval 链。"
      boundary: "UB 批量广播是另一备选，不能把它等同于标量分支选表；state 初始化副作用、动态 offset 支持及表规模门槛仍需核验。"
bound_hint: "SCALARLDST、SCALAR"
applicability: "热循环内存在 getval 结果再作为下一个 getval 偏移的链式查表，或每轮读取 state 类守卫标量；表项 launch 后不变且表很小，守卫条件首轮后恒成立"
target_api_gate: "仅限 Ascend 950PR 或 950DT；须核验 pl.getval/pl.setval 动态偏移与运行期标量分支"
---

# 技术卡片 scalar-02：热循环查表与守卫的 load-store 减载

- **适用 bound**：SCALARLDST / Scalar
- **一句话**：热循环里每次"读索引再查表"是两次带地址依赖的标量 load，每次"读守卫再判断"是一次 load 加比较加分支；把不变的表项提到循环外读一次、把初始化守卫提到循环外做一次，热循环只保留真正逐块变化的单次读取。

## 何时用（诊断特征）

- profiler 显示 SCALARLDST 占比高，热点在工作循环内的标量访存。
- `pl.getval` 从 GM 张量读标量：`meta`、`lut`、`state` 均为 kernel 入参 GM
  张量，每次调用是一次走 SCALARLDST 通路的 GM 标量 load，承受 GM 访存延迟
  而非片上读取；热循环内逐块调用会直接堆高该泳道。
- 源码中存在链式查表：`idx = pl.getval(meta, t)` 后紧跟 `pl.getval(lut, idx)`，
  第二次 load 的地址依赖第一次的结果（串行延迟叠加）。
- 循环内每轮执行 `flag = pl.getval(state, 0)` 加 `if flag == 0` 的惰性初始化
  守卫，而该守卫只在首轮成立。

## 何时不适用

- 表项数较多（如大于 8 项）：常量选择的分支链自身成为 Scalar 分支开销，
  收益可能反转，需按目标 profiling 权衡。
- `lut` 或 `meta` 的内容在循环内可能被 `pl.setval` 改写（不再是准常量或
  独立索引）。
- 表项值要到运行期才能确定（无法在循环前捕获），或索引域开放无法用有限
  分支覆盖。
- 循环块数很少时，外提与分支链的改写不划算。
- SCALAR（ALU/分支）占比明显高于 SCALARLDST 时，常量选择的分支链会加重
  标量 ALU 负担，本卡可能负收益；先走 [scalar-01](scalar-01-scalar-bound-checklist.md)
  的外提与预计算路线并复测 profiling。

## 原理

- 每次 `pl.getval` 是一次 SCALARLDST 标量 load；链式查表是"load 地址依赖
  load"的串行链，延迟逐级累加。
- 表项 launch 后不变（准常量）且表很小时，循环外一次性捕获全部表项为
  Python 标量，循环内用运行期标量分支选择，消除第二跳 load。
- 守卫类标量（如一次性初始化标志）改为循环外无条件 `pl.setval` 一次完成，
  热循环内不再有"读-比较-分支"三连。
- 真正逐块变化的索引读取（`pl.getval(meta, t)`）保留在循环内——减载的是
  不变量与依赖链，不是动态数据本身。

## 怎么改（before / after）

以下为嵌入片段（`pypto_pro` 0.2.0、FP32、tile `[8, 128]`、Vec memory、
单 block 启动、`@pl.jit(auto_mutex=True)` 上下文中核验；`meta` 为
逐块索引、`lut` 为 4 项缩放表、`state` 为初始化标志，TR=8、NN=128）。

### before：链式查表 + 守卫在热循环内

```python
with pl.section_vector():
    nt = (m + TR - 1) // TR
    xi = ing.next()
    yi = og.next()
    for t in pl.range(0, nt, 1):
        idx = pl.getval(meta, t)          # 链式查表第 1 跳
        scale = pl.getval(lut, idx)       # 依赖 load 第 2 跳
        flag = pl.getval(state, 0)        # lazy-init 守卫每轮 load
        if flag == 0:
            pl.setval(state, 0, 1)        # 惰性初始化在热循环内
        ro = t * TR
        vr = pl.min(TR, m - ro)
        pl.set_validshape(xi, [vr, NN])
        pl.load(xi, x, [ro, 0])
        pl.set_validshape(yi, [vr, NN])
        pl.mul(yi, xi, scale)
        pl.store(y, yi, [ro, 0])
```

### after：准常量捕获 + 守卫清理 + 主尾分离

```python
with pl.section_vector():
    nt_full = m // TR
    # 准常量捕获：表项 launch 后不变，循环外一次读出
    s0 = pl.getval(lut, 0)
    s1 = pl.getval(lut, 1)
    s2 = pl.getval(lut, 2)
    s3 = pl.getval(lut, 3)
    # 守卫清理：初始化移出热循环，一次完成
    pl.setval(state, 0, 1)
    xi = ing.next()
    yi = og.next()
    for t in pl.range(0, nt_full, 1):
        idx = pl.getval(meta, t)          # 真动态：仍需逐块读
        # 小表改常量选择，消除第 2 跳依赖 load
        if idx == 0:
            scale = s0
        elif idx == 1:
            scale = s1
        elif idx == 2:
            scale = s2
        else:
            scale = s3
        pl.set_validshape(xi, [TR, NN])
        pl.load(xi, x, [t * TR, 0])
        pl.set_validshape(yi, [TR, NN])
        pl.mul(yi, xi, scale)
        pl.store(y, yi, [t * TR, 0])
    rem = m - nt_full * TR                # Epilogue：尾块单独处理
    if rem > 0:
        idx = pl.getval(meta, nt_full)
        if idx == 0:
            scale = s0
        elif idx == 1:
            scale = s1
        elif idx == 2:
            scale = s2
        else:
            scale = s3
        ro = nt_full * TR
        pl.set_validshape(xi, [rem, NN])
        pl.load(xi, x, [ro, 0])
        pl.set_validshape(yi, [rem, NN])
        pl.mul(yi, xi, scale)
        pl.store(y, yi, [ro, 0])
```

`meta`、`lut`、`state` 为 GM 张量，经 `pl.getval`/`pl.setval` 直接读写，
不占用 tile group；`ing`/`og` 为主数据流的常规 tile group，声明不受本卡
改写影响。

## 性能与验证指标

- 正确性：改写前后输出逐元素一致，且与 golden（逐块 `lut[meta[t]] * x`）
  一致；守卫标志改写前后均按预期置位；整除、带尾块、仅尾块 shape 均须覆盖，
  索引须覆盖全表域。
- 性能观察项：SCALARLDST 泳道占比、`aiv_scalar_ratio`、kernel 总时长。
  每块净减的 load 次数可从源码静态推出；收益随块数与原查表密度增长，
  未实测前不作承诺。

## 技术限制与风险

- 常量选择分支链在表项多或索引分布集中时退化为多次比较；必要时回到
  链式查表或与 profiling 对照选择。
- `pl.setval(state, 0, 1)` 对 GM 张量有跨 launch 可见副作用：依赖 state
  初值的调用方约定须同步调整（如 host 侧重置）。
- `getval` 的动态偏移、运行期标量分支（`if idx == 0` 等）与分支内变量
  赋值依赖当前 parser 支持；升级版本后需重新核验。
- 主循环与尾块的常量选择分支链为重复展开；表项增多时该冗余与分支开销
  一起放大，须回到不适用条件重新评估。

## 参考资料

- 语法：`pypto_pro.language` 的 `getval` / `setval`（标量读写，支持动态
  偏移）、`min` / `max`（标量最值）、`range`（循环迭代）、`set_validshape`
  （部分块有效形状）
