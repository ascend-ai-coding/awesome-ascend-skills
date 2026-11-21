---
type: "PyPTO Performance Optimization Card"
title: "BF16 行处理改用 UNPK_B16/PACK 单行寄存器，替代 even/odd 双子行 interleave 洗牌"
description: "对 bf16（fp16）行数据逐行做逐元素 VF 计算时，把『一次搬入整个寄存器（128 个 bf16 = 两行）+ astype(CastLayout.ZERO/ONE) + interleave 展开、再 astype + de_interleave 收窄』改成『LoadDist.UNPK_B16 单行展开搬入 + StoreDist.PACK 单行压缩写回』，消除跨 lane 洗牌并使每条搬入/搬出正好一行。"
status: "stable"
tags: ["pypto-pro", "vec", "bf16", "widen", "pack", "shuffle"]
item_id: "vec-19"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/vf.md#distinguish-numerical-conversion-from-port-bit-packing-first"
      scope: "以 B16/32-bit 容器、有效 lane、数值 Cast 和 PACK 位提取分别描述单行 UNPK_B16/PACK 通路。"
      boundary: "UNPK/PACK 本身不等于浮点数值转换；BF16 的舍入、行宽及逐行独立性不能从 FP16 示例继承。"
bound_hint: "VEC"
applicability: "VF 逐行处理 bf16/fp16 数据且一行恰好填满一个 b16 寄存器（如 64 元素），各行相互独立、不跨行归约；当前实现用『加载整寄存器（两行）+ astype(CastLayout.ZERO/ONE) + interleave』展开、用『astype(低精度) + de_interleave』收窄"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖已核验的 vf.load_align(dist=pl.LoadDist.UNPK_B16)、vf.astype(→DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)、vf.astype(→DT_BF16, round_mode=pl.VFRoundMode.CAST_ROUND) 与 vf.store_align(dist=pl.StoreDist.PACK)；仓库内 engram_gate / engram_gate_bwd 已有同用法"
---

# 技术卡片 vec-19：BF16 行处理改用 UNPK_B16/PACK 单行寄存器

- **适用 bound**：VEC
- **一句话**：逐行处理 bf16 时，搬入用 `LoadDist.UNPK_B16`（搬入即展开成 fp32）、搬出用 `StoreDist.PACK`（搬出即压缩回 bf16），让每条 load/store 恰好一行，去掉「整寄存器两行 + 偶/奇 `astype` + `interleave`/`de_interleave`」的跨 lane 洗牌。

## 何时用（诊断特征）

- 目标 VF 处理的是 bf16（或 fp16）行数据，且**每行宽度恰好等于一个 b16 寄存器（VL16=64 个 bf16）**，典型为 `[N,64]` 的 D-half 行或 `[1,64]` 片段。
- 从源码或生成物看，该 VF 呈「**一次 `load_align` 拉满整寄存器（128 个 bf16 = 两行）→ `astype` 取 `CastLayout.ZERO`/`CastLayout.ONE` 得偶/奇两半 → `interleave` 还原自然序**」；输出侧对称地「`astype` 到 bf16 → `de_interleave` 拼回 128 个 bf16 再 `store_align`」。
- 各行为**独立的逐元素/逐行计算**（比较、`exp`、乘加、逐行缩放），**没有跨行依赖或跨行归约**。
- `aiv_vec` 是热点，且上述洗牌/偶奇 `astype` 在指令构成里占可观比例（按「洗牌 op 数 × 行对数」初判）。

## 何时不适用

- 计算本身要**跨行归约/扫描**（列内前缀和、按行求全局统计等）——单行拆开后仍需跨行合并，收益与正确性另论。
- 行宽**不是一个 b16 寄存器**（如 32 或 128 列）：`UNPK_B16`/`PACK` 粒度按 B16（64 元素）定义，需按实际宽度分段或另选分布模式并证明 mask 覆盖。
- 数据是 **fp32 输入输出**：本卡针对 bf16/fp16 的展开/收窄，fp32 无 widen/pack 需求。
- 现有 even/odd 写法与下游**物理布局契约**强绑定（packed/蝶式布局，`INTLV`/`DINTLV` 是刻意为之）：改单行会改变布局，须先核对消费方，或继续用 `StoreDist.INTLV_*` 显式表达。

## 原理

- `LoadDist.UNPK_B16` 在一次 `load_align` 内把 B16 元素展开到 32-bit 槽位（搬入即 widen），因此**一次搬入正好得到一行 64 个 fp32**；`StoreDist.PACK` 反向把 32-bit 低半压缩回 B16（搬出即收窄）。二者合起来替代了「整寄存器两行 + 偶/奇 `astype` + `interleave`/`de_interleave`」的**跨 lane 洗牌链**。
- 收益来源：① 去掉每两行各一次的 `interleave` 与 `de_interleave`（纯 shuffle）；② 去掉为凑满整寄存器而必须的偶/奇两次 `astype`；③ 逐行处理使每条搬入/搬出**恰好一行**，与行级 mask/offset 天然对齐。
- 数值上：bf16↔fp32 的展开是**精确**的（bf16 即 fp32 高 16 位），收窄沿用 `CAST_ROUND`；保持与原 even/odd 路径**同操作数、同顺序、同舍入**，可做到位级等价。

## 怎么改（before / after）

**嵌入片段**（已在 Ascend 950PR 交付算子的 VF 中核验；`x` bf16 `[N,64]`、`y` bf16 `[N,64]`、`cs` fp32 `[N,64]`，`preg` 为 B32 全掩码、`pregbf` 为 B16 全掩码）：

before（整寄存器两行 + 洗牌）：

```python
for m2 in pl.range(0, ROWS // 2):
    base = (r0 + 2 * m2) * 64
    s16 = vf.load_align(x, base, dtype=pl.DT_BF16)          # 一次 128 bf16 = 两行
    se = vf.astype(s16, pregbf, dtype=pl.DT_FP32)
    so = vf.astype(s16, pregbf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    lo, hi = vf.interleave(se, so)                          # 跨 lane 洗牌
    c_lo = vf.load_align(cs, base)
    c_hi = vf.load_align(cs, base + 64)
    t_lo = vf.mul(lo, vf.exp(c_lo, preg), preg)
    t_hi = vf.mul(hi, vf.exp(c_hi, preg), preg)
    l16 = vf.astype(t_lo, pregbf, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_ROUND)
    h16 = vf.astype(t_hi, pregbf, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_ROUND)
    packed, _ = vf.de_interleave(l16, h16)                  # 跨 lane 洗牌
    vf.store_align(y + 2 * m2 * 64, packed, pregbf)
```

after（单行 + `UNPK_B16` / `PACK`）：

```python
for row in pl.range(0, ROWS):
    off = row * 64
    xr = vf.load_align(x, off, dist=pl.LoadDist.UNPK_B16)   # 搬入即展开成一行 fp32
    xf = vf.astype(xr, preg, dtype=pl.DT_FP32,
                   round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
    cf = vf.load_align(cs, off)
    tf = vf.mul(xf, vf.exp(cf, preg), preg)
    tb = vf.astype(tf, pregbf, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_ROUND)
    vf.store_align(y + off, tb, pregbf, dist=pl.StoreDist.PACK)  # 搬出即收窄
```

注：`LoadDist.UNPK_B16` 后仍接一次 `astype(→FP32)`（与仓库既有用法一致）；若某处 `UNPK` 已直接给出可用 fp32，可省这一步。仅 fp32 的逐元素 VF（无 bf16 展开/收窄）不适用本卡。

## 性能与验证指标

- 预期指标：`aiv_vec_time` / VEC ratio 下降（洗牌与偶/奇 `astype` 被删除）；`aiv_mte2`/`aiv_mte3` 可能因搬运更细（每行一次）略升——净收益以 kernel 总时长与 `aiv_vec_time` 同时判定。
- 正确性覆盖：本改动是**布局/原语等价改写**，须对代表性 layout（含非对齐尾块）与改动前做**逐位对比**（`torch.equal`，`out` 与 `state` 的 `n_diff=0`），而非仅看容差门；尾块行由既有 mask/tail 逻辑处理，须单独覆盖。
- 机制证据：对照生成物中 `vlds`/`vsts`（UNPK/PACK 形态）替代原 `interleave`/`de_interleave` 序列；以 `aiv_vec` 绝对时间变化佐证。
- 逐元素算子收益随「洗牌 op / 总 op」占比放大；寄存器压力变化须复核（压力上升时见 vec-12）。

## 技术限制与风险

- `UNPK_B16`/`PACK` 粒度按 B16（64 元素）定义：行宽非 64 时须分段或另行证明 mask 覆盖，避免越界或漏写。
- 输出侧 `PACK` 要求先把 fp32 取整成 BF16 再压缩；若原路径舍入不是 `CAST_ROUND`，须对齐舍入模式以保持位级等价。
- 改动**只适用于逐行独立**的 VF；一旦引入跨行消费（归约/扫描）即失效。
- 死代码 VF（无调用点）不必改：无收益且无法被测试验证。
- 需目标版本支持这两个分布模式；版本差异时以 `pl.LoadDist` / `pl.StoreDist` 的实际枚举与生成物为准。

## 参考资料

- 语法引用：`vf.load_align(..., dist=pl.LoadDist.UNPK_B16)`、`vf.store_align(..., dist=pl.StoreDist.PACK)`、`vf.astype(..., round_mode=pl.VFRoundMode.CAST_RINT/CAST_ROUND, saturate=pl.SaturateMode.OFF)`、`pl.LoadDist` / `pl.StoreDist` 分布模式文档。
- 仓库内参考实现：[engram_gate.py](https://gitcode.com/cann/pypto-gym/blob/ccd64a44f843d077feab9a81d894a09bda3c5af2/ops/pro/deepseek_v4_1/engram_gate/op_kernel/engram_gate.py) 是 UNPK_B16/PACK 用法实例；[engram_gate_bwd.py](https://gitcode.com/cann/pypto-gym/blob/ccd64a44f843d077feab9a81d894a09bda3c5af2/ops/pro/deepseek_v4_1/engram_gate_bwd/op_kernel/engram_gate_bwd.py) 是使用 CastLayout.ZERO/ONE 拆分的对照实现。
