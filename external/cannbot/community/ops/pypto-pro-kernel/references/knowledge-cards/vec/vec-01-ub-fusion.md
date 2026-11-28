---
type: "PyPTO Performance Optimization Card"
title: "Vec Tile / VF 内融合，消除 GM 往返"
description: "相邻 Vector 操作在同一 Vec Tile 与同一 VF 中接力，使无外部消费者的中间结果不写回 GM。"
status: "stable"
tags: ["pypto-pro", "vec", "memory", "fusion"]
item_id: "vec-01"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-row-reduce.md#choose-where-dependent-phases-share-state"
      scope: "相邻 Vector 阶段在同一 JIT kernel 内共享寄存器或 UB 状态，比较删除中间 GM 往返与额外 launch。"
      boundary: "仅覆盖同一 owner 可容纳全部 live set 的融合；外部消费者、跨 worker 依赖和收益仍须分别核验。"
bound_hint: "VEC、MTE2、MTE3"
applicability: "相邻 Vector 链的中间结果无外部消费者，生成物仍有中间 GM 回写再读入，且 Vec 容量可容纳融合 live set"
target_api_gate: "仅限已核验 VF、TileGroup 与 auto_mutex 的 Ascend 950PR 或 950DT 工具链；其它 SoC 重新查表并验证"
---
# 技术卡片 vec-01：Vec Tile / VF 内融合，消除 GM 往返

- **适用 bound**：VEC / 访存
- **一句话**：相邻 Vector 操作在同一 Vec Tile 与同一 VF 中接力，中间结果不写回 GM，省掉多次 MTE2/MTE3 往返。

## 何时用（诊断特征）

- trace 中两个 VECTOR 活跃段之间插入 **MTE3（写回）+ MTE2（读入）**，说明中间结果经过 GM。
- `ai*_mte2_ratio` / `ai*_mte3_ratio` 偏高但有效算术很少。
- elementwise/activation 链由多个 kernel 或多段 load/store 拼接，且中间值没有外部消费者。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 中间结果有外部消费者、跨核可见性、调试输出或必须保留的合同边界。
- 融合后的中间值与所有轮转槽位超出 Vec 容量。
- 生成物已消除中间往返，或当前瓶颈不受这段 MTE2/MTE3 流量影响。

## 原理

PyPTO-Pro 的外层 kernel 用 Tile API 完成 GM↔Vec 搬运，`@pl.vector_function` 用 `vf.*` 在向量寄存器上完成数值计算。把 N 步链路压成“一次 `pl.load` → 一次 VF → 一次 `pl.store`”，即可同时减少 GM 往返、VF 启停和中间 Tile 生命周期。

## 怎么改（before / after）

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def fused_vf(src_tile, bias_tile, dst_tile, valid: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    x = vf.load_align(src_tile, 0)
    bias = vf.load_align(bias_tile, 0)
    x = vf.exp(x, preg)                 # Compute1，结果留寄存器
    out = vf.add(x, bias, preg)         # Compute2，直接消费
    vf.store_align(dst_tile, out, preg)

@pl.jit(auto_mutex=True)
def fused_kernel(
    src: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dst: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Vec,
                     valid_shape=[-1, -1])
    src_group = pl.make_tile_group(type=tt, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    bias_group = pl.make_tile_group(type=tt, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])
    out_group = pl.make_tile_group(type=tt, addrs=[0x0400, 0x0500], mutex_ids=[4, 5])
    valid = pl.min(64, src.shape[1])
    with pl.section_vector():
        src_tile = src_group.next()
        bias_tile = bias_group.next()
        out_tile = out_group.next()
        pl.set_validshape(src_tile, [1, valid])
        pl.set_validshape(bias_tile, [1, valid])
        pl.set_validshape(out_tile, [1, valid])
        pl.load(src_tile, src, [0, 0])
        pl.load(bias_tile, bias, [0, 0])
        fused_vf(src_tile, bias_tile, out_tile, valid)
        pl.store(dst, out_tile, [0, 0])
```

这是结构较完整的单行**示意代码**，用于展示 typed `@pl.jit`、GM↔Tile↔VF 数据流和融合方法，不保证可直接编译或运行，更不能原样作为算子交付。三个 GM 参数需要 `pl.Tensor[[shape...], dtype]` 注解；实际采用时还要按真实 Tensor、shape、地址、tail 与外层循环重新设计。示意约束为三个 Tensor 至少有一行，且 `bias`/`dst` 第二维不短于 `src` 的本次有效长度；推广到多行时在外层按行轮转三个 TileGroup，并为每行重新设置 valid shape 与 GM offset。可用性必须由当前工具链编译、完整正确性测试和健康设备数值验证共同确认，Python AST 或历史 compile-only 结果都不构成保证。

before 形态是 Compute1 后 `pl.store(gm_mid, ...)`，下一个 kernel 再 `pl.load(..., gm_mid, ...)`；after 将两段数值计算合入一个 VF。多 tile 循环沿用相同结构并轮转 group，不能用三个单 `make_tile` 冒充流水 buffer。

## 性能与验证指标

同条件比较 `Task Duration(us)`，确认中间 MTE3→MTE2 往返消失，并检查融合后的寄存器 spill。**待实测**：中间落 GM 的步数越多，预期收益越大。

## 技术限制与风险

- 中间结果与所有轮转槽位必须放得下 Vec 内存；使用 N-buffer 后按槽位总量预算。
- 中间结果若有其他消费者、跨核可见性或调试输出，不能删除。
- 数值计算只在 `@pl.vector_function` 中用 `vf.*`；kernel 只做 Tile 搬运/控制。
- 方法验证包括 `test_{op}.py` 的完整正确性测试与同条件性能比较。

## 参考资料

- PyPTO 仓库：CV 融合与片上数据传递见 `docs/zh/guide/programming_guide/pro/development/cube_vector_fusion.md`；VF API 实现见 `python/pypto_pro/language/_vf_api.py`。
