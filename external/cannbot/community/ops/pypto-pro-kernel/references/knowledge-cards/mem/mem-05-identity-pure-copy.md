---
type: "PyPTO Performance Optimization Card"
title: "Identity/Pure-Copy 快路径路由"
description: "identity（恒等映射）快路径：tiling 证明输出是输入的等价拷贝时路由 pure-copy kernel 整块搬运，跳过通用模板的坐标恢复与小段搬运。"
status: "stable"
tags: ["pypto-pro", "mem", "identity", "routing"]
item_id: "mem-05"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/datamove.md#host-tensor-strides-and-boundary-guards"
      scope: "identity/pure-copy 路由须同时保留真实 shape、stride、offset、dtype、alias 与输出 guard 合同。"
      boundary: "该物理契约不证明算子已退化为 identity；数学谓词和通用 fallback 仍需独立推导，不能按测试形状白名单路由。"
bound_hint: "SCALAR、MTE2"
applicability: "算子存在退化语义（identity reshape、单段 split、参数使变换退化为 copy），且该条件可由 shape/axis/layout/属性在 tiling 阶段完整证明"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 pl.load/pl.store 二维整块搬运与 host 侧模板路由"
---
# 技术卡片 mem-05：Identity/Pure-Copy 快路径路由

- **适用 bound**：访存 / Scalar
- **一句话**：先在 tiling 判定"输出只是输入的等价拷贝"，路由到不含元数据计算的 pure-copy kernel；通用模板留给真正的非平凡映射。

## 何时用（诊断特征）

- 算子存在退化语义：reshape 前后逻辑索引不变、split 恰为连续块、参数使重排/插值/归约退化为 copy。
- 通用路径仍在逐段恢复逻辑坐标（div/mod、stride 表），而最终输出元素与输入一一直接映射。
- 该条件可由 shape、axis、layout 和属性在 tiling 阶段完整证明。

## 何时不适用

- 映射涉及 format/stride 变换：不是 identity，必须使用保持物理布局的通用模板。
- 输入输出可能重叠：需满足 overlap 语义的实现，不能默认并行 copy 安全。
- 判定条件只能在运行中逐元素确认时，不能用 shape 白名单硬路由。

## 原理

通用变换模板按小段搬运并逐段恢复逻辑坐标（`//`、`%`、stride 表等标量元数据），服务于非平凡映射。identity 条件成立时这些准备只增加标量指令与小 copy 开销；pure-copy kernel 把整块数据按最大 Tile 直接搬运，标量开销与 copy 次数同时下降。路由发生在 host/tiling：谓词成立走 pure-copy kernel，否则走通用 kernel，两条路径各自独立验证。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的 identity 复制 kernel 对；尺寸、地址与循环边界须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

# before：通用模板——逐段恢复逻辑坐标并小段搬运
# for s in pl.range(core_id, total_segs, num_cores):
#     i = s // segs_per_row
#     j = s % segs_per_row
#     xt = x_g.next(); yt = y_g.next()
#     pl.load(xt, x, [i, j * SEG])
#     pl.mul(yt, xt, 1.0)
#     pl.store(y, yt, [i, j * SEG])

# after：identity 谓词成立时路由到 pure-copy kernel
tile_type = pl.TileType(shape=[TILE_M, N], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec)
x_g = pl.make_tile_group(type=tile_type, addrs=0x00000, mutex_ids=[0, 1])

with pl.section_vector():
    blocks = x.shape[0] // TILE_M
    for b in pl.range(core_id, blocks, num_cores):
        xt = x_g.next()
        pl.load(xt, x, [b * TILE_M, 0])
        pl.store(y, xt, [b * TILE_M, 0])


# host 侧路由（tiling 谓词）
def is_identity_mapping(x_shape, y_shape):
    return list(x_shape) == list(y_shape)   # 实际谓词须覆盖 axis/stride/layout/dtype/offset/alias
```

## 性能与验证指标

分别记录 identity 与非 identity 两组的 copy 次数、标量指令数与同条件 `Task Duration`。有效快路径应删除通用模板特有的逐段元数据计算；非 identity 路径的性能与结果不能受影响。

## 技术限制与风险

- 判定必须覆盖 shape、axis、layout、stride、dtype、offset 和 alias 约束；仅元素数相同不足以证明可直接 copy。
- 必须保留通用 fallback 并由运行时谓词路由，禁止按测试 shape 白名单切换。
- pure-copy 仍要正确处理空 tensor、尾块和多核分片边界。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/resource_management/make_tile_group.md`
