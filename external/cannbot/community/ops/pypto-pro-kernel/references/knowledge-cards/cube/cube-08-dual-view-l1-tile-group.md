---
type: "PyPTO Performance Optimization Card"
title: "同一 L1 地址提供两种布局"
description: "同一份数据如果既要按原布局使用，又要按转置布局使用，可以给同一 L1 地址声明两个视角，避免再搬一份数据。"
status: "stable"
tags: ["pypto-pro", "cube", "memory", "layout"]
item_id: "cube-08"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/memory-allocation.md#aliasing-and-mutual-exclusion"
      scope: "同一 L1 分配的 NZ/ZN 双视角必须保持一致的地址、容量和同步所有权。"
      boundary: "别名合法不代表任意矩形 shape 可以转置搬运；两个消费视角、单写者和 L0 输入形状须逐条证明。"
bound_hint: "MTE1"
applicability: "同一份数据要同时作为 Cube 两条输入路径的操作数，一条使用 NZ 视角，另一条使用 ZN 视角，并且两边都能使用同形搬运。"
target_api_gate: "需要当前版本支持同地址 TileGroup、NZ/ZN 视角和对应的 L0 搬运；共享地址下的 mutex 和写入顺序必须在目标版本上复核。"
---

# 技术卡片 cube-08：同一 L1 地址提供两种布局

## 何时用（诊断特征）

- 一份 L1 数据要给两个 matmul 使用。
- 一条路径需要原布局，另一条路径需要转置布局。
- 当前代码为两种布局各搬运或读取一次。

## 何时不适用

- 数据只会被一种布局消费。
- 两个消费 tile 的 shape 不匹配，不能各自做同形搬运。
- 写入并非唯一，或者不能证明两个消费者都在写入完成后读取。

## 原理

在同一组物理地址上声明两个 tile 视角：一个是 `[M, N]` 的 NZ 视角，另一个是 `[N, M]` 的 ZN 视角。Vector 只写一次，Cube 分别从两个视角搬到 L0A 和 L0B。这里不是把 `[M, N]` 的 tile 直接搬到 `[N, M]`，而是让两个目的 tile 各自和自己的视角同形。

## 怎么改（before / after）

下面是能力门控伪码，地址、mutex、布局和 `next()` 形式需要按目标版本补证。

```能力门控伪码
# before：两份数据
write(l1_nz, data)
write(l1_zn, transpose(data))
move(l0a, l1_nz)
move(l0b, l1_zn)

# after：一份数据，两个视角
l1_nz = make_tile_group(shape=[M, N], layout=NZ, addrs=[A0, A1])
l1_zn = make_tile_group(shape=[N, M], layout=ZN, addrs=[A0, A1])
write(l1_nz.next(), data)
move(l0a, l1_nz.next())
move(l0b, l1_zn.next())
```

实际实现中应保存同一轮的槽位句柄，避免两个 `next()` 取到不同槽位；每个物理槽仍需有清晰的生产、消费和释放顺序。

## 性能与验证指标

比较两种写法的 L1/L0 搬运次数、MTE 空泡和 kernel 时间。正确性需要覆盖完整 tile、尾块、多轮复用和并发 subblock，并逐元素比较两条消费路径。

目前只有生产先例，尚无当前算子的量化收益；采用前需要做交错 A/B 测试。

## 技术限制与风险

- 两个视角必须分别和 L0A、L0B 的输入 shape 对齐；跨 shape 搬运不能替代转置。
- 同地址只允许一个生产者写入；共享地址的 mutex 合同待验证。
- ZN 视角能否被目标版本正确消费，需要看生成物和 value test，不能只凭声明判断。
