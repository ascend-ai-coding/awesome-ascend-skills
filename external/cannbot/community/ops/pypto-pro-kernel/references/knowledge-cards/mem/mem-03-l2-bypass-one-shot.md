---
type: "PyPTO Performance Optimization Card"
title: "一次性数据绕过 L2（能力门控）"
description: "确认某块数据后续不会被当前核或后续 wave 再读时才绕过 L2，避免流式读污染可复用 cache；PyPTO-Pro 当前未暴露 L2 策略接口。"
status: "stable"
tags: ["pypto-pro", "mem", "cache", "capability-gated"]
item_id: "mem-03"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/datamove.md#7-move_sync--mutex_dcci-two-synchronization-mechanisms"
      scope: "讨论一次性读取的 cache 策略前先区分 system.dcci 缓存维护与同步接口。"
      boundary: "此引用不提供 L2 hint/bypass 能力；目标版本没有公开 bypass API 时仅保留能力缺口，不生成替代 kernel 改法。"
bound_hint: "MTE2、L2Cache"
applicability: "大权重/scale/输入块只被完整消费一次且可由 shape/schedule 谓词证明无后续复用，热数据 L2 命中被流式读挤压"
target_api_gate: "能力门控：PyPTO-Pro 当前版本无 L2 cache hint/bypass 公开 API；`pl.system.dcci` 仅做缓存清理失效，不构成 bypass"
---
# 技术卡片 mem-03：一次性数据绕过 L2（能力门控）

- **适用 bound**：访存 / Cache
- **一句话**：仅在可证明"后续没有同一物理区域的读取"时绕过 L2，使流式一次性读不驱逐可复用热数据；有任何复用可能都保留正常 cache。

## 何时用（诊断特征）

- 大权重、scale 或输入块只被一个完整 problem/tile 消费一次，而热数据的 L2 命中受这些流式读挤压。
- 能在运行时由合法 shape/schedule 谓词证明无后续复用，而不是凭个别 benchmark case 猜测。
- Tensor 物理行跨度满足该机制的 cache-line 对齐要求。

## 何时不适用

- 数据会被后续 M tile、其他 wave、其他核或重算路径再读：L2 命中本身就是收益来源。
- 带宽已饱和且无热数据被驱逐：该手段不减少 GM 读字节，通常无收益。
- 没有可证明的无复用谓词：禁止按 case ID、固定 shape 列表或单次 profiling 硬路由。

## 原理

流式一次性读填充 L2 会驱逐之后真正要复用的数据。对这样的读取设置 bypass 使数据不填入 L2；但 problem 被拆成多 tile、多 wave 或热点数据会被重复处理时，bypass 退化为重复 DDR 读。谓词（如"单 wave 覆盖完整数据、单 M tile、工作集大于 L2"）须由 tiling/调度导出。

## 怎么改（能力门控伪码）

**能力缺口**：PyPTO-Pro 当前版本没有公开的 L2 cache hint / bypass 接口（`pl.system.dcci` 只对指定地址做缓存清理并失效，用于跨核共享可见性，不等价于读路径 bypass）。以下伪码仅表达路由结构，**禁止直接复制实现**；采用前须在目标版本确认等价 API 存在并补端到端验证，否则记录 capability gap。

```python
# 能力门控伪码：L2Hint 代表目标版本尚待确认的 cache 策略接口，当前 PyPTO-Pro 无此 API。
# before: 全部读取沿用默认 L2 策略
weight_tile = load(weight_gm)            # 一次性流式读也填充 L2

# after: 仅在可证明无复用且物理行对齐时 bypass
no_later_reuse = covers_whole_problem and m_tiles == 1   # 由 tiling/调度导出
row_aligned = is_physical_row_cache_line_aligned(weight_layout)
if no_later_reuse and row_aligned:
    weight_tile = load_with_l2_hint(weight_gm, hint="BYPASS")   # 伪码，无对应公开 API
else:
    weight_tile = load_with_l2_hint(weight_gm, hint="NORMAL")   # 伪码，无对应公开 API
```

## 性能与验证指标

比较 L2 命中率、DDR 流量与同条件 `Task Duration`，至少覆盖"完整一次性读取"与"拆成多 tile/wave"两类 case；后者必须验证回退 normal cache 后不退化。

## 技术限制与风险

- 对齐按物理布局而非逻辑 shape 判断；ND/NZ 与 scale 的 stride 分别检查，cache line 大小按目标 ini 复核。
- 该手段优化 cache 污染，不减少 GM 读字节。
- PyPTO-Pro 能力补齐前，本卡不产生任何 kernel 改动；在账本上记录 capability gap 与待验证项。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/cache_control/dcci.md`（仅缓存清理失效，非 bypass）
