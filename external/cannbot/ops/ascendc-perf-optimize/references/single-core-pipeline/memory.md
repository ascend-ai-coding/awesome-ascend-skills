# 访存 Bound 诊断页

> 通用诊断适用于 arch22 / arch35，具体指标与卡片按目标能力选用；RegBase / Reg / MicroAPI / VF / RVEC 相关内容仅适用于 arch35（A5），示例阈值不作为跨代默认值。


> 本页只做**诊断**；优化手段见下方技术卡片。采集与指标口径见 `metrics.md`。

## 判定条件

- MTE2 利用率高（AIC 带宽接近理论峰值）
- Vector/Cube 单元存在等待数据的空闲气泡
- 计算密度低（Ops/Byte 低于硬件能力）

> ⚠️ **MC² 通算融合场景注意**：`CrossCoreWaitFlag` 挂在 MTE2 流水线上会导致 MTE2 ratio 含通信等待空转。须先用隔离测试排除污染（参考 [bound_diagnosis.md](../comm-compute/bound_diagnosis.md)「MC² 场景 MTE2 污染判定」），确认 MTE2 ratio 反映真实访存瓶颈后再套用以下策略。若 `mte2_polluted = true`，本文档策略无效，请路由到 [comm-compute/](../comm-compute/) 通信掩盖策略。

## 仿真图分析要点

- 查看 MTE2 搬移时间线与计算时间线的重叠度
- 识别大数据量搬移窗口，标记可合并或优化的小搬移序列

---

## 优化手段（详见技术卡片库 `techniques/`）

| 诊断特征 | 首选卡片 |
| --- | --- |
| 连续主区未走对齐搬运 / 非对齐规格拖慢主路径 | [mem-01-aligned-contiguous-copy](techniques/mem-01-aligned-contiguous-copy.md) |
| UB 内可原地消费、消除搬入搬出往返 | [vec-01-ub-fusion](techniques/vec-01-ub-fusion.md)、[vec-05-eliminate-ub-staging](techniques/vec-05-eliminate-ub-staging.md) |
| 反复 load 不变量 | [vec-06-broadcast-once-vl-merge](techniques/vec-06-broadcast-once-vl-merge.md) |
| copy 与计算串行、下一 tile 无数据依赖 | [mem-02-ping-pong-prefetch](techniques/mem-02-ping-pong-prefetch.md) |
| 小行/成对张量反复单独搬运 | [mem-03-batched-row-copy](techniques/mem-03-batched-row-copy.md) |
| 可证明仅消费一次的大块数据污染 L2 | [mem-05-l2-bypass-one-shot](techniques/mem-05-l2-bypass-one-shot.md) |
| 参数使通用变换退化为直接 copy | [mem-07-identity-pure-copy](techniques/mem-07-identity-pure-copy.md) |
| 临时值写 GM 后立刻读回 / 为清零创建 GM 中转 | [mem-04-ub-buffer-reuse](techniques/mem-04-ub-buffer-reuse.md) |
| 输出完整连续却分段 gather/copy out | [mem-06-direct-contiguous-output](techniques/mem-06-direct-contiguous-output.md) |

## Tiling 修正建议

- 调整 UB tile 大小以提升 L2 命中率
- 优化数据搬移粒度与对齐参数
- 调整多核切分以减少单核数据量
