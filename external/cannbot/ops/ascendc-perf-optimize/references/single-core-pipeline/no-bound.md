# 无 Bound 诊断页

> 通用诊断适用于 arch22 / arch35，具体指标与卡片按目标能力选用；RegBase / Reg / MicroAPI / VF / RVEC 相关内容仅适用于 arch35（A5），示例阈值不作为跨代默认值。


> 本页只做**诊断**；优化手段见技术卡片 [techniques/nobound-01-nobound.md](techniques/nobound-01-nobound.md)。采集与指标口径见 `metrics.md`。

## 判定条件

- 各硬件单元利用率均不高
- 无明显单一瓶颈
- 总吞吐量未达预期
- **重叠因子低（本页主判据）**：`Σ(各泳道耗时)/aiv_time` 远低于理想上界 `Σ/max`。实测算例见 `metrics.md` §2.1——四条泳道各 ~0.5、Σ=1.634 而上界 3.186，达成率仅 51%
- 判定成立后**先起仿真再动手**：msprof 到此为止只能说明"重叠不够"，阻塞源要靠指令级 trace 定，命令见 `techniques/nobound-02`
- **写回泳道在等计算**：`aiv_mte3_wait_ratio` > 0.4，多半是消费阶段紧跟发起它的异步计算（见 nobound-02）

## 仿真图分析要点

- 识别流水线气泡与等待区间
- 检查搬移-计算 overlap 度
- 看**标量泳道空窗**期间谁在忙：计算泳道满负荷 = 消费阻塞，搬运泳道满负荷但只放行 1 条指令 = 发射阻塞（见 nobound-02）
- 标记 UnitFlag 同步点，检查是否可以进一步减少

---

## 优化手段（详见技术卡片库 `techniques/`）

| 诊断特征 | 首选卡片 |
| --- | --- |
| 搬算未重叠 / 同步过多 / Preload / 提早发射 / PingPong | [nobound-01-nobound](techniques/nobound-01-nobound.md) |
| 三阶段顺序不当：写回等刚发出的计算 / 批量搬运发射反压住计算 | [nobound-02-stage-order-rescheduling](techniques/nobound-02-stage-order-rescheduling.md) |
| 归约累加器串行链型空泡（非无 bound，属 VEC 依赖链） | [vec-08-multi-accumulator-unroll](techniques/vec-08-multi-accumulator-unroll.md) |
| 同一数据重复搬入、Vec 融合消往返 | [vec-01-ub-fusion](techniques/vec-01-ub-fusion.md) |
| 小定长循环 | [vec-10-small-loop-unroll](techniques/vec-10-small-loop-unroll.md) |

## Tiling 修正建议

- 调整 tile 粒度以实现搬移-计算平衡
- 优化 PingPong 分段大小
- 调整 Preload 窗口大小与时间节点
