# Scalar Bound / 小 case 诊断页

> 通用诊断适用于 arch22 / arch35，具体指标与卡片按目标能力选用；RegBase / Reg / MicroAPI / VF / RVEC 相关内容仅适用于 arch35（A5），示例阈值不作为跨代默认值。


> 本页只做**诊断**；优化手段见技术卡片 [techniques/scalar-01-scalar-bound.md](techniques/scalar-01-scalar-bound.md)。采集与指标口径见 `metrics.md`。

## 判定条件

- 总计算量极小（如 element count < 阈值）
- Scalar 指令占比高，Vector/Cube 单元空闲
- 指令发射率低，IPC 偏低
- **tile 数巨大、每 tile 计算量小**：每 tile 的固定开销（DataCopyPad/Broadcast/EnQue/launch 等）摊不开 → `scalar_ratio` 高、发射 bound，**甚至慢于更粗粒度的基线**（典型：小 shape 广播每 tile 只含一个小平面、`totalTiles` 上万）

## 仿真图分析要点

- 定位 Scalar 指令占比较大的时间窗口
- 识别 Scalar 与 Vector 间不必要的同步等待

## 优化手段（详见技术卡片库 `techniques/`）

| 诊断特征 | 首选卡片 |
| --- | --- |
| 冗余 scalar / 小循环 / 减循环轴 / 去封装开销 | [scalar-01-scalar-bound](techniques/scalar-01-scalar-bound.md) |
| 标量 round-trip（GetValue/SetValue、V_S/S_V 同步） | [vec-04-scalar-roundtrip-vectorize](techniques/vec-04-scalar-roundtrip-vectorize.md) |
| 外层 for 在 VF 外反复启停 / setup 重复 | [vec-09-loop-sink-into-vf](techniques/vec-09-loop-sink-into-vf.md) |
| 小定长循环 | [vec-10-small-loop-unroll](techniques/vec-10-small-loop-unroll.md) |
| **Matmul API 常量折叠**（kernel 使用 `AscendC::Matmul`，TCubeTiling 运行时解析致 aic_scalar 偏高） | 见下方「Matmul API 常量折叠（Scalar Bound 专项）」 |

### Matmul API 常量折叠（Scalar Bound 专项）

**适用门禁（同时满足才推荐实施 Go）**：
1. Kernel 使用 Matmul API（如 `AscendC::Matmul`）；
2. `aic_scalar_ratio ≥ 0.15`（Scalar Bound / 小 case 门禁——**只看 scalar 占比是否过阈值，不与其它泳道占比比大小**；即使 MTE2/Vector 占比更高，只要 scalar 过阈值即命中本策略，与访存 Bound 路线可叠加分析）；
3. shape 可固化或可取上界（调用点能全程 constexpr）。

**手段**：把 `TCubeTiling` 的运行时解析迁移到编译期——用 constexpr `MatmulApiStaticTiling` 作为模板参数，`Init/REGIST` 传 `nullptr` 后再 `SetOrgShape`（顺序不可颠倒）、singleCore 填上界、尾块维度不可常量化，从而降低 `aic_scalar_time`。

**分工边界**：本技能负责策略选择与 tiling 修正建议；逐步改代码的实施细节需参考 `ascendc-performance-best-practices` 的 matmul 族优化指南（[constant_folding_design.md](../../../ascendc-performance-best-practices/references/matmul/constant_folding_design.md)）。

**验证**：改后按完整调用 msprof Task Duration 与 scalar 分段核验收益，精度按原口径复测。

## Tiling 修正建议

- 适当增大单次处理粒度，减少循环次数
- 考虑与其他 kernel 融合以减少 launch 开销
- **每 tile 打包多 batch**：小 tile 过多致固定开销 bound 时，一个 tile 装多个 batch（`batchPerTile = elemNum/plane`）、整块一次计算/搬运，把 `totalTiles` 降一个量级（准入 `plane % 对齐 == 0`，非对齐回落）。
