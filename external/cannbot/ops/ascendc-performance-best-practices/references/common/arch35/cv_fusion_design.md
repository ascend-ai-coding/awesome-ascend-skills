# CV 融合（Cube-Vector 协同）设计路由

Cube（MMAD）与 Vector（激活 / Cast / 量化 / 归约）需要协同的融合算子，先按下表定位编程模型与数据通路材料，再进入对应算子族实施。本文只做路由，实现依据见各分册。

## 能力路由

| 能力 | 说明 | 材料 |
|---|---|---|
| `__mix__(1,2)` 混合核编程模型 | Cube/Vector 混合核的入口形态与模板适配 | [Depthwise mix 模板](../../conv/templates/dwconv2d_mix.md)，Conv 族整体入口见 [Conv 性能优化指南](../../conv/guide.md) |
| 数据通路选型（分离 / GM 中转 / L0C→UB 直通） | 中间结果回 UB 还是经 GM 中转的取舍 | [A5 基本块](../../matmul/arch35/basic_block_design.md)、[A5 性能建模](../../matmul/arch35/performance_modeling.md) |
| CrossCore 同步纪律 | Cube/Vector 分段交接的同步与配平 | [StreamK 设计](../../matmul/arch35/streamk_design.md)、[MC² 流水配平](../../mc2/pipeline_balancing_design.md) |
| 双 AIV 分工 | 多 AIV 场景的 Vector 侧分工与 VF Bound 判定 | [A5 性能建模 VF Bound](../../matmul/arch35/performance_modeling.md) |
| Conv 场景 CV 协同实例 | Conv3DDX/Conv3DDW 中的 Cube-Vector 协同落地 | [Conv 通用性能经验](../../conv/conv_common_perf_experience.md) |

## 使用约束

- 选型必须同时核对 UB 容量与 L0C 复用约束（见 [A5 常驻复用与 Bank 冲突](ub_resident_design.md)），通路选型结论不得跨 shape 直接复用。
- 归约类 Vector 后处理与 Cube 主计算的配平，先读 [MC² 流水配平](../../mc2/pipeline_balancing_design.md) 的卡间/核间配平口径再裁剪。
