# Broadcast 广播类参考算法

面向 Add、Mul、Sub 等广播计算。先按 [代际路由](../../../architecture-routing.md) 确认目标，随后读取 [tiling-flow.md](tiling-flow.md) 和 [tiling-fields.md](tiling-fields.md)。

[参考脚本](script/broadcast_tiling.py) 显式接收 `is_dav3510`、目标可用 `ub_size` 和 `core_num`，不探测平台、不默认 A5。A2/A3 只选 OneDim/静态 UB 候选，rank/对齐及目标 API 合法性仍需核验；A5 可按条件选择动态 UB/NDDMA。脚本结果是建模候选，不代替目标构建与运行验证。
