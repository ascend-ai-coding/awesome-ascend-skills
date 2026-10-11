# 典型调优案例库

本目录存放开箱调优阶段的典型案例，供调优时参考。

## 案例索引

| 案例 | 文件 | 优化类型 | 收益 |
|------|------|---------|------|
| Decode Attention Vector 合轴+合图 | [vector-axis-merge-softmax.md](vector-axis-merge-softmax.md) | 合轴优化 + sg_set_scope 合图 | -6.0%（275→259 us） |
| 多 Matmul 独立 TileShape | [per-matmul-tile-shapes.md](per-matmul-tile-shapes.md) | 按 M/K/N 特征独立设 cube_tile_shapes | -7.8%（257→237 us） |
| 尾轴 Broadcast 合轴优化 | [combine-axis-broadcast.md](combine-axis-broadcast.md) | `combine_axis=True` 尾轴 broadcast inline | 0%（Cube 占主导时无收益） |
