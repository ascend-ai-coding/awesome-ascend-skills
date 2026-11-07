> FA 族 CANNBot-DSL 第 4 层「积木模块」的 SFA 实例，按类边界拆分为独立文件以便逐模块复用；定位与使用规则见 [组装层索引](../index.md)。

# 积木模块 M0b：设备侧寻址/视图纯函数（源 L586-606）

> **为什么**：无状态纯函数可单独单测，且 host/device 共享同一实现，避免两处推导漂移。
```python
def _native_head_tile(gm, layout, query_id, head_offset, rows):
    """Fix one physical token in native BSND/TND storage; tile the G,D axes."""
    if const_expr(layout == "BSND"):
        span = gm[query_id // gm.shape[1], query_id % gm.shape[1], None, None]
    else:
        span = gm[query_id, None, None]
    base = tile_slice(span, (gm.shape[-2], gm.shape[-1]), (0, 0))
    return tile_slice(base[head_offset:, None],
                     (rows, gm.shape[-1]), (0, 0))


def _align16_py(value: int) -> int:
    return (value + 15) // 16 * 16


def _bounded_local(buffer, shape, capacity, offset=0, stride=None):
    # Preserve the allocation pitch while narrowing the runtime extent.
    root = reinterpret(buffer, shape=capacity, offset=offset, stride=stride)
    return reinterpret(root, shape=make_tiler(shape, alignment=(1, 1)), stride=stride)


```

