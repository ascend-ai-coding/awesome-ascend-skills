# conv3d kernel reference

> Note: N（batch）轴 loop 切分；空间/深度/通道轴整块在 tile 内，由 3D conv tiling 处理。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def conv3d_kernel(fmap: pypto.Tensor(list(fmap_shape), dtype),
           weight: pypto.Tensor(list(weight_shape), dtype),
           out: pypto.Tensor(list(out_shape), dtype)):
    for n in pypto.loop(batch, name="N", unroll_list=[1]):
        fmap_s = pypto.view(fmap, [1] + fmap_inner, [n] + [0] * len(fmap_inner))
        pypto.set_conv_tile_shapes(tile_l1, tile_l0)
        result = pypto.conv(fmap_s, weight, dtype, strides, pads, dilations, extend_params={}, groups=groups)
        pypto.assemble(result, [n] + [0] * (len(out_shape) - 1), out)
```
