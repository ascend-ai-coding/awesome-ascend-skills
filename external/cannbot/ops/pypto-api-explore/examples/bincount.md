# bincount kernel reference

> Note: 无可切轴——1D 输入全归约到定长直方图（bin 轴），输入轴与 bin 轴相互依赖，整 tile 处理。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def bincount_kernel(a: pypto.Tensor([n], pypto.DT_INT32),
           out: pypto.Tensor([num_bins], pypto_dtype)):
    pypto.set_vec_tile_shapes(n, num_bins)
    oh = pypto.cast(pypto.one_hot(a, num_classes=num_bins), pypto_dtype)
    out[:] = pypto.sum(oh, dim=0)
```
