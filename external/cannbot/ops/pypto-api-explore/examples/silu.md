# silu kernel reference

> Note: batch 轴 loop 切分；last-dim 计算轴整块；sigmoid 组合展开，中间 FP32。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def silu_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        x = pypto.cast(a_s, pypto.DT_FP32)
        e = pypto.add(pypto.exp(pypto.mul(x, -1.0)), 1.0)
        ones = pypto.full([1] + inner, 1.0, pypto.DT_FP32)
        r = pypto.mul(x, pypto.div(ones, e, pypto.PrecisionType.INTRINSIC))
        pypto.assemble(pypto.cast(r, pypto_dtype), [i] + [0] * len(inner), out)
```
