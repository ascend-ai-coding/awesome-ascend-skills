# softplus kernel reference

> Note: batch 轴 loop 切分；last-dim 计算轴整块。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softplus_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.log(pypto.add(pypto.exp(a_s), pypto.full([1] + inner, 1.0, a_s.dtype)))
        pypto.assemble(r, [i] + [0] * len(inner), out)
```
