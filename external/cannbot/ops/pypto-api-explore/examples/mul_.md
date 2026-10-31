# mul_ kernel reference

> Note: batch 轴 loop 切分；last-dim 计算轴整块；pypto 无 inplace，结果写回 out。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def mul__kernel(a: pypto.Tensor(sl, pypto_dtype),
                b: pypto.Tensor(sl, pypto_dtype),
                out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        b_s = pypto.view(b, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.mul(a_s, b_s)
        pypto.assemble(r, [i] + [0] * len(inner), out)
```
