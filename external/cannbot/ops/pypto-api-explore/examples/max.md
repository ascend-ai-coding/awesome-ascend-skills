# max kernel reference

> Note: batch 轴 loop 切分；最后轴（归约轴）整块在 tile 内，由 pypto.amax 归约。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def max_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.amax(a_s, dim=-1, keepdim=True)
        pypto.assemble(r, [i] + [0] * (len(ol) - 1), out)
```
