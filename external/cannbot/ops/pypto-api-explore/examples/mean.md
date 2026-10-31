# mean kernel reference

> Note: batch 轴 loop 切分；归约轴（last-dim）整块；sum 后除以轴长。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def mean_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        s = pypto.sum(a_s, dim=-1, keepdim=True)
        n = pypto.full([1] + inner_out, float(inner[-1]), pypto_dtype)
        r = pypto.div(s, n)
        pypto.assemble(r, [i] + [0] * (len(ol) - 1), out)
```
