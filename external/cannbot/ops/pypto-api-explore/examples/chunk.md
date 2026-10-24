# chunk kernel reference

> Note: batch 轴 loop 搬运；沿切分轴取其中一片（view 切片），其余轴整块。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def chunk_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner_out)
        r = pypto.view(a_s, [1] + inner_out, [0] * len([1] + inner))
        pypto.assemble(r, [i] + [0] * (len(ol) - 1), out)
```
