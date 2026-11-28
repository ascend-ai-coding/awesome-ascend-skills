# expand_as kernel reference

> Note: batch 轴 loop 搬运；目标 shape 取自 other，广播复制由 expand_clone 完成。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def expand_as_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner_out)
        r = pypto.expand_clone(a_s, [1] + inner_out)
        pypto.assemble(r, [i] + [0] * (len(ol) - 1), out)
```
