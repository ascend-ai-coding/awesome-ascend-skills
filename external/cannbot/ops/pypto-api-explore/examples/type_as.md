# type_as kernel reference

> Note: batch 轴 loop 切分；cast 逐元素转换，轴整块（dst_dtype 取自目标张量 dtype）。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def type_as_kernel(a: pypto.Tensor(sl, src_dtype),
                   out: pypto.Tensor(sl, dst_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.cast(a_s, dst_dtype)
        pypto.assemble(r, [i] + [0] * len(inner), out)
```
