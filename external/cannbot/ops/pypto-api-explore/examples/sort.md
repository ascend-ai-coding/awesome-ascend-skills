# sort kernel reference

> Note: batch 轴 loop 切分；排序轴（最后轴）整块由专用 sort HW（sort32 + mrgsort）处理，输出为交错值+索引（out_interleaved）。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def sort_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(out_interleaved, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        sorted_interleaved = pypto.sort32(a_s, -1)
        fully_sorted = pypto.mrgsort(sorted_interleaved, 32)
        pypto.assemble(fully_sorted, [i] + [0] * (len(out_interleaved) - 1), out)
```
