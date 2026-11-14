# linspace kernel reference

> Note: 无 batch 轴，沿输出轴按 tile loop 切分；每 tile 生成一段等差序列。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def linspace_kernel(out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(num_tiles, name="out_tile", unroll_list=[1]):
        step = pypto.full([tile_len], 1.0 / max(n - 1, 1), pypto_dtype)
        idx = pypto.cast(pypto.arange(tile_len), pypto_dtype)
        seg = pypto.mul(idx, step)
        pypto.assemble(seg, [i * tile_len], out)
```
