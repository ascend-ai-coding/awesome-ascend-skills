# eye kernel reference

> Note: 无 batch 轴，沿输出行（第 0 轴）loop 切分；每行是位置 i 的 one-hot，列轴整块。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def eye_kernel(out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(n, name="row", unroll_list=[1]):
        idx = pypto.full([1], i, pypto.DT_INT32)
        row = pypto.cast(pypto.one_hot(idx, n), pypto_dtype)
        pypto.assemble(row, [i, 0], out)
```
