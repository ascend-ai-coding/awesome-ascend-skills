# softmax kernel reference

> Note: batch 轴 loop 切分；归约轴（last-dim）整块在 tile 内；中间 FP32，非 FP32 输入首尾各一次 `cast`。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        x = pypto.cast(a_s, pypto.DT_FP32)
        e = pypto.exp(pypto.sub(x, pypto.amax(x, -1, True)))
        r = pypto.div(e, pypto.sum(e, -1, True), pypto.PrecisionType.INTRINSIC)
        pypto.assemble(pypto.cast(r, pypto_dtype), [i] + [0] * len(inner), out)
```
