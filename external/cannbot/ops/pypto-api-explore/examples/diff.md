# diff kernel reference

> Note: batch 轴 loop 切分；最后轴滑窗相减（out 末轴 = 输入末轴 - 1），轴整块。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def diff_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner_out)
        a_right = pypto.view(a_s, [1] + inner_out, [0] * len(inner) + [1])
        a_left = pypto.view(a_s, [1] + inner_out, [0] * (len(inner) + 1))
        r = pypto.sub(a_right, a_left)
        pypto.assemble(r, [i] + [0] * (len(ol) - 1), out)
```
