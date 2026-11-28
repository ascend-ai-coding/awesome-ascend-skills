# stack kernel reference

> Note: 输入 batch 轴 loop 切分；新增 stack 轴上各输入占一片（此处 2 输入），输入内轴整块。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def stack_kernel(a: pypto.Tensor(sl, pypto_dtype),
           b: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        b_s = pypto.view(b, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        pypto.assemble(a_s, [0, i] + [0] * (len(ol) - 2), out)
        pypto.assemble(b_s, [1, i] + [0] * (len(ol) - 2), out)
```
