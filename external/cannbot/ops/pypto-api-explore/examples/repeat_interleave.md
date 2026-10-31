# repeat_interleave kernel reference

> Note: 输入 batch 轴 loop 切分；每行复制 rep 份写到输出（输出 batch = rep×输入），元素轴整块。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def repeat_interleave_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        pypto.assemble(a_s, [i * rep] + [0] * (len(ol) - 1), out)
        pypto.assemble(a_s, [i * rep + 1] + [0] * (len(ol) - 1), out)
```
