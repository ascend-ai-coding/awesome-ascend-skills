# cat kernel reference

> Note: batch 轴 loop 搬运；本算子为 metadata/搬运语义（view 取片 + assemble 拼回），无逐元素计算；多输入沿 concat 轴顺序搬运到输出对应偏移。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def cat_kernel(a: pypto.Tensor(sl, pypto_dtype),
               b: pypto.Tensor(sl, pypto_dtype),
               out_tensor: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        b_s = pypto.view(b, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        pypto.assemble(a_s, [i] + [0] * (len(ol) - 1), out_tensor)
        pypto.assemble(b_s, [i] + [0] * (len(ol) - 1), out_tensor)
```
