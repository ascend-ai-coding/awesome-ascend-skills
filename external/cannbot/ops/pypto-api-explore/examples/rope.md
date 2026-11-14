# rope kernel reference

> Note: batch 轴 loop 切分；last-dim 折半做旋转变换，cos/sin 与输入同行参与乘加。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rope_kernel(a: pypto.Tensor(sl, pypto_dtype),
           cos: pypto.Tensor(sl, pypto_dtype),
           sin: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        cos_s = pypto.view(cos, [1] + inner, [i] + [0] * len(inner))
        sin_s = pypto.view(sin, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        neg_a2 = pypto.neg(pypto.view(a_s, [1] + inner[:-1] + [half], [0] * len([1] + inner[:-1]) + [half]))
        a1 = pypto.view(a_s, [1] + inner[:-1] + [half], [0] * len([1] + inner))
        rot = pypto.concat([neg_a2, a1], dim=-1)
        r = pypto.add(pypto.mul(a_s, cos_s), pypto.mul(rot, sin_s))
        pypto.assemble(r, [i] + [0] * len(inner), out)
```
