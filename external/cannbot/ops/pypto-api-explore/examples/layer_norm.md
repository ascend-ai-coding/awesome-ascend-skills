# layer_norm kernel reference

> Note: batch 轴 loop 切分；归约轴（last-dim）整块；mean/var 以 keepdim 在行内计算。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def layer_norm_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        mean = pypto.div(pypto.sum(a_s, dim=-1, keepdim=True), pypto.full([1] + inner_out, float(inner[-1]), pypto_dtype))
        centered = pypto.sub(a_s, mean)
        var = pypto.div(pypto.sum(pypto.mul(centered, centered), dim=-1, keepdim=True), pypto.full([1] + inner_out, float(inner[-1]), pypto_dtype))
        std = pypto.sqrt(pypto.add(var, pypto.full([1] + inner_out, 1e-5, pypto_dtype)))
        normed = pypto.div(centered, std)
        pypto.assemble(normed, [i] + [0] * len(inner), out)
```
