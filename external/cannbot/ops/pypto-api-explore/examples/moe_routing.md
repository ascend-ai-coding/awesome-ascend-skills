# moe_routing kernel reference

> Note: token（行）轴 loop 切分；专家轴（最后轴）整块在 tile 内（归约）；sigmoid 组合展开，中间 FP32。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def moe_routing_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="token", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        x = pypto.cast(a_s, pypto.DT_FP32)
        e = pypto.add(pypto.exp(pypto.mul(x, -1.0)), 1.0)
        ones = pypto.full([1] + inner, 1.0, pypto.DT_FP32)
        w = pypto.div(ones, e, pypto.PrecisionType.INTRINSIC)
        s = pypto.sum(w, dim=-1, keepdim=True)
        r = pypto.div(w, s)
        pypto.assemble(r, [i] + [0] * len(inner), out)
```
