# glu kernel reference

> Note: batch 轴 loop 切分；最后轴折半，前半 × sigmoid(后半)；sigmoid 组合展开，中间 FP32。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def glu_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner_out)
        a1 = pypto.view(a_s, [1] + inner_out, [0] * (len(inner) + 1))
        a2 = pypto.view(a_s, [1] + inner_out, [0] * len(inner) + [half])
        x = pypto.cast(a2, pypto.DT_FP32)
        e = pypto.add(pypto.exp(pypto.mul(x, -1.0)), 1.0)
        ones = pypto.full([1] + inner_out, 1.0, pypto.DT_FP32)
        r = pypto.mul(a1, pypto.cast(pypto.div(ones, e, pypto.PrecisionType.INTRINSIC), pypto_dtype))
        pypto.assemble(r, [i] + [0] * (len(ol) - 1), out)
```
