# outer kernel reference

> Note: 沿第一输入元素（输出行）loop 切分；每行 = a[i] × b 向量，第二输入轴整块。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def outer_kernel(a: pypto.Tensor(al, pypto_dtype),
           b: pypto.Tensor(bl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    b_row = pypto.unsqueeze(b, 0)
    for i in pypto.loop(batch, name="row", unroll_list=[1]):
        a_i = pypto.view(a, [1, 1], [i, 0])
        r = pypto.mul(a_i, b_row)
        pypto.assemble(r, [i, 0], out)
```
