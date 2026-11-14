# repeat kernel reference

> Note: 沿输出新增（复制）轴 loop 切分；每次迭代把整块输入搬到输出第 i 份，被复制的数据轴整块。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def repeat_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(rep, name="rep", unroll_list=[1]):
        a_s = pypto.view(a, sl, [0] * len(sl))
        pypto.assemble(a_s, [i] + [0] * (len(ol) - 1), out)
```
