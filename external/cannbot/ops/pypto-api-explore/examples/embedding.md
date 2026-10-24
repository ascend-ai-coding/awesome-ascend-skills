# embedding kernel reference

> Note: 输入行（[B,S]）轴 loop 切分；词表轴整块保留，由 gather HW 查表。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def embedding_kernel(weight: pypto.Tensor(wl, pypto.DT_FP32),
                     indices: pypto.Tensor(il, pypto.DT_INT32),
                     out: pypto.Tensor(ol, pypto.DT_FP32)):
    for i in pypto.loop(batch, name="row", unroll_list=[1]):
        idx_s = pypto.view(indices, [1] + idx_inner, [i] + [0] * len(idx_inner))
        r = pypto.gather(weight, 0, idx_s)
        pypto.assemble(r, [i] + [0] * (len(ol) - 1), out)
```
