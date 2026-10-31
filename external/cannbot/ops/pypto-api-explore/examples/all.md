# all kernel reference

> Note: batch 轴 loop 切分；归约轴（last-dim）整块；bool 经 where 转 FP32 求和后与 S−0.5 比较实现 all。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def all_kernel(a: pypto.Tensor(sl, pypto.DT_BOOL),
           out: pypto.Tensor(ol, pypto.DT_BOOL)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        mask_fp32 = pypto.where(a_s, pypto.full([1] + inner, 1.0, pypto.DT_FP32), pypto.full([1] + inner, 0.0, pypto.DT_FP32))
        sum_val = pypto.sum(mask_fp32, dim=-1, keepdim=True)
        pypto.set_vec_tile_shapes(1, *inner_out)
        r = pypto.gt(sum_val, pypto.full([1] + inner_out, S - 0.5, pypto.DT_FP32))
        pypto.assemble(r, [i] + [0] * (len(ol) - 1), out)
```
