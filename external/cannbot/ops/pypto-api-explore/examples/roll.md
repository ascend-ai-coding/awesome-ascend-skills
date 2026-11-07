# roll kernel reference

> Note: batch 轴 loop 切分；last-dim 整块；`shift` 为滚动偏移，`split` 为切分点（= 末轴长度 − shift），index_select 取两段后 concat。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def roll_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        idx_all = pypto.arange(inner[-1])
        part_head = pypto.index_select(a_s, -1, pypto.view(idx_all, [shift], [split]))
        part_tail = pypto.index_select(a_s, -1, pypto.view(idx_all, [split], [0]))
        r = pypto.concat([part_head, part_tail], dim=-1)
        pypto.assemble(r, [i] + [0] * len(inner), out)
```
