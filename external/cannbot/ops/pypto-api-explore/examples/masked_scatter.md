# masked_scatter kernel reference

> Note: batch 轴 loop 切分；最后轴 cumsum 索引依赖整块在 tile 内（此处以 where 示意选择）。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def masked_scatter_kernel(x: pypto.Tensor(sl, pypto.DT_FP32),
                          mask: pypto.Tensor(sl, pypto.DT_BOOL),
                          source: pypto.Tensor(sl, pypto.DT_FP32),
                          out: pypto.Tensor(sl, pypto.DT_FP32)):
    for i in pypto.loop(batch, name="row", unroll_list=[1]):
        x_s = pypto.view(x, [1] + inner, [i] + [0] * len(inner))
        mask_s = pypto.view(mask, [1] + inner, [i] + [0] * len(inner))
        src_s = pypto.view(source, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.where(mask_s, src_s, x_s)
        pypto.assemble(r, [i] + [0] * len(inner), out)
```
