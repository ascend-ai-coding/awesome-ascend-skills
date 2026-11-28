# attention kernel reference

> Note: batch/head 轴 loop 切分；序列与 head_dim 轴整块在 tile 内，matmul 由 cube tiling 处理（非 batch-row vector loop）。

```python
scores_shape = [1] + inner[:-1] + [inner[-2]]  # 单迭代 [1, S, S]

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def attention_kernel(q: pypto.Tensor(sl, pypto_dtype),
           k: pypto.Tensor(sl, pypto_dtype),
           v: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        q_s = pypto.view(q, [1] + inner, [i] + [0] * len(inner))
        k_s = pypto.view(k, [1] + inner, [i] + [0] * len(inner))
        v_s = pypto.view(v, [1] + inner, [i] + [0] * len(inner))
        pypto.set_vec_tile_shapes(1, *inner)
        scores = pypto.matmul(q_s, k_s, pypto_dtype, b_trans=True)
        scale_t = pypto.full(scores_shape, scale, pypto_dtype)
        scores = pypto.mul(scores, scale_t)
        x = pypto.cast(scores, pypto.DT_FP32)
        e = pypto.exp(pypto.sub(x, pypto.amax(x, -1, True)))
        attn = pypto.cast(pypto.div(e, pypto.sum(e, -1, True), pypto.PrecisionType.INTRINSIC), pypto_dtype)
        r = pypto.matmul(attn, v_s, pypto_dtype)
        pypto.assemble(r, [i] + [0] * len(inner), out)
```
