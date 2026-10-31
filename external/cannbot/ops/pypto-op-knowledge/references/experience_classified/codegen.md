# CODEGEN 组件经验

> 对应错误码范围：F6XXXX

---

## 1. 3 层嵌套 `pypto.loop` 导致 workspace 累积失败

**触发场景**：3 层嵌套顺序 `pypto.loop` 且最内层循环体较重（20+ ops）时，所有写回方式在恰好 **16 次最内层执行**后失败。

**根因**：codegen 硬编码 `MAX_LOOP_DEPTH = 3`，3 层嵌套时 workspace 累积速度超过 `WorkspaceRecyclePeriod`（`stitch_function_max_num × MAX_UNROLL_TIMES`）回收速度。"16" 是经验观察到的失败边界，不是源码中的命名常量。

**错误关键词**：无特定错误码，表现为 **16 次迭代后数据损坏**（精度失败），且与写回方式无关（concat/assemble/batch assemble 均在 16 次后失败）

**诊断方法**：改变内层 tile 大小，观察失败边界。若 N_TILE=16（8 内层迭代）→ 2 个 S-tile 后失败（2×8=16）；N_TILE=32（4 内层迭代）→ 4 个 S-tile 后失败（4×4=16）。边界不变 → 确认是 16 次最内层执行的硬限制。

**解决方案**：中间层加 `parallel=True`，使迭代独立调度、各自拥有独立 workspace：

```python
# ❌ 3 层顺序嵌套 → 16 次后失败
for b in pypto.loop(B, name="batch"):
    for n_idx in pypto.loop(N // N_TILE, name="heads"):      # 顺序
        for s_idx in pypto.loop(num_s_tiles, name="seq"):     # 顺序
            ...  # 20+ ops

# ✅ 中间层 parallel=True → 独立 workspace → 通过
for b in pypto.loop(B, name="batch"):
    for s_idx in pypto.loop(num_s_tiles, name="seq", parallel=True):  # 并行
        for n_idx in pypto.loop(N // N_TILE, name="heads"):
            ...  # 20+ ops
```

> `parallel=True` 需要配合 `device_sched_parallelism` 运行时选项（默认 1，建议 4）。轻循环体（<5 ops）可能不触发（workspace 累积慢）。
