# TileLang/PTO SIMT 优化

## 使用方式

在一维 T.Kernel 内进入 with T.SimtVF(threads=N)，局部数据用 T.alloc_fragment，并行维用 T.Parallel。连续规则逐元素优先 T.SimdVF；reduction、离散索引、复杂控制流和字节 transpose 使用目标仓库已验证的 SIMT 路径。

模式分支由 Python factory 生成独立 T.prim_func。threads 在 64/128/256 等已验证候选中实测，元素尾部用 if i<valid 或完整 padding，不能假设线程数自动屏蔽越界。

## 门禁

覆盖每个模式、threads、lane/warp 边界和尾部。检查生成源码的分支分化、向量化、fragment 大小和 spill；只有精度通过且 latency 改善才替换 SIMD/fallback。可执行例子见实际 TileLang 源码 `examples/ascend/example_rmsnorm.py`、`testing/ascend/language/test_tilelang_ascend_reduce.py`；转置布局语义另查 `testing/ascend/layout/test_ascend_l0_transpose.py`。
