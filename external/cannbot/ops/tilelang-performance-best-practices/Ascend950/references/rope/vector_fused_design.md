# RoPE SIMD 融合

## 实现流程

将 load、旋转与 store 放在同一 tile 中，不写中间 GM。half-split 读取 left/right，计算 left*cos-right*sin 与 right*cos+left*sin；interleaved 读取相邻偶奇元素并使用相同公式。代码直接从 references/rope/code/rope_vf_common.py 派生。

sin/cos 若由调用方按 half 维存储则直接 T.copy；若存储为 full dim，factory 明确索引映射。当前基线使用 T.SimtVF；T.SimdVF 版本只有在消除后端 unsupported scalar instruction、通过 lowering 与性能 A/B 后才能替换。

## 门禁

运算提升 fp32，覆盖极值角度、重复 position、非整 SIMD 尾部和 alias。比较融合前后 GM 字节、lane 利用率、寄存器压力和端到端 latency。
