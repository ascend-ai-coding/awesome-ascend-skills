# Sklansky Prefix Network

## 适用条件

扫描轴静态、完整 tile 可驻留 UB，并且 PTO SIMD 的 lane shift/gather/broadcast 能高效表达树形 combine。带 argmin/argmax 索引传播的 scan 不直接使用该模板。

## 算法

对长度 `R` 的 resident tile 展开 `ceil(log2(R))` 层。第 `k` 层的半组宽为 `2^k`，组宽为 `2^(k+1)`；每组使用下半组最后一个元素作为 anchor，与上半组全部 target 做结合：

```text
half  = 1 << k
group = half << 1
anchor = group_start + half - 1
target = [group_start + half, min(group_start + group, R))
```

Cumsum 的 combine 是加法，Cumprod 是乘法，无索引 Cummin/Cummax 分别是 min/max。尾组只更新真实 target，padding 不参与结果。`dav310/cum_oneway_sklansky.py::level_pairs` 给出每层准确的 anchor/target 关系。

跨 tile 时，当前 tile 的最后一个 prefix 作为 fp32 carry，由同一个 row owner 串联到下一 tile。没有经过 PTO SIMD microtest 前，使用 `scan_base.py` 的串行实现作为精度 oracle，而不是把它称为 Sklansky kernel。

## 精度与性能门禁

覆盖每个 `2^k-1/2^k/2^k+1` 边界、非整尾部、多 tile carry、正负抵消及接口规定的溢出和 NaN/Inf 语义。性能上与 Kogge-Stone 比较 fanout、指令数、寄存器压力和生成代码长度。
