# Row Kogge-Stone Scan

## 适用条件

扫描轴静态且 tile 可驻留 UB；更重视低逻辑深度，能够接受每层较多 combine 和寄存器流量。

## 算法

第 `k` 层距离为 `d=2^k`，所有 `i>=d` 的元素并行执行：

```text
next[i] = combine(prev[i - d], prev[i])
```

每层必须从上一层快照读取，不能原地让本层新值污染后续 lane。共执行 `ceil(log2(R))` 层；非整尾部只更新真实元素。`dav310/cum_row_kogge_stone.py::level_pairs` 给出每层依赖边。

跨 tile carry 与 Sklansky 相同，由唯一 row owner 保持 fp32 状态。若 PTO SIMD 无法安全表达层间快照或产生寄存器 spill，回退到 streaming correctness baseline。

## 精度与性能门禁

覆盖 `R=1`、每个二次幂边界、tile±1、多 tile carry、抵消、大小量和 dtype 溢出契约。与 Sklansky 比较 latency、SIMD 指令数、临时 UB/寄存器和代码体积。
