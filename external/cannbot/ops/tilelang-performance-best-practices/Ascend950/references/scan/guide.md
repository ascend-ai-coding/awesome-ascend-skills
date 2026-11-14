# PTO Prefix Scan

## 适用条件

inclusive scan、cummin/cummax 及带 index 变体。先固定 axis、方向、dtype、累加和输出语义。

## TileLang/PTO 实现

行数足够时使用 row-owned streaming；短行可 full-load；极少超长行才使用 local-prefix、chunk-total scan、fixup 三 kernel 方案。lane-parallel 仅在对应 SIMD gather/shift 微测通过后启用。

正确性基线见 `references/scan/templates/dav310/scan_base.py`：每行由唯一 kernel task 持有，沿最后连续轴分 tile，fp32 carry 常驻，GM/UB 使用 `T.copy`。实现不调用 `T.cumsum`，也不依赖 grid barrier。该 serial inner scan 只用于精度回归；没有附带硬件、版本、命令和原始结果的性能数字不作为生产 dispatch 依据。

## 精度门禁

cummin/cummax 的 tie-break、NaN 和 index 传播需独立 reference。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry。

## 性能门禁

按短/中/长行建立 dispatch，不能以单一 shape 决定。生产路径必须以已通过 lowering 的 SIMD lane scan 或端到端更快的 split-row 替换 serial inner scan。

先通过 PTO lowering 与定向精度测试。比较 serial correctness、SIMD lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。
