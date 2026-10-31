# UB Resident Scan

## 适用条件

整行及 fp32 工作区可放入 UB。

## TileLang/PTO 实现

一次搬入整行，在 UB/fragment 完成 prefix 后一次写回。按 dtype lane 对齐分配 padding，但只处理 valid 元素。

正确性基线见 references/scan/templates/dav310/scan_base.py：每行由唯一 kernel task 持有，沿最后连续轴分 tile，fp32 carry 常驻，GM/UB 使用 T.copy。实现不调用 T.cumsum，也不依赖 grid barrier。

## 精度门禁

验证最后有效 lane 作为输出而非 padding。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry。

## 性能门禁

适合短中行；比较 full-load 与 streaming 的 launch、copy 和 UB 占用。

先通过 PTO lowering 与定向精度测试。比较 serial correctness、SIMD lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。
