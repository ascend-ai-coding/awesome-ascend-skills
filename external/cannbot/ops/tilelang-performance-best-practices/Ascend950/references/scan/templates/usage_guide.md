# Scan Kernel 接入流程

## 适用条件

向当前仓库新增 cumulative operator 时使用。

## TileLang/PTO 实现

Python wrapper 处理空 tensor 并选择 row-owned 或 split-row。factory 固化 tile_cols、dtype 和模式；动态 rows/cols 仅在后端已验证时启用。

正确性基线见 references/scan/templates/dav310/scan_base.py：每行由唯一 kernel task 持有，沿最后连续轴分 tile，fp32 carry 常驻，GM/UB 使用 T.copy。实现不调用 T.cumsum，也不依赖 grid barrier。

## 精度门禁

先声明 inclusive/exclusive、forward/reverse、axis 和输出 dtype。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry。

## 性能门禁

定向精度测试覆盖全部 dispatch，再运行同口径 benchmark。

先通过 PTO lowering 与定向精度测试。比较 serial correctness、SIMD lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。
