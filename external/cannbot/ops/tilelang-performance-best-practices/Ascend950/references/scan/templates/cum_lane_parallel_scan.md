# SIMD Lane-parallel Scan

## 适用条件

同一向量内部希望用 log2(lanes) 个 shift-add 层替代串行 lane 循环。

## TileLang/PTO 实现

使用 PTO 已验证的 SIMD gather/index primitive 构造 offset=1,2,4... 的静态移位向量，每层只对 lane>=offset 更新；向量间仍通过 fp32 carry 串联。该路径必须先独立 lowering 微测。

正确性基线见 references/scan/templates/dav310/scan_base.py：每行由唯一 kernel task 持有，沿最后连续轴分 tile，fp32 carry 常驻，GM/UB 使用 T.copy。实现不调用 T.cumsum，也不依赖 grid barrier。

## 精度门禁

每一层 mask 必须阻止跨向量串行污染，尾向量只提取最后有效 lane。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry。

## 性能门禁

测 lane scan 指令数、寄存器压力和短行开销；失败或不增益时保持 streaming fallback。

先通过 PTO lowering 与定向精度测试。比较 serial correctness、SIMD lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。
