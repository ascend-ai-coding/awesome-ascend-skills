# Quantization 优化工作流

> 适用：算子本体就是 quant/dequant 的情况。**注意**：量化更多是 trait——量化 matmul / 量化 FA 请回主干族 workflow（[matmul.md](matmul.md) M10 / [flashattention.md](flashattention.md)），只把本文件当支路参考。

## Step 0 — 族判定确认

命中 [README.md](README.md) 判定表第 4 行（scale/zero_point、quant/dequant 对、int8/fp8 张量）且主干不是 matmul/FA/conv。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。Quantization 族常见命中：**C3**（D4 转换路径 + A8 量化精度）。

## Step 2 — 族内阶段（按序推进）

| 阶段 | 判定条件（满足才做） | 命中用卡 | 跳过条件 |
|---|---|---|---|
| Q1 转换路径 | FP8/INT4/INT8 转换由软件逐元素实现 | D4（FP8/INT4 转换）、A8（量化精度：舍入/bias 处理） | 已是硬件转换 |
| Q2 多模式分发 | 支持多种量化模式（per-tensor/per-channel/int8/fp8） | P48（多量化模式编译期分发） | 单一模式 |
| Q3 反量化 | dequant 由 Mul+Add 向量链实现 | P49（AscendDequant 硬件反量化） | 已用硬件指令 |
| Q4 内联量化 | 输出端 requant 走独立 pass | P70（FixPipe buffer 内联量化，Cube 输出直接量化写回） | 无 requant |
| Q5 引擎分流与 Cube 修复 | weight-quant GEMM/GEMV 类，M 跨度大 | 小 M（1-8）→ X12（纯 VEC 流式，scale 提出 K 循环）；M 跨度大 → X13（shape 自适应双引擎分流，实测校准 M 阈值）；cube 利用率异常低（ND drain 碎片）→ X11（CType 路由 ND + baseM 扩至 L0C 上限） | 纯 fp16/bf16 无量化 |
| Q6 dequant pass 消除 | 独立 VEC dequant kernel + cube kernel 两趟 | X14（mix-kernel 跨核生产者/消费者融合，AIV 产 strip / AIC 消费，launch 3→2） | 已单 kernel 融合 |

## 组合约束

- 推荐组合：量化全链路（P48+P49+P70）见 [../strategy-compatibility.md](../strategy-compatibility.md) §3
