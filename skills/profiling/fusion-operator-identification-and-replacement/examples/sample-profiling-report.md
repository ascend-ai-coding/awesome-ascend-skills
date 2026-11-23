# 示例 Profiling 数据与分析演示

> ## ⚠️ 虚构示例声明
> **本文件中的全部数据均为虚构，仅用于演示输入格式、分析思路与报告结构，不代表任何真实芯片、模型、框架或版本的实测结果。** 虚构数字不得在任何分析中被引用为证据；真实分析必须使用用户提供的 Profiling 数据。演示中的耗时/占比也不保证物理自洽（如带宽反推值可能偏离真实量级），仅作格式示范。

---

## 1. 虚构环境信息（示例：真实分析时用户提供或从环境自查的等价信息）

| 项 | 虚构取值 |
|---|---|
| 模型 | DemoLLM-S（虚构：32 层 Transformer，32 头 GQA，RMSNorm，SwiGLU，RoPE，BF16，未量化） |
| 推理框架 | transformers + torch_npu，**eager 模式**（虚构版本号 v0.0） |
| 硬件 | 单卡 Ascend 910B3（虚构标注），Atlas 800I A2 服务器（虚构） |
| 软件栈 | CANN 8.0.0（虚构）、驱动 24.1.rc1（虚构）、torch 2.1.0（虚构）、torch_npu 0.0（虚构） |
| 采集方式 | msprof，单次 **prefill step**（seq_len=2048, batch=1），导出 op_summary 聚合表与时间线片段 |

## 2. 虚构 op_summary 聚合表（按 Op Type 聚合，单 step）

| Op Type | Task Type | Count | Total(us) | Avg(us) | Share |
|---|---|---|---|---|---|
| MatMul | AI Core-Cube | 129 | 89,800 | 697 | 68.8% |
| Mul | AI Core-Vector | 192 | 9,600 | 50 | 7.4% |
| Add | AI Core-Vector | 96 | 5,500 | 57 | 4.2% |
| TransData | AI Core-Vector | 64 | 4,400 | 69 | 3.4% |
| Sigmoid | AI Core-Vector | 32 | 3,400 | 106 | 2.6% |
| SoftmaxV2 | AI Core-Vector | 32 | 3,300 | 103 | 2.5% |
| BatchMatMul | AI Core-Cube | 64 | 2,900 | 45 | 2.2% |
| Rsqrt | AI Core-Vector | 64 | 2,700 | 42 | 2.1% |
| ReduceSum | AI Core-Vector | 64 | 2,200 | 34 | 1.7% |
| Cast | AI Core-Vector | 32 | 1,500 | 47 | 1.1% |
| Gather | AI Core-Vector | 35 | 900 | 26 | 0.7% |
| 其他（约 20 种小算子） | — | 214 | 3,800 | — | 2.9% |
| **合计** | | | **130,500** | | ~100% |

> 注：Share 因四舍五入合计约 99.6%。列名与真实工具输出可能不同，以实际表头为准。

## 3. 虚构时间线片段（第 17 层，节选）

> "推断语义"列仅为按常见模型结构做出的假设，**未经源码确认前不是结论**。

```
t(us)    Op Type       Dur(us)  输入/输出（示意）                推断语义（待源码确认）
41,020   Add           57       [2048,2048]+[2048,2048]         残差相加
41,082   Mul           50       [2048,2048]×[2048,2048]         RMSNorm: x²
41,137   ReduceSum     34       [2048,2048]→[2048,1]            RMSNorm: 按行求和
41,176   Rsqrt         42       [2048,1]                        RMSNorm: 1/√(mean+ε)
41,223   Mul           50       [2048,2048]×[2048,1]            RMSNorm: 归一化
41,277   Mul           50       [2048,2048]×[1,2048]            RMSNorm: ×γ
41,332   MatMul        698      [2048,2048]×[2048,3072]         QKV 投影
42,030   TransData     69       [2048,3072] ND→NZ               布局转换
42,105   Mul           50       q×1/√d                          注意力缩放
42,160   BatchMatMul   46       Q·Kᵀ                            注意力得分
42,211   Add           57       +mask                           因果掩码
42,273   SoftmaxV2     103      [32,2048,2048]                  行 softmax
42,381   BatchMatMul   44       A·V                             加权求和
42,430   TransData     69       NZ→ND                           布局回转
42,502   MatMul        695      [2048,2048]×[2048,2048]         O 投影
...
44,010   Sigmoid       106      [2048,5632]                     SwiGLU: σ(gate)
44,121   Mul           105      gate×σ(gate)                    SwiGLU: silu
44,231   Mul           105      silu(gate)×up                   SwiGLU: 门控乘
44,341   MatMul        690      [2048,5632]×[5632,2048]         下投影
```

## 4. 分析演示（简版，真实分析按 SKILL.md 的 10 节报告结构完整输出）

### 4.1 能与不能（数据边界）

- **能**：热点排序、瓶颈分类、序列模式识别（有时间线片段）、访存量级估算（有 shape）、带宽反推标定
- **不能**：确认融合关系的最高置信级（无模型源码、无图 dump，证据最多到 E1+E3 的"高置信"）；图编译器自动融合无从谈起（eager 模式，E4 框架配置证据）

### 4.2 现有融合识别（演示）

| 观察 | 结论 | 置信度 |
|---|---|---|
| 全部 kernel 均为单算子语义，无与 `npu-fused-op-catalog.md` 匹配的融合 kernel | 未见算子库融合算子 | 高置信（E1+E3；⚠ torch_npu 适配层是否有隐式替换需核实） |
| 框架为 eager 模式（环境信息自述） | 无图编译器自动融合 | 高置信（E4，前提是"eager 模式"信息属实） |
| 无项目前缀 kernel | 未见自定义融合算子 | 待核实（需源码） |

### 4.3 瓶颈分类（演示）

- MatMul/BatchMatMul 合计 ~71%：GEMM 为主体（计算与权重读取）
- elementwise 链（Mul+Add+Rsqrt+ReduceSum+Sigmoid+Cast）合计 ~19%：访存型开销，融合候选集中区
- TransData 3.4%：布局转换开销，能否消除需布局信息
- 结论：**prefill 阶段除 GEMM 主体外，存在约四分之一的访存型可优化空间**（虚构数据演示）

### 4.4 候选清单条目示例（8 字段格式演示）

| 字段 | 内容（虚构数据演示） |
|---|---|
| 候选算子或算子序列 | Add→Mul→ReduceSum→Rsqrt→Mul→Mul（RMSNorm 展开链，64 处/step） |
| 当前耗时证据 | 序列 wall time ≈283us/处 × 64 ≈ 18.1ms，占 step 13.9%（时间线实测，虚构） |
| 融合可行性 | 高：catalog 已有 npu_add_rms_norm（含 cast/quant 变体），先验证 shape/dtype/版本支持；⚠ 现成算子在虚构 CANN 版本的可用性需核实 |
| 预期收益来源 | 消除 5 个中间张量的读写：未融合流量 ≈84MB/处 → 融合后 ≈25~34MB/处，ΔB≈50~59MB/处；由 Mul kernel 反推 BW_eff≈336GB/s（16.8MB/50us，虚构） → 理论节省 ≈150~175us/处 |
| 实现成本 | 低（改调用现成接口） |
| 风险 | 精度（累加顺序变化）；版本兼容（⚠ 需核实）；torch_npu 适配层与 eager 的兼容行为 |
| 预测可信程度 | 中（理论估算；带宽从本份数据反推标定；未含 tiling 损失） |
| 验证方法 | microbenchmark（64 处序列 vs 融合算子，≥100 次取中位数）+ 端到端 prefill 耗时对比 + atol/rtol 精度对齐 |

端到端口径（演示）：ΔT ≈ 9.6~11.2ms → step 理论加速 ≈ 7~9%（**理论估算，非实测**）。

### 4.5 本虚构数据内嵌的演示要点

1. RMSNorm 与 SwiGLU 均为手写展开（未融合序列模式）——热点的融合候选
2. attention 路径为 BMM+Softmax 展开，未用 FA 类算子——"改用现成 FA 接口"方向的候选
3. TransData 出现但缺少布局信息——演示"数据不足时不硬下结论"
4. GEMM 占主体——演示"融合收益有上限，要给端到端口径"
5. 环境版本全部虚构——演示"版本兼容性结论一律标 ⚠ 需核实"

---

> **再次提醒：以上全部数字为虚构演示，不得引用为任何真实场景的证据。**
