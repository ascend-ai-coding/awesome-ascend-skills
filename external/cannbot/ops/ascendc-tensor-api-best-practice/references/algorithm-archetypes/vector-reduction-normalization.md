+# 向量归约、归一化与融合

本 archetype 是路由层，覆盖 add_rms_norm_dynamic_quant、dequant_swiglu_quant、engram_gate_fusion、mhc_post_grad、mhc_sinkhorn。它不是完整实现手册；实施前必须继续读任务合同、DESIGN/PLAN 模板、precision-model、experimental-reg-vector，以及 source-map 中对应参考实现的路线审计结论。

## 路线选择

| 场景 | 执行单元 | 说明 |
|---|---|---|
| 行内 sum/max/amax、softmax、Norm | AIV | row 内归一化，Reg/Mem Vector 均可 |
| Norm + dynamic quant + 多输出 | AIV | row max 与量化输出必须在同一 owner 闭合 |
| Matmul + 归一化/激活 | __mix__ | 先验证 L0C→UB 直通是否满足依赖 |
| 小方阵 Sinkhorn | AIV | fp32、row/col 交替归一化 |

## 关键设计点

1. 归约顺序是数值合同：fp32 累加、分段树归约或串行累加不可混用后仍宣称 bit 级一致。
2. RMSNorm 冻结 mean/rms 的计算域、sqrt(eps) 位置、gamma dtype 和最终 cast。
3. 动态量化冻结 amax/127、clamp_min、round 模式、-128/127 边界和 scale 输出 dtype。
4. SwiGLU 冻结 split 方向、activate_left 语义、SiLU 计算精度和可选 quant_scale 顺序。
5. 多输出必须区分公开输出与诊断中间量；每个输出只有一个 writer。
6. Softmax 和 Sinkhorn 的 eps 位置、迭代次数、收敛条件与 NaN/Inf 处理写入 DESIGN。

## 验证顺序

1. 单行、单 token、小 H；再用全零、全负、极大、极小值测边界。
2. H 对齐与非对齐、rank 1/2/3、重复 launch。
3. 多输出逐个对比，不允许一个输出失败被另一个输出平均掩盖。
4. 对融合优化做 bit 级回归；对算法路径变化只做误差门。

## 事实源

- asc-devkit：softmax/reduction/gather 等 Experimental Reg Vector 样例
- ops-nn：norm/rms_norm_dynamic_quant、quant/dynamic_quant、quant/swiglu_mx_quant
- cann-samples：Samples/2_Performance/{rms_norm_quant_story,kv_rms_norm_rope_cache_story,softmax_regbase_story}
- ops-transformer：mhc/mhc_sinkhorn、mhc/mhc_post_backward

L3 映射：add_rms_norm_dynamic_quant、dequant_swiglu_quant、engram_gate_fusion、mhc_post_grad、mhc_sinkhorn。
