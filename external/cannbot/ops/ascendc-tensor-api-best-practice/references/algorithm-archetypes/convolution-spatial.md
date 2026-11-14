+# 卷积与空间采样

本 archetype 是路由层，覆盖 conv_2d、conv_3d_backprop_filter、depthwise_conv_2d、dilation_2d、roi_align、aggregate_hidden。它不是完整实现手册；实施前必须继续读任务合同、DESIGN/PLAN 模板、api-dataflow-layout，以及 source-map 中对应参考实现的路线审计结论。

## 路线选择

| 场景 | 执行单元 | 说明 |
|---|---|---|
| 大 Cout×Cin、规整 conv | AIC | 转 im2col/矩阵收缩，复用 Mmad pipeline |
| depthwise、小核、通道独立 | AIV | 每通道滑窗往往比强行 Cube 更直接 |
| dilation/max pooling | AIV | 加窗口、边界 mask、max 归约 |
| filter grad | AIC/AIV 两阶段或 __mix__ | 局部窗口乘加 + 跨输出位置归约 |
| ROIAlign | AIV | 坐标变换、双线性采样、bin pooling |

## 关键设计点

1. 输入/输出坐标映射先写成显式公式，再映射到 Layout；不要从样例函数名推断 NCHW/NHWC。
2. stride、padding、dilation、ceil_mode 和 asymmetric pad 的边界必须冻结；每条边单独判断。
3. depthwise 的 C 独立性可以利用通道并行，但不能把跨 C 归约误写成普通 conv。
4. filter grad 需要明确归约 owner：核内归约、workspace 归约或 AIV finalize，且输出 writer 唯一。
5. 小窗口卷积（W=3 aggregate_hidden）优先按 shifted load + 向量乘加实现，不必套完整 GEMM 框架。
6. ROIAlign 必须冻结 aligned 半像素偏移、sampling_ratio=0 的启发式、双线性插值边界和 box 越界策略。

## 验证顺序

1. 1×1、3×3、stride=1、pad=0、dilation=1。
2. stride/pad/dilation 组合，再加非对称边界和 ceil_mode。
3. 通道尾块、空间尾块、Cin/Cout 非对齐、groups。
4. filter grad 覆盖空输出位置和全 padding case；Golden 不包含非法采样。

## 事实源

- asc-devkit：examples/01_simd_cpp_api/07_tensor_api/conv2d_forward_tensor_api
- ops-nn：conv/conv2d_v2、conv/conv3d_backprop_filter_v2、conv/depthwise_conv2d_backprop_input、pooling/dilation2_d
- cann-samples：SIMT gather 与 CV datapath 样例可作为坐标访问和融合形态参考

L3 映射：aggregate_hidden、aggregate_hidden_grad、conv_2d、conv_3d_backprop_filter、depthwise_conv_2d、dilation_2d、roi_align。
