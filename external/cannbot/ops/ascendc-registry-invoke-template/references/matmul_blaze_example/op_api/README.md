# op_api 说明

## 描述
quant_matmul_gelu_example算子的ACLNN接口定义与实现，包括L0和L2两级API.

## 实现说明
本目录中部分文件未实现，可参考references/add_example/op_api/实现，差异点如下：
- quant_matmul_gelu_example算子的aclnn接口中，除基础的shape/dtype/format校验外，需额外校验输入矩阵的k轴是否匹配，矩阵形状和转置情况是否匹配，以及scale的shape和对应矩阵的shape是否匹配。

## 目录结构
op_api/                                   # ACLNN 接口
├── aclnn_quant_matmul_gelu_example.h     # L2 API 头文件
├── aclnn_quant_matmul_gelu_example.cpp   # L2 API 实现
├── quant_matmul_gelu_example.h           # L0 API 头文件
└── quant_matmul_gelu_example.cpp         # L0 API 实现



