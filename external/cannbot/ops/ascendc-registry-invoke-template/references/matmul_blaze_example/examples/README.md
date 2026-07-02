# examples 说明

## 描述
examples目录提供一个可快速验证quant_matmul_gelu_example算子功能的最小调用样例。

## 实现说明
本目录留空，可参考references/add_example/examples/实现，差异点如下：
- matmul的golden计算耗时较高，需要在gen_golden.py中基于torch/numpy等计算库的matmul实现加速。不能直接用三重for循环实现matmul golden计算，耗时太多，shape较大时效率极低。

## 目录结构
examples/           # 算子调用示例
├── CMakeLists.txt
├── run.sh          # 快速编译运行脚本，可配置输入矩阵的shape和转置等参数
├── gen_golden.py   # 生成输入和计算golden，将输入和golden保存为二进制文件，供样例调用。
└── test_aclnn_quant_matmul_gelu_example.cpp # aclnn接口的调用示例，并将npu输出和golden进行精度比对



