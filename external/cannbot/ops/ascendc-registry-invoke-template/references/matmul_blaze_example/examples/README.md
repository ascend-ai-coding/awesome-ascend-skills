# examples

examples目录提供一个可快速验证 matmul_blaze_example 算子功能的最小调用样例。

## 文件结构

```
examples/
├── CMakeLists.txt                        # 构建配置
├── run.sh                                # 编译运行脚本
├── gen_golden.py                         # 生成golden数据的脚本
└── test_aclnn_matmul_blaze_example.cpp   # aclnn接口的调用示例，并将npu输出和golden进行精度比对
```
