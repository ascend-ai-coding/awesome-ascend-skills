# tests

matmul_blaze_example 算子测试，包括单元测试 ut 和系统测试 st，具体可参考 references/add_example/tests/

## 文件结构

```
tests/
├── st/                              # 系统测试（ST，上板精度验证）
│   ├── CMakeLists.txt
│   ├── run.sh
│   ├── gen_golden.py
│   └── test_aclnn_matmul_blaze_example.cpp
└── ut/                              # 单元测试（UT）
    ├── op_host/                     # Host侧UT
    └── op_api/                      # API侧UT
```
