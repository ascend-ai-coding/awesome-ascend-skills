# tests 说明

## 描述
quant_matmul_gelu_example算子测试，包括单元测试ut和系统测试st，具体可参考references/add_example/tests/

## 实现说明
本目录留空，可参考references/add_example/tests/实现.

## 目录结构
tests/                               # 测试
├── st/                              # 系统测试（ST，上板精度验证）
│   ├── CMakeLists.txt
│   ├── run.sh
│   ├── README.md
│   └── test_aclnn_quant_matmul_gelu_example.cpp
└── ut/                              # 单元测试（UT）
    ├── CMakeLists.txt
    ├── run.sh
    ├── README.md
    ├── cmake/                      # GoogleTest 构建脚本
    ├── common/                     # tiling/infershape 测试框架（faker/executor）
    ├── op_api/                     # ACLNN 接口 UT
    └── op_host/                    # tiling/infershape UT
