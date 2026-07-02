# QuantMatmulGeluExample

## 基于blaze的matmul算子脚手架

本工程是一个 **基于blaze的matmul算子的registry-invoke 脚手架模板**，
基于 Blaze/tensor_api 实现 matmul 的 AIC 侧流水，通过自定义 Epilogue 实现 AIV 侧 eltwise 后处理，
通过 MatmulKernelFused 模板桥接 AIC/AIV 双侧，形成完整融合流水。

本脚手架可供 ops-registry-invoke 工作流在开发 CV 融合算子时参考复用。

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend950|√|

## 功能说明

- 算子功能：量化矩阵乘 + 激活函数融合计算。

- 计算公式：

$$
y = \text{gelu\_tanh}(\text{matmul\_int32}(x1, x2^T) \times \text{scale}[n] \times \text{pertoken\_scale}[m] + \text{bias}[n])
$$

- 融合结构：
  - **AIC 侧**: BlockMmad 做 int8×int8 → int32 matmul，结果经 CopyL0C2UB 落到 UB
  - **AIV 侧**: ScaleGeluEpilogueRegBase 读取 UB int32，做 scale/pertoken/gelu，写回 GM bf16
  - **CV 同步**: MatmulKernelFused 内部 CrossCoreSetFlag/WaitFlag 同步 AIC/AIV

## 目录结构

```
quant_matmul_gelu_example/
├── op_host/                             # Host 侧代码
│   ├── quant_matmul_gelu_example_def.cpp         # 算子定义
│   ├── quant_matmul_gelu_example_infershape.cpp  # 形状推导
│   ├── CMakeLists.txt
│   └── arch35/                          # arch35 架构 Tiling
├── op_kernel/                           # Kernel 侧代码
│   ├── quant_matmul_gelu_example_arch35.cpp   # MIX Kernel 入口（AIC+AIV）
│   ├── custom_compile_options.ini       # opbuild 编译选项注入
│   ├── CMakeLists.txt
│   └── arch35/                          # arch35 架构实现
├── op_api/                              # ACLNN 接口
│   ├── aclnn_quant_matmul_gelu_example.h     # L2 API 头文件
│   ├── aclnn_quant_matmul_gelu_example.cpp   # L2 API 实现
│   ├── quant_matmul_gelu_example.h           # L0 API 头文件
│   └── quant_matmul_gelu_example.cpp         # L0 API 实现
├── op_graph/                            # Graph mode proto
│   ├── CMakeLists.txt
│   └── quant_matmul_gelu_example_proto.h
├── examples/                            # 算子调用示例
│   ├── CMakeLists.txt
│   ├── run.sh
│   ├── gen_golden.py
│   └── test_aclnn_quant_matmul_gelu_example.cpp
├── tests/                               # 测试
│   ├── st/                              # 系统测试（ST，上板精度验证）
│   └── ut/                              # 单元测试（UT）
├── third_party/
│   └── tensor_api/                      # Blaze/tensor_api 库（需从外部仓库拉取）
├── CMakeLists.txt                       # 构建配置
├── build.sh                             # 构建脚本, 参考references/add_examples/build.sh实现
└── README.md                            # 本文档
```


## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>左矩阵，shape=(M,K)，不转置。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>右矩阵，shape=(N,K)，转置语义（物理行主序=列主序）。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>perchannel 反量化因子，shape=(N)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pertoken_scale</td>
      <td>输入</td>
      <td>pertoken 反量化因子，shape=(M)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>perchannel bias，shape=(N)。aclnn 外部接口为 BF16，L2 层 cast 为 float 后下发。</td>
      <td>BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>融合输出，shape=(M,N)。</td>
      <td>BF16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持 2D GEMM（无 batch 维度）
- M、N、K 必须 >= 16（BLOCK_CUBE 对齐约束）
- 仅支持 Ascend950 (DAV_3510) 架构
- x2 为 (N,K) 转置语义，物理行主序即列主序（DNExtLayoutPtn）
- bias 外部接口 dtype 为 BF16，框架自动 cast 为 float

