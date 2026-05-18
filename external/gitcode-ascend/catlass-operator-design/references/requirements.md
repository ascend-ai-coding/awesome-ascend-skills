# Catlass MatmulGelu 算子需求文档

## 1. 功能需求

### 1.1 核心功能
- **矩阵乘法与GELU融合计算**: 实现矩阵乘法（Matmul）与高斯误差线性单元（GELU）激活函数的融合执行
- **多维度支持**: 支持不同形状的矩阵输入（m×k × k×n → m×n）

### 1.2 数据类型支持
- **输入**: 半精度浮点数（half）
- **中间计算**: 单精度浮点数（float）
- **输出**: 半精度浮点数（half）

### 1.3 布局支持
- 支持灵活的矩阵布局配置（LayoutA、LayoutB、LayoutC、LayoutD），example中均使用RowMajor布局

## 2. 依赖与约束

### 2.1 硬件依赖
- 目标平台: Ascend AtlasA2

### 2.2 软件依赖
- Catlass库核心组件
- Ascend Toolkit 8.3及以上版本

### 2.3 设计约束
- 必须使用模板化设计实现
- 遵循Catlass库编码规范
- 支持静态编译优化