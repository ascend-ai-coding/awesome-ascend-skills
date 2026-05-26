---
name: external-cannbot-ops-pypto-precision-compare
description: PyPTO 算子精度问题调试技能。提供两种精度对比方法：文件保存方法（使用 pypto.pass_verify_save 和 torch.save）和二分对比方法（使用检查点
  tensor）。当需要调试 PyPTO 算子精度、定位精度差异来源、进行中间结果对比时使用此技能。
original-name: pypto-precision-compare
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# PyPTO 算子精度问题调试技能

提供两种精度对比方法，用于快速定位 PyPTO 算子中导致精度问题的具体 op。

## 方法选择

### 方法对比

| 特性 | 文件保存方法 | 二分对比方法 |
|------|--------------|--------------|
| **实现方式** | 使用 `pypto.pass_verify_save()` 保存到文件，使用 `torch.save()` 保存 golden | 使用检查点 tensor 作为输入参数，在内存中直接对比 |
| **适用场景** | 初次定位精度问题，需要快速找到问题范围 | 需要对比多个循环迭代数据，减少文件 IO 开销 |
| **循环支持** | 只保存 `idx=0` 的数据 | 支持保存所有循环迭代数据 |
| **代码修改** | 不需要修改 kernel 函数签名 | 需要修改 kernel 函数签名，添加检查点参数 |
| **数据类型** | 直接保存原始类型，对比时统一转换 | 在内存中直接对比，类型需一致 |
| **使用难度** | 简单，只需添加检查点调用 | 较复杂，需要管理检查点 tensor |

### 选择指南

**优先使用文件保存方法**（参考 [verify.md](reference/verify.md)）：
- 需要快速找到问题范围
- 只需要对比单次循环的数据，不需要对比多个循环迭代
- 不想修改 kernel 函数签名，保持代码简洁
- 复杂算子，循环较多，二分对比方法难以将数据搬运到循环外

**切换到二分对比方法**（参考 [binary-search.md](reference/binary-search.md)）：
- 需要对比多个循环迭代的数据（如 idx=0,1,2,...）
- 保存的数据文件过大时，用二分法在内存中直接对比减少文件 IO 开销

## 快速开始

### 使用文件保存方法

1. 阅读 [verify.md](reference/verify.md) 了解详细步骤
2. 在 kernel 中添加 `pypto.pass_verify_save()` 调用
3. 在 golden 中添加 `torch.save()` 调用
4. 运行测试生成数据
5. 使用对比工具分析结果

### 使用二分对比方法

1. 阅读 [binary-search.md](reference/binary-search.md) 了解详细步骤
2. 修改 kernel 函数签名，添加检查点 tensor 参数
3. 修改 golden 函数，返回检查点数据
4. 修改测试函数，创建检查点 tensor 并对比
5. 运行测试并分析结果

## 核心原则

### 数据对齐原则

无论使用哪种方法，都必须确保：
- golden 和 kernel 的计算逻辑、切块方式、数据维度完全一致
- 如果实现不一致，改写 golden 函数使其与 kernel 一致
- 检查点的位置和顺序必须一一对应

### 检查点命名原则

- 使用有意义的名称，反映计算步骤
- 按计算顺序添加数字前缀（如 `1_after_matmul`, `2_after_softmax`）
- 确保命名约定一致，便于对比工具自动匹配

## 参考资料

- PyPTO API: `docs/api/`
- pass_verify_save API: `docs/api/others/pypto-pass_verify_save.md`
