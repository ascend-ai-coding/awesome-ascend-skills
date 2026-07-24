---
name: external-cannbot-ops-ops-precision-standard
description: 算子精度标准。描述 Ascend C 算子各种 dtype 输出对应的精度比对标准(atol/rtol)。当需要(1)评估算子精度是否达标,(2)编写
  ST 测试验证精度,(3)处理 FP16/FP32/BF16 等不同数据类型精度问题,(4)确认算子精度验收标准时触发。
original-name: ops-precision-standard
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# 选择精度标准

```
随机数生成算子?
  ├─ 是 → 随机数生成类标准(reference/random_generation.md)
  └─ 否 → 包含数值计算?
           ├─ 否 → 非计算类标准(reference/non_compute.md)
           └─ 是 → 检查输入输出dtype
                    ├─ 均为整型 → 整数计算类标准(reference/integer_compute.md)
                    ├─ 整型↔浮点 → 量化计算类标准(reference/quantization.md)
                    └─ 均为浮点 → 浮点计算类标准(reference/float_compute.md)
```

## 辅助文档

- **[标杆构造方法](reference/benchmark_construction.md)** - CPU Golden 或昇腾小算子拼接标杆构造
- **[测试用例生成](reference/test_case_generation.md)** - 测试用例设计与边界覆盖
