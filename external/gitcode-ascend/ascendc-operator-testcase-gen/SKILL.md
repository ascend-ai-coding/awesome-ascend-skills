---
name: external-gitcode-ascend-ascendc-operator-testcase-gen
description: 完成AscendC算子验证用例生成 - 帮助用户完成testcase设计。当用户提到用例设计、泛化用例生成、算子标杆、UT用例、精度用例、性能用例时，使用此skill。
original-name: ascendc-operator-testcase-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---

# AscendC Operator Testcase Gen Skill

根据算子设计文档（design.md）生成验证用例和标杆，供后续UT代码生成、精度验证以及性能验证skill使用。

## 使用场景

在 `ascendc-operator-design` 创建`design.md`之后、`ascendc-operator-ut-gen`/`ascendc-operator-precision-eval`/`ascendc-operator-performance-eval`调用之前调用。

## 调用接口

### 必需参数

调用此技能时，必须明确提供以下参数：

**参数1：设计文档**
- 根据用户提供的`design.md`设计用例
- 必须包含算子逻辑、输入输出、约束条件、所支持数据类型
- 如果未指定，默认项目路径下的design.md

## 设计流程

### 1. 算子设计文档解析

1. 读取设计文档中[算子接口]、[计算逻辑]、[参考实现]章节
2. 根据算子实际计算逻辑以及pyTorch已有实现、参考文档等，生成算子标杆，了解算子各个输入输出之间shape的相互约束。
3. 基于约束，生成模型场景使用该算子时的常见shape典型用例、以及根据边界条件生成的泛化用例。

### 2. 测试用例生成

**MANDATORY**: 在生成测试用例之前，必须读取以下参考文档：

1. **必读**: 
- `templates/test-cases-template.md` — 测试用例格式参考
- `design.md` — 用户提供的算子设计文档

#### 2.1 设计原则

1. **全 dtype 覆盖**：每个 shape / 每个边界值都遍历算子支持的全部dtype，`SUPPORTED_DTYPES`必须包含`design.md`中[参数说明]支持的所有数据类型
2. **shape 由算子决定**：根据算子支持的维度选择合适的 shape，不要写固定维度
3. **shape 不要过大**：单个用例元素数控制在合理范围，避免不必要的大 tensor
4. **用例总数 = (len(TEST_SHAPES) + len(GENERAL_SHAPES)) × len(SUPPORTED_DTYPES) ≥ 30**

### Part A: 常规 Shape 测试（TEST_SHAPES）

根据算子支持的维度以及算子类型，从以下维度池中选取适合的 shape，组成 `TEST_SHAPES` 列表。

**MUST** 根据算子实际支持的维度来选，不支持的维度不要选。

#### shape 选择参考池

| 维度 | 推荐 shape | 适用算子类型 |
|------|-----------|-------------|
| 1D | (128,), (1024,), (4096,), (8192,) | elementwise, reduction |
| 2D | (32, 512), (64, 768), (128, 1024) | elementwise, matmul, linear |
| 3D | (8, 16, 64), (4, 128, 256) | elementwise, attention, conv1d |
| 4D | (4, 8, 32, 16), (2, 64, 32, 32) | conv2d, elementwise |
| 5D | (2, 3, 4, 5, 6) | conv3d, elementwise |

> **shape 不要过大**：推荐单个 shape 元素数 ≤ 200K。

#### TEST_SHAPES 格式

```python
TEST_SHAPES = [
    ("category_name", "description", (dim0, dim1, ...)),
    # ...
]
```

Category 名称自定义，建议按维度或场景命名。示例：

```python
# elementwise 算子（支持任意维度）
TEST_SHAPES = [
    ("1D",         "128 elements",         (128,)),
    ("1D",         "1024 elements",        (1024,)),
    ("1D",         "4096 elements",        (4096,)),
    ("1D",         "8192 elements",        (8192,)),
    ("2D",         "32x512",              (32, 512)),
    ("3D",         "8x16x64",             (8, 16, 64)),
    ("3D",         "4x128x256",           (4, 128, 256)),
    ("4D",         "4x8x32x16",           (4, 8, 32, 16)),
]
```

### Part B: 泛化Shape 测试（GENERAL_SHAPES）

根据算子支持的维度以及算子类型，从以下维度池中选取适合的 shape，组成 `GENERAL_SHAPES` 列表。

**MUST** 根据算子实际支持的维度来选，不支持的维度不要选。

#### shape 选择参考池

**小Shape场景**（边界测试、极小值、非对齐测试）

| 维度 | 小 shape | 测试目的 |
|------|---------|---------|
| 1D | (1,), (2,), (4,), (8,) | 极小元素、边界值 |
| 1D | (3,), (5,), (7,) | 非对齐长度 |
| 2D | (1, 1), (2, 2), (4, 4) | 极小2D tensor |
| 2D | (1, 128), (2, 256) | 单行/单列场景 |
| 3D | (1, 1, 1), (2, 2, 2) | 极小3D tensor |
| 3D | (1, 8, 64), (2, 16, 128) | 单batch场景 |

**大Shape场景**（生产环境、压力测试、模型典型场景）

| 维度 | 大 shape | 适用模型 | 元素数 |
|------|---------|---------|--------|
| 1D | (3072,), (4096,) | BERT FFN中间层 | 3K-4K |
| 1D | (5120,), (6400,) | GPT-2 FFN中间层 | 5K-6K |
| 2D | (512, 768) | BERT-base full sequence | 393K |
| 2D | (512, 1024) | BERT-large full sequence | 524K |
| 2D | (1024, 768) | GPT-2 sequence | 786K |
| 2D | (1024, 1024) | GPT-2 medium sequence | 1M |
| 2D | (1024, 1600) | GPT-2 XL sequence | 1.6M |
| 3D | (8, 512, 768) | BERT-base batch | 3.1M |
| 3D | (8, 197, 768) | ViT-base batch | 1.2M |
| 3D | (16, 1024, 1024) | GPT-2 large batch | 16.7M |

> **注意事项**：
> - 生产环境、模型典型场景推荐shape应视每个算子具体应用场景而变化
> - 小shape用于边界测试，确保算子在极小输入下正确工作
> - 大shape用于性能测试和验证大规模数据的正确性
> - 元素数超过200K的shape仅用于泛化测试，不在常规测试中使用

#### GENERAL_SHAPES 格式

```python
GENERAL_SHAPES = [
    ("Small", "description", (dim0, dim1, ...)),
    ("Large", "description", (dim0, dim1, ...)),
    # ...
]
```

Category 建议使用 "Small" 和 "Large" 区分场景。示例：

```python
# elementwise 算子（支持任意维度）
GENERAL_SHAPES = [
    # 小Shape场景
    ("Small", "single element",         (1,)),
    ("Small", "tiny vector 2",          (2,)),
    ("Small", "tiny vector 4",          (4,)),
    ("Small", "unaligned length 3",     (3,)),
    ("Small", "unaligned length 5",     (5,)),
    ("Small", "2x2 matrix",             (2, 2)),
    ("Small", "1x128 single row",       (1, 128)),
    ("Small", "1x1x1 scalar",           (1, 1, 1)),
    ("Small", "1x8x64 single batch",    (1, 8, 64)),
    
    # 大Shape场景（生产环境）
    ("Large", "BERT-base FFN 3072",     (3072,)),
    ("Large", "BERT-large FFN 4096",    (4096,)),
    ("Large", "BERT-base 512x768",      (512, 768)),
    ("Large", "BERT-large 512x1024",    (512, 1024)),
    ("Large", "GPT-2 1024x768",         (1024, 768)),
    ("Large", "GPT-2 1024x1024",        (1024, 1024)),
    ("Large", "ViT-base 8x197x768",     (8, 197, 768)),
]
```

#### 2.2 设计用例输出

基于收集的信息，读取 `templates/test-cases-template.md` 模板，填充所有章节，输出到 `csrc/ops/[op-name]/test/[op-name]-test-cases.md`。

**输出位置**: `ascend-kernel/csrc/ops/[op-name]/test/[op-name]-test-cases.md`

## 注意事项

1. **严格遵循模板章节**：只输出模板中定义的章节（算子标杆、典型用例、泛化用例、使用说明）
2. **禁止输出无关内容**：
   - ❌ 精度验证标准（由 `ascendc-operator-precision-eval` 负责）
   - ❌ 性能验证标准（由 `ascendc-operator-performance-eval` 负责）
   - ❌ UT测试代码（由其他skill负责）
   - ❌ 任何超出模板章节定义的内容
3. **输出文件位置**：`ascend-kernel/csrc/ops/[op-name]/test/[op-name]-test-cases.md`