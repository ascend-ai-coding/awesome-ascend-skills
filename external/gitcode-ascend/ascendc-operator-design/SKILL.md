---
name: external-gitcode-ascend-ascendc-operator-design
description: 完成AscendC算子设计 - 帮助用户完成算子的架构设计、接口定义和性能规划。当用户提到算子设计、算子开发、tiling策略、内存规划、AscendC
  kernel设计、两级tiling、核间切分、核内切分时，使用此skill。
original-name: ascendc-operator-design
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# AscendC Operator Design Skill

根据算子需求生成完整的设计文档（design.md），供后续 code-gen skill 消费。

## 使用场景

在 `ascendc-operator-project-init` 创建骨架后、`ascendc-operator-code-gen` 生成代码之前调用。

## 设计流程

### 1. 算子需求分析

**如果由调度 skill 调用**，算子名称和功能描述已确定，直接进入步骤 2。

**如果独立调用**，与用户确认以下信息：

| 信息 | 必填 | 说明 |
|------|------|------|
| 算子名称（snake_case） | 是 | 如 `acosh`, `rms_norm` |
| 功能描述 / 数学公式 | 是 | 如 "acosh(x) = ln(x + sqrt(x²-1))" |
| 支持的数据类型 | 否 | 默认 float16 + float32 |

**MANDATORY**: 检查 PyTorch / NumPy 是否存在同名接口。如果存在，接口签名和语义 **必须** 与之对齐（如 `torch.acosh`、`torch.softmax`）。

### 2. 选择实现路径

根据算子特性，推荐合适的实现方式：

| 实现路径 | 适用场景 | 判断标准 |
|----------|---------|---------|
| **AscendC Kernel** | 纯 vector 算子 | 不涉及矩阵乘法 |
| **CATLASS 模板库** | GEMM / FlashAttention | 含 cube 矩阵计算 |
| **ACLNN 封装** | CANN 已有内置算子 | 无需自定义 kernel |

> **新算子默认使用 AscendC Kernel 路径**，除非明确涉及矩阵乘法。

### 3. 详细设计文档生成

**MANDATORY**: 在生成设计文档之前，必须读取以下参考文档：

1. **必读**: `templates/design-template.md` — 设计文档模板
2. **按算子类型选读**:
   - 逐元素操作（add/relu/acosh/sigmoid...）→ `references/elementwise-tiling.md`
   - 归约操作（softmax/layernorm...）→ `references/reduction-tiling.md`
   - 索引操作（gather/index_select/scatter...）→ `references/index-tiling.md`
   - 排序操作（sort/topk...）→ `references/sort-tiling.md`
   - 池化操作（avgpool/maxpool/...）→ `references/pooling-tiling.md`
3. **通用参考**: `references/general-tiling-principles.md`

**绝对不要跳过参考文档的阅读。**

#### 3.1 设计文档结构

设计文档包含以下核心章节：

1. **算子接口定义** — 函数签名、参数说明、支持的数据类型
2. **计算逻辑设计** — 算法描述、**AscendC API 调用伪代码**、实现路径选择
3. **Tiling策略** — 两级Tiling设计（Block级 + UB级）、UB分配表、tileLength计算
4. **Workspace需求** — workspace大小计算
5. **性能优化** — 关键优化点、算子特性分析
6. **Kernel端实现要点** — 偏移计算、执行流程、**FP16/BF16 升精度流程**
7. **实现检查清单** — 文件结构、代码要点、测试要点

#### 3.2 计算逻辑伪代码（关键产出）

**必须** 将数学公式分解为 AscendC API 调用序列。这是 code-gen skill 的直接输入。

**常见数学函数到 AscendC API 映射**：

| 数学运算 | AscendC API | 备注 |
|----------|-------------|------|
| x + y | `Add(dst, src0, src1, len)` | 双输入 |
| x - y | `Sub(dst, src0, src1, len)` | 双输入 |
| x * y | `Mul(dst, src0, src1, len)` | 双输入 |
| x / y | `Div(dst, src0, src1, len)` | 双输入 |
| x + scalar | `Adds(dst, src, scalar, len)` | 标量运算，优先使用 |
| x * scalar | `Muls(dst, src, scalar, len)` | 标量运算，优先使用 |
| abs(x) | `Abs(dst, src, len)` | |
| exp(x) | `Exp(dst, src, len)` | |
| ln(x) | `Ln(dst, src, len)` | |
| sqrt(x) | `Sqrt(dst, src, len)` | |
| 1/x | `Reciprocal(dst, src, len)` | |
| 1/sqrt(x) | `Rsqrt(dst, src, len)` | |
| tanh(x) | `Tanh(dst, src, len)` | |
| relu(x) | `Relu(dst, src, len)` | |
| max(x,y) | `Max(dst, src0, src1, len)` | |
| min(x,y) | `Min(dst, src0, src1, len)` | |
| fp16→fp32 | `Cast(dst, src, CAST_NONE, len)` | 升精度无损 |
| fp32→fp16 | `Cast(dst, src, CAST_ROUND, len)` | 降精度有损 |

**示例 — acosh(x) 的 API 调用序列**：
```cpp
Mul(tmp, x, x, len);        // tmp = x²
Adds(tmp, tmp, -1.0f, len); // tmp = x² - 1
Sqrt(tmp, tmp, len);         // tmp = sqrt(x² - 1)
Add(tmp, tmp, x, len);      // tmp = x + sqrt(x² - 1)
Ln(y, tmp, len);             // y = ln(x + sqrt(x² - 1))
```

> **注意**：如果计算序列中某步的 dst 与 src 相同（原地操作），大部分 AscendC API 支持，但需确认具体 API。

#### 3.2 Tiling 策略

**重要**: AscendC 算子采用两级 Tiling 策略，根据算子类型参考相应文档：

```
┌─────────────────────────────────────────────────────────────┐
│                    全局内存 (GM)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              totalLength 元素数据                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Core 0  │     │  Core 1  │ ... │ Core 39  │   ← Block级Tiling (核间切分)
    └──────────┘     └──────────┘     └──────────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   UB 0   │     │   UB 1   │     │  UB 39   │   ← UB级Tiling (核内切分)
    └──────────┘     └──────────┘     └──────────┘
```

**Block级Tiling（核间切分）**:
- 将数据分配到多个 AI Core 并行处理
- **负载均衡**: 整核/尾核策略，尾核处理数据量小于等于整核

**UB级Tiling（核内切分）**:
- 每个 Core 内部分块处理数据
- **UB 对齐**: 32 字节
- **UB 容量**: 不超过 UB_SIZE_LIMIT（实际编码时通过接口获取，示例值 192KB）

**参考文档**:
- **逐元素操作**: 阅读 `references/elementwise-tiling.md`（包含完整两级 Tiling 实现）
- **归约操作**: 阅读 `references/reduction-tiling.md`
- **索引操作**: 阅读 `references/index-tiling.md`
- **排序操作**: 阅读 `references/sort-tiling.md`
- **池化操作**: 阅读 `references/pooling-tiling.md`
- **通用原则**: 参考 `references/general-tiling-principles.md`

#### 3.3 硬件约束说明

- **UB 缓冲区**: 必须按 32 字节对齐，即使逻辑上只需要存储少量数据
- **归约类算子**: 单值缓冲区需要开辟 32B 空间
- **精度处理**:
  - **FP32 输入**: 无需升精度，直接计算
  - **FP16 输入**: **必须升精度到 FP32 计算**，保证计算精度
  - **BF16 输入**: **必须升精度到 FP32 计算**，vector 计算单元不支持 bfloat16 直接计算
- **Workspace 需求**:
  - **elementwise 类**: SYSTEM_WORKSPACE_SIZE（通常为 16MB）
  - **其他类算子**: 根据实际 tiling data 大小计算

#### 3.4 常见算子类型的 UB 分配速查表

根据算子输入数量和数据类型，快速确定 bufferCoefficient：

**单输入单输出 elementwise（acosh, relu, sigmoid, exp, ln, sqrt, abs...）**：

| 数据类型 | UB 布局 | bufferCoefficient |
|----------|---------|-------------------|
| float32 | inQ(2×4) + outQ(2×4) + tmpBuf(1×4) = 20 | **20** |
| float16 | inQ(2×2) + outQ(2×2) + tmpBuf1(1×4) + tmpBuf2(1×4) = 16 | **16** |

**双输入单输出 elementwise（add, mul, sub, div...）**：

| 数据类型 | UB 布局 | bufferCoefficient |
|----------|---------|-------------------|
| float32 | inQ_X(2×4) + inQ_Y(2×4) + outQ(2×4) + tmpBuf(2×4) = 32 | **32** |
| float16 | inQ_X(2×2) + inQ_Y(2×2) + outQ(2×2) + tmpBuf(3×4) = 24 | **24** |

> **实战经验**：bufferCoefficient 是 code-gen 阶段最关键的参数。设计文档中 **必须** 明确给出每种 dtype 的值，否则代码生成无法正确计算 tileLength。

#### 3.5 生成设计文档

基于收集的信息，读取 `templates/design-template.md` 模板，填充所有章节，输出到 `csrc/ops/<op_name>/design.md`。

**输出位置**: `ascend-kernel/csrc/ops/<op_name>/design.md`（覆盖初始化阶段的占位文件）

## 交互流程

**被调度 skill 调用时**（推荐流程）：
1. 接收算子名称和功能描述
2. 自动选择实现路径
3. 读取参考文档，生成完整设计文档
4. 输出到 design.md

**独立调用时**：
1. **需求收集**: 通过对话了解算子需求
2. **方案推荐**: 基于需求推荐实现路径
3. **详细设计**: 生成完整的设计文档
4. **检查确认**: 与用户确认设计要点
5. **移交开发**: 生成检查清单，准备进入编码阶段

## 注意事项

### Tiling 参数设计原则

1. **参数结构化**:
   ```cpp
   // 好的做法：使用结构体
   struct MyOperatorTilingData {
       int64_t totalLength;        // 总数据长度

       int64_t formerNum;          // 整核数量
       int64_t formerLength;       // 整核数据长度
       int64_t tailNum;            // 尾核数量
       int64_t tailLength;         // 尾核数据长度

       int64_t tileLength;         // UB单次处理长度
   };

   // 避免：使用大量独立参数
   void KernelFunc(int64_t totalLength, int64_t tileNum, int64_t tileLength, ...);
   ```

2. **两级对齐**:
   ```cpp
   // Block级: Cache Line 对齐 (512字节)
   constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
   int64_t totalLengthCoreAlign = ((totalLengthCore + CACHE_LINE_BYTE_LENGTH - 1) / CACHE_LINE_BYTE_LENGTH) * CACHE_LINE_BYTE_LENGTH;

   // UB级: 32字节对齐
   int64_t ubAlignElements = 32 / dtypeSize;
   int64_t tileLengthAligned = ((tileLength + ubAlignElements - 1) / ubAlignElements) * ubAlignElements;
   ```

3. **UB 分配表**: 每个算子设计**必须**包含 UB 分配表，明确列出：
   - 所有 buffer 名称和用途
   - 每个 buffer 的大小（字节）
   - buffer 数量（单 buffer 或 double buffer）
   - 总 UB 使用量和约束验证

4. **Double Buffer**: 使用 BUFFER_NUM=2 实现 double buffer，隐藏内存延迟

### 其他注意事项

1. **数据类型对齐**: 确保PyTorch tensor类型和AscendC kernel类型匹配（half ↔ float16, float ↔ float32）
2. **内存对齐**: AscendC要求内存地址对齐（UB 32B, Cache Line 512B）
3. **Shape约束**: 某些算子对shape有特殊要求（如需要被tile size整除）
4. **性能权衡**: 在代码复杂度和性能之间找到平衡点
5. **接口定义**: 检查PyTorch/Numpy等库是否存在类似算子接口，如果存在，接口定义参考PyTorch/Numpy等库
6. **测试输入范围**: 在设计文档中注明算子的有效输入范围（如 acosh 要求 x >= 1），测试用例需据此生成数据

## 交付标准（DoD）

设计文档生成后，必须包含以下关键产出物（供 code-gen skill 直接消费）：

- [ ] **函数签名**: `at::Tensor op_name(const at::Tensor &self, ...)` 完整声明
- [ ] **支持的数据类型**: 明确列出（如 float16, float32）
- [ ] **AscendC API 调用伪代码**: 每步计算映射到具体 API
- [ ] **UB 分配表**: 每种 dtype 的 buffer 布局和 bufferCoefficient
- [ ] **Tiling 参数结构体**: 字段定义和计算公式
- [ ] **FP16/BF16 升精度流程**: 如支持半精度，必须描述 Cast 路径

## 下一步

设计完成后，使用 `ascendc-operator-code-gen` skill 生成具体代码实现。
