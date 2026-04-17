---
name: external-gitcode-ascend-simple-vector-triton-gpu-to-npu
description: 将简单Vector类型Triton算子从GPU迁移到昇腾NPU。当用户需要迁移Triton代码到NPU、提到GPU到NPU迁移、Triton迁移、昇腾适配时使用。注意：无法自动迁移存在编译问题的算子。
original-name: simple-vector-triton-gpu-to-npu
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-03-26'
synced-commit: 0a97c6e3999cf97425ca5ab07678e48089d79ff5
license: UNKNOWN
---


# Triton算子GPU到昇腾NPU迁移技能

本技能帮助用户将简单Vector类型的Triton算子从GPU迁移到昇腾NPU平台，提供完整的迁移流程、代码转换指南和精度验证方法。

## When to use this Skill

当出现以下情况时使用本技能：

- 用户需要将Triton算子从GPU迁移到昇腾NPU
- 用户提到"GPU到NPU迁移"、"Triton迁移"、"昇腾适配"
- 用户需要优化Triton算子在NPU上的性能
- 用户遇到NPU特有的编译错误（如coreDim超限、UB溢出）
- 用户需要了解GPU和NPU的架构差异

## Quick start

```python
# 1. 分析源代码
# 使用 templates/analysis_template.md 生成语义分析报告

# 2. 最小化迁移
# 只修改设备：device='cuda' -> device='npu'

# 3. 运行测试
python test_your_kernel.py

# 4. 根据错误调整
# 参考 reference/troubleshooting.md 解决问题
```

## 适用范围

**✅ 支持迁移的算子类型**:
- 简单Vector类算子（仅使用Vector Core）
- 不涉及复杂控制流的算子
- 1D/2D网格的算子

**❌ 暂不支持自动迁移的情况**:
- 存在编译错误的算子
- 复杂的循环嵌套和条件分支
- 需要特殊内存对齐的CV算子（同时使用Cube和Vector Core）
- 依赖外部库或特殊runtime的算子

## Instructions

### Step 1: 代码逻辑语义分析（关键步骤）

**⚠️ 在迁移前，必须先分析源代码的语义逻辑，生成分析报告**

使用 [templates/analysis_template.md](templates/analysis_template.md) 生成分析报告。

分析要点：
1. **tl.load 分析**：指针计算、mask语义、other值选择
2. **tl.store 分析**：mask是否只检查输出边界
3. **Mask 逻辑**：每个mask的语义含义、广播维度
4. **数据流**：输入如何变换为输出

详细分析指南请参考 [reference/analysis_guide.md](reference/analysis_guide.md)。

### Step 2: 环境准备

```bash
# 安装依赖
pip uninstall triton  # 卸载社区Triton
pip install triton-ascend
pip install torch-npu

# 验证安装
python -c "import torch_npu; print(torch_npu.npu.is_available())"
```

### Step 3: 最小化迁移尝试

**核心原则：先按照GPU源代码尝试，如果尝试有报错再进行后续调整**

```python
# 第一步：只修改设备指定
# device='cuda' -> device='npu'
x = torch.rand(size, device='npu')

# 第二步：运行测试
try:
    result = kernel_npu(**test_inputs)
    print("✅ 基础运行成功")
except Exception as e:
    print(f"❌ 运行失败: {e}")
```

### Step 4: 根据错误类型调整

根据遇到的错误类型，选择对应的解决方案：

| 错误类型 | 错误信息关键词 | 解决方案 |
|---------|--------------|---------|
| **编译错误** | compilation failed | 检查Triton语法兼容性 |
| **coreDim超限** | coreDim > UINT16_MAX | 增大BLOCK_SIZE或设置环境变量 |
| **UB溢出** | ub overflow | 使用子块切分策略 |
| **精度问题** | NaN, Inf, 不匹配 | 检查逻辑运算符、mask使用 |
| **性能问题** | 运行缓慢 | 优化内存访问、使用Tiling |

详细解决方案请参考 [reference/troubleshooting.md](reference/troubleshooting.md)。

### Step 5: 精度验证

迁移完成后**必须**进行精度验证：

```python
def verify_accuracy(result, ref, dtype):
    # 检查NaN/Inf
    assert not torch.isnan(result).any(), "结果包含NaN"
    assert not torch.isinf(result).any(), "结果包含Inf"
    
    # 设置容差
    if dtype in [torch.float16, torch.bfloat16]:
        rtol, atol = 1e-3, 1e-3
    elif dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 0, 0
    
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)
```

## Examples

### 完整迁移示例：向量加法

**迁移前（GPU版本）**:
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

x = torch.rand(98432, device='cuda')
y = torch.rand(98432, device='cuda')
```

**迁移后（NPU版本）**:
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
    y = tl.load(y_ptr + offsets, mask=mask, care_padding=False)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

x = torch.rand(98432, device='npu')
y = torch.rand(98432, device='npu')
```

更多示例请参考 [reference/examples.md](reference/examples.md)。

## Best practices

1. **Grid优先使用1D**: 2D grid会被合并为1D
2. **内存对齐**: VV算子32字节对齐，CV算子512字节对齐
3. **添加care_padding=False**: 提升数据加载并行度
4. **精度验证必须**: 迁移后必须验证精度
5. **tl.load的other值选择**: 索引加载时`other`应设为超出有效范围的值（如`N`）
6. **BLOCK_SIZE调整**: 精度问题时尝试减小BLOCK_SIZE
7. **tl.load与tl.store的mask语义不同**: `tl.load`检查输入有效性，`tl.store`检查输出边界

### tl.load 与 tl.store 的 mask 使用规范

```python
# ✅ 正确示例：load 和 store 使用各自的 mask
out_mask = rows_mask & cols_mask[None, :]  # 输出边界检查
final_mask = out_mask & index_valid_mask[None, :]  # 输入有效性检查
selected = tl.load(inp + inp_off, mask=final_mask, other=0.0)
tl.store(out + out_off, selected, mask=out_mask)  # 正确！
```

| 操作 | mask 含义 | 应检查的内容 |
|------|----------|-------------|
| `tl.load` | 哪些**输入位置**需要读取 | 索引有效性、输入边界 |
| `tl.store` | 哪些**输出位置**需要写入 | 输出边界、行列范围 |

## Requirements

```bash
pip install triton-ascend torch-npu
```

## Advanced usage

- 详细架构差异：[reference/architecture.md](reference/architecture.md)
- 完整故障排查：[reference/troubleshooting.md](reference/troubleshooting.md)
- 迁移案例分析：[reference/examples.md](reference/examples.md)
- 分析模板：[templates/analysis_template.md](templates/analysis_template.md)
