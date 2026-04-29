---
name: external-gitcode-ascend-simple-vector-triton-gpu-to-npu
description: 将简单Vector类型Triton算子从GPU迁移到昇腾NPU。当用户需要迁移Triton代码到NPU、提到GPU到NPU迁移、Triton迁移、昇腾适配时使用。注意：无法自动迁移存在编译问题的算子。
original-name: simple-vector-triton-gpu-to-npu
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
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

## 迁移策略

本技能采用**NPU专用迁移策略**，将GPU实现迁移为NPU实现。

### 策略特点
- **适用场景**：将GPU代码迁移到NPU平台
- **代码特点**：直接替换GPU API为NPU API，代码简洁清晰
- **维护性**：代码量小，易于维护
- **使用场景**：生产部署、性能优化场景

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

### Step 3.1: 关键API迁移

**⚠️ 必须迁移的GPU专用API**：

#### 0. CUDA API迁移对照表

| GPU API | NPU API | 说明 |
|---------|---------|------|
| `torch.cuda.is_available()` | `torch.npu.is_available()` | 检查设备是否可用 |
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` | 清空缓存 |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` | 同步设备 |
| `torch.cuda.mem_get_info()` | `torch.npu.mem_get_info()` | 获取内存信息 |
| `device="cuda"` | `device="npu"` | 设备指定 |
| `@torch.compile` | **删除** | NPU暂不支持torch.compile训练 |

#### 1. 设备上下文迁移

```python
# 直接使用NPU API
import torch_npu

with torch_npu.npu.device(device_index):
    kernel[grid](...)
```

#### 2. 设备属性访问迁移

```python
props = torch_npu.npu.get_device_properties(device)
sm_count = props.vector_core_num  # Ascend910为48
```

**NPU设备属性对照表**：
| GPU属性 | NPU属性 | 典型值（Ascend910） |
|---------|---------|-------------------|
| multi_processor_count | vector_core_num | 48 |
| total_memory | total_memory | 62740MB |
| name | name | 'Ascend910_9392' |
| - | cube_core_num | 24 |
| - | L2_cache_size | '192MB' |

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
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

x = torch.rand(98432, device='npu')
y = torch.rand(98432, device='npu')
```

**说明**：示例中未添加`care_padding=False`，遵循"先确保功能正确"的原则。

更多示例请参考 [reference/examples.md](reference/examples.md)。

## Best practices

### 核心原则

1. **Grid优先使用1D**: 2D grid会被合并为1D
2. **内存对齐**: VV算子32字节对齐，CV算子512字节对齐
3. **精度验证必须**: 迁移后必须验证精度
4. **tl.load的other值选择**: 索引加载时`other`应设为超出有效范围的值（如`N`）
5. **BLOCK_SIZE调整**: 精度问题时尝试减小BLOCK_SIZE
6. **tl.load与tl.store的mask语义不同**: `tl.load`检查输入有效性，`tl.store`检查输出边界
7. **设备API必须迁移**: `torch.cuda.*` 必须替换为 `torch.npu.*` 或 `torch_npu.npu.*`
8. **设备属性不同**: NPU使用 `vector_core_num` 而非 `multi_processor_count`
9. **删除torch.compile**: NPU暂不支持`@torch.compile`装饰器，必须删除

### CUDA到NPU API转换规则

| GPU API | NPU API | 说明 |
|---------|---------|------|
| `torch.cuda.is_available()` | `torch.npu.is_available()` | 检查设备是否可用 |
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` | 清空缓存 |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` | 同步设备 |
| `torch.cuda.mem_get_info()` | `torch.npu.mem_get_info()` | 获取内存信息 |
| `device="cuda"` | `device="npu"` | 设备指定 |
| `@torch.compile` | **删除** | NPU暂不支持torch.compile训练 |

### care_padding参数使用规范

**原则**：先确保功能正确，再考虑性能优化

**推荐流程**：
1. **初始迁移**：不添加`care_padding=False`
2. **功能验证**：确保精度正确
3. **性能优化**：如需提升性能，再尝试添加`care_padding=False`
4. **回归验证**：添加后必须重新验证精度

**示例**：
```python
# Step 1: 初始迁移（不添加）
x = tl.load(x_ptr + offsets, mask=mask)

# Step 2: 功能验证通过后，可选的性能优化
# x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
```

**⚠️ 警告**：直接添加`care_padding=False`可能导致输出全为0或精度问题。

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

## Grid优化模式（高级主题）

**⚠️ 注意**：本节内容适用于以下场景：
- 原始GPU代码使用2D/3D网格
- 需要充分利用NPU物理核心
- 简单1D网格算子通常不需要此优化

对于简单Vector算子，通常只需要：
1. 将`device='cuda'`改为`device='npu'`
2. 使用固定核数：`grid = (num_core,)`

### 核心概念

GPU使用逻辑网格（如3D网格），而NPU使用物理核心网格（1D网格）。为了优化NPU性能，需要将GPU风格的网格适配到NPU的物理核心架构。

### 网格适配模式

**原始GPU风格：**
```python
grid = (NV, NK, N * H)  # 3D逻辑网格
kernel[grid](...)
```

**优化后的NPU风格：**
```python
import torch_npu
import triton.runtime.driver as driver

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

num_core = get_npu_properties()["num_vectorcore"]
grid = (num_core,)  # 1D物理核心网格
```

### 任务分发模式

**GPU内核入口：**
```python
i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
i_n, i_h = i_nh // H, i_nh % H
```

**带任务分发的NPU内核入口：**
```python
core_id = tl.program_id(0)
task_num = NV * NK * N * H
knh_step = NK * N * H
nh_step = N * H

for task_id in tl.range(core_id, task_num, num_core):
    i_v = task_id // knh_step
    i_k = task_id % knh_step // nh_step
    i_nh = task_id % knh_step % nh_step
    i_n = task_id % knh_step % nh_step // H
    i_h = task_id % knh_step % nh_step % H
    # ... 原有内核逻辑
```

### 内核参数适配

**额外的NPU参数：**
```python
def kernel(...,    
    knh_step: tl.constexpr,
    nh_step: tl.constexpr,
    N: tl.constexpr,
    task_num: tl.constexpr,
    num_core: tl.constexpr,
    ...):
```

### Grid优化步骤

#### 步骤1：分析原始网格结构

1. 识别网格维度：`grid = (dim1, dim2, dim3, ...)`
2. 识别program_id用法：`tl.program_id(0)`, `tl.program_id(1)`等
3. 将program_id索引映射到逻辑维度

#### 步骤2：计算任务分发参数

```python
# 计算总任务数
task_num = dim1 * dim2 * dim3 * ...  # 所有网格维度的乘积

# 计算每个维度的步长
# 3D网格(dim1, dim2, dim3)示例：
step_dim2_dim3 = dim2 * dim3
step_dim3 = dim3

# 在内核中：
# task_id = core_id + i * num_core
# dim1_idx = task_id // step_dim2_dim3
# dim2_idx = (task_id % step_dim2_dim3) // step_dim3
# dim3_idx = task_id % step_dim3
```

#### 步骤3：修改内核入口点

用任务分发循环替换直接的program_id索引：

```python
# 之前：
i0 = tl.program_id(0)
i1 = tl.program_id(1)
i2 = tl.program_id(2)

# 之后：
core_id = tl.program_id(0)
for task_id in tl.range(core_id, task_num, num_core):
    i0 = task_id // step_dim2_dim3
    i1 = (task_id % step_dim2_dim3) // step_dim3
    i2 = task_id % step_dim3
    # ... 所有使用这些变量的代码都在循环内部
```

#### 步骤4：处理变量作用域（关键步骤）

**重要**：从GPU风格迁移到NPU任务分发模式时，变量作用域发生变化。必须将所有使用循环内部变量的代码移到循环内部。

**需要检查的常见变量**：
- 索引变量：`pid_b`, `pid_h`, `pid_seq`, `i0`, `i1`, `i2`等
- 长度变量：`seq_len`, `T`, `B`等（特别是当`IS_VARLEN`时可能被修改）
- 偏移变量：`seq_offset`, `bos`, `eos`等

**需要移动的代码**：
1. 所有使用上述变量的计算
2. 指针偏移计算
3. 循环控制变量（如`nchunks`）
4. 状态初始化
5. 结果存储

**错误示例**：
```python
# ❌ 错误：变量在循环内部定义，但在外部使用
core_id = tl.program_id(0)
for task_id in tl.range(core_id, task_num, num_core):
    pid_b = task_id // h_step
    pid_h = task_id % h_step
    # ... 其他变量定义

# 错误：在循环外部使用循环内部定义的变量
nchunks = tl.cdiv(seq_len, CHUNK_SIZE)  # seq_len未定义
ANGLE += pid_b * stride_angle_batch  # pid_b未定义
```

**正确示例**：
```python
# ✅ 正确：所有使用循环内部变量的代码都在循环内部
core_id = tl.program_id(0)
for task_id in tl.range(core_id, task_num, num_core):
    pid_b = task_id // h_step
    pid_h = task_id % h_step
    seq_len = ...  # 在循环内部定义
    
    # 所有使用这些变量的代码都在循环内部
    nchunks = tl.cdiv(seq_len, CHUNK_SIZE)
    angle_ptr = ANGLE + pid_b * stride_angle_batch  # 使用局部变量
    # ... 后续所有计算
```

#### 步骤5：更新内核启动配置

```python
# 之前：
grid = (dim1, dim2, dim3)
kernel[grid](...)

# 之后：
num_core = get_npu_properties()["num_vectorcore"]
grid = (num_core,)
kernel[grid](
    ...,
    knh_step=step_dim2_dim3,
    nh_step=step_dim3,
    N=dim1,  # 或适当的映射
    task_num=task_num,
    num_core=num_core,
)
```

### Grid优化最佳实践

1. **优先使用1D网格**：NPU物理核心是1D结构，使用1D网格可以获得最佳性能
2. **任务均衡分发**：确保计算任务均匀分配到所有核心
3. **注意变量作用域**：所有使用循环内部变量的代码必须在循环内部
4. **使用局部指针变量**：在任务分发循环中，使用局部变量存储偏移后的指针
5. **避免编译时未定义错误**：检查所有变量是否在正确的作用域内定义和使用

### Grid优化示例

**原始GPU版本（2D网格）：**
```python
@triton.jit
def kernel_gpu(x_ptr, output_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # 计算偏移
    x = x_ptr + pid_n * M + pid_m * BLOCK_SIZE
    # ... 计算逻辑

# 启动内核
grid = (N, M // BLOCK_SIZE)
kernel_gpu[grid](x, output, N, M, BLOCK_SIZE=128)
```

**优化后的NPU版本（1D网格）：**
```python
@triton.jit
def kernel_npu(x_ptr, output_ptr, N, M, BLOCK_SIZE: tl.constexpr,
               m_step: tl.constexpr, task_num: tl.constexpr, num_core: tl.constexpr):
    core_id = tl.program_id(0)
    
    for task_id in tl.range(core_id, task_num, num_core):
        pid_n = task_id // m_step
        pid_m = task_id % m_step
        
        # 计算偏移
        x = x_ptr + pid_n * M + pid_m * BLOCK_SIZE
        # ... 计算逻辑（所有代码都在循环内部）

# 启动内核
num_core = get_npu_properties()["num_vectorcore"]
m_step = M // BLOCK_SIZE
task_num = N * m_step
grid = (num_core,)
kernel_npu[grid](x, output, N, M, BLOCK_SIZE=128, 
                m_step=m_step, task_num=task_num, num_core=num_core)
```

## Advanced usage

- 详细架构差异：[reference/architecture.md](reference/architecture.md)
- 完整故障排查：[reference/troubleshooting.md](reference/troubleshooting.md)
- 迁移案例分析：[reference/examples.md](reference/examples.md)
- 分析模板：[templates/analysis_template.md](templates/analysis_template.md)

## NPU专用代码生成规范

本技能采用NPU专用迁移策略，将GPU实现迁移为NPU实现。

### 核心原则

1. **输出文件位置**：在同目录下新建文件（如`xxx_optimized.py`或`xxx_npu.py`）
2. **函数名保持一致**：所有函数名与原始GPU版本保持一致，不添加`_npu`后缀
3. **注释全英文**：所有新增注释使用英文

### 代码转换规则

#### 1. 函数命名规范

**错误示例**：
```python
# ❌ 不要添加_npu后缀
def _layer_norm_fwd_1pass_kernel_npu(...):
    ...
```

**正确示例**：
```python
# ✅ 保持原始函数名
def _layer_norm_fwd_1pass_kernel(...):
    # NPU optimized implementation
    ...
```

#### 2. 注释转换规范

| 中文注释 | 英文注释 |
|---------|---------|
| `# NPU优化版本` | `# NPU optimized version` |
| `# NPU支持` | `# NPU support` |
| `# NPU任务分发参数` | `# NPU task dispatch parameters` |
| `# 使用1D物理核心网格` | `# Use 1D physical core grid` |
| `# 从task_id重建原始索引` | `# Reconstruct original indices from task_id` |
| `# 计算指针偏移` | `# Calculate pointer offsets` |
| `# 获取设备属性` | `# Get device properties` |

### 完整转换示例

**原始GPU代码（layernorm_gated.py）**：
```python
def _layer_norm_fwd_1pass_kernel(...):
    # GPU kernel implementation
    row = tl.program_id(0)
    group = tl.program_id(1)
    ...

def _layer_norm_fwd(...):
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](...)
```

**优化后NPU代码（layernorm_gated_optimized.py）**：
```python
# NPU support
import torch_npu
import triton.runtime.driver as driver

def get_npu_properties():
    """Get NPU device properties, including number of cores"""
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

def _layer_norm_fwd_1pass_kernel(..., 
    # NPU task dispatch parameters
    ngroups_step: tl.constexpr,
    task_num: tl.constexpr,
    num_core: tl.constexpr,
):
    # NPU optimization: Use 1D physical core grid with task dispatch
    core_id = tl.program_id(0)
    
    for task_id in tl.range(core_id, task_num, num_core):
        # Reconstruct original 2D indices from task_id
        row = task_id // ngroups_step
        group = task_id % ngroups_step
        
        # Calculate pointer offsets
        X_ptr = X + row * stride_x_row + group * N
        # ... kernel logic

def _layer_norm_fwd(...):
    # NPU optimization: Use 1D physical core grid
    npu_props = get_npu_properties()
    num_core = npu_props["num_vectorcore"]
    grid = (num_core,)
    
    _layer_norm_fwd_1pass_kernel[grid](...)
```

### 转换检查清单

在生成NPU专用代码时，确保：

- [ ] 输出文件在同目录下新建
- [ ] 所有函数名与原始GPU版本一致（无`_npu`后缀）
- [ ] 移除了`with torch.cuda.device(...)`上下文
- [ ] 所有中文注释已转换为英文
- [ ] 设备属性使用`num_vectorcore`而非`multi_processor_count`
- [ ] Grid使用1D物理核心网格`(num_core,)`
- [ ] 添加了任务分发参数（`ngroups_step`, `task_num`, `num_core`）
- [ ] `torch.cuda.is_available()` 替换为 `torch.npu.is_available()`
- [ ] `torch.cuda.empty_cache()` 替换为 `torch.npu.empty_cache()`
- [ ] `torch.cuda.synchronize()` 替换为 `torch.npu.synchronize()`
- [ ] `torch.cuda.mem_get_info()` 替换为 `torch.npu.mem_get_info()`
- [ ] `device="cuda"` 替换为 `device="npu"`
- [ ] 删除所有 `@torch.compile` 装饰器
