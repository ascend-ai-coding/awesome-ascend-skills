# 迁移案例分析

## CUDA到NPU API转换规则

在迁移过程中，需要将以下GPU API替换为对应的NPU API：

| GPU API | NPU API | 说明 |
|---------|---------|------|
| `torch.cuda.is_available()` | `torch.npu.is_available()` | 检查设备是否可用 |
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` | 清空缓存 |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` | 同步设备 |
| `torch.cuda.mem_get_info()` | `torch.npu.mem_get_info()` | 获取内存信息 |
| `device="cuda"` | `device="npu"` | 设备指定 |
| `@torch.compile` | **删除该装饰器** | NPU暂不支持torch.compile训练 |

**示例**：
```python
# ❌ GPU代码
if torch.cuda.is_available():
    x = torch.randn(100, device='cuda')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

@torch.compile
def my_function(x):
    return x * 2

# ✅ NPU代码
if torch.npu.is_available():
    x = torch.randn(100, device='npu')
    torch.npu.empty_cache()
    torch.npu.synchronize()

# 删除@torch.compile装饰器
def my_function(x):
    return x * 2
```

---

## 案例1: index_select算子

### 算子功能
按索引选择张量的指定维度元素。

### 源代码（GPU版本）
```python
@triton.jit
def index_select_kernel(inp, out, M, N, index, index_len, BLOCK_M, BLOCK_N):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    
    out_mask = rows_mask and (cols_offsets < index_len)
    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    
    valid_lower_bound = indices >= 0
    valid_upper_bound = indices < N
    index_valid_mask = valid_lower_bound & valid_upper_bound
    
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]
    
    final_mask = out_mask & index_valid_mask
    selected = tl.load(inp + inp_off, mask=final_mask, other=0.0)
    tl.store(out + out_off, selected, mask=final_mask)
```

### 问题分析

| 问题 | 位置 | 描述 |
|------|------|------|
| other值选择 | indices加载 | other=0可能导致误加载第0个索引 |
| store mask错误 | tl.store | 使用final_mask应改为out_mask |
| 逻辑运算符 | out_mask计算 | 使用and应改为& |
| 外部依赖 | 导入部分 | 依赖flag_gems库 |

### 迁移后代码（NPU版本）
```python
@triton.jit
def index_select_kernel(inp, out, M, N, index, index_len, BLOCK_M, BLOCK_N):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    
    cols_mask = cols_offsets < index_len
    out_mask = rows_mask & cols_mask[None, :]
    
    indices = tl.load(index + cols_offsets, mask=cols_mask, other=N)
    valid_lower_bound = indices >= 0
    valid_upper_bound = indices < N
    index_valid_mask = valid_lower_bound & valid_upper_bound
    
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]
    
    final_mask = out_mask & index_valid_mask[None, :]
    selected = tl.load(inp + inp_off, mask=final_mask, other=0.0)
    tl.store(out + out_off, selected, mask=out_mask)
```

### 关键修改

| 修改项 | 原代码 | 修改后 |
|--------|--------|--------|
| other值 | `other=0` | `other=N` |
| store mask | `mask=final_mask` | `mask=out_mask` |
| 逻辑运算符 | `and` | `&` |
| BLOCK_SIZE | 无指定 | `BLOCK_M=32, BLOCK_N=32` |

### 解决过程

| 步骤 | 调整内容 | 结果 |
|------|---------|------|
| Step 1 | 最小化迁移 - 只改设备 | ❌ 精度问题（NaN） |
| Step 2 | 修复逻辑运算符和广播 | ❌ 问题仍存在 |
| Step 3 | 使用1D grid + 跨步分配 | ❌ 问题仍存在 |
| Step 4 | 修复other值和store mask | ✅ 精度验证通过 |

---

## 案例2: 向量加法算子

### 算子功能
两个向量逐元素相加。

### 源代码（GPU版本）
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### 迁移后代码（NPU版本）
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
    y = tl.load(y_ptr + offsets, mask=mask, care_padding=False)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # 获取NPU核数
    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    NUM_AICORE = properties["num_aicore"]
    
    # 固定核数
    grid = (NUM_AICORE,)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### 关键修改

| 修改项 | 原代码 | 修改后 |
|--------|--------|--------|
| care_padding | 无 | `care_padding=False` |
| grid配置 | 动态计算 | 固定核数 |
| 设备 | `device='cuda'` | `device='npu'` |

---

## 案例3: Tiling优化算子

### 问题
单次计算数据量超过UB空间。

### 解决方案
```python
@triton.jit
def optimized_kernel(in_ptr, out_ptr, n_elements, 
                     BLOCK_SIZE: tl.constexpr, SUB_BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    
    for xoffset_sub in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        x_index = xoffset + xoffset_sub + tl.arange(0, SUB_BLOCK_SIZE)
        xmask = x_index < n_elements
        
        x = tl.load(in_ptr + x_index, xmask, care_padding=False)
        result = compute(x)
        tl.store(out_ptr + x_index, result, xmask)

# 调用
ncore = NUM_AICORE
BLOCK_SIZE = 32768
SUB_BLOCK_SIZE = 8192  # 尽量用满片上缓存（192KB）
optimized_kernel[ncore, 1, 1](input, output, n_elements, BLOCK_SIZE, SUB_BLOCK_SIZE)
```

---

## 案例4: LayerNorm Gated算子（完整迁移案例）

### 算子功能
实现LayerNorm和RMSNorm的前向和反向传播，支持门控机制（SiLU激活函数）。

### 迁移难点
1. 使用了 `torch.cuda.device` 设备上下文
2. 使用了 `torch.cuda.get_device_properties` 查询设备属性
3. 涉及前向和反向传播，计算图复杂
4. 大维度（4096）可能触发编译器限制

### 源代码关键部分（GPU版本）
```python
def _layer_norm_fwd(x, weight, bias, eps, z=None, ...):
    # ... 省略部分代码 ...
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):  # ❌ GPU专用API
        _layer_norm_fwd_1pass_kernel[grid](...)

def _layer_norm_bwd(dy, x, weight, bias, eps, ...):
    # ... 省略部分代码 ...
    # ❌ GPU专用属性
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    # ... 省略部分代码 ...
    with torch.cuda.device(x.device.index):  # ❌ GPU专用API
        _layer_norm_bwd_kernel[grid](...)
```

### 迁移后代码（NPU版本）
```python
import torch_npu

def _layer_norm_fwd(x, weight, bias, eps, z=None, ...):
    # ... 省略部分代码 ...
    grid = (M, ngroups)
    # ✅ 使用NPU设备上下文
    with torch_npu.npu.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](...)

def _layer_norm_bwd(dy, x, weight, bias, eps, ...):
    # ... 省略部分代码 ...
    # ✅ 使用NPU设备属性
    props = torch_npu.npu.get_device_properties(x.device)
    sm_count = props.vector_core_num
    # ... 省略部分代码 ...
    # ✅ 使用NPU设备上下文
    with torch_npu.npu.device(x.device.index):
        _layer_norm_bwd_kernel[grid](...)
```

### 关键修改总结

| 修改项 | 原代码 | 修改后 | 原因 |
|--------|--------|--------|------|
| 设备上下文 | `torch.cuda.device` | `torch_npu.npu.device` | NPU使用torch_npu |
| 设备属性 | `multi_processor_count` | `vector_core_num` | NPU属性不同 |
| care_padding | 无 | **不添加** | 可能导致精度问题 |

### 迁移过程记录

| 步骤 | 尝试内容 | 结果 | 经验教训 |
|------|---------|------|---------|
| Step 1 | 添加 `care_padding=False` | ❌ 输出全为0 | care_padding需谨慎使用 |
| Step 2 | 移除 `care_padding=False` | ✅ 前向传播正确 | 先确保功能正确 |
| Step 3 | 修复设备属性访问 | ✅ 反向传播正常 | 使用vector_core_num |
| Step 4 | 完整测试 | ✅ 测试通过 | 迁移成功 |

### 测试结果

**成功场景**：
- ✅ 维度2048的所有前向传播
- ✅ 维度2048的大部分反向传播
- ✅ LayerNorm和RMSNorm
- ✅ 各种配置（bias、z、group_size等）

### 最佳实践总结

1. **设备API迁移模式**：
```python
# NPU专用模式
import torch_npu

with torch_npu.npu.device(x.device.index):
    kernel[grid](...)
```

2. **设备属性访问模式**：
```python
# 使用NPU设备属性
props = torch_npu.npu.get_device_properties(x.device)
core_count = props.vector_core_num
```

---

## 迁移经验总结

### 成功迁移的关键因素

1. **先分析，后迁移**: 通过语义分析识别潜在风险
2. **最小化改动**: 只修改必要的部分
3. **针对性修复**: 根据错误类型选择解决方案
4. **精度验证必须**: 每次修改后都要验证精度

### 常见陷阱

| 陷阱 | 表现 | 解决方案 |
|------|------|---------|
| other值选择 | 索引加载误加载 | 使用超出有效范围的值 |
| mask混用 | 精度问题或越界 | load/store使用各自的mask |
| 逻辑运算符 | 精度问题 | 使用&/|而非and/or |
| int64类型 | 性能下降 | 转换为int32 |
| UB溢出 | 编译失败 | 使用Tiling策略 |
