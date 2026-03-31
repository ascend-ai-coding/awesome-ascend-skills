# 故障排查指南

## 常见错误及解决方案

### 1. coreDim超限问题

**错误信息**:
```
coreDim=xxxx can't be greater than UINT16_MAX
```

**原因**: grid维度或大小超过NPU限制（65535）

**解决方案1 - 设置环境变量**:
```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

**解决方案2 - 增大BLOCK_SIZE**:
```python
N = x.numel()
min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
BLOCK_SIZE = max(32768, min_block_size)
```

**解决方案3 - 使用固定核数**:
```python
# 获取NPU核数
device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_AICORE = properties["num_aicore"]

# 使用固定核数
grid = (NUM_AICORE,)
```

### 2. UB空间溢出

**错误信息**:
```
ub overflow, requires xxxx bits while 1572684 bits available!
```

**原因**: 单次计算数据量超过UB空间（约192KB）

**解决方案 - 使用子块切分**:
```python
@triton.jit
def kernel_func(..., BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    base_offset = pid * BLOCK_SIZE
    
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < n_elements
        
        data = tl.load(input_ptr + offsets, mask=mask, care_padding=False)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)

# 调用
BLOCK_SIZE = 32768
BLOCK_SIZE_SUB = 8192  # 尽量用满片上缓存
kernel_func[grid](..., BLOCK_SIZE, BLOCK_SIZE_SUB)
```

### 3. 精度问题（NaN/Inf）

**错误现象**: 输出包含NaN或Inf值

**可能原因及解决方案**:

#### 原因1: 逻辑运算符使用错误
```python
# ❌ 错误：使用 and/or
mask = mask1 and mask2

# ✅ 正确：使用位运算符 &/|
mask = mask1 & mask2
```

#### 原因2: tl.load的other值选择不当
```python
# ❌ 错误：other=0 可能误加载
indices = tl.load(index + offsets, mask=mask, other=0)

# ✅ 正确：other=N 使无效索引超出范围
indices = tl.load(index + offsets, mask=mask, other=N)
```

#### 原因3: mask使用错误
```python
# ❌ 错误：tl.store使用了tl.load的mask
final_mask = index_valid_mask[None, :]
selected = tl.load(inp + inp_off, mask=final_mask, other=0.0)
tl.store(out + out_off, selected, mask=final_mask)  # 错误！

# ✅ 正确：tl.store使用输出边界mask
out_mask = rows_mask & cols_mask[None, :]
final_mask = out_mask & index_valid_mask[None, :]
selected = tl.load(inp + inp_off, mask=final_mask, other=0.0)
tl.store(out + out_off, selected, mask=out_mask)  # 正确！
```

#### 原因4: BLOCK_SIZE过大
```python
# 尝试减小BLOCK_SIZE
BLOCK_SIZE = 32  # 从64或128减小
```

### 4. 编译错误

**错误信息**:
```
compilation failed
```

**检查项**:
1. Triton语法兼容性
2. 是否使用了NPU不支持的特性
3. 依赖的外部库是否存在

**解决方案**:
```python
# 移除GPU特有的代码
# DEVICE = triton.runtime.driver.active.get_active_torch_device()  # 删除

# 移除外部依赖
# from flag_gems import runtime  # 删除，自行实现
```

### 5. 性能问题

**现象**: 运行速度慢

**优化方案**:

#### 优化1: 添加care_padding=False
```python
x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
```

#### 优化2: 使用Tiling策略
```python
# 控制单次计算数据量
BLOCK_SIZE = 32768
SUB_BLOCK_SIZE = 8192
```

#### 优化3: 避免int64
```python
# ❌ int64导致scalar退化
position = tl.arange(block_num)  # 输出int64

# ✅ 转换为int32
position = tl.arange(block_num)
positions = positions.to(tl.int32)
```

## 调试技巧

### 1. 分步验证
```python
# Step 1: 验证基础运行
try:
    result = kernel_npu(**test_inputs)
    print("✅ 基础运行成功")
except Exception as e:
    print(f"❌ 运行失败: {e}")
    return

# Step 2: 检查NaN/Inf
has_nan = torch.isnan(result).any().item()
has_inf = torch.isinf(result).any().item()
print(f"NaN: {has_nan}, Inf: {has_inf}")

# Step 3: 对比精度
torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
```

### 2. 打印中间结果
```python
# 在kernel中打印（仅用于调试）
print("pid:", pid)  # Triton支持print调试
```

### 3. 简化问题
```python
# 使用最小数据集测试
x = torch.randn(16, 16, device='npu')  # 小数据集
```

## 错误诊断流程

```
1. 运行基础测试
   ↓
2. 检查错误类型
   ↓
3. 根据错误类型查找解决方案
   ├── 编译错误 → 检查语法、依赖
   ├── coreDim超限 → 增大BLOCK_SIZE或固定核数
   ├── UB溢出 → 使用子块切分
   ├── 精度问题 → 检查mask、other值、运算符
   └── 性能问题 → 优化内存访问、Tiling
   ↓
4. 验证修复效果
   ↓
5. 迭代优化
```
