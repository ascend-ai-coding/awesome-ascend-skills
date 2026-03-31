# Triton算子代码逻辑语义分析报告模板

## 使用说明

在迁移Triton算子前，使用此模板生成详细的语义分析报告，识别潜在风险点。

---

# [算子名称] 代码逻辑语义分析报告

## 1. 算子基本信息

- **算子名称**: [kernel名称]
- **算子功能**: [一句话描述算子的功能]
- **Grid维度**: [1D/2D/3D]
- **特殊特性**: [是否使用tl.dot、原子操作等]

## 2. 输入输出分析

| 变量名 | 类型 | 形状 | 含义 |
|--------|------|------|------|
| [变量1] | [tensor/int] | [形状] | [含义描述] |
| [变量2] | [tensor/int] | [形状] | [含义描述] |
| ... | ... | ... | ... |

## 3. 核心逻辑流程

1. [步骤1描述]
2. [步骤2描述]
3. [步骤3描述]
...

## 4. 内存访问模式分析

### 4.1 tl.load 分析

| 加载操作 | 指针计算 | mask语义 | other值 | 潜在问题 |
|----------|----------|----------|---------|----------|
| [加载操作1] | [指针计算公式] | [mask的含义] | [other值] | [⚠️ 或 ✅] |
| [加载操作2] | [指针计算公式] | [mask的含义] | [other值] | [⚠️ 或 ✅] |

**检查要点**:
- [ ] 指针计算是否正确
- [ ] mask是否检查了所有必要的边界条件
- [ ] other值选择是否合理（特别是索引类操作）

### 4.2 tl.store 分析

| 存储操作 | 指针计算 | mask语义 | 潜在问题 |
|----------|----------|----------|----------|
| [存储操作1] | [指针计算公式] | [mask的含义] | [⚠️ 或 ✅] |
| [存储操作2] | [指针计算公式] | [mask的含义] | [⚠️ 或 ✅] |

**检查要点**:
- [ ] mask是否只检查输出边界
- [ ] 是否与tl.load的mask混用

## 5. Mask逻辑分析

### 5.1 Mask定义

- **[mask名称1]**: [计算逻辑] - [语义含义]
- **[mask名称2]**: [计算逻辑] - [语义含义]
...

### 5.2 Mask使用检查

- [ ] tl.load的mask是否正确检查输入有效性
- [ ] tl.store的mask是否正确检查输出边界
- [ ] mask广播维度是否正确
- [ ] 是否使用了正确的运算符（&/| vs and/or）

### 5.3 Mask关系图

```
[mask1] 
    ↓
[mask2] = [mask1] & [条件]
    ↓
[final_mask] = [mask2] & [其他条件]
```

## 6. 数据流分析

```
输入: [输入变量列表]
    ↓
[处理步骤1]
    ↓
[处理步骤2]
    ↓
输出: [输出变量列表]
```

## 7. 潜在迁移风险点

| 风险类型 | 代码位置 | 问题描述 | 建议修改 | 优先级 |
|----------|----------|----------|----------|--------|
| [风险类型1] | [代码位置] | [问题描述] | [建议修改] | [高/中/低] |
| [风险类型2] | [代码位置] | [问题描述] | [建议修改] | [高/中/低] |

**常见风险类型**:
- other值选择不当
- mask使用错误
- 逻辑运算符错误
- 外部依赖
- int64类型
- UB溢出风险

## 8. 迁移建议

### 8.1 必须修改项

1. **[修改项1]**:
   - 原代码: `[原代码片段]`
   - 修改为: `[修改后代码片段]`
   - 原因: [修改原因]

2. **[修改项2]**:
   - 原代码: `[原代码片段]`
   - 修改为: `[修改后代码片段]`
   - 原因: [修改原因]

### 8.2 建议优化项

1. **[优化项1]**:
   - 建议: [优化建议]
   - 收益: [预期收益]

### 8.3 迁移策略

- [ ] 最小化迁移（只改设备）
- [ ] 根据分析报告预先修复已知问题
- [ ] 其他策略: [描述]

## 9. 验证计划

### 9.1 测试用例

| 测试名称 | 输入规格 | 预期输出 | 验证方法 |
|---------|---------|---------|---------|
| [测试1] | [输入描述] | [预期输出] | [验证方法] |
| [测试2] | [输入描述] | [预期输出] | [验证方法] |

### 9.2 精度要求

- 数据类型: [float16/float32/bfloat16]
- 相对容差: [rtol]
- 绝对容差: [atol]

---

## 分析完成检查清单

- [ ] 已分析所有tl.load操作
- [ ] 已分析所有tl.store操作
- [ ] 已检查所有mask逻辑
- [ ] 已识别所有潜在风险点
- [ ] 已生成迁移建议
- [ ] 已制定验证计划

---

# 示例：index_select_kernel 分析报告

## 1. 算子基本信息

- **算子名称**: index_select_kernel
- **算子功能**: 按索引选择张量的指定维度元素
- **Grid维度**: 2D (rows × cols)
- **特殊特性**: 无

## 2. 输入输出分析

| 变量名 | 类型 | 形状 | 含义 |
|--------|------|------|------|
| inp | tensor | (M, N) | 输入张量（经过dim_compress处理） |
| out | tensor | (M, index_len) | 输出张量 |
| index | tensor | (index_len,) | 索引张量 |
| M | int | - | 压缩后的行数 |
| N | int | - | 压缩后的列数 |
| index_len | int | - | 索引长度 |

## 3. 核心逻辑流程

1. 获取program_id确定当前处理的块位置 (pid_x, pid_y)
2. 计算rows_offsets和cols_offsets确定处理的数据位置
3. 加载索引值indices
4. 验证索引有效性（0 <= indices < N）
5. 计算输入输出偏移量inp_off和out_off
6. 加载选中数据selected
7. 存储结果到输出张量

## 4. 内存访问模式分析

### 4.1 tl.load 分析

| 加载操作 | 指针计算 | mask语义 | other值 | 潜在问题 |
|----------|----------|----------|---------|----------|
| 加载indices | index + cols_offsets | cols < index_len | 0 | ⚠️ other=0可能导致误加载 |
| 加载selected | inp + inp_off | final_mask | 0.0 | ✅ 正常 |

### 4.2 tl.store 分析

| 存储操作 | 指针计算 | mask语义 | 潜在问题 |
|----------|----------|----------|----------|
| 存储selected | out + out_off | final_mask | ⚠️ 应使用out_mask |

## 5. Mask逻辑分析

### 5.1 Mask定义

- **rows_mask**: rows_offsets < M - 行边界检查
- **cols_mask**: cols_offsets < index_len - 列边界检查
- **out_mask**: rows_mask & cols_mask - 输出边界检查
- **index_valid_mask**: indices >= 0 & indices < N - 索引有效性检查
- **final_mask**: out_mask & index_valid_mask - 综合检查

### 5.2 Mask使用检查

- [x] tl.load的mask检查了输入有效性
- [ ] ⚠️ tl.store的mask应使用out_mask而非final_mask
- [ ] ⚠️ mask广播维度需检查

## 6. 数据流分析

```
输入: inp(M,N), index(index_len)
    ↓
indices = load(index + cols_offsets)
    ↓
inp_off = rows * N + indices
    ↓
selected = load(inp + inp_off)
    ↓
输出: store(out, selected)
```

## 7. 潜在迁移风险点

| 风险类型 | 代码位置 | 问题描述 | 建议修改 | 优先级 |
|----------|----------|----------|----------|--------|
| other值选择 | indices加载 | other=0可能误加载 | 改为other=N | 高 |
| store mask | tl.store | 使用final_mask应改为out_mask | mask=out_mask | 高 |
| 逻辑运算符 | out_mask计算 | 使用and应改为& | 使用位运算符 | 中 |
| 外部依赖 | 导入部分 | 依赖flag_gems库 | 自行实现 | 中 |

## 8. 迁移建议

### 8.1 必须修改项

1. **修复other值**:
   - 原代码: `indices = tl.load(index + cols_offsets, mask=cols_mask, other=0)`
   - 修改为: `indices = tl.load(index + cols_offsets, mask=cols_mask, other=N)`
   - 原因: other=0可能导致无效索引误加载第0个元素

2. **修复store mask**:
   - 原代码: `tl.store(out + out_off, selected, mask=final_mask)`
   - 修改为: `tl.store(out + out_off, selected, mask=out_mask)`
   - 原因: tl.store应只检查输出边界

3. **修复逻辑运算符**:
   - 原代码: `out_mask = rows_mask and (cols_offsets < index_len)`
   - 修改为: `out_mask = rows_mask & cols_mask[None, :]`
   - 原因: 应使用位运算符&而非逻辑运算符and

### 8.2 建议优化项

1. **调整BLOCK_SIZE**:
   - 建议: 使用较小的BLOCK_SIZE（如32）
   - 收益: 提高精度稳定性

## 9. 验证计划

### 9.1 测试用例

| 测试名称 | 输入规格 | 预期输出 | 验证方法 |
|---------|---------|---------|---------|
| 基础float32 | (128,64), dim=1 | 正确选择 | torch.index_select对比 |
| float16 | (256,128), dim=1 | 正确选择 | 精度验证 |
| 3D张量 | (16,32,64), dim=1 | 正确选择 | torch.index_select对比 |

### 9.2 精度要求

- 数据类型: float16, float32, bfloat16
- 相对容差: 1e-3 (float16), 1e-4 (float32)
- 绝对容差: 1e-3 (float16), 1e-4 (float32)
