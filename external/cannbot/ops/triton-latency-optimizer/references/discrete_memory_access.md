# 离散访存优化模式

## 概述

在 Triton NPU kernel 中，当线程通过非连续或不可预测的索引向量访问全局内存时，会导致访存效率低下，显著降低带宽利用率。先将整块数据读取到UB（Unified Buffer），再取非连续或不可预测的索引可以显著提升计算效率。

## 触发条件

**当代码中存在以下场景时，应考虑应用此优化**：

1. **存在访存语句**：使用了`tl.load`/ `tl.store`读写了全局内存
2. **访存的索引为随机向量/矩阵**：当且仅当访存地址序列无法被静态解析为连续/步长/分块连续模式时，判定为随机访存，注意如果读取单个索引标量并不会引起离散访存，关于随机性的判定如下：

```
┌─────────────────┬─────────────────────────────────────┬──────────────┐
│ 索引来源类型     │ 示例代码                              │ 随机性判定   │
├─────────────────┼─────────────────────────────────────┼──────────────┤
│ 程序ID线性变换   │ pid * BLOCK + arange                 │ 确定性连续   │
│                 │ tl.program_id(0) * 256 + tl.arange   │             │
├─────────────────┼─────────────────────────────────────┼──────────────┤
│ 循环变量线性     │ for i in range(N): ptr + i * stride  │ 确定性步长   │
│                 │                                      │             │
├─────────────────┼─────────────────────────────────────┼──────────────┤
│ 内存加载值/      │ idx = tl.load(indices_ptr + offset)  │ ⚠️ 潜在随机 │
│ kernel入参      │ val = tl.load(data_ptr + idx)        │ 需检查来源   │
└─────────────────┴─────────────────────────────────────┴──────────────┘
```

## 实验性优化方法

下面的优化方法仅能在特定软件配套版本下成功生效，如果尝试后发生报错请跳转到##基于普通接口的优化方法

### 随机读优化方法
随机读可以尝试使用`gather_out_to_ub`接口替换，该接口功能为：根据索引张量（index）从全局内存（Global Memory, GM）的源张量（src）采集数据到统一缓冲区（Unified Buffer, UB） 中。该函数输入：

| 参数名             | 类型                        | 是否必填 | 说明                                   |
| --------------- | ------------------------- | ---- | ------------------------------------ |
| src             | tl.tensor（指针类型）           | 是    | 全局内存（GM）中的源张量指针，数据将从此张量中采集           |
| index           | tl.tensor                 | 是    | UB 中的索引，表示待采集的原始索引集合                 |
| index\_boundary | int64                     | 是    | 索引值的上边界，超过该边界的索引视为越界                 |
| dim             | int32                     | 是    | 采集操作沿源张量的维度，需满足0 ≤ dim < index.rank  |
| src\_stride     | Tuple\[int64]             | 是    | 源张量各维度的步长（stride）                    |
| end\_offset     | Tuple\[int32]             | 是    | 索引张量的每个维度的结束偏移量                      |
| start\_offset   | Tuple\[int32]             | 是    | 索引张量的每个维度的起始偏移量                      |
| other           | Optional\[numbers.Number] | 否    | 索引越界时的兜底填充值（标量），默认值为None（需确保越界时显式指定） |

输出：
* 类型：tensor
* 形状：形状与 index 一致
* 数据类型：与源张量相同
* 内存位置：统一缓冲区（UB）

支持dtype：fp16	fp32 bf16
Shape 支持：仅支持 1~5维 tensor

特殊限制说明：
* 维度一致性：src 与index的秩（rank）必须相同。
* 数据类型限制：src 仅支持 float16、bfloat16、float32 三种浮点类型；index 必须为整数类型张量。
* 索引张量约束：index 的秩需在 1~5 之间；dim 需为有效维度（0 ≤ dim < index_tile.rank）。
* 维度尺寸约束：对于非采集维度（i ≠ dim），原始索引张量的尺寸index_shape[i]不得超过源张量对应维度尺寸src.shape[i]。
* 兜底值约束：other 必须为标量（非张量），若存在越界场景则必须显式指定。

下面是该接口替换tl.load的使用例：

#### 原始代码（随机访存）

```python
index = tl.load(index_ptr + y0_local*2 + x1_local, mask) # idx是一个完全无法预测的随机向量
val = tl.load(src_ptr + index, mask=mask)  # 直接从global中离散访问取数
```

#### 优化后代码（整块访存+访存）

```python
	# Load index tile to UB
	index = tl.load(index_ptr + y0_local*2 + x1_local, mask)

	# Call gather_out_to_ub: gather values from src along dim=0
	val = gather_out_to_ub(
		src=src_ptr,
		index=index,
		index_boundary=4,
		dim=0,
		src_stride=(2, 1),
		end_offset=(2, 2),
		start_offset=(0, 0)
	)
```

### 随机写优化方法

随机写可以尝试使用`scatter_ub_to_out`接口替换，该接口功能为：将统一缓冲区（Unified Buffer, UB） 中的数值张量（value）根据索引张量（index）沿目标张量的指定维度（dim），分散存储到全局内存（Global Memory, GM）的目标张量（ptr）中。该函数输入：

| 参数名             | 类型              | 是否必填 | 说明                                   |
| --------------- | --------------- | ---- | ------------------------------------ |
| ptr             | tl.tensor（指针类型） | 是    | 全局内存（GM）中的目标张量指针，数据将分散写入此张量中         |
| value           | tl.tensor       | 是    | UB 中的值，表示待分散写入目标张量的数值集合              |
| index           | tl.tensor       | 是    | UB 中的索引，表示待分散操作的原始索引集合               |
| index\_boundary | int64           | 是    | 索引值的上边界，用于索引边界检查，确保分散操作的索引有效性        |
| dim             | int32           | 是    | 分散操作沿目标张量的维度，需满足0 ≤ dim < index.rank |
| dst\_stride     | Tuple\[int64]   | 是    | 目标张量各维度的步长（stride）                   |
| end\_offset     | Tuple\[int32]   | 是    | 索引张量的每个维度的结束偏移量                      |
| start\_offset   | Tuple\[int32]   | 是    | 索引张量的每个维度的起始偏移量                      |

返回值：无返回值

支持dtype：fp16	fp32 bf16
Shape 支持：仅支持 1~5维 tensor

特殊限制说明：
* 维度一致性：ptr、value 与index的秩（rank）必须相同。
* 数据类型限制：ptr 仅支持 float16、bfloat16、float32 三种浮点类型；index 必须为整数类型张量。
* 索引张量约束：index 的秩需在 1~5 之间；dim 需为有效维度（0 ≤ dim < index.rank）。
* 维度尺寸约束：对于非分散维度（i ≠ dim），索引分片的尺寸index.size[i]不得超过目标张量对应维度尺寸ptr.size[i]。

下面是该接口替换tl.store的使用例：

#### 原始代码（随机访存）

```python
index = tl.load(index_ptr + y0_local*2 + x1_local, mask) # offset是一个完全无法预测的随机标量
tl.store(dst_ptr + index, value)  # 直接从global中离散访问取数
```

#### 优化后代码（整块访存+访存）

```python
	index = tl.load(index_ptr + y0_local*2 + x1_local, mask)

	scatter_ub_to_out(
		ptr=dst_ptr,
		value=value,
		index=index,
		index_boundary=4,
		dim=0,
		dst_stride=(2, 1),
		end_offset=(2, 2),
		start_offset=(0, 0)
	)
```

### 关键点

1. **识别无法预测的随机值**：溯源访存命令的索引计算过程，找到是否有随机值风险
2. **替换优化**：将 `tl.load`的替换为`gather_out_to_ub`，将`tl.store`的替换为`scatter_ub_to_out`

## 基于普通接口的优化方法

### 原始代码（随机访存）

```python
offset = tl.load(offset_ptr) # offset是一个完全无法预测的随机标量
idx = tl.load(idx_ptr + rn * stride_idx) # idx是一个完全无法预测的随机向量
val = tl.load(x_ptr + offset + idx * stride_x, mask=mask)  # 直接从global中离散访问取数
```

### 优化后代码（整块访存+访存）

```python
offset = tl.load(offset_ptr) # offset是一个完全无法预测的随机标量
idx = tl.load(idx_ptr + rn * stride_idx) # idx是一个完全无法预测的随机值向量
rm = tl.arange(0, M) # rm包含了所有的值，M为x张量的总长度
x_shared = tl.load(x_ptr + offset + rm * stride_x) # 将x对应偏移的所有数据从global搬至UB
val = tl.gather(x_shared.to(tl.float16), idx, 0).to(tl.int32)  # 再从UB中select目标值，注意数据类型的切换
```

### 关键点

1. **识别无法预测的随机值**：溯源`tl.load`的输入索引计算过程，找到是否有无法预测的随机值，例如被`tl.load`读进来的值
2. **自动优化**：将 `tl.load`的输入指针中的随机值剔除，改为读取一大块内存（注意不能超过UB限制），然后使用`tl.gather`输入随机值，得到最终需要取的值
3. **注意gather的数据类型**：`tl.gather`不支持`int类型`，最好将输入强转成`tl.float16`再执行`tl.gather`,最后再转回原有的数据类型。如果提示精度报错，可以尝试强转成`tl.float32`。

### 模式 2：循环内通过随机索引访问小查找表

**问题代码（极易误判）**：

```python
for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
    mask = offsets < (topk_numel - x * BLOCK_SIZE)
    expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)
    if HAS_EXPERT_MAP:
        # 错误：expert_ids 来自 tl.load，是运行时随机值
        # expert_map + expert_ids 产生随机全局内存访问，每次循环迭代都可能跨步
        expert_map_ptrs = expert_map + expert_ids
        expert_ids = tl.load(expert_map_ptrs, mask=mask, other=-1)
    ...
```

**优化后代码**：

```python
# 循环外一次性将 expert_map（小查找表）加载到 UB
expert_map_data = tl.load(expert_map + tl.arange(0, num_experts)).to(tl.float32)
for x in range(cntx):
    mask = offsets < (topk_numel - x * BLOCK_SIZE)
    e_ids = tl.load(topk_ids_ptr + x * BLOCK_SIZE + offsets, mask=mask, other=0)
    # 对 UB 中的局部 buffer 使用 gather，替代全局离散加载
    expert_ids = tl.gather(expert_map_data, e_ids, 0)
    ...
```

**关键差异**：
- 原始版本：每次循环迭代执行 `tl.load(expert_map + expert_ids)`，索引 `expert_ids` 随机，导致全局离散访存
- 优化版本：循环外一次 `tl.load` 将 expert_map 整体搬到 UB，循环内用 `tl.gather` 从 UB 局部查找

**识别要点**：当代码出现 `tl.load(small_table + tl.load(random_indices))` 这种"双层加载-索引"模式时，几乎总是可以通过"预加载 + gather"优化。


## 性能收益

先将整块数据读取到share memory，再取非连续或不可预测的索引可在 NPU 上获得显著的性能提升。

## 风险提示

| 风险 | 说明 | 缓解措施 |
|------|------|----------|
| 精度下降 | `tl.gather`输入强转成`tl.float16`可能会丢失精度 | 尝试将`tl.gather`输入强转成`tl.float32`或者放弃优化 |
