# Broadcast 族 — TilingData 字段语义

> Broadcast 族各分支（OneDim / UB-Broadcast / NDDMA-Broadcast）的 TilingData 字段并集。

---

## 1. 通用基础字段（所有分支必备）

| 字段 | 类型 | 含义 | 推导出处 |
|------|------|------|---------|
| `output_shape` | int[] | 合轴后输出shape | DimensionCollapse |
| `input_shapes` | int[][] | 合轴后各输入shape | DimensionCollapse |
| `input_strides` | int[][] | 合轴后各输入strides (broadcast轴=0) | 右乘累加 |
| `scalar_flags` | bool[] | 各输入是否为标量 (所有dim=1) | 合轴后判定 |
| `shape_len` | int | 合轴后总维度数 | — |
| `branch` | str | 所选分支: onedim/ub_broadcast/nddma_broadcast/dynamic_ub | 决策链 |
| `core_num` | int | 实际使用的AIC核数 | 多核均分计算 |

## 2. UB切分字段

| 字段 | 含义 | 出现条件 | 推导 |
|------|------|---------|------|
| `ub_split_axis` | UB切分所在的轴索引 | 多维分支 | 从内向外累乘找到放不下的轴 |
| `ub_former` | UB切分轴上每次处理的元素数 | 所有分支 | `maxElemNum / innerProduct` |
| `ub_outer` | UB切分轴上的tile数 | 所有分支 | `ceil(dim[ubSplitAxis] / ubFormer)` |
| `ub_tail` | UB切分轴上最后一个tile的大小 | 所有分支 | `dim - (ubOuter-1)*ubFormer` |
| `max_elem_num` | UB可容纳的最大元素数 | 所有分支 | `(ubSize-extraSize)*8 / (bufferNum*maxDtypeBits)` |

## 3. 多核切分字段

| 字段 | 含义 | 推导 |
|------|------|------|
| `fused_product` | ubSplitAxis及其外层轴展平的tile总数 | `ubOuter × outerDims` |
| `block_former` | 每个虚拟block的tile数 | `ceil(fusedProduct / coreNum)` |
| `block_num` | 虚拟block总数 | `ceil(fusedProduct / blockFormer)` |
| `block_tail` | 最后一个block的tile数 | `fusedProduct - (blockNum-1)*blockFormer` |

## 4. NDDMA专属字段

| 字段 | 含义 | 取值规则 |
|------|------|---------|
| `sch_mode` | NDDMA调度模式 | 1=WithoutLoop (≤5轴), 2=WithLoop (>5轴) |

## 5. 分支—字段交叉表

| 字段 | OneDim | UB-Broadcast | NDDMA | Dynamic-UB |
|------|:------:|:----------:|:-----:|:----------:|
| output_shape/input_shapes/strides | ✅ | ✅ | ✅ | ✅ |
| scalar_flags | ✅ | ➖ | ➖ | ➖ |
| ub_split_axis | ➖ (固定0) | ✅ | ✅ | ✅ |
| ub_former/ub_outer/ub_tail | ✅ | ✅ | ✅ | ✅ |
| max_elem_num | ✅ | ✅ | ✅ | ✅ |
| block_former/block_num/block_tail | ✅ | ✅ | ✅ | ✅ |
| sch_mode | ➖ | ➖ | ✅ | ➖ |

## 6. 对齐规则

| 场景 | 对齐粒度 | 说明 |
|------|---------|------|
| OneDim | 128B (CACHE_LINE) | `(ubSize/bufferNum / 128) * 128 / dtype_bytes` |
| 多维 | 256B (REPEAT) | `maxElemNum对齐到 256*8/minDtypeBits` |
| NDDMA blockLen | 32B | DataCopy blockLen对齐 |
