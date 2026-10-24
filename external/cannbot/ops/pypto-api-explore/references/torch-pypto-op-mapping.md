# Torch ↔ Pypto 算子对标手册

## 同名映射

仅含 `pypto/op/` 下的内置原子接口；`sigmoid` / `softmax` / `rms_norm` 为 `pypto/operator.py` 前端组合算子，不属于原子接口，见「组合方案」。

### 同名同参

以下 Torch 算子与 Pypto 原子接口同名且核心参数一致，直接以 `pypto.{op}` 调用（双方的可选修饰参数不视为差异）：

| 分组 | 算子 |
|------|------|
| 逐元素（单输入） | `abs` `acos` `acosh` `asin` `asinh` `atan` `atanh` `bitwise_not` `ceil` `clone` `cos` `cosh` `erf` `erfc` `exp` `exp2` `expm1` `floor` `isfinite` `isnan` `log` `log1p` `log2` `log10` `logical_not` `neg` `reciprocal` `relu` `rsqrt` `sign` `signbit` `sin` `sinh` `sqrt` `tan` `tanh` `trunc` |
| 逐元素（双输入） | `add` `atan2` `bitwise_and` `bitwise_left_shift` `bitwise_or` `bitwise_right_shift` `bitwise_xor` `div` `eq` `fmod` `gcd` `ge` `gt` `hypot` `le` `logical_and` `lt` `maximum` `minimum` `mul` `ne` `pow` `remainder` `sub` `where` |
| 归约 | `amax` `amin` `cumprod` `cumsum` `var` |
| 索引 | `gather` `index_add` `index_add_` `index_put_` `index_select` `scatter` `scatter_` |
| 形状 | `concat` `permute` `reshape` `transpose` `unsqueeze` |
| 排序/裁剪 | `argsort` `clip` |
| 创建 | `arange` `full` `ones` `zeros` |
| 特殊 | `one_hot`（torch 侧为 `torch.nn.functional.one_hot`） `prelu` `tril` `triu` |

### 同名不同参

名字相同但核心参数存在差异，调用时按下表适配：

| 算子 | 参数差异 |
|------|---------|
| `argmax` `argmin` `sum` `prod` | pypto `dim` 必填；torch 可省略 `dim` 做全归约 |
| `topk` | pypto 无 `sorted` 参数，另有 `algo: TopKAlgo` |
| `matmul` | pypto `out_dtype` 必填，另有 `a_trans` / `b_trans` / `c_matrix_nz` / `extend_params`（cube 侧） |
| `round` | pypto `decimals` 必填；torch 默认 `decimals=0` |
| `pad` | torch 侧为 `torch.nn.functional.pad`，其 `pad` 参数在 pypto 名为 `padding` |
| `normal` | 语义不同：torch 从 N(mean, std) 分布采样；pypto 为状态式随机生成 `normal(shape, key, counter, alg, dtype)` |
| `dequantize` | 语义不同：torch `dequantize(tensor)` 为 QTensor 反量化；pypto `dequantize(input, scale, otype, axis, zero_points)` 带 scale/zero_points |
| `scatter_update` | torch 侧实为 torch_npu 接口 `scatter_update(data, indices, updates, axis)`，参数名与 pypto `(input, dim, index, src)` 不同 |

## 命名映射

Torch 算子有对应的 Pypto API，但名字不同。

**纯换名**（参数调用一致，直接替换）：

- `cat` → `concat`：[cat.md](../examples/cat.md)
- `clamp` → `clip`：[clip.md](../examples/clip.md)
- `to`(dtype) → `cast`：[to.md](../examples/to.md)

**差异映射**（单 API 对应，但语义或参数存在差异，调用时需换算）：

- `max`（归约） → `amax`：torch 返回 `(values, indices)` 二元组，`amax` 仅返回值：[max.md](../examples/max.md)
- `mul_` → `mul`：无 inplace 语义：[mul_.md](../examples/mul_.md)
- `expand` → `expand_clone`：实际分配内存并复制，非视图：[expand.md](../examples/expand.md)
- `type_as` → `cast`：需先取 `other.dtype` 再 `cast`：[type_as.md](../examples/type_as.md)
- `sort` → `sort32` + `mrgsort`：两个专用 HW 算子接力：[sort.md](../examples/sort.md)
- `flatten` → `reshape`：需按轴合并换算目标 shape：[flatten.md](../examples/flatten.md)
- `squeeze` → `reshape` / `view`：需定位 size=1 轴后换算 shape：[squeeze.md](../examples/squeeze.md)
- `t` → `transpose`：`t()` 等价于 `transpose(input, 0, 1)`：[t.md](../examples/t.md)
- `expand_as` → `expand_clone`：需取 `other.shape` 作为目标：[expand_as.md](../examples/expand_as.md)
- `movedim` → `permute`：需把 source/destination 换算为完整 dims 序列：[movedim.md](../examples/movedim.md)
- `split` / `chunk` → `view`（切片）：需按份数计算各片偏移：[split.md](../examples/split.md)、[chunk.md](../examples/chunk.md)

## 组合方案

Torch 算子需要通过多个 Pypto 算子组合实现。

### 形状变换

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `unbind` | `view` + `reshape` | [unbind.md](../examples/unbind.md) |
| `stack` | `unsqueeze` + `concat` | [stack.md](../examples/stack.md) |
| `repeat` | `unsqueeze` + `expand_clone` + `reshape` | [repeat.md](../examples/repeat.md) |
| `repeat_interleave` | `unsqueeze` + `expand_clone` + `reshape` | [repeat_interleave.md](../examples/repeat_interleave.md) |

### 数学运算

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `outer` | `unsqueeze` + `mul` | [outer.md](../examples/outer.md) |

### 归约操作

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `mean` | `sum` + `div` | [mean.md](../examples/mean.md) |
| `norm` | `mul` + `sum` + `sqrt` | [norm.md](../examples/norm.md) |
| `all` | `where` + `sum` + `gt` | [all.md](../examples/all.md) |
| `any` | `where` + `sum` + `gt` | [any.md](../examples/any.md) |

### 矩阵运算

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `nn.Linear` / `linear` | `matmul` + `add` | [linear.md](../examples/linear.md) |
| `nn.Conv2d` | `conv` + TileL1/L0 | [conv2d.md](../examples/conv2d.md) |
| `nn.Conv3d` | `conv` + TileL1/L0 | [conv3d.md](../examples/conv3d.md) |
| `scaled_dot_product_attention` | `matmul` + `amax` + `sub` + `exp` + `sum` + `div` | [attention.md](../examples/attention.md) |

### 激活函数

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `sigmoid` | `cast` + `exp` + `add` + `div` | [sigmoid.md](../examples/sigmoid.md) |
| `softmax` | `amax` + `sub` + `exp` + `sum` + `div` | [softmax.md](../examples/softmax.md) |
| `silu` | `exp` + `add` + `div` + `mul` | [silu.md](../examples/silu.md) |
| `softplus` | `exp` + `add` + `log` | [softplus.md](../examples/softplus.md) |
| `nn.GELU` | `mul` + `tanh` + `add` | [gelu_tanh.md](../examples/gelu_tanh.md) |
| `glu` | `view` + `exp` + `add` + `div` + `mul` | [glu.md](../examples/glu.md) |

### 索引/查表

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `nn.Embedding` | `gather` + padding | [embedding.md](../examples/embedding.md) |

### 归一化

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `nn.LayerNorm` | `sum` + `div` + `sub` + `mul` + `sqrt` | [layer_norm.md](../examples/layer_norm.md) |

### 索引操作

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `masked_fill` | `gt` + `where` | [masked_fill.md](../examples/masked_fill.md) |
| `masked_fill_` | `gt` + `where` | [masked_fill_inplace.md](../examples/masked_fill_inplace.md) |
| `masked_scatter` | `where` + `cumsum` + `cast` + `gather` | [masked_scatter.md](../examples/masked_scatter.md) |

### 特殊操作

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `roll` | `index_select` + `concat` | [roll.md](../examples/roll.md) |
| `diff` | `view` + `sub` | [diff.md](../examples/diff.md) |
| `bincount` | `one_hot` + `sum` | [bincount.md](../examples/bincount.md) |

### 张量创建

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `eye` | `arange` + `one_hot` + `cast` | [eye.md](../examples/eye.md) |
| `linspace` | `arange` + `mul` | [linspace.md](../examples/linspace.md) |

### 其他

| Torch 算子 | Pypto 组合方案 | 参考实现 |
|-----------|---------------|---------|
| `rope` | `view` + `neg` + `concat` + `mul` + `add` | [rope.md](../examples/rope.md) |
| `l2_norm` | `mul` + `sum` + `rsqrt` | [l2_norm.md](../examples/l2_norm.md) |
| `moe_routing` | `matmul` + `exp` + `add` + `div` + `topk` + `sum` | [moe_routing.md](../examples/moe_routing.md) |
