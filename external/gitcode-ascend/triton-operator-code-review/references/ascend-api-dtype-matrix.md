# Ascend Triton API 数据类型支持矩阵

来源：官方文档 + 实际测试用例使能情况验证

## 符号说明

- ✓ 支持（测试用例使能）
- × 不支持
- ✓* bool 内部转为 int8 计算
- ⚠ 有条件支持（见备注）

## 静态检视重点 Op

以下是检视时最常遇到、最容易出错的 Op 数据类型约束：

### tl.dot（矩阵乘法）

| int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 |
|------|-------|-------|-------|------|------|------|
| ✓ | × | × | × | ✓ | ✓ | ✓ |

- **累加器类型**：浮点用 `tl.float32`，int8 用 `tl.int32`
- `dot_scaled` 完全不支持

### tl.arange

| int32 |
|-------|
| ✓ |

**仅支持 int32**，返回值为 int32。

### permute / trans

| int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
|------|-------|-------|-------|------|------|------|------|
| ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ⚠ |

- 不支持 int64
- 3D (2,1,0) 不相邻轴转置：pytest_ut 使能，generalization_cases 注释掉（"not support yet: need bisheng support later"），使用时需注意兼容性
- bool：trans 不支持

### gather

| fp16 | fp32 | bf16 |
|------|------|------|
| ✓ | ✓ | ✓ |

- generalization_cases 中 `tl.gather` 支持 axis 0~4（多轴）
- pytest_ut 中 `extension.gather` 标记 skip（"waiting for the compiler to support"）

### sort

| int8 | int16 | fp16 | fp32 | bf16 | bool |
|------|-------|------|------|------|------|
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

- generalization_cases 使用 `tl.sort`（1D 排序，支持 1D~5D shape）
- pytest_ut 使用 `extension.sort`（支持多维 dim 参数）

### Atomic Ops

| Op | int8 | int16 | int32 | uint32 | int64 | fp16 | fp32 | bf16 |
|----|------|-------|-------|--------|-------|------|------|------|
| atomic_add | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| atomic_cas | × | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| atomic_max/min | ✓ | ✓ | ✓ | × | × | ✓ | ✓ | ✓ |

- `atomic_or/xor/and/xchg/cas` **不支持在 loop 中使用**
- `atomic_add` 不支持多核 add + 保存中间结果

## 完整支持矩阵（参考）

### Creation Ops

| Op | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
|----|------|-------|-------|-------|------|------|------|------|
| arange | × | × | ✓ | × | × | × | × | × |
| cat/full/zeros | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cast | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Memory Ops

| Op | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
|----|------|-------|-------|-------|------|------|------|------|
| load/store | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| make_block_ptr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |

### Math Ops

| Op 类别 | int 系列 | fp16 | fp32 | bf16 |
|---------|----------|------|------|------|
| add/sub/mul/div | ✓ | ✓ | ✓ | ✓ |
| cos/sin/exp/log/sigmoid | × | ✓ | ✓ | ✓ |
| sqrt/rsqrt/fma | × | ✓ | ✓ | ✓ |

### Reduction Ops

| Op | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 |
|----|------|-------|-------|-------|------|------|------|
| sum/max/min | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| argmax/argmin | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| reduce | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Scan Ops

| Op | int8 | int16 | int32 | fp16 | fp32 | bf16 |
|----|------|-------|-------|------|------|------|
| associative_scan | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| cumsum/cumprod | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 精度验证参考容差

| 数据类型 | rtol | atol |
|----------|------|------|
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-3 | 1e-3（转 float32 后比较） |
| float32 | 1e-4 | 1e-4 |
| 整数/bool | exact match | exact match |
