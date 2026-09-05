# API 支持矩阵（API Support Matrix）

## 概述

本文档列出所有 Triton Python API 在 Ascend NPU 上的支持状态。支持状态分为四类：

| 状态 | 标记 | 说明 |
|------|------|------|
| 完全支持 | ✓ | 功能与 GPU 版本一致 |
| 部分支持 | ✓* | 功能可用但有数据类型或使用限制 |
| 不支持 | × | 当前版本不可用 |
| Ascend 扩展 | A | Triton-Ascend 独有扩展功能 |

> 数据类型列：i8=int8, i16=int16, i32=int32, i64=int64, f16=fp16, f32=fp32, bf16=bfloat16, bool=布尔型
> uint8/uint16/uint32/uint64/fp64 在 Ascend 上普遍不支持（硬件限制），不再单独列出

## 内存/指针操作（Memory/Pointer Ops）

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.load` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.store` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | cache_modifier/eviction_policy 对 NPU 无效 |
| `tl.make_block_ptr` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 不允许算术运算；复杂循环+分支可能编译失败 |
| `tl.make_tensor_descriptor` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 需与 load/store_tensor_descriptor 配套使用 |
| `tl.load_tensor_descriptor` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 同上 |
| `tl.store_tensor_descriptor` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 同上 |
| `tl.advance` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 不允许通过调整 stride 顺序实现转置 |

## 数学操作（Math Ops）

### 算术运算

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `+ (add)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `- (sub)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `* (mul)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `/ (div)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `// (floordiv)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓* | - |
| `% (mod)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | int64 仅支持 -2^24 ~ 2^24 |
| `- (neg)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | - |

### 位运算

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `& (and)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `\| (or)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `^ (xor)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `~ (invert)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `! (not)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `<< (lshift)` | ✓ | ✓ | ✓ | ✓ | × | × | × | × | - |
| `>> (rshift)` | ✓ | ✓ | ✓ | ✓ | × | × | × | × | - |

### 比较运算

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `>` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `>=` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `<` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `<=` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `==` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `!=` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |

### 浮点数学函数

| API | f16 | f32 | bf16 | 限制说明 |
|-----|:---:|:---:|:----:|---------|
| `tl.abs` | ✓ | ✓ | ✓ | 整数类型也支持 |
| `tl.ceil` | ✓ | ✓ | ✓ | - |
| `tl.clamp` | ✓ | ✓ | ✓ | - |
| `tl.cos` | ✓ | ✓ | ✓ | - |
| `tl.div_rn` | ✓ | ✓ | ✓ | - |
| `tl.erf` | ✓ | ✓ | ✓ | - |
| `tl.exp` | ✓ | ✓ | ✓ | 近似计算 |
| `tl.exp2` | ✓ | ✓ | ✓ | 近似计算 |
| `tl.fdiv` | ✓ | ✓ | ✓ | - |
| `tl.floor` | ✓ | ✓ | ✓ | - |
| `tl.fma` | ✓ | ✓ | ✓ | - |
| `tl.log` | ✓ | ✓ | ✓ | - |
| `tl.log2` | ✓ | ✓ | ✓ | - |
| `tl.maximum` | ✓ | ✓ | ✓ | 整数类型也支持 |
| `tl.minimum` | ✓ | ✓ | ✓ | 整数类型也支持 |
| `tl.rsqrt` | ✓ | ✓ | ✓ | - |
| `tl.sigmoid` | ✓ | ✓ | ✓ | - |
| `tl.sin` | ✓ | ✓ | ✓ | - |
| `tl.softmax` | ✓ | ✓ | ✓ | - |
| `tl.sqrt` | ✓ | ✓ | ✓ | - |
| `tl.sqrt_rn` | ✓ | ✓ | ✓ | - |
| `tl.round` | × | ✓ | × | 仅 fp32 |
| `tl.umulhi` | × | × | × | 仅 i32，不支持负数输入 |

### 逻辑运算

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.where` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓* | - |
| `logical_and` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `logical_or` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |

## 归约操作（Reduction Ops）

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.sum` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `tl.max` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `tl.min` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `tl.argmax` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | - |
| `tl.argmin` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | - |
| `tl.reduce` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `tl.xor_sum` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓* | - |

## 线性代数操作（Linear Algebra Ops）

| API | i8 | i16 | i32 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.dot` | ✓ | × | × | ✓ | ✓ | ✓ | × | acc 不支持 fp16；max_num_imprecise_acc 暂不支持；out_dtype 缺乏 int8/fp16 |
| `tl.dot_scaled` | × | × | × | × | × | × | × | fp4/fp8 在 A2/A3 不支持；910_95 支持 fp8 的 dot_scaled；scale 为 int8（GPU 为 uint8） |

## 原子操作（Atomic Ops）

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|---------|
| `tl.atomic_add` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | 不支持多核 add+保存中间结果 |
| `tl.atomic_and` | ✓ | ✓ | ✓ | ✓ | × | × | × | 不支持在 loop 中使用 |
| `tl.atomic_cas` | × | ✓ | ✓ | ✓ | ✓ | ✓ | × | 不支持在 loop 中使用 |
| `tl.atomic_max` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | - |
| `tl.atomic_min` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | - |
| `tl.atomic_or` | ✓ | ✓ | ✓ | ✓ | × | × | × | 不支持在 loop 中使用 |
| `tl.atomic_xchg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 不支持在 loop 中使用 |
| `tl.atomic_xor` | ✓ | ✓ | ✓ | ✓ | × | × | × | 不支持在 loop 中使用 |

> 所有 atomic op：`sem` 只支持默认值 `"acq_rel"`，`scope` 只支持默认值 `"gpu"`

## 形状操作（Shape Manipulation Ops）

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.broadcast` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.broadcast_to` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.expand_dims` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.interleave` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.join` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.permute` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | 不支持不相邻轴转置 |
| `tl.ravel` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.reshape` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.split` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.trans` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | 不支持不相邻轴转置 |
| `tl.view` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |

## 创建操作（Creation Ops）

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.arange` | × | × | ✓ | × | × | × | × | × | 仅 i32 |
| `tl.cat` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.full` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.zeros` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.zeros_like` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.cast` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Ascend 扩展 overflow_mode 参数 |

## 索引操作（Indexing Ops）

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.flip` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.swizzle2d` | ✓ | ✓ | ✓ | ✓ | × | × | × | × | - |
| `tl.gather` | × | × | × | × | ✓ | ✓ | ✓ | × | 仅支持 axis=n-1 |

## 扫描/排序操作（Scan/Sort Ops）

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.associative_scan` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | ✓ | - |
| `tl.cumprod` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.cumsum` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.histogram` | × | × | ✓ | ✓ | × | × | × | × | - |
| `tl.sort` | × | × | × | × | × | × | × | × | 毕昇编译器限制 |

## 随机数生成（Random Number Generation）

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.randint4x` | ✓ | ✓ | ✓ | × | × | × | × | ✓ | 输出类型 |
| `tl.randint` | ✓ | ✓ | ✓ | × | × | × | × | ✓ | 输出类型 |
| `tl.rand` | × | × | × | × | ✓ | ✓ | ✓ | ✓ | 输出类型 |
| `tl.randn` | × | × | × | × | ✓ | ✓ | ✓ | ✓ | 输出类型 |

## Ascend 扩展操作

| API | 模块 | 说明 |
|-----|------|------|
| `al.custom()` | `extra.cann.extension` | 调用自定义算子 |
| `al.register_custom_op` | `extra.cann.extension` | 注册自定义算子 |
| `al.builtin` | `extra.cann.extension` | 包装为内置操作 |
| `al.compile_hint` | `extra.cann.extension` | 编译提示（如 "dot_pad_only_k"） |
| `al.multibuffer` | `extra.cann.extension` | 多缓冲区设置 |
| `al.parallel` | `extra.cann.extension` | 并行迭代器 |
| `al.fixpipe` | `extra.cann.extension` | Fixpipe 后处理 |
| `al.sync_block_all` | `extra.cann.extension` | 块内全核心同步 |
| `al.sync_block_set` | `extra.cann.extension` | 设置同步事件 |
| `al.sync_block_wait` | `extra.cann.extension` | 等待同步事件 |
| `al.int64` | `extra.cann.extension` | 64位整数常量 |
| `al.gather_out_to_ub` | `extra.cann.extension` | 聚合加载到 UB |
| `al.scatter_ub_to_out` | `extra.cann.extension` | 从 UB 散射到输出 |
| `al.index_select` | `extra.cann.extension` | 索引选择 |
| `al.index_put` | `extra.cann.extension` | 索引放置 |
| `al.atan2` | `extra.cann.extension` | 反正切2 |
| `al.isfinited` | `extra.cann.extension` | 有限值判断（双精度） |
| `al.finitef` | `extra.cann.extension` | 有限值判断（单精度） |

## 通用约束

- **UB 空间**：A2/A3 系列所有 tensor 总和不能超过 96KB（开启 double buffer），关闭 double buffer 时不能超过 192KB；910_95 系列不能超过 128KB（开启 double buffer），关闭 double buffer 时不能超过 256KB
- **Shape 限制**：所有 tensor 不允许某个 shape 的 size 小于 1
- **int8 特殊处理**：int8 类型会占用更大的片上空间，编译时容易造成 UB overflow，通常调整 tiling 即可解决
- **标量 tensor**：不支持使用 shape 为 `[[]]` 的标量 tensor 进行计算
- **维度限制**：默认最高支持 8 维 tensor

## 相关文档

- [02-data-type-matrix.md](./02-data-type-matrix.md) - 数据类型支持矩阵
- [03-error-codes.md](./03-error-codes.md) - 错误码参考
- 源码参考：[outline.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/sources/python-api/outline.md)
