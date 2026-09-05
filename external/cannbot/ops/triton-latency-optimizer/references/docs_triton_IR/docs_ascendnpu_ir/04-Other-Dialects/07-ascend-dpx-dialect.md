# AscendDPX 方言

## 1. 概述

AscendDPX 方言定义了 Ascend NPU 的 SIMT（Single Instruction Multiple Thread）虚拟 ISA 操作集，提供线程索引查询、内存读写、Warp Shuffle、Warp 归约、原子操作、类型转换和数学函数等操作。

- **方言名称**：`ascend_dpx`
- **C++ 命名空间**：`::mlir::ascend_dpx`

> 源码参考：[AscendDPXOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/AscendDPX/IR/AscendDPXOps.td)、[AscendDPXBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/AscendDPX/IR/AscendDPXBase.td)、[AscendDPXAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/AscendDPX/IR/AscendDPXAttrs.td)

## 2. 方言定义

```tablegen
def AscendDPX_Dialect : Dialect {
  let name = "ascend_dpx";
  let description = [{
    Ascend David SIMT virtual ISA definitions.
  }];
  let cppNamespace = "::mlir::ascend_dpx";
  let useDefaultAttributePrinterParser = 1;
}
```

## 3. 程序信息操作

所有程序信息操作继承自 `AscendDPX_InfoOp`，返回 `I32` 类型（clock64 返回 `I64`），无操作数。

### 3.1 线程索引

| 操作 | 说明 |
|------|------|
| `ascend_dpx.thread_id_x` | 当前线程的 X 维线程索引 |
| `ascend_dpx.thread_id_y` | 当前线程的 Y 维线程索引 |
| `ascend_dpx.thread_id_z` | 当前线程的 Z 维线程索引 |

### 3.2 Block 索引

| 操作 | 说明 |
|------|------|
| `ascend_dpx.block_idx_x` | 当前 Block 的 X 维索引 |
| `ascend_dpx.block_idx_y` | 当前 Block 的 Y 维索引 |
| `ascend_dpx.block_idx_z` | 当前 Block 的 Z 维索引 |
| `ascend_dpx.block_idx` | 当前 Block 的线性索引 |

### 3.3 Block 维度

| 操作 | 说明 |
|------|------|
| `ascend_dpx.block_dim_x` | Block 的 X 维大小 |
| `ascend_dpx.block_dim_y` | Block 的 Y 维大小 |
| `ascend_dpx.block_dim_z` | Block 的 Z 维大小 |

### 3.4 Grid 维度

| 操作 | 说明 |
|------|------|
| `ascend_dpx.grid_dim_x` | Grid 的 X 维大小 |
| `ascend_dpx.grid_dim_y` | Grid 的 Y 维大小 |
| `ascend_dpx.grid_dim_z` | Grid 的 Z 维大小 |

### 3.5 其他信息

| 操作 | 返回类型 | 说明 |
|------|----------|------|
| `ascend_dpx.clock32` | I32 | 32 位时钟计数 |
| `ascend_dpx.clock64` | I64 | 64 位时钟计数 |
| `ascend_dpx.core_id` | I32 | Core ID |

### 3.6 MLIR 示例

```mlir
%tid = ascend_dpx.thread_id_x : i32
%bid = ascend_dpx.block_idx_y : i32
%bdim = ascend_dpx.block_dim_z : i32
%gdim = ascend_dpx.grid_dim_x : i32
%cid = ascend_dpx.core_id : i32
```

## 4. 内存操作

### 4.1 ascend_dpx.load

#### 功能

SIMT 模式下的加载操作，支持掩码和缓存策略。

#### 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `ptr` | `LLVM_AnyPointer` | 源地址指针 |
| `mask` | `Optional<I1>` | 掩码 |
| `falseVal` | `Optional<LLVM_Type>` | 掩码为假时的回退值 |
| `cache` | `AscendDPX_LoadCachePolicyAttr` (默认: L2_CACHE_HINT_NORMAL_FV) | L2 缓存策略 |
| `result` | `LLVM_Type` | 加载结果 |

#### MLIR 示例

```mlir
%val = ascend_dpx.load %ptr : !llvm.ptr<f32> -> f32
%val = ascend_dpx.load %ptr, %mask, %false cacheModifier = <L2_CACHE_HINT_NORMAL_CG> : !llvm.ptr<f32>, i1, f32 -> f32
```

### 4.2 ascend_dpx.store

#### 功能

SIMT 模式下的存储操作，支持掩码和缓存策略。

#### 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `ptr` | `LLVM_AnyPointer` | 目标地址指针 |
| `value` | `LLVM_Type` | 待存储的值 |
| `mask` | `Optional<I1>` | 掩码 |
| `cache` | `AscendDPX_StoreCachePolicyAttr` (默认: L2_CACHE_HINT_NORMAL_FV) | L2 缓存策略 |

#### MLIR 示例

```mlir
ascend_dpx.store %ptr, %val : !llvm.ptr<f32>, f32
```

## 5. Warp 操作

### 5.1 同步

| 操作 | 说明 |
|------|------|
| `ascend_dpx.sync_threads` | 线程间同步屏障 |

### 5.2 Warp Shuffle

所有 Warp Shuffle 操作共享相同的操作签名：

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `LLVM_Type` | 源值 |
| `lane_mask` | `I32` | 通道掩码 |
| `clamp` | `I32` | 钳位值 |
| `index` | `I32` | 索引 |
| `res` | `LLVM_Type` | 结果 |

| 操作 | 说明 |
|------|------|
| `ascend_dpx.shfl.up` | 向上 Shuffle（从低通道获取值） |
| `ascend_dpx.shfl.down` | 向下 Shuffle（从高通道获取值） |
| `ascend_dpx.shfl.bfly` | 蝶形 Shuffle（按位异或索引） |
| `ascend_dpx.shfl.idx` | 索引 Shuffle（按指定索引获取值） |

#### MLIR 示例

```mlir
%result = ascend_dpx.shfl.up %src, %mask, %clamp, %idx : (f32, i32, i32, i32) -> f32
```

### 5.3 Warp 归约

所有 Warp 归约操作共享相同的操作签名：

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `LLVM_Type` | 源值 |
| `res` | `LLVM_Type` | 结果 |

| 操作 | 说明 |
|------|------|
| `ascend_dpx.reduce.add` | Warp 级加法归约 |
| `ascend_dpx.reduce.max` | Warp 级有符号最大值归约 |
| `ascend_dpx.reduce.min` | Warp 级有符号最小值归约 |
| `ascend_dpx.reduce.umax` | Warp 级无符号最大值归约 |
| `ascend_dpx.reduce.umin` | Warp 级无符号最小值归约 |

## 6. 原子操作

### 6.1 通用原子操作

所有通用原子操作（除 CAS 外）共享相同的操作签名：

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `LLVM_AnyPointer` | 目标地址 |
| `data` | `LLVM_Type` | 操作数 |
| `cache` | `AscendDPX_StoreCachePolicyAttr` (默认: L2_CACHE_HINT_NORMAL_FV) | 缓存策略 |
| `res` | `LLVM_Type` | 原始值 |

| 操作 | 说明 |
|------|------|
| `ascend_dpx.atomic_and` | 原子按位与 |
| `ascend_dpx.atomic_or` | 原子按位或 |
| `ascend_dpx.atomic_xor` | 原子按位异或 |
| `ascend_dpx.atomic_inc` | 原子递增 |
| `ascend_dpx.atomic_dec` | 原子递减 |
| `ascend_dpx.atomic_max` | 原子有符号最大值 |
| `ascend_dpx.atomic_min` | 原子有符号最小值 |
| `ascend_dpx.atomic_add` | 原子加 |
| `ascend_dpx.atomic_sub` | 原子减 |
| `ascend_dpx.atomic_umax` | 原子无符号最大值 |
| `ascend_dpx.atomic_umin` | 原子无符号最小值 |
| `ascend_dpx.atomic_exchange` | 原子交换 |

### 6.2 原子 CAS

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `LLVM_AnyPointer` | 目标地址 |
| `data` | `LLVM_Type` | 期望值 |
| `other` | `LLVM_Type` | 新值 |
| `cache` | `AscendDPX_StoreCachePolicyAttr` | 缓存策略 |
| `res` | `LLVM_Type` | 原始值 |

#### MLIR 示例

```mlir
%old = ascend_dpx.atomic_cas %ptr, %expected, %new_val : (!llvm.ptr<i32>, i32, i32) -> i32
```

## 7. 类型转换

### 7.1 ascend_dpx.cast

#### 功能

通用类型转换操作，支持多种转换模式。

#### 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `in` | `LLVM_Type` | 输入值 |
| `kind` | `AscendDPX_CastKindAttr` (默认: SIGNED_TO_FLOAT) | 转换类型 |
| `out` | `LLVM_Type` | 输出值 |

#### CastKind 枚举

| 枚举值 | 助记符 | 说明 |
|--------|--------|------|
| `SIGNED_TO_FLOAT` | `si_to_fp` | 有符号整数转浮点 |
| `UNSIGNED_TO_FLOAT` | `ui_to_fp` | 无符号整数转浮点 |
| `FLOAT_TO_SIGNED` | `fp_to_si` | 浮点转有符号整数 |
| `FLOAT_TO_UNSIGNED` | `fp_to_ui` | 浮点转无符号整数 |

#### MLIR 示例

```mlir
%out = ascend_dpx.cast %in kind <si_to_fp> : i32 to f32
```

## 8. 数学函数

### 8.1 一元数学操作

所有一元数学操作共享相同的操作签名：

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `ins` | `LLVM_Type` | 输入 |
| `res` | `LLVM_Type` | 结果 |

| 操作 | 说明 |
|------|------|
| `ascend_dpx.atan` | 反正切 |
| `ascend_dpx.ceil` | 向上取整 |
| `ascend_dpx.cos` | 余弦 |
| `ascend_dpx.erf` | 误差函数 |
| `ascend_dpx.exp2` | 2^x |
| `ascend_dpx.exp` | e^x |
| `ascend_dpx.floor` | 向下取整 |
| `ascend_dpx.ilogb` | 指数部分 |
| `ascend_dpx.log1p` | log(1+x) |
| `ascend_dpx.log2` | 以 2 为底对数 |
| `ascend_dpx.log` | 自然对数 |
| `ascend_dpx.recip` | 倒数 |
| `ascend_dpx.relu` | ReLU 激活 |
| `ascend_dpx.rint` | 四舍五入到偶数 |
| `ascend_dpx.round` | 四舍五入远离零 |
| `ascend_dpx.rsqrt` | 平方根倒数 |
| `ascend_dpx.sin` | 正弦 |
| `ascend_dpx.sqrt` | 平方根 |
| `ascend_dpx.tanh` | 双曲正切 |
| `ascend_dpx.tan` | 正切 |

### 8.2 二元数学操作

所有二元数学操作共享相同的操作签名：

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `in1` | `LLVM_Type` | 左操作数 |
| `in2` | `LLVM_Type` | 右操作数 |
| `res` | `LLVM_Type` | 结果 |

| 操作 | 说明 |
|------|------|
| `ascend_dpx.div` | 浮点除法 |
| `ascend_dpx.udiv` | 无符号整数除法 |
| `ascend_dpx.pow` | 幂运算 |
| `ascend_dpx.umulhi` | 无符号乘法高位 |
| `ascend_dpx.ldexp` | ldexp 函数 |

### 8.3 数值检查操作

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `in` | `LLVM_Type` | 输入 |
| `res` | `I1` | 布尔结果 |

| 操作 | 说明 |
|------|------|
| `ascend_dpx.isfinite` | 判断是否为有限值 |
| `ascend_dpx.isinf` | 判断是否为无穷 |
| `ascend_dpx.isnan` | 判断是否为 NaN |

## 9. L2 缓存策略

### 9.1 加载缓存策略

| 枚举值 | 说明 |
|--------|------|
| `L2_CACHE_HINT_NORMAL_FV` | 正常访问，首次有效 |
| `L2_CACHE_HINT_NORMAL_LV` | 正常访问，末次有效 |
| `L2_CACHE_HINT_NORMAL_PERS` | 正常访问，持久化 |
| `L2_CACHE_HINT_NORMAL_PREF` | 正常访问，预取 |
| `L2_CACHE_HINT_NOTALLOC_KEEP` | 不分配，保持 |
| `L2_CACHE_HINT_NOTALLOC_CLEAN` | 不分配，清除 |
| `L2_CACHE_HINT_NOTALLOC_DROP` | 不分配，丢弃 |
| `L2_CACHE_HINT_IDS_FV` | IDS 首次有效 |
| `L2_CACHE_HINT_IDS_LV` | IDS 末次有效 |
| `L2_CACHE_HINT_IDS_PERS` | IDS 持久化 |
| `L2_CACHE_HINT_IDS_PREF` | IDS 预取 |
| `L2_CACHE_HINT_EXCLUSIV_FV` | 独占首次有效 |
| `L2_CACHE_HINT_EXCLUSIV_LV` | 独占末次有效 |
| `L2_CACHE_HINT_EXCLUSIV_PERS` | 独占持久化 |
| `L2_CACHE_HINT_EXCLUSIV_PREF` | 独占预取 |
| `L2_CACHE_HINT_INVALID` | 无效 |

### 9.2 存储缓存策略

| 枚举值 | 说明 |
|--------|------|
| `L2_CACHE_HINT_NORMAL_FV` | 正常写回，首次有效 |
| `L2_CACHE_HINT_NORMAL_LV` | 正常写回，末次有效 |
| `L2_CACHE_HINT_NORMAL_PERS` | 正常写回，持久化 |
| `L2_CACHE_HINT_NORMAL_RED` | 正常写回，归约 |
| `L2_CACHE_HINT_NOTALLOC_CI` | 不分配，缓存无关 |
| `L2_CACHE_HINT_NOTALLOC_PW` | 不分配，部分写 |
| `L2_CACHE_HINT_NOTALLOC_PI` | 不分配，部分无关 |
| `L2_CACHE_HINT_NOTALLOC_RED` | 不分配，归约 |
| `L2_CACHE_HINT_WBH_FV` | 写回提示，首次有效 |
| `L2_CACHE_HINT_WBH_LV` | 写回提示，末次有效 |
| `L2_CACHE_HINT_WBH_PERS` | 写回提示，持久化 |
| `L2_CACHE_HINT_WBH_RED` | 写回提示，归约 |
| `L2_CACHE_HINT_WTS_FV` | 写穿透提示，首次有效 |
| `L2_CACHE_HINT_WTS_LV` | 写穿透提示，末次有效 |
| `L2_CACHE_HINT_WTS_PERS` | 写穿透提示，持久化 |
| `L2_CACHE_HINT_WTS_RED` | 写穿透提示，归约 |
| `L2_CACHE_HINT_INVALID` | 无效 |

## 10. 地址空间

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `GLOBAL_MEM` | 1 | 全局内存 |
| `SHARED_MEM` | 3 | 共享内存 |
