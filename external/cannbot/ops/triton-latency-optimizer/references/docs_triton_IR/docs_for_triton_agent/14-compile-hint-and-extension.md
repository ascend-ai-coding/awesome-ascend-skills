# compile_hint 与 Ascend 扩展 API 速查

## 触发条件

当 agent 需要使用 NPU 特有扩展 API 或编译提示时查阅本文档。典型场景包括：
- 需要打断 Vector Fusion 防止 UB 溢出
- 需要使用 `insert_slice` / `extract_slice` / `get_element` 进行张量切片
- 需要控制 NaN 传播行为
- 需要使用 Cube-Vector 同步操作
- 需要注册或调用自定义算子
- 需要使用 Buffer 模型直接操作片上存储

---

## 扩展 API 导入方式

```python
import triton.language.extra.cann.extension as extension
```

按需导入：

```python
from triton.language.extra.cann.extension import (
    compile_hint, multibuffer,
    insert_slice, extract_slice, get_element,
    sync_block_set, sync_block_wait, sync_block_all,
    sub_vec_id, sub_vec_num,
    custom, register_custom_op,
    PIPE, CORE, MODE,
    fixpipe, copy, cast, sort, flip,
    index_put, gather_out_to_ub, scatter_ub_to_out, index_select_simd,
)
```

Buffer 编程模型：

```python
import triton.extension.buffer.language as bl
```

libdevice 数学函数库：

```python
from triton.language.extra.cann import libdevice
```

---

## 核心知识：扩展 API 速查表

| API | 类别 | 一句话说明 |
|-----|------|-----------|
| `compile_hint` | 编译提示 | 为张量附加编译器元数据，指导优化 |
| `multibuffer` | 编译提示 | 设置多缓冲（ping-pong），隐藏搬运延迟 |
| `insert_slice` | 向量操作 | 将子张量插入到指定偏移位置 |
| `extract_slice` | 向量操作 | 从指定偏移位置提取子张量 |
| `get_element` | 向量操作 | 按索引获取张量中的单个元素 |
| `sort` | 向量操作 | 沿最后一维排序（硬件加速） |
| `flip` | 向量操作 | 沿指定维度翻转 |
| `cast` | 向量操作 | 增强类型转换（支持 overflow_mode） |
| `sync_block_set` | 同步操作 | 生产者核心发送同步信号 |
| `sync_block_wait` | 同步操作 | 消费者核心等待同步信号 |
| `sync_block_all` | 同步操作 | 全局屏障同步 |
| `sub_vec_id` | 子向量操作 | 获取当前 Vector 核心编号（0 或 1） |
| `sub_vec_num` | 子向量操作 | 获取每个 AI Core 中 Vector 核心数量（通常为 2） |
| `fixpipe` | Cube-Vector | L0C 到 UB 数据搬运（仅 910_95） |
| `custom` | 自定义算子 | 调用自定义算子 |
| `register_custom_op` | 自定义算子 | 注册自定义算子类 |
| `bl.alloc` | Buffer 模型 | 分配缓冲区 |
| `bl.to_buffer` | Buffer 模型 | Tensor 转 Buffer |
| `bl.to_tensor` | Buffer 模型 | Buffer 转 Tensor |
| `bl.subview` | Buffer 模型 | 创建缓冲区子视图 |
| `tl.PropagateNan` | 枚举 | NaN 传播策略（NONE / ALL） |

---

## 代码模式：每个 API 的使用示例

### 1. extension.compile_hint(tensor, hint)

**函数签名：**

```python
extension.compile_hint(ptr, hint_name, hint_val=None, _builder=None)
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr` | `tensor` | 需要附加提示的张量 |
| `hint_name` | `str` 或 `constexpr` | 提示名称（必须为字符串） |
| `hint_val` | `None`/`bool`/`int`/`constexpr`/`list[int]` | 提示值，默认 None |

**常用 hint 名称：**

| hint_name | hint_val 类型 | 说明 | 使用场景 |
|-----------|--------------|------|----------|
| `"break_vf"` | `None` | 在此位置打断 Vector Fusion | 防止过长的 VF 融合导致 UB 溢出 |
| `"hivm.multi_buffer"` | `int` | 多缓冲标记 | 流水线双缓冲优化（由 multibuffer 内部使用） |
| `"overflow_mode"` | `str`（constexpr） | 溢出处理模式 | 整数类型转换溢出控制（"saturate"/"trunc"） |
| `"mem_unique"` | `None` | 核心独占内存标记 | 多核心并行时独立缓冲区 |
| `"saturate_src_unsigned"` | `bool` | saturate 模式下源类型是否为无符号 | cast overflow_mode="saturate" 时自动设置 |
| `"saturate_dst_unsigned"` | `bool` | saturate 模式下目标类型是否为无符号 | cast overflow_mode="saturate" 时自动设置 |

**hint_val 类型与 IR 映射：**

| hint_val 类型 | IR 表示 | 示例 |
|---------------|---------|------|
| `None` | `unit_attr` | `compile_hint(t, "break_vf")` |
| `bool` | `bool_attr` | `compile_hint(t, "hint_c", True)` |
| `int` | `int32_attr` | `compile_hint(t, "hint_b", 42)` |
| `constexpr` (str) | `str_attr` | `compile_hint(t, "overflow_mode", constexpr("saturate"))` |
| `list[int]` | `i64_array_attr` | `compile_hint(t, "hint_d", [128, 32])` |

**示例：break_vf 打断 Vector Fusion**

Flash Attention 反向传播中，当 VF 融合链过长导致 UB 溢出时，在关键位置插入 `break_vf` 提示：

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

@triton.jit
def bwd_q_kernel(q_ptr, k_ptr, v_ptr, dq_ptr, do_ptr, ...,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    for start_n in range(begin, end, BLOCK_N):
        ds = p * (dp - d[:, None])
        ds = tl.where(mask, ds, 0.0)
        ds = ds.to(dtype)
        extension.compile_hint(ds, "break_vf")
        dq += tl.dot(ds, k)
```

**示例：多种 hint 组合使用**

```python
@triton.jit
def hint_demo_kernel(in_ptr, out_ptr, xnumel,
                     XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr + xindex, xmask)

        extension.compile_hint(tmp0, "hint_a")
        extension.compile_hint(tmp0, "hint_b", 42)
        extension.compile_hint(tmp0, "hint_c", True)
        extension.compile_hint(tmp0, "hint_d", [XBLOCK, XBLOCK_SUB])

        tl.store(out_ptr + xindex, tmp0, xmask)
```

**注意事项：**
- SIMT 模式下 `compile_hint` 无效，会被直接跳过
- `compile_hint` 是非侵入式的，不改变计算语义，仅添加元数据
- 同一张量可多次标注不同名称的提示
- `list` 类型仅支持整数数组

---

### 2. extension.get_element / insert_slice / extract_slice

**insert_slice - 将子张量插入到指定偏移位置**

```python
result = extension.insert_slice(ful, sub, offsets, sizes, strides)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `ful` | `tensor` | 目标张量（接收插入） |
| `sub` | `tensor` | 要插入的子张量 |
| `offsets` | `tuple[int]` | 各维偏移量 |
| `sizes` | `tuple[int]` | 各维大小 |
| `strides` | `tuple[int]` | 各维步长 |

**extract_slice - 从指定偏移位置提取子张量**

```python
sub = extension.extract_slice(ful, offsets, sizes, strides)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `ful` | `tensor` | 源张量 |
| `offsets` | `tuple[int]` | 各维偏移量 |
| `sizes` | `tuple[int]` | 各维大小（决定输出形状） |
| `strides` | `tuple[int]` | 各维步长 |

**get_element - 按索引获取单个元素**

```python
elem = extension.get_element(src, indice)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor` | 源张量 |
| `indice` | `tuple[int]` | 索引元组，长度必须等于张量的维度数 |

**示例：**

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

@triton.jit
def slice_ops_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])

    sub = extension.extract_slice(data, offsets=[0, 0], sizes=[BLOCK_M // 2, BLOCK_N], strides=[1, 1])

    full = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    result = extension.insert_slice(full, sub, offsets=[0, 0], sizes=[BLOCK_M // 2, BLOCK_N], strides=[1, 1])

    elem = extension.get_element(data, indice=[0, 0])

    tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], result)
```

**约束：**
- `ful` 和 `sub` 的维度数必须相同
- `offsets`、`sizes`、`strides` 的长度必须与 `ful` 的维度数相同
- 所有 `sizes` 必须 >= 1，所有 `strides` 必须 >= 0
- `offset` 参数支持编译期常量（`constexpr`）和运行时张量（`tensor`）

---

### 3. tl.PropagateNan 枚举

**定义：** `tl.PropagateNan` 是从 `ir.PROPAGATE_NAN` 导出的枚举类型，定义在 [core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L22)。

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `tl.PropagateNan.NONE` | 0 | 不传播 NaN（默认），NaN 被视为"安静"值 |
| `tl.PropagateNan.ALL` | 0xFFFF | 传播所有 NaN，任一操作数为 NaN 则结果为 NaN |

**适用函数：**

| 函数 | propagate_nan 参数 | 默认值 | 说明 |
|------|-------------------|--------|------|
| `tl.maximum(x, y)` | `propagate_nan` | `PropagateNan.NONE` | 逐元素最大值 |
| `tl.minimum(x, y)` | `propagate_nan` | `PropagateNan.NONE` | 逐元素最小值 |
| `tl.clamp(x, min, max)` | `propagate_nan` | `PropagateNan.NONE` | 钳位操作 |
| `tl.max(input, axis)` | `propagate_nan` | `False` | 沿轴最大值归约 |

**IR 映射：**

| propagate_nan 值 | maximum/minimum 生成的 IR 操作 |
|-----------------|-------------------------------|
| `PropagateNan.NONE` | `arith::MinNumFOp` / `arith::MaxNumFOp` |
| `PropagateNan.ALL` | `arith::MinimumFOp` / `arith::MaximumFOp` |

**示例：Flash Attention 中的 NaN 传播**

在 Flash Attention 的 online softmax 计算中，需要正确处理 NaN 值以避免数值错误：

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

@triton.jit
def flash_attn_fwd(q_ptr, k_ptr, v_ptr, o_ptr, ...,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = tl.full((BLOCK_M,), value=-2**30, dtype=tl.float32)
    for start_n in range(begin, end, BLOCK_N):
        s = tl.dot(q, k)
        s = s * qk_scale + tl.where(mask, 0.0, -2.0**30)
        m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        p = tl.math.exp(s - m_new[:, None])
        alpha = tl.math.exp(m - m_new)
        acc = acc * alpha[:, None] + pv
        l = l * alpha + p_sum
        m = m_new
```

**示例：maximum/minimum 中的 NaN 传播**

```python
@triton.jit
def nan_propagation_demo(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
    y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))

    result_none = tl.maximum(x, y, propagate_nan=tl.PropagateNan.NONE)
    result_all = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)

    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result_all)
```

**NPU 适配要点：**
- 在 NPU 上，`PropagateNan.ALL` 生成 `arith::MaximumFOp`/`arith::MinimumFOp`，符合 IEEE 754 NaN 传播语义
- `PropagateNan.NONE` 生成 `arith::MaxNumFOp`/`arith::MinNumFOp`，NaN 被视为安静值
- Flash Attention 等涉及 online softmax 的算子中，`tl.max` 归约的 `propagate_nan=True` 和 `tl.maximum` 的 `propagate_nan=tl.PropagateNan.ALL` 配合使用，确保 NaN 值正确传播

---

### 4. sub_vec 操作

**sub_vec_id() - 获取当前 Vector 核心编号**

```python
vec_id = extension.sub_vec_id()  # 返回 tl.tensor (int64)，值为 0 或 1
```

**sub_vec_num() - 获取 Vector 核心数量**

```python
vec_num = extension.sub_vec_num()  # 返回 tl.constexpr (int)，通常为 2
```

**示例：CV 融合算子中的工作负载分配**

```python
@triton.jit
def cv_fusion_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    with extension.scope(core_mode="cube"):
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + ...)
            b = tl.load(b_ptr + ...)
            acc = tl.dot(a, b, acc)
        ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], extension.ascend_address_space.UB)
        extension.fixpipe(acc, ub_buf)
        extension.sync_block_set("cube", "vector", 0)

    with extension.scope(core_mode="vector"):
        extension.sync_block_wait("cube", "vector", 0)
        vec_id = extension.sub_vec_id()
        vec_num = extension.sub_vec_num()
        half_m = BLOCK_M // vec_num
        start_row = vec_id * half_m
        end_row = start_row + half_m
        acc_tensor = bl.to_tensor(ub_buf)
        my_slice = acc_tensor[start_row:end_row, :]
        bias_slice = tl.load(bias_ptr + tl.arange(0, BLOCK_N))
        result = my_slice + bias_slice
        tl.store(c_ptr + ..., result)
```

**示例：parallel 循环中的 bind_sub_block**

```python
@triton.jit
def parallel_sub_vec_kernel(data_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    with extension.scope(core_mode="vector"):
        for i in extension.parallel(0, N, BLOCK_N, bind_sub_block=True):
            vec_id = extension.sub_vec_id()
            offset = i + vec_id * BLOCK_N
            data = tl.load(data_ptr + offset + tl.arange(0, BLOCK_N))
            result = data * 2.0
            tl.store(out_ptr + offset + tl.arange(0, BLOCK_N), result)
```

**注意事项：**
- `sub_vec_id()` 应仅在 `extension.scope(core_mode="vector")` 上下文中使用
- `sub_vec_id()` 返回 `tl.tensor`（int64），`sub_vec_num()` 返回 `tl.constexpr`
- 昇腾 NPU 每个 AI Core 包含 1 个 Cube + 2 个 Vector，硬件比例为 1:2

---

### 5. custom_op 机制概述

**注册自定义算子：**

```python
@extension.register_custom_op
class my_elementwise_op:
    name = 'my_elementwise_op'
    core = extension.CORE.VECTOR
    pipe = extension.PIPE.PIPE_V
    mode = extension.MODE.SIMT
    symbol = 'my_elementwise_kernel'
    bitcode = '/path/to/my_op.bc'

    def __init__(self, input, output):
        assert input.dtype == output.dtype
```

**调用自定义算子：**

```python
@triton.jit
def my_kernel(in_ptr, out_ptr, N):
    x = tl.load(in_ptr + tl.arange(0, 128))
    result = extension.custom('my_elementwise_op', x, out=tl.zeros([128], dtype=tl.float32))
    tl.store(out_ptr + tl.arange(0, 128), result)
```

**自定义算子类必需属性：**

| 属性 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | `str` | 是 | 算子名称，全局唯一 |
| `core` | `CORE` | 是 | 执行核心类型 |
| `pipe` | `PIPE` | 是 | 执行流水线类型 |
| `mode` | `MODE` | 是 | 执行模式 |
| `symbol` | `str` | 非内置必需 | CANN bitcode 中的符号名 |
| `bitcode` | `str` | 非内置必需 | CANN bitcode 文件路径 |

**内置自定义算子（无需注册）：**

| 算子名 | 说明 |
|--------|------|
| `__builtin_index_select` | 索引选择（GM -> UB，2D-5D） |
| `__builtin_index_put` | 索引写入（UB -> GM，2D-5D） |
| `__builtin_gather_load` | 索引收集加载（GM -> UB，1D-5D） |
| `__builtin_scatter_store` | 索引散列存储（UB -> GM，2D-5D） |

**int64 类型包装：** 自定义算子的 Python int 参数默认转换为 int32，如果设备端需要 int64，使用 `extension.int64(x)` 包装。

---

### 6. Buffer 模型概述

Buffer 模型允许直接控制数据在 NPU 片上存储中的分配和访问。

**地址空间：**

| 地址空间 | 属性名 | 说明 | 容量 | 访问速度 |
|----------|--------|------|------|----------|
| `extension.ascend_address_space.UB` | UB | Unified Buffer，Vector 工作存储 | ~1MB | 最快 |
| `extension.ascend_address_space.L1` | L1 | L1 缓存，Cube 缓存 | ~1MB | 快 |
| `extension.ascend_address_space.L0C` | L0C | Cube 输出缓冲区 | Cube 输出 | 快 |
| `extension.ascend_address_space.GM` | GM | Global Memory | GB级 | 慢 |

**核心 API：**

```python
ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], extension.ascend_address_space.UB)
data_buf = bl.to_buffer(tensor, extension.ascend_address_space.UB, bind_buffer=ub_buf)
result_tensor = bl.to_tensor(data_buf, writable=True)
sub_buf = bl.subview(src_buf, offsets=[0, 0], sizes=[M, N], strides=[1, 1])
```

**示例：fixpipe 中的 Buffer 使用**

```python
@triton.jit
def matmul_buffer_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + ...)
        b = tl.load(b_ptr + ...)
        acc = tl.dot(a, b, acc)

    ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], extension.ascend_address_space.UB)
    extension.fixpipe(acc, ub_buf)
    acc_tensor = bl.to_tensor(ub_buf)
    tl.store(c_ptr + ..., acc_tensor)
```

**注意事项：**
- `to_buffer` 和 `to_tensor` 是零拷贝视图转换，无数据搬运开销
- `subview` 要求 32 字节对齐：偏移值必须 32 字节对齐，所有步长必须为 1
- 不支持分配 int1（bool）类型的缓冲区

---

### 7. 同步操作

**sync_block_set - 生产者发送同步信号**

```python
extension.sync_block_set(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `sender` | `str` | "cube" 或 "vector" |
| `receiver` | `str` | "cube" 或 "vector" |
| `event_id` | `int` | 事件 ID，范围 0-15 |
| `sender_pipe` | `PIPE` | 发送方流水线（可自动推断） |
| `receiver_pipe` | `PIPE` | 接收方流水线（可自动推断） |

**sync_block_wait - 消费者等待同步信号**

```python
extension.sync_block_wait(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None)
```

**sync_block_all - 全局屏障同步**

```python
extension.sync_block_all(mode, event_id)
```

| mode | 说明 |
|------|------|
| `"all_cube"` | 所有 Cube 核心同步 |
| `"all_vector"` | 所有 Vector 核心同步 |
| `"all"` | 所有核心同步 |
| `"all_sub_vector"` | 所有子 Vector 核心同步 |

**PIPE 默认值：**

| 方向 | sender_pipe 默认 | receiver_pipe 默认 |
|------|-----------------|-------------------|
| cube -> vector | `PIPE_FIX` | `PIPE_MTE2` |
| vector -> cube | `PIPE_MTE3` | `PIPE_MTE2` |

**示例：基本 Cube-Vector 同步**

```python
@triton.jit
def basic_sync_kernel(a_ptr, b_ptr, c_ptr, N):
    with extension.scope(core_mode="cube"):
        a = tl.load(a_ptr + tl.arange(0, 128))
        b = tl.load(b_ptr + tl.arange(0, 128))
        result = tl.dot(a, b)
        extension.sync_block_set("cube", "vector", 0)

    with extension.scope(core_mode="vector"):
        extension.sync_block_wait("cube", "vector", 0)
        processed = result * 2.0
        tl.store(c_ptr + tl.arange(0, 128), processed)
```

**注意事项：**
- `event_id` 范围 0-15，共 16 个独立事件通道
- `sender` 和 `receiver` 不能相同（cube->cube 或 vector->vector 不允许）
- `set/wait` 必须成对使用，且 `sender`、`receiver`、`event_id` 必须一致
- 同步操作必须在 `extension.scope(core_mode="cube")` 或 `extension.scope(core_mode="vector")` 上下文中使用

---

## 910_95 特别注意

### fixpipe 仅限 910_95

`fixpipe` 将 Cube 计算结果从 L0C 搬运到 UB，**仅在 Ascend 910_95 系列上支持**。完整 API 说明、对齐约束和代码示例见 [11-fixpipe-and-bias-fusion.md](11-fixpipe-and-bias-fusion.md)。

### cast 的 overflow_mode 平台差异

- **910_95**：saturate 模式直接使用硬件 `int_cast` 指令，通过 `compile_hint` 附加 `saturate_src_unsigned` 和 `saturate_dst_unsigned` 属性
- **非 910_95**：saturate 模式通过 fp32 中转实现，精度可能略有差异

### libdevice 函数的平台差异

- **910_95**：`libdevice.rint` 使用 HMF 硬件实现（`__hmf_rint`）
- **其他平台**：`libdevice.rint` 使用软件实现（银行家舍入逻辑）

### fp8/fp64 限制

在非 910_95 平台上，`cast` 操作不支持 fp8 和 fp64 类型。

---

## 相关文档链接

### 源文档

- [01-extension-overview.md](../docs_triton_ascend/03-Ascend-Extensions/01-extension-overview.md) - 扩展 API 总览与导入方式
- [02-pipe-and-core.md](../docs_triton_ascend/03-Ascend-Extensions/02-pipe-and-core.md) - PIPE/CORE/MODE/IteratorType 枚举
- [03-fixpipe.md](../docs_triton_ascend/03-Ascend-Extensions/03-fixpipe.md) - fixpipe 操作详解
- [04-sync-operations.md](../docs_triton_ascend/03-Ascend-Extensions/04-sync-operations.md) - 同步操作详解
- [05-sub-vec-ops.md](../docs_triton_ascend/03-Ascend-Extensions/05-sub-vec-ops.md) - sub_vec_id/sub_vec_num
- [06-custom-op.md](../docs_triton_ascend/03-Ascend-Extensions/06-custom-op.md) - 自定义算子注册与使用
- [07-buffer-model.md](../docs_triton_ascend/03-Ascend-Extensions/07-buffer-model.md) - Buffer 编程模型
- [08-aux-ops.md](../docs_triton_ascend/03-Ascend-Extensions/08-aux-ops.md) - parallel/compile_hint/multibuffer
- [09-vec-ops.md](../docs_triton_ascend/03-Ascend-Extensions/09-vec-ops.md) - insert_slice/extract_slice/get_element/sort/flip/cast
- [10-mem-ops.md](../docs_triton_ascend/03-Ascend-Extensions/10-mem-ops.md) - index_put/gather_out_to_ub/scatter_ub_to_out/index_select_simd
- [11-libdevice.md](../docs_triton_ascend/03-Ascend-Extensions/11-libdevice.md) - libdevice 数学函数库

### 实际使用参考

- `flash_attention_npu_v8.py` - Flash Attention 中 compile_hint("break_vf") 和 PropagateNan.ALL 的使用
- `fused_matmul_npu_v3.py` - Fused Matmul 中编译参数的使用

### 源码参考

- [aux_ops.py: compile_hint](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/aux_ops.py#L135-L151) - compile_hint 函数定义
- [aux_ops.py: multibuffer](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/aux_ops.py#L153-L162) - multibuffer 函数定义
- [vec_ops.py: insert_slice](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L47-L92) - insert_slice 函数定义
- [vec_ops.py: extract_slice](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L95-L137) - extract_slice 函数定义
- [vec_ops.py: get_element](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L139-L177) - get_element 函数定义
- [core.py: PropagateNan](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L22) - PropagateNan 枚举定义
- [core.py: maximum/minimum](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1935-L1975) - maximum/minimum 函数定义
- [standard.py: max](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/standard.py#L174) - max 归约函数定义（含 propagate_nan 参数）
- [core.py: sync_block_set/wait/all](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L202-L244) - 同步操作函数定义
- [core.py: sub_vec_id/sub_vec_num](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L166-L171) - sub_vec 操作定义
- [custom_op.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/custom_op.py) - 自定义算子注册与调用
- [buffer/core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/extension/buffer/language/core.py) - Buffer 编程模型核心定义
