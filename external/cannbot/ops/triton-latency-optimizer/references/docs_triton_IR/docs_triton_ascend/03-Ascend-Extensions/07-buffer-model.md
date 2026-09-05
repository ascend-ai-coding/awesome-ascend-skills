# Buffer 编程模型

## 概述

Buffer 编程模型是 Triton-Ascend 在标准 Tensor 编程模型之外提供的另一种数据抽象。与 Tensor 模型关注计算语义不同，Buffer 模型关注内存布局和地址空间，允许开发者直接控制数据在 NPU 片上存储（UB、L1、L0C）中的分配和访问。

Buffer 模型是 fixpipe、copy 等底层操作的基础，也是 Cube-Vector 协同计算中数据共享的关键机制。

## 关键概念

### Buffer 与 Tensor 的区别

| 维度 | Tensor | Buffer |
|------|--------|--------|
| 抽象层次 | 计算语义（值） | 存储语义（内存区域） |
| 地址空间 | 隐式（编译器管理） | 显式（UB/L1/L0C/GM） |
| 数据排布 | 逻辑形状 | 物理排布（含步长） |
| 可写性 | 默认可写 | 通过 `writable` 参数控制 |
| 使用场景 | 标准计算操作 | fixpipe/copy/子视图等底层操作 |
| 转换关系 | `bl.to_buffer(tensor)` | `bl.to_tensor(buffer)` |

### address_space - 缓冲区地址空间

`address_space` 是 Buffer 编程模型的核心概念，表示缓冲区所在的物理存储区域。

```python
class address_space:
    """Represents a buffer's address space.
    The address_space of a buffer is a target-specific attribute.
    """
    def to_ir(self, builder: ir.builder) -> ir.type:
        raise NotImplementedError("Abstract address_space cannot be converted to ir")
```

昇腾 NPU 的地址空间通过 `ascend_address_space` 对象访问：

| 地址空间 | 属性名 | 说明 | 容量级别 | 访问速度 |
|----------|--------|------|----------|----------|
| `ascend_address_space.UB` | UB | Unified Buffer，Vector 核心工作存储 | 片上存储（~1MB） | 最快 |
| `ascend_address_space.L1` | L1 | L1 缓存，Cube 核心缓存 | 片上存储（~1MB） | 快 |
| `ascend_address_space.L0C` | L0C | L0C 缓冲区，Cube 输出 | Cube 输出 | 快 |
| `ascend_address_space.GM` | GM | Global Memory，全局内存 | 片外存储（GB级） | 慢 |

**注意**：`ascend_address_space` 是一个 `ascend_address_space_group` 对象，其属性由 `ascend_ir.AddressSpace` 枚举动态生成。具体可用的地址空间取决于硬件平台，上述四个是最常用的。

### buffer_type - 缓冲区类型

`buffer_type` 描述缓冲区的完整类型信息，包括元素类型、形状、地址空间和步长。

```python
class buffer_type(tl.dtype):
    def __init__(self, element_ty: tl.dtype, shape: List, space: address_space = None, strides: List = None):
        self.element_ty = element_ty   # 元素数据类型
        self.shape = shape             # 缓冲区形状
        self.space = space             # 地址空间
        self.strides = strides         # 内存步长（可选）
```

### buffer - 缓冲区对象

`buffer` 是 Buffer 编程模型的核心数据结构，表示一块物理内存区域。

```python
class buffer(tl._value):
    def __init__(self, handle, buffer_ty: buffer_type):
        self.type = buffer_ty           # 缓冲区类型
        self.dtype = buffer_ty.element_ty.scalar  # 元素标量类型
        self.shape = buffer_ty.shape    # 形状
        self.space = buffer_ty.space    # 地址空间
        self.strides = buffer_ty.strides  # 步长
```

### mem_unique 标注

`mem_unique` 是 Buffer 分配时的一个重要标注，表示该缓冲区在多核心间不共享（每个核心有独立副本）。通过 `alloc` 函数的 `is_mem_unique` 参数设置。

```python
if is_mem_unique:
    builder.create_annotation_mark(handle, "mem_unique", builder.get_unit_attr())
```

## API 参考

### alloc - 分配缓冲区

```python
@builtin
def alloc(
    etype: tl.dtype,
    shape: List[tl.constexpr],
    _address_space: address_space = None,
    is_mem_unique: bool = False,
    _builder=None,
) -> buffer:
    """
    Allocates a region of local memory with the specified shape and type.
    """
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `etype` | `tl.dtype` | 必需 | 元素数据类型（不支持 int1） |
| `shape` | `List[tl.constexpr]` | 必需 | 缓冲区形状 |
| `_address_space` | `address_space` | `None` | 地址空间（UB/L1 等） |
| `is_mem_unique` | `bool` | `False` | 是否为核心独占缓冲区 |

**限制：** 不支持分配 int1 类型的缓冲区。

### to_buffer - Tensor 转 Buffer

```python
@builtin
def to_buffer(
    tensor: tl.tensor,
    space: address_space = None,
    bind_buffer: buffer = None,
    _builder=None,
) -> buffer:
    """
    Convert a tensor to a buffer.
    """
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tensor` | `tl.tensor` | 必需 | 要转换的张量（不支持标量） |
| `space` | `address_space` | `None` | 目标地址空间 |
| `bind_buffer` | `buffer` | `None` | 绑定到已有缓冲区 |

**bind_buffer 语义：** 如果指定了 `bind_buffer`，则将 tensor 的数据绑定到该 buffer，后续对 tensor 的写操作会直接写入该 buffer。

### to_tensor - Buffer 转 Tensor

```python
@builtin
def to_tensor(
    memref: buffer,
    writable: bool = True,
    target_shape=None,
    _builder=None,
) -> tl.tensor:
    """
    Create a tl.tensor from a bl.buffer.
    """
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `memref` | `buffer` | 必需 | 输入缓冲区 |
| `writable` | `bool` | `True` | 生成的 tensor 是否可写 |
| `target_shape` | `list` | `None` | 目标形状（用于布局转换） |

**target_shape 语义：** 如果指定了与 buffer 原始形状不同的 `target_shape`，会生成 `convert_layout` 操作进行布局转换。

### subview - 创建子视图

```python
@builtin
def subview(
    src: buffer,
    offsets: List[tl.constexpr],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    _builder=None,
) -> buffer:
    """Creates a subview of the source buffer with the specified offsets, sizes, and strides."""
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `buffer` | 源缓冲区 |
| `offsets` | `List[tl.constexpr]` | 各维偏移量（必须非负） |
| `sizes` | `List[tl.constexpr]` | 各维大小 |
| `strides` | `List[tl.constexpr]` | 各维步长 |

**对齐约束：** subview 操作要求数据满足 32 字节对齐：
1. 偏移值必须 32 字节对齐
2. 所有步长必须为 1
3. 最后一维第二行的起始偏移必须 32 字节对齐

## 代码示例

### 示例 1：基本 Buffer 分配与使用

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl

@triton.jit
def buffer_basic_kernel(in_ptr, out_ptr, N,
                        BLOCK_SIZE: tl.constexpr):
    x = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))

    ub_buf = bl.alloc(tl.float32, [BLOCK_SIZE], al.ascend_address_space.UB)

    x_buf = bl.to_buffer(x, al.ascend_address_space.UB, bind_buffer=ub_buf)

    result_tensor = bl.to_tensor(x_buf)
    result = result_tensor * 2.0

    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

### 示例 2：fixpipe 中的 Buffer 使用

```python
@triton.jit
def matmul_buffer_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + ...)
        b = tl.load(b_ptr + ...)
        acc = tl.dot(a, b, acc)

    ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
    al.fixpipe(acc, ub_buf)

    acc_tensor = bl.to_tensor(ub_buf)
    tl.store(c_ptr + ..., acc_tensor)
```

### 示例 3：Buffer 子视图

```python
@triton.jit
def subview_kernel(in_ptr, out_ptr, M, N,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)

    data = tl.load(in_ptr + ...)
    data_buf = bl.to_buffer(data, al.ascend_address_space.UB, bind_buffer=ub_buf)

    half_buf = bl.subview(data_buf,
                          offsets=[0, 0],
                          sizes=[BLOCK_M, BLOCK_N // 2],
                          strides=[1, 1])

    half_tensor = bl.to_tensor(half_buf)
    tl.store(out_ptr + ..., half_tensor)
```

### 示例 4：UB 到 L1 的数据拷贝

```python
@triton.jit
def copy_ub_l1_kernel(in_ptr, out_ptr, N,
                      BLOCK_SIZE: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))

    ub_buf = bl.alloc(tl.float32, [BLOCK_SIZE], al.ascend_address_space.UB)
    l1_buf = bl.alloc(tl.float32, [BLOCK_SIZE], al.ascend_address_space.L1)

    data_ub = bl.to_buffer(data, al.ascend_address_space.UB, bind_buffer=ub_buf)
    al.copy(data_ub, l1_buf)

    result = bl.to_tensor(l1_buf)
    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

### 示例 5：mem_unique 标注

```python
@triton.jit
def mem_unique_kernel(in_ptr, out_ptr, N,
                      BLOCK_SIZE: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))

    ub_buf = bl.alloc(tl.float32, [BLOCK_SIZE], al.ascend_address_space.UB,
                      is_mem_unique=True)

    data_buf = bl.to_buffer(data, al.ascend_address_space.UB, bind_buffer=ub_buf)
    result = bl.to_tensor(data_buf) * 2.0
    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

### 示例 6：to_tensor 布局转换

```python
@triton.jit
def layout_convert_kernel(in_ptr, out_ptr, M, N,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)

    data = tl.load(in_ptr + ...)
    data_buf = bl.to_buffer(data, al.ascend_address_space.UB, bind_buffer=ub_buf)

    result = bl.to_tensor(data_buf, target_shape=[BLOCK_N, BLOCK_M])
    tl.store(out_ptr + ..., result)
```

## NPU 适配要点

1. **地址空间选择**：
   - UB：Vector 核心的工作存储，所有向量操作的数据源和目的地
   - L1：Cube 核心的缓存，用于存储矩阵乘法的输入
   - L0C：Cube 核心的输出缓冲区，fixpipe 的数据源
   - GM：全局内存，容量最大但延迟最高

2. **Buffer 与 Tensor 的转换开销**：`to_buffer` 和 `to_tensor` 操作本身不产生数据搬运，只是视图转换。实际的数据搬运发生在 `copy`、`fixpipe` 等操作中。

3. **bind_buffer 的用途**：通过 `bind_buffer` 可以将 tensor 的计算结果直接写入指定的 buffer，避免额外的内存分配和拷贝。

4. **subview 的 32 字节对齐**：subview 操作要求偏移和步长满足 32 字节对齐约束，这是昇腾 NPU 硬件的限制。不满足对齐要求会抛出 `TypeError`。

5. **is_mem_unique 的使用**：在多核心并行场景中，如果每个核心需要独立的缓冲区副本，设置 `is_mem_unique=True`。编译器会为每个核心分配独立的物理内存。

6. **int1 不支持 alloc**：由于硬件限制，不支持分配 int1（bool）类型的缓冲区。

## 常见问题

**Q: 什么时候应该使用 Buffer 而不是 Tensor？**
A: 当需要：(1) 显式控制数据的物理存储位置（UB/L1）；(2) 使用 fixpipe/copy 等底层操作；(3) 在 Cube-Vector 间共享数据；(4) 精细控制内存布局和步长时，使用 Buffer 模型。

**Q: to_buffer 和 to_tensor 有性能开销吗？**
A: 两者都是零拷贝的视图转换，没有数据搬运开销。但如果指定了 `target_shape`，`to_tensor` 会生成布局转换操作。

**Q: bind_buffer 的数据生命周期如何管理？**
A: bind_buffer 将 tensor 绑定到指定的 buffer，tensor 的计算结果会直接写入该 buffer。buffer 的生命周期由其分配点决定，在 kernel 结束时自动释放。

**Q: subview 的步长为什么必须为 1？**
A: 这是昇腾 NPU 硬件对 memref subview 操作的限制，要求连续的内存访问模式以确保 32 字节对齐。

## 相关文档

- [01-extension-overview.md](./01-extension-overview.md) - 扩展 API 总览
- [03-fixpipe.md](./03-fixpipe.md) - fixpipe 操作（Buffer 在 fixpipe 中的使用）
- [04-sync-operations.md](./04-sync-operations.md) - 同步操作（Buffer 在 Cube-Vector 间共享）

## 源码参考

- [buffer/core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/extension/buffer/language/core.py) - Buffer 编程模型核心定义（address_space/buffer_type/buffer/alloc/to_buffer/to_tensor/subview）
- [buffer/semantic.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/extension/buffer/language/semantic.py) - Buffer 操作的 IR 生成
- [extension/core.py: ascend_address_space](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L143-L163) - 昇腾地址空间定义
