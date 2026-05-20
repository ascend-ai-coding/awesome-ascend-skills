# Load 指令重排序

## 优化原理

该文件旨在说明，编译器不会修改用户load指令的顺序，在前面的load指令被其他指令阻塞时，可以将没有数据依赖的load语句放在前面以提前发射，提升并行度。

## 优化思路

在循环中，原本的语句顺序是：

```
load B
load A
calc
store O
store B
```

由于当前的 load B 会等待上一次循环的 store B，load A 不能提前与 load B 执行，所以 load A 与 store B 不能并行。

将语句顺序改为：

```
load A
load B
calc
store O
store B
```

load A 即可与上一次循环的 store B 并行。

## 代码示例

优化前（load B 在前，load A 在后）：

```python
@triton.jit
def AB_load_kernel(
    A,
    B,
    B_index,
    O,
    B_DIM: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_range = tl.arange(0, B_DIM)

    for i in range(HEAD_NUM):
        # calc index
        p_A = A + i * HEAD_DIM + i_n * B_DIM + i_range
        p_O = O + i * HEAD_DIM + i_n * B_DIM + i_range
        p_B_index = B_index + i

        # load B (在前，会阻塞 load A)
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        b_B = tl.load(p_B)

        # load A (在后，必须等 load B 完成)
        b_A = tl.load(p_A)

        # calculation
        b_B += tl.sum(b_A)
        b_O = b_A * b_B

        # store O
        tl.store(p_O, b_O)

        # store B
        tl.store(p_B, b_B)
```

优化后（load A 在前，load B 在后）：

```python
@triton.jit
def AB_load_kernel(
    A,
    B,
    B_index,
    O,
    B_DIM: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_range = tl.arange(0, B_DIM)

    for i in range(HEAD_NUM):
        # calc index
        p_A = A + i * HEAD_DIM + i_n * B_DIM + i_range
        p_O = O + i * HEAD_DIM + i_n * B_DIM + i_range
        p_B_index = B_index + i

        # load A (在前，可以与上一次循环的 store B 并行)
        b_A = tl.load(p_A)

        # load B (在后，但不会阻塞 load A)
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        b_B = tl.load(p_B)

        # calculation
        b_B += tl.sum(b_A)
        b_O = b_A * b_B

        # store O
        tl.store(p_O, b_O)

        # store B
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        tl.store(p_B, b_B)
```

## 为什么有效

| 优化前 | 优化后 |
|--------|--------|
| load B 等待上一次 store B 完成后才能执行 | load A 无需等待，可以提前发射 |
| load A 必须等 load B 完成后才能执行 | load A 可以与上一次循环的 store B 并行执行 |
| 串行执行，并行度低 | 并行执行，并行度高 |

核心原理：load A 与 store B 之间没有数据依赖，调整顺序后 load A 可提前发射，充分利用流水线并行能力。