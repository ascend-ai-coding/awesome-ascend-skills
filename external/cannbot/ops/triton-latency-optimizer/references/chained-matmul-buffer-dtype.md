# Matmul 链中间 buffer dtype 优化

## 概述

在由多个 `tl.dot` 串联而成的算子中，中间 buffer 的 dtype 会直接影响下一段 matmul 能否以 Cube 峰值吞吐执行。

**最优模式**：
- 输入 A、B 保持 fp16/bf16。
- 累加器显式声明为 fp32（`tl.dot` 默认 `out_dtype=tl.float32`）。

当中间 buffer 被声明为 fp32 时，下一段 matmul 的输入也会变成 fp32。Cube 仍可执行，但会付出三重代价：
1. **算力下降**：fp32 在 Cube 上的有效吞吐通常远低于 fp16/bf16。
2. **带宽翻倍**：fp32 buffer 的读写量是 fp16/bf16 的两倍。
3. **额外转换**：若另一输入仍为低精度，编译器需在 kernel 内将其提升到 fp32 才能匹配。

## 适用条件与命中特征

**适用范围**：
- 算子内存在两段及以上串联 matmul（`A @ B → C @ D`）。
- 输入/输出 dtype 为 fp16/bf16，但链式 matmul 之间的中间 buffer 被声明为 fp32。
- 该中间 buffer 直接作为下一段 `tl.dot` 的输入。

**命中特征（代码中出现以下任一模式）**：
- Host 侧将前一段 matmul 的输出声明为 `torch.float32`，并作为下一段 `tl.dot` 的输入。
- Kernel 内对 `tl.dot` 的输入张量显式 `.to(tl.float32)` 后再计算。
- 中间 buffer 由低精度 tensor 计算得到，但 dtype 被提升为 fp32，且后续仍作为低精度 weight 的 matmul 输入。

## 优化方法

### 优化前

```python
# Host 侧：中间 buffer 声明为 fp32
mid = torch.empty((M, K2), dtype=torch.float32, device=x.device)

# Kernel 内：一个输入为 fp32，另一个为 bf16/fp16
a_tile = tl.load(a_ptr + ...)          # fp32
b_tile = tl.load(b_ptr + ...)          # bf16/fp16
# 编译器需将 b_tile 提升到 fp32，无法利用 Cube 低精度高吞吐路径
acc += tl.dot(a_tile, b_tile)
```

### 优化后

```python
# Host 侧：中间 buffer 保持输入 dtype
mid = torch.empty((M, K2), dtype=x.dtype, device=x.device)  # fp16/bf16

# Kernel 内：两个输入同 dtype
a_tile = tl.load(a_ptr + ...)          # fp16/bf16
b_tile = tl.load(b_ptr + ...)          # fp16/bf16
# Cube 低精度矩阵乘：fp16/bf16 × fp16/bf16 → fp32
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
acc += tl.dot(a_tile, b_tile)
```

## 关键点

1. **Cube 效率**：fp32 中间量会迫使 Cube 以 fp32 吞吐执行，无法利用低精度高吞吐路径。
2. **精度不丢失**：输入 tensor 本身只有 fp16/bf16 精度，cast 到 fp32 不会增加有效数字；两个 fp16/bf16 数的乘积可精确表示为 fp32。只要 `tl.dot` 累加器为 fp32，结果与 fp32 输入基本一致，差异仅来自最后写回低精度时的一次舍入。
3. **内存减半**：fp16/bf16 中间 buffer 的读写量只有 fp32 的一半。
4. **量化路径例外**：int8 量化中间 buffer 仍为 int8，不受本条约束。

## 性能基准（示例）

| 中间 buffer dtype | 几何平均加速比 |
|------------------|---------------|
| fp32 | 0.19x ~ 0.38x |
| fp16/bf16 + 其他优化 | 0.85x ~ 1.03x |

