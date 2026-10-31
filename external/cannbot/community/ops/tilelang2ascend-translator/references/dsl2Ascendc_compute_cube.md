## Cube 计算详细参考：Init 与 Process

本文档适用于纯 Cube 算子，或设备侧包含独立 Cube 计算阶段的 AscendC 实现。
概览与判断规则见 `@references/dsl2Ascendc.md`。

---

## 第三章：Kernel 入口 `xxx.cpp`

纯 Cube 算子的 kernel 入口通常较简单：

```cpp
extern "C" __global__ __aicore__ void kernel_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    KernelClass kernel;
    kernel.Init(a, b, c, tiling, &pipe);
    kernel.Process();
}
```

---

## 第四章：主 Kernel 类（Cube）

纯 Cube 算子通常把主要逻辑放在主 `Kernel` 类和若干 Cube 子模块中。建议按 `Init()` 和 `Process()` 两个大阶段组织实现。

### 1. `Init()`：接收 tiling 字段并初始化 GM / Queue / 子模块

`Init()` 主要负责：

- 读取并保存 tiling 字段
- 绑定 GM tensor
- 初始化 L1/L0 队列
- 初始化 Cube 子模块

#### A. tiling 字段与运行时参数

常见模式：

- `CopyTiling(&tiling_, tilingGM)`
- `SetGlobalBuffer(...)` 绑定输入 / 输出 GM tensor
- 从 tiling 派生 `baseM`、`baseN`、`baseK`、`l1Prefetch`、`tileSize` 等运行时参数

#### B. TPosition 与物理内存映射（Cube 侧）

| TPosition | 物理存储 | 说明 |
|:---|:---|:---|
| **A1** / **B1** | L1 Buffer | 大块矩阵 |
| **A2** / **B2** | L0A / L0B | 小块矩阵（MMA 输入） |
| **CO1** | L0C | 矩阵计算结果 |

#### C. Queue 与 Buffer 初始化

Cube 侧通常围绕 L1/L0/CO1 队列组织：

- `A1/B1`：承载 GM -> L1 后的大块输入
- `A2/B2`：承载 L1 -> L0 后的 MMA 输入
- `CO1`：承载 MMA 结果

若存在独立 Cube 子模块，例如 `matmul.h`，通常也在 `Init()` 中把 base 参数和 `TPipe` 传给该子模块。

### 2. `Process()`：组织工作负载循环与 Cube 流水

`Process()` 负责把工作负载循环和 Cube 侧实际流水串起来。

#### A. Cube 侧典型组织方式

- **CopyA / CopyB**：GM → L1，把一批 K tiles 搬到 `A1/B1`
- **SplitA / SplitB**：L1 → L0，从大块 L1 buffer 中拆出当前 `baseK` 子块到 `A2/B2`
- **Compute**：对当前 `A2/B2` 调用 `Mmad`
- **CopyOut**：把 `CO1` 结果通过 `Fixpipe` 写回 GM 或 workspace

#### B. Matmul / Cube 实现要点

- 每个 K_L1 迭代只 `DeQue` 一次，向内层循环传递偏移指针
- 所有内层迭代完成后只 `FreeTensor` 一次
- `K_total` 和 `dstStride` 推荐按调用传递，而非在 `Init()` 中固定
- `Fixpipe` 写入大矩阵时，`dstStride` 必须是完整行宽
