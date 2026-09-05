## Vector 计算详细参考：Init 与 Process

本文档适用于纯 Vector 算子，或设备侧包含 Vector 计算阶段的 AscendC 实现。
概览与判断规则见 `@references/dsl2Ascendc.md`。

---

## 第三章：Kernel 入口 `xxx.cpp`

纯 Vector 算子的 kernel 入口通常较简单：

```cpp
extern "C" __global__ __aicore__ void kernel_custom(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    KernelClass kernel;
    kernel.Init(x, y, tiling, &pipe);
    kernel.Process();
}
```

### `vec_num` 与 block 组成

| DSL `vec_num` | KERNEL_TYPE | 每个 block 组成 |
|:---|:---|:---|
| 1 | `KERNEL_TYPE_MIX_AIC_1_1` | 1 AIC + 1 AIV |
| 2 | `KERNEL_TYPE_MIX_AIC_1_2` | 1 AIC + 2 AIV |

---

## 第四章：主 Kernel 类（Vector）

需要按 `Init()` 和 `Process()` 两个大阶段组织实现。

### 1. `Init()`：接收 tiling 字段并初始化 Buffer / Queue

`Init()` 主要负责三件事：

- 读取并保存 tiling 字段
- 绑定 GM tensor
- 初始化 Buffer / Queue

#### A. tiling 字段与运行时参数

常见模式：

- `CopyTiling(&tiling_, tilingGM)`
- `SetGlobalBuffer(...)` 绑定输入 / 输出 GM tensor
- 从 tiling 派生本核要用的运行时参数，如 `tileM`、`tileN`、`tileSize`

如果存在多子块划分，也通常在 `Init()` 中完成与 tile 大小相关的派生量计算。

#### B. Buffer 和 Queue 初始化

必须根据数据流中的用途，严格选择 TBuf（计算缓冲区）或 TQue（数据队列）类型。

TPosition 与物理内存映射（Vector 侧）：

| TPosition | 物理存储 | 说明 |
|:---|:---|:---|
| **VECIN** / **VECOUT** | Unified Buffer | 矢量计算输入/输出 |
| **VECCALC** | Unified Buffer | 矢量计算临时变量 |

| TileLang DSL | AscendC 实现 | TPosition | 物理存储 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `x_ub = T.alloc_ub((tileM, tileN), dtype)` | `TQue<TPosition::VECIN, 0> inQueue;` | **VECIN** | Unified Buffer | 输入数据缓冲区 |
| `y_ub = T.alloc_ub((tileM, tileN), dtype)` | `TQue<TPosition::VECOUT, 0> outQueue;` | **VECOUT** | Unified Buffer | 输出数据缓冲区 |
| `tmp_ub = T.alloc_ub((tileM, tileN), "uint8")` | `TBuf<TPosition::VECCALC> tmpBuf;` | **VECCALC** | Unified Buffer | 中间计算临时存储 |

Buffer / Queue 选型指令：

- 如果某个 UB buffer 位于 `T.serial` 这类逐轮处理的循环里，并且出现在 copy API 中作为“从 GM 读入到 UB”或“从 UB 写回到 GM”的对象，就默认翻译成 `TQue + VECIN/VECOUT`，并按 `AllocTensor -> LoadGmToUb(DataCopyPad) -> EnQue -> DeQue -> Compute -> EnQue/DeQue -> StoreUbToGm(DataCopyPad)` 组织。
- 如果某个 UB buffer 只服务于当前 `Compute` 内部的临时计算，例如中间结果、cast buffer、reduce buffer、broadcast 临时 buffer，就默认翻译成 `TBuf + VECCALC`，不要放进 `TQue`。
- 如果某个 UB buffer 是“在 `Init()` 或循环外从 GM 预加载一次，后续多个循环反复复用”的只读缓存，就默认翻译成 `TBuf`；由于它不经过队列同步，`DataCopyPad` 后必须补 `PipeBarrier<PIPE_MTE2>()`，再进入后续 Vector 计算。

可直接套用的经验规则：

- TileLang 里“位于 `T.serial` 中，且出现在 GM<->UB copy API 里”的 UB buffer，默认按 `TQue` 处理。
- TileLang 里“只在本轮计算内部使用”的 UB buffer，默认按 `TBuf` 处理。
- TileLang 里“先加载一次、后续复用多次”的 UB buffer，默认按 `TBuf + PipeBarrier<PIPE_MTE2>()` 处理。

#### C. `TQue` depth=0 约束

VECIN/VECOUT 队列（depth=0）必须使用引用形式：

```cpp
queue.AllocTensor<T>(localVar);
queue.DeQue<T>(localVar);
```

不要对 depth=0 队列使用返回值形式 API。

### 2. `Process()`：组织调度、三阶段流程与 Vector API

`Process()` 负责把实际工作负载循环和具体计算阶段串起来。

#### A. 工作负载循环与子块调度

AscendC 代码通常把工作负载循环封装在 `Process()` 中。

- `Process()` 管理分配给当前核的数据单元
- 若存在 block / sub-block 划分，通常也在 `Process()` 中结合 `GetSubBlockIdx()` 或 `vid` 计算数据偏移
- AIV 侧需要在 `Kernel` 类的 `Process()` 中用 `GetSubBlockIdx()` 区分数据偏移

#### B. 数据和缓冲区管理三阶段

- **CopyInX**：`AllocTensor` → `DataCopyPad`（GM→UB） → `EnQue`
- **ComputeX**：`DeQue` → 计算 → `EnQue`（或 `FreeTensor`）
- **CopyOutX**：`DeQue` → `DataCopyPad`（UB→GM） → `FreeTensor`

即使是纯 Vector 算子，也建议按这三阶段组织函数。

#### C. Vector 侧常用 API 注意事项

- `T.tile.cast(dst, src, mode, count)` 可稳定使用 `count`
- 参考 `archive_tasks/rms_norm/kernel/vector_tile.h`，Vector 侧 GM<->UB 搬运默认优先封装成 `DataCopyPad`
- `TBuf` 从 GM `DataCopyPad` 后若马上被 Vector 侧消费，必须插入 `PipeBarrier<PIPE_MTE2>()`
