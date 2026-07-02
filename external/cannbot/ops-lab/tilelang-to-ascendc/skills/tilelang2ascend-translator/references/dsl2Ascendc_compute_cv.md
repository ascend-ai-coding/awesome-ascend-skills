## C/V 融合计算总参考：Init 与 Process

本文档适用于 C/V 融合算子，或设备侧存在多个 `Scope`、多个计算阶段、AIC/AIV 协同的 AscendC 实现。
它不是用来替代 `vector` 或 `cube` 详细参考，而是给出融合场景下的组合阅读顺序与协同组织方式。
概览与判断规则见 `@references/dsl2Ascendc.md`。

---

## 第三章：Kernel 入口（C/V 融合总览）

### 1. 阅读顺序

- 纯 Vector 算子：只看 `@references/dsl2Ascendc_compute_vector.md`
- 纯 Cube 算子：只看 `@references/dsl2Ascendc_compute_cube.md`
- C/V 融合算子：先看本文，再分别看 `@references/dsl2Ascendc_compute_cube.md` 和 `@references/dsl2Ascendc_compute_vector.md`

### 2. kernel 入口形态

C/V 融合算子的入口通常同时接收输入、输出、workspace 和 tiling：

```cpp
extern "C" __global__ __aicore__ void kernel_custom(GM_ADDR ...inputs..., GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::TPipe pipe;
    KernelClass kernel;
    kernel.Init(..., workspace, tiling, &pipe);
    kernel.Process();
}
```

### 3. `vec_num` 与 block 组成

| DSL `vec_num` | KERNEL_TYPE | 每个 block 组成 | GetSubBlockNum() |
|:---|:---|:---|:---|
| 1 | `KERNEL_TYPE_MIX_AIC_1_1` | 1 AIC + 1 AIV | 2 |
| 2 | `KERNEL_TYPE_MIX_AIC_1_2` | 1 AIC + 2 AIV | 3 |

---

## 第四章：主 Kernel 类（C/V 融合）

参考 `archive_tasks/matmul_leakyrelu/kernel/matmul_leakyrelu.h`，C/V 融合主 `Kernel` 类建议按 `Init()` 和 `Process()` 两个大阶段组织。

### 1. `Init()`：接收 tiling 字段并初始化 GM / workspace / 子模块

`Init()` 主要负责：

- 读取并保存 tiling 字段
- 绑定输入 / 输出 GM tensor
- 初始化调度器与 workspace
- 分别初始化 Cube 子模块和 Vector 子模块

#### A. tiling 字段、GM 绑定与调度

常见模式：

- `CopyTiling(&tiling_, tilingGM)`
- `SetGlobalBuffer(...)` 绑定 A/B/C 等 GM tensor
- 根据 `GetBlockIdx()`、`GetSubBlockNum()` 派生 `coreIdx`
- 初始化调度器，如 `sched_.Init(...)`

#### B. workspace 与跨核协同

如果 C/V 之间通过 workspace 传递中间结果，通常在 `Init()` 中完成：

- workspace 基址和每个 core 的偏移计算
- ring buffer / `WorkspaceQueue` 初始化
- C/V 协同所需 flag 或队列的初始化

若存在跨核同步或 producer / consumer 关系，继续结合 `@references/dsl2Ascendc_cross_core_sync.md`。

#### C. 子模块初始化

融合场景下，通常同时存在：

- Cube 子模块：如 `matmul.h`
- Vector 子模块：如 `leakyrelu.h`、`scale.h`

推荐在 `Init()` 中按分支初始化：

- `ASCEND_IS_AIC` 分支初始化 Cube 子模块
- `ASCEND_IS_AIV` 分支初始化 Vector 子模块

### 2. `Process()`：组织调度、AIC/AIV 分支与阶段调用

`Process()` 负责把工作负载循环、AIC/AIV 分支和模块调用串起来。

#### A. 工作负载循环

常见骨架：

```cpp
__aicore__ inline void KernelClass::Process()
{
    int mIdx, nIdx;
    while (sched_.HasNext()) {
        sched_.Next(mIdx, nIdx);

        if ASCEND_IS_AIC {
            // Cube 侧
        }

        if ASCEND_IS_AIV {
            // Vector 侧
        }
    }
}
```

#### B. AIC 分支

AIC 分支通常负责：

- 从 GM 取当前 tile 的输入
- 获取 workspace 生产者槽位
- 调用 Cube 子模块，如 `mm_.ComputeBlock(...)`
- 释放生产者槽位或发送完成信号

#### C. AIV 分支

AIV 分支通常负责：

- 获取 workspace 消费者槽位
- 根据 `GetSubBlockIdx()` 计算当前子块偏移
- 从 workspace 中取本子块负责的数据
- 调用 Vector 子模块完成后处理并写回 GM
- 释放消费者槽位或发送完成信号

#### D. 何时拆单独子模块

当满足以下任一条件时，建议拆出单独计算子模块文件：

- TileLang 设备侧有多个职责清晰的 `Scope`
- 同时存在 Cube 计算阶段和 Vector 后处理阶段
- 需要在主 `Kernel` 类中复用某段计算逻辑

建议让 TileLang 中一个主要 `Scope` 对应 AscendC 中一个子模块。
