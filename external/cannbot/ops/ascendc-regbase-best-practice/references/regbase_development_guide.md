---
title: RegBase 开发指南
purpose: 串联 RegBase 算子的需求分析、参考实现检查、Host/Kernel/UB/VF 分层、tail/mask 与同步设计。
read_when:
  - 需要从需求描述形成 DAV_3510 / RegBase 设计或实现方案。
  - 需要检查 RegBase 工作流是否覆盖参考实现、分层、数据流、VF body 和审查要点。
not_for:
  - 替代 SDK header 或真实源码做 API 签名验证。
keywords:
  - RegBase
  - DAV_3510
  - design
  - workflow
next_reads:
  - api/index.md
  - pitfalls/index.md
  - dev-experience/index.md
  - reference-ops/open_source_operator_table.md
depth: foundation
topic_type: guide
---

# RegBase 开发指南

这是开发 RegBase 算子的实践主线。需要从需求描述快速走到可实现的 `DAV_3510 / RegBase` 方案时使用它。

本指南刻意以真实算子代码为基础。不要把 RegBase 开发当成空白页设计。

## 1. 从真实 RegBase 参考实现开始

不要凭想象启动新 RegBase 算子。写设计或代码前，至少检查一个现有 RegBase 算子实现。

通过 [[reference-ops/open_source_operator_table]] 找候选算子，先确认目录中存在 RegBase route、RegBase body 或 VF 证据，再阅读真实源码：

- 外层入口和路由：通常是 `op_kernel/*_apt.cpp` 或 `op_kernel/*.cpp`。
- RegBase 主体：通常是 `op_kernel/arch35/*_regbase.h`、其他 `arch35` 头文件，或带有 `__simd_vf__` / `__VEC_SCOPE__` 的实现文件。

第一次阅读至少要确认：

1. kernel 入口如何声明，以及如何用 `TILING_KEY_IS(...)` 路由。
2. `Init` 与 `Process` 如何分工。
3. UB 层 `CopyIn -> Compute -> CopyOut` 如何组织。
4. `Compute` 如何进入 `__VEC_SCOPE__`，并使用 `RegTensor`、`MaskReg`、`LoadDist`、`StoreDist`。
5. dtype 专属 VF 路径如何拆分。

适合作为起点的参考形态：

- 小型 unary RegBase 实现，入口 `_apt.cpp` 清晰，主体位于 `arch35/*_regbase.h`。
- 融合寄存器链实现，显式展示 VF 路径，而不是被框架结构完全遮住。
- 只有任务确实需要手工 overlap 或 ping-pong 协调时，才优先读显式同步的高级实现。

参考代码用于学习现有 RegBase 写法，不等于允许复制整段实现。

## 2. 正确的分层模型

RegBase 最常见错误是把所有层压成一层。应使用四层模型：

1. **Host 与 tiling 层**
   - 选择 dtype 路由、block 切分、tile 形状和 launch 参数。
2. **Kernel outer shell**
   - 管理 `TPipe`、`GlobalTensor`、`TQue`、`LocalTensor`、`Init` 和 `Process`。
3. **UB 层数据流**
   - 通过 `CopyIn -> Compute -> CopyOut` 在 GM 与 UB 之间搬运 tile。
4. **VF / 寄存器计算内层**
   - 运行在 `__VEC_SCOPE__` 内。
   - 使用 `RegTensor`、`MaskReg`、`LoadDist`、`StoreDist`，以及 `AscendC::Reg::*` 或常见别名 `AscendC::MicroAPI::*` 暴露的 VF-safe 寄存器计算调用。

这意味着 RegBase 不是“只有 VF”，也不是“不使用队列”。真实 RegBase 实现通常仍然有外层 shell 和 UB stage 搬运。RegBase 的身份来自 compute 核心写成 VF/寄存器计算代码，而不是以 `LocalTensor` 为中心的 compute 主体。

典型文件职责：

| 层 | 典型位置 | 职责 |
|---|---|---|
| Host / tiling | host 代码或 tiling 文件 | shape、dtype route、blockDim、tile、tail |
| kernel entry | `*_apt.cpp` 或 kernel `.cpp` | 解析 tiling，选择 `TILING_KEY_IS(...)` 分支 |
| RegBase body | `arch35/*_regbase.h` | 定义 `Init`、`Process`、`CopyIn`、`Compute`、`CopyOut` |
| VF body | header 内 `__simd_vf__` 或 `__VEC_SCOPE__` | 使用 `RegTensor` / `MaskReg` 运行寄存器级计算 |

推荐设计顺序：

1. 先确定算子数学和 dtype route。
2. 再确定 Host tiling packet 需要传什么。
3. 再确定 kernel route 和 outer shell。
4. 再设计 UB tile pipeline。
5. 最后写 VF body 的寄存器链。

DESIGN.md 至少记录：

- 触发 RegBase 的依据。
- Host / Kernel / UB / VF 的职责边界。
- 每条 route 的 dtype、shape 范围和 tail 策略。
- VF body 是否独立成函数或宏，以及 `asc_vf_call` / `__simd_vf__` 的调用关系。

相关细节见：

- [[dev-experience/regbase_programming_notes]]

## 3. 先准备 Host 与 tiling packet

写 VF 代码前，Host 侧必须能说明：

- 当前 dtype 路由是什么。
- 应进入哪个 `TILING_KEY_IS(...)` 分支。
- 使用多少 block/core。
- 每个 block 负责多少工作量。
- tail 元素如何处理。

简化路由示例：

```cpp
extern "C" __global__ __aicore__ void op_entry(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TilingData);
    GET_TILING_DATA_WITH_STRUCT(TilingData, tilingData, tiling);

    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        KernelOp<DTYPE_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    }
}
```

通过参考算子判断：

- 一个 `TILING_KEY` 是否足够，还是需要多个路由。
- dtype 差异只影响 VF 内部，还是也影响 UB 规划。
- 任务只需要一个 kernel 分支，还是要拆成 simple/advanced 路由。
- 简单 elementwise 通常是一个 `TILING_KEY` 加一个 dtype route。
- 多 dtype 或多 shape family 可能需要多个 `TILING_KEY` 或模板实例。
- 融合链路通常 outer shell 不复杂，但 VF body 中寄存器临时变量更多。
- 高级 overlap 只有在参考实现能证明必要时，才在 pipeline 层引入显式同步。

更多 tiling packet、effective length、aligned length 和 tail 检查见 [[dev-experience/tiling_review_notes]]。

## 4. 构建 Kernel outer shell

Outer shell 中的 RegBase 代码仍然很像普通 kernel。典型职责包括：

- 绑定 GM tensor。
- 初始化 `TPipe`。
- 分配 `TQue` 或本地 UB buffer。
- 决定 tile loop count 和 tail count。
- 调用 `CopyIn`、`Compute`、`CopyOut`。

最小结构：

```cpp
class KernelOp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t count);
    __aicore__ inline void Compute(uint32_t count);
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t count);

    TPipe* pipe_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    TQue<QuePosition::VECIN, 2> queIn_;
    TQue<QuePosition::VECOUT, 2> queOut_;
};
```

不要把寄存器级数学直接塞进 `Process`。保持 outer shell 可读，并让 VF 核心作为独立 stage 可见。

## 5. 组织 UB 层 `CopyIn -> Compute -> CopyOut`

在 UB 层，RegBase kernel 仍然需要稳定的 tile pipeline：

1. `CopyIn`：把当前 tile 从 GM 搬到 UB-local state。
2. `Compute`：消费 UB-local tile 并运行 VF body。
3. `CopyOut`：把结果 tile 从 UB-local state 写回 GM。

该层关注 UB ownership，而不是寄存器指令。这里常见对象是：

- `GlobalTensor<T>`
- `LocalTensor<T>`
- `TQue`
- 从 UB buffer 派生出的 raw `__ubuf__` 指针
- 参考实现中也可能写 `__local_mem__` 指针；CANN 9.0.0 的 `reg_compute` 公共头里将它定义为 `__ubuf__` 别名

因此，kernel 层应使用“CopyIn / Compute / CopyOut”语言；而“Load / Compute / Store”属于更深的 VF body。

参见：

- [[dev-experience/regbase_programming_notes]]

## 6. 把寄存器级数学放进 VF

`Compute` 内的 RegBase 专属部分是 VF body。新算子通常通过 `asc_vf_call` 调用 `__simd_vf__` 入口，也可能在参考实现中看到 `__VEC_SCOPE__` 片段；两者的核心都是操作寄存器对象，而不是把 UB buffer 当成最终 compute 模型。

代表性片段：

```cpp
__simd_vf__ inline void ThresholdVf(__ubuf__ float* inAddr, __ubuf__ float* outAddr,
                                    uint32_t size, float lambd)
{
    using namespace AscendC::Reg;

    constexpr uint32_t VL = 64;
    uint32_t vfLoopNum = (size + VL - 1) / VL;
    RegTensor<float> vregIn, vregAbs, vregZero, vregOut;

    for (uint32_t i = 0; i < vfLoopNum; i++) {
        uint32_t remain = size - i * VL;
        uint32_t active = remain > VL ? VL : remain;
        MaskReg mask = UpdateMask<float>(active);
        MaskReg cmpMask;

        LoadAlign<float, LoadDist::DIST_NORM>(vregIn, inAddr + i * VL);
        Abs(vregAbs, vregIn, mask);
        Compares<float, CMPMODE::GT>(cmpMask, vregAbs, lambd, mask);
        Duplicate(vregZero, 0.0f, mask);
        Select<float>(vregOut, vregIn, vregZero, cmpMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(outAddr + i * VL, vregOut, mask);
    }
}
```

核心写法是：

- 用 `LoadAlign` / `LoadUnAlign*` 等 VF load API 从 UB address load 到 `RegTensor`。
- 在 `RegTensor` 上计算。
- 用 `MaskReg` 表示有效通道。
- 用 `StoreAlign` / `StoreUnAlign*` 等 VF store API 从 `RegTensor` store 回 UB address。

fp16/bf16 路径的常见模式：

1. 从 UB load packed narrow type。
2. unpack 或 cast 到更宽的寄存器类型。
3. 必要时以 fp32 计算。
4. cast 回窄类型。
5. pack 并 store。

参见：

- [[api/regbase_api_whitelist]]
- [[dev-experience/regbase_programming_notes]]

## 7. 按域规划 buffer

好的 RegBase 设计必须说明每类状态属于哪个域。

典型域：

- **GM 域**
  - 输入 tensor。
  - 输出 tensor。
  - tiling packet。
- **UB 域**
  - stage tile。
  - ping-pong 输入/输出 buffer。
  - tile stage 之间共享的临时 UB 存储。
- **寄存器域**
  - `RegTensor`。
  - `MaskReg`。
  - 短生命周期的 cast、compare、select 和 fused arithmetic 状态。

两个实用规则：

- 如果数据会在一个 tile 内被多个 VF step 复用，通常应放在 UB 或寄存器，而不是 GM。
- 如果某个值只服务于一次 VF 链，优先保持为寄存器对象，而不是额外物化一个 UB buffer。

参见：

- [[dev-experience/regbase_programming_notes]]
- [[reference-ops/open_source_operator_table]]

## 8. 按层处理同步

同步决策只有在先确定所在层之后才有意义。

在 **UB pipeline 层**：

- stage 交接通常由 queue 生命周期或 `Process` 中的显式 stage 顺序表达。
- 当 kernel 使用基于 queue 的 UB 搬运时，`EnQue` / `DeQue` 表示本地 stage 就绪。

在 **VF 层**：

- `MaskReg` 是有效通道控制，不是同步。
- `LoadDist` / `StoreDist` 描述 load/store 分布，不是 stage 间 fence。
- 大多数简单 VF body 不需要显式事件 flag。

在 **高级 pipeline 层**：

- 当设计手工 overlap MTE 与 vector work 时，才会出现显式 `SetFlag` / `WaitFlag`。
- 只有确认参考算子确实需要时才使用。

在 **cross-core 层**：

- cross-core flag 用于真实跨 core 协调。
- 不要把它当成本地 UB stage 管理的替代品。

参见：

- [[api/regbase_api_sync]]

## 9. 写代码前检查什么

第一版实现前必须能回答：

1. 选了哪个真实 RegBase 参考算子。
2. 它的 outer shell 文件和 RegBase body 文件在哪里。
3. 当前 `TILING_KEY` 和 dtype route 如何设计。
4. `CopyIn -> Compute -> CopyOut` 如何组织。
5. VF loop 形状和 `VL` 是什么。
6. 哪些值留在寄存器，哪些值留在 UB。
7. 任务只需要默认 stage 顺序，还是需要高级显式同步。

回答不了这七项，说明设计仍然太模糊。

## 10. 常见错误

- 把整个 kernel 写成一个巨大的 VF 函数。
- 混淆 UB 层搬运与寄存器级计算。
- Host 与 tiling packet 尚未稳定就写 VF 内部。
- 在数据流和路由清楚前先选 API。
- 误以为“RegBase 就是不需要 `TPipe`、`TQue`、`LocalTensor`”。
- 复制完整参考实现，而不是借鉴其已验证写法。
- 只用参考算子的名称做匹配，不读真实 kernel 代码。

## 相关文档

- [[dev-experience/regbase_programming_notes]]
- [[dev-experience/tiling_review_notes]]
- [[api/regbase_api_whitelist]]
- [[api/regbase_api_reference]]
- [[reference-ops/open_source_operator_table]]
