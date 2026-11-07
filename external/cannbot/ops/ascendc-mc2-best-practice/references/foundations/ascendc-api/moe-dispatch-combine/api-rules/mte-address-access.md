# MTE 通信窗口地址获取方法

本文直接说明 `winContext` / `mc2Context` 的来源、与 HCCL 约定的底层结构体、平台差异，以及它和数据区、状态区之间的关系。kernel 中的通信窗口地址获取逻辑由这些关系决定。

通用 DataCopy 等接口请查阅 `ascendc-api-best-practices` skill；本文只记录 MoE 直调场景下的 window 资源结构和地址获取逻辑。

## 先明确什么是 `winContext`

- `winContext` 不是 Ascend C 的通用公共 API 对象，而是 host 侧 HCCL / mc2 通信资源创建后返回给 kernel 的一块通信资源描述结构
- `winContext` 的底层布局不是 sample 自己发明的临时结构，而是当前算子和 HCCL 通信库之间约定好的平台固定结构
- 在 sample 里，kernel 侧通常把 host 传入的 `mc2Context` 转成 `__gm__ HcclOpParam *` 后使用；这里的 `HcclOpParam` 不是随意定义的业务结构，而是 compat 层对“当前平台 HCCL 通信资源结构”的统一别名
- 真正决定“数据区地址怎么拿、状态区地址怎么拿”的，是 `HcclOpParam` 结构体里保存了哪些 window 起始地址，以及这些地址和数据区 / 状态区的排布关系

可以把它理解成：

- host 侧通过 `HcclAllocComResourceByTiling` 拿到一块由 HCCL 负责创建和填充的通信资源上下文
- kernel 侧按平台约定，把这块内存解释成 `HcclA3OpResParam` 或 `HcclA5OpResParam`
- 工程中的结构体声明，只是把这份既有约定显式写出来，方便 kernel 侧按同一份 ABI 读取字段

> **重要警告**：sample 中的 `HcclA3OpResParam` 和 `HcclA5OpResParam` 是**精简定义**，仅覆盖 sample 需要访问的字段。A3 平台 SDK 实际的 `HcclOpResParam` 结构体（定义在 `moe_distribute_base.h`）包含更多中间字段（`mc2WorkSpace`、`hcomId[128]`、`rWinStart`/`rWinOffset`/`version`/`reservedStruct`/`topoInfo`/`config` 等），导致 `winExpSize` 之后的字段偏移与 sample 定义不同。sample 中的定义仅适用于 mock/test 场景（自建 context），**不适用于解析真实 HCCL 返回的 context**。若需解析真实 context，必须使用 SDK 完整的 `HcclOpResParam` 定义。同理，`HcclRankRelationResV2` 在 SDK 中还有尾部 `ListCommon nextTagRes` 字段（32B）。

因此，这里最重要的不是“工程里用了什么结构体名字”，而是“当前平台下 `mc2Context` 的字段布局是确定的，kernel 解析时必须与 HCCL 返回的结构保持一致”。

## `winContext` 从哪里来

生成或阅读直调工程时，默认按下面这条链路理解：

1. host 侧先组好完整的 mc2 tiling 数据，包括 `mc2InitTiling`、`mc2CcTiling` 和算子自己的 tiling 信息
2. host 侧通过通信域句柄和 stream，调用 `HcclAllocComResourceByTiling` 创建通信资源
3. `HcclAllocComResourceByTiling` 返回的 `mc2Context` 本身就是 device 地址，host 不需要再额外分配一份 device buffer 去拷贝它
4. host 侧把这个 `mc2Context` 直接作为 kernel 入参传下去
5. kernel 入口把这个 `mc2Context` 解释成 `__gm__ HcclOpParam *winContext`
6. 后续所有数据区 / 状态区地址计算，都从这个 `winContext` 出发

更完整的 host 侧语义链路是：

- 先通过 `HcomGetCommHandleByGroup` 或等价方式拿到 `HcclComm`
- 用 `Mc2CcTilingConfig` 或等价流程生成通信相关 tiling
- 调 `HcclAllocComResourceByTiling(commHandle, stream, tilingData, &mc2Context)`
- 检查返回值，并确认 `mc2Context != nullptr`
- 把 `mc2Context` 当成 kernel 的 `GM_ADDR` 入参直接传下去
- 这块上下文由 HCCL 管理生命周期，不应被当成普通 device buffer 自行释放或覆写

最小语义示意：

```c++
// host 侧：先拿 comm handle，再基于 tiling 创建通信资源
HcclComm commHandle = ...;
void *stream = ...;
MoeDispatchTilingData *tilingData = ...;
void *mc2Context = nullptr;
HcclAllocComResourceByTiling(commHandle, stream, tilingData, &mc2Context);

// 注意：mc2Context 返回的就是 device address，可直接传给 kernel
kernel<<<...>>>(..., (uint8_t *)mc2Context, ...);

// kernel 侧
__gm__ Mc2Kernel::HcclOpParam *winContext =
    (__gm__ Mc2Kernel::HcclOpParam *)mc2Context;
```

sample 里还有一个调试动作值得注意：host 侧可以把 `mc2Context` 从 device 拷回 host，打印 `rankId`、`rankDim`、`winSize`、`windowsIn[]` 等字段，帮助确认平台上的真实 window 结构。这不是运行时必须步骤，但对阅读和排错很有价值。

因此，`winContext` 的获取方式本身就是“host 显式创建并传入”，不是 kernel 内部临时查出来的公共上下文对象。

## `winContext` 的底层结构

`winContext` 在 kernel 侧按平台解释成 HCCL 约定的通信资源结构。这里直接按结构语义说明，不把 sample 当成定义主体。

也就是说：

- HCCL 决定 `mc2Context` 的实际内存布局
- kernel 必须按该平台对应的固定结构解释这块内存
- 工程代码只是把这层约定显式写成 `struct`，便于 kernel 代码按字段访问

当前工程中，compat 层的映射关系是明确的：

- A5 平台：`using HcclOpParam = MoeDispatchImpl::HcclA5OpResParam`
- A3 平台：`using HcclOpParam = MoeDispatchImpl::HcclA3OpResParam`

因此，`(__gm__ HcclOpParam *)mc2Context` 这一步不是普通的类型转换，而是在按平台把 HCCL 返回的上下文解释成约定好的通信资源结构。

### A5 平台：`HcclA5OpResParam`

这是 A5 平台下 `mc2Context` 对应的 HCCL 通信资源结构。kernel 至少依赖下面这些字段语义：

```c++
struct HcclA5OpResParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM];
    uint64_t xnAddr;
    uint64_t ckeAddr;
    uint64_t msAddr;
    uint64_t msSize;
};
```

字段语义可以直接理解为：

- `workSpace` / `workSpaceSize`：HCCL 为当前通信资源关联的 workspace 基址和大小
- `rankId` / `rankDim`：当前 rank 标识和通信域大小
- `winSize`：单个 rank 对应的完整 window 容量
- `windowsIn[]`：按 rank 编排的输入 window 基址表。对 MoE dispatch/combine 而言，这一组地址最关键
- `windowsOut[]`：另一组 window 指针表，是否参与当前实现取决于通信模式；当前直调样例主要围绕 `windowsIn[]` 取数和发状态
- `xnAddr` / `ckeAddr` / `msAddr` / `msSize`：由 HCCL 管理的其他通信资源地址和规模信息，不是当前 MoE window 地址推导的主路径

对当前 skill 最关键的结论是：

- `windowsIn[rankId]` 给出的不是“纯数据区地址”，而是该 rank 完整 window 的起始地址
- A5 平台的数据区和状态区不是两个独立字段，而是共享同一块 `windowsIn[rankId]`
- 状态区位于前部固定区域，数据区位于其后。当前工程里的固定偏移常量为：

```c++
constexpr uint64_t A5_MTE_STATE_WIN_SIZE = 1024UL * 1024UL;
```

- 因此 kernel 取数据区地址时要在 `windowsIn[rankId]` 基础上再加 `A5_MTE_STATE_WIN_SIZE`
- 换句话说，A5 平台下的 `winContext` 本质上就是“按 rank 编排的完整 window 描述表”

### A3 平台：`HcclA3OpResParam`

这是 A3 平台下 `mc2Context` 对应的 HCCL 通信资源结构。kernel 至少依赖下面这些字段语义：

```c++
struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HcclA3OpResParam {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
};
```

字段语义可以直接理解为：

- `localUsrRankId` / `rankSize`：当前 rank 标识和通信域大小
- `winSize`：单个 rank 的 window 容量
- `localWindowsIn`：本卡数据 window 基址
- `localWindowsOut`：本卡另一组 window 基址，是否使用取决于通信模式
- `winExpSize` / `localWindowsExp`：状态 window 的规模和本卡状态 window 基址
- `remoteResNum` / `remoteRes[]`：远端资源关系表。`remoteRes[rankId].nextDevicePtr` 指向远端关系结构 `HcclRankRelationResV2`
- `HcclRankRelationResV2.windowsIn`：远端 rank 数据 window 基址
- `HcclRankRelationResV2.windowsOut`：远端 rank 另一组 window 基址
- `HcclRankRelationResV2.windowsExp`：远端 rank 状态 window 基址

对当前 skill 最关键的结论是：

- A3 平台把本卡数据区和本卡状态区直接拆成了不同字段：`localWindowsIn` 和 `localWindowsExp`
- 远端 rank 的 window 地址不直接平铺在顶层结构里，而是通过 `remoteRes[rankId].nextDevicePtr` 继续跳转
- 因此 A3 平台的“本卡地址”和“远端地址”获取路径天然不对称
- 换句话说，A3 平台下的 `winContext` 更像“本卡 window + 远端关系入口表”

## 为什么这里要把结构体写死到平台级别

- 因为 kernel 侧对 `mc2Context` 的读取不是反射式或自描述式的，而是按字段偏移直接访问
- 只要平台确定，`HcclOpParam` 对应哪套结构、哪些字段表示 rank/window/远端关系，也随之确定
- 这也是 compat 层能够稳定封装 `GetRankId()`、`GetBaseWindAddrByRankId()`、`GetBaseWindStateAddrByRankId()` 的前提

实践上应把这层理解为“与 HCCL 的 ABI 约定”，而不是“sample 里的一个可自由改写的数据结构”。如果字段顺序、字段语义或平台映射关系被改坏，kernel 对 window 的解析就会直接失效。

## `winContext` 和数据区 / 状态区的关系

先记住一个原则：`winContext` 只提供每张卡 window 的基址；真正的数据区布局、状态区布局、slot 偏移，仍要由 layout 公式决定。

### Dispatch 场景

- 向目标 rank 发送 token：先从 `winContext` 取目标 rank 的数据区基址，再叠加 `sourceRank -> localExpert -> tokenSlot` 的布局偏移
- 向目标 rank 发布状态：先从 `winContext` 取目标 rank 的状态区基址，再叠加 `localExpert -> sourceRank -> stateSlot` 的布局偏移
- 等待本地接收状态：从 `winContext` 取本卡状态区基址，再按本地状态矩阵布局轮询
- 从本地 window 回搬数据：从 `winContext` 取本卡数据区基址，再按接收布局计算偏移

### Combine 场景

- 向来源 rank 回传数据：先从 `winContext` 取目标 rank 数据区基址，再按 `token -> slot` 布局定位
- 向来源 rank 发布 ready：先从 `winContext` 取目标 rank 状态区基址，再按 `token -> slot` 状态布局定位
- 等待本地 token 聚合条件满足：从 `winContext` 取本卡状态区基址，再按 token 级状态布局轮询

换句话说，`winContext` 决定“去哪一张卡的哪一块 window”，layout 文档决定“进了这块 window 以后具体偏移多少”。

## 平台差异（关键）

两个平台的 window 地址语义不同。compat 层可以统一访问方式，但底层差异仍然决定了数据区和状态区基址的真实含义。

### A3 平台（默认）

`HcclOpParam = HcclA3OpResParam`，相关结构展开如下：

```
HcclA3OpResParam
    .localWindowsIn                  → 本卡自己的数据接收 window
    .localWindowsExp                 → 本卡自己的状态 window（兼做 GetStatusDataSpaceGm）
    .remoteRes[rankId].nextDevicePtr → 指向远端 rank 的 HcclRankRelationResV2

HcclRankRelationResV2
    .windowsIn   → 远端 rank 数据 window
    .windowsOut  → 远端 rank 另一组 window
    .windowsExp  → 远端 rank 状态 window
```

- 本卡数据区基址：`localWindowsIn`
- 本卡状态区基址：`localWindowsExp`
- 远端数据区 / 状态区基址：通过 `remoteRes[rankId].nextDevicePtr` 取得远端关系结构，再读其中的 `windowsIn` / `windowsExp`

### A5 平台（`__NPU_ARCH__ == 3510`）

`HcclOpParam = HcclA5OpResParam`，相关常量和字段展开如下：

```
HcclA5OpResParam
    .windowsIn[rankId]              → 每张卡的完整 window 起始地址（包含状态区 + 数据区）

A5_MTE_STATE_WIN_SIZE = 1024UL * 1024UL
    状态区位于前 1MB，数据区位于这 1MB 之后
```

- 每张卡的完整 window 起始地址：`windowsIn[rankId]`
- 状态区基址：`windowsIn[rankId]`
- 数据区基址：`windowsIn[rankId] + A5_MTE_STATE_WIN_SIZE`
- 因此 A5 平台拿数据区地址时必须额外跳过前 1MB 状态区

## compat 封装的职责

工程中的 compat 封装没有创造新的通用 API，它只是把上面的平台差异收敛成统一的读取方式，减少 kernel 主流程里的平台分支。

本地工程里通常会提供一层 compat helper，例如：

- 读取当前 rank / rank 数 / 单 rank window 大小
- 读取本卡状态区基址
- 读取指定 rank 的数据区基址
- 读取指定 rank 的状态区基址

这些 helper 本质上是工程代码对 `winContext` 的一层适配，不代表 HCCL 会直接提供同名 kernel API。

## 常见封装分层

围绕 `winContext` 的读取和使用，常见会拆成四层：

### 1. 上下文解析层

这一层把 `GM_ADDR mc2Context` 解释成可以读取字段的 `winContext`，并缓存最常用的上下文信息。

这一层通常至少封装：

- `winContext` 指针绑定
- `rankId` 读取
- `rankDim` / `winSize` 读取
- 平台差异选择（A3 / A5）

sample 里的 `InitHcclContextByAddr` 就属于这一层；它对应的是“统一初始化通信上下文”这类职责。

### 2. window 基址获取层

这一层只回答“本卡 / 远端的数据区基址和状态区基址分别在哪里”，不掺入 expert、slot、token 等业务偏移。

这一层通常至少封装：

- 本卡状态区基址
- 指定 rank 的数据区基址
- 指定 rank 的状态区基址

sample 的 `GetStatusDataSpaceGm`、`GetBaseWindAddrByRankId`、`GetBaseWindStateAddrByRankId` 属于这一层。这里对应的是“数据区基址”和“状态区基址”分开封装，而不是在主流程里反复手写平台分支。

### 3. 业务地址拼装层

这一层在拿到基址后，再结合 layout 公式拼出 dispatch / combine 真正要访问的 slot 地址。

这一层通常至少封装：

- dispatch 数据地址拼装
- dispatch 状态地址拼装
- 本地回搬地址拼装
- combine 回传地址 / 状态地址拼装

sample 的 `GetDispatchDataAddr`、`GetDispatchStateAddr`、`GetLocalWindowDataAddr` 属于这一层。这里对应的是“先拿基址，再叠加布局偏移”的组织方式。

### 4. 通信动作层

这一层把状态写入、状态等待、状态复位这些动作与地址获取解耦。

这一层通常至少封装：

- 发布远端状态
- 轮询本地状态区
- 状态消费后复位

sample 的 `SetRemoteStatus`、`WaitRemoteStatus`、`ClearLocalStatus` 属于这一层。这里对应的是“通信动作层依赖地址拼装层，但不直接关心平台字段细节”。

## kernel 中的最小使用顺序

### 初始化（必须第一步，在 Init() 中调用）

```c++
// kernel 入口传入 mc2Context（GM_ADDR 类型）
comm_.InitHcclContextByAddr(tilingData->mc2Context, tilingData->tilingInfo.epWorldSize);
// 等价于：
winContext = (__gm__ Mc2Kernel::HcclOpParam*)mc2Context;
rankId     = 从 winContext 读取当前 rank；
rankDim    = 来自 tiling，或与 winContext 中的 rank 数保持一致；
```

这一步的本质不是“必须调 sample 的某个初始化函数”，而是先完成“上下文解析层”的工作：把 host 传下来的 `mc2Context` 解释成能读 window 基址的结构体，再开始做后续地址计算。

### 发送数据到目标 rank

```c++
// 第一步：通过 window 基址获取层，拿目标 rank 数据区基址
// 第二步：通过业务地址拼装层，叠加 dispatch 数据区布局偏移
```

### 发布状态到目标 rank

```c++
// 第一步：通过 window 基址获取层，拿目标 rank 状态区基址
// 第二步：通过业务地址拼装层，叠加 dispatch 状态区布局偏移
```

### 轮询本地接收状态

```c++
__gm__ int32_t *localStateBase = ...; // 通过 window 基址获取层，解出本卡状态区基址
// 再通过 DataCopyPad 搬回 UB 做判断，不要直接 GetValue
```

### 从本地 window 回搬数据

```c++
// 第一步：通过 window 基址获取层，拿本卡数据区基址
// 第二步：通过业务地址拼装层，叠加 dispatch 数据区布局偏移
```

## 易错点

1. **A5 数据区偏移 1MB**：A5 平台数据区基址 = `windowsIn[rankId] + 1MB`，不加偏移将读写到状态区。

2. **A3 远端访问通过链表**：A3 远端 rank 地址通过 `remoteRes[rankId].nextDevicePtr` 间接获取，不是数组直接索引。

3. **不要把 sample helper 当成公共标准 API**：sample 中的 compat 函数只是对 `winContext` 读取逻辑的封装；底层仍然是 `HcclOpParam` 的字段关系在决定本卡 / 远端的数据区和状态区基址。

4. **不要在直调工程中使用 `GetHcclContext<...>()`**：direct invoke 工程 `mc2Context` 由 host 显式传入；registry invoke 才是框架自动注入上下文那套路径。

5. **本地状态区和本地数据区不是一回事**：A3 平台本卡状态区是 `localWindowsExp`，本卡数据区是 `localWindowsIn`；A5 平台两者共享同一块 `windowsIn[rankId]` 基址，但状态区在前、数据区在后。

6. **读取本地状态区必须走 GM 可见路径**：不能在 `SyncAll` 后直接 `GetValue()` 读取共享状态，必须先搬回 UB 再读取。

7. **先拿 window 基址，再套 layout 公式**：`winContext` 不会直接给你某个 slot 的最终地址，它只提供 window 基址；真正的 slot 地址还要结合 `window-memory-layout.md` 里的布局公式计算。

## 与其他文档的关系

- 地址偏移计算公式：`window-memory-layout.md`
- DataCopyPad、SyncAll 使用规则：`sync-and-visibility.md`
- sample 中的 helper 分层：`../samples/sample-helper-map.md`
- compat 头文件实现：`../samples/moe_dispatch_direct_invoke_sample/include/moe_dispatch_base_compat.h`
