# arch22 FlashAttention Scalar 与启动开销设计

> 适用架构：arch22（DAV_2201 / Atlas A2/A3）。只迁移有 arch22 源码/API 证据的机制，不迁移 A5 的 SCALAR/SCALARLDST 百分比阈值和收益数字。

## 1. 优化目标

针对 FIA 启动、FlashDecode 与长循环中的取指、Init 和 Scalar 重复计算开销，只在目标 trace 证明对应瓶颈时减少热路径指令。该路线不改变 FA 的 QK/PV 数学语义和 tiling ABI；arch22 不使用 A5 的 SCALAR/SCALARLDST 百分比阈值选路。

## 2. 架构概览

| 阶段 | 待区分的开销 | 候选手段 |
|---|---|---|
| 启动与 FlashDecode | ICache miss、同步等待或 Scalar GM load | 取指停顿成立时调整预取 |
| Init | 固定布局的 buffer 仍走动态管理 | 静态 LocalTensor，并证明地址区间不重叠 |
| TilingData 读取 | 大结构体栈拷贝或热循环反复经成员解引用 | GM 指针直读、热点字段局部缓存 |
| KV 热循环 | 地址乘除法和公共前缀重复计算 | 不变量外提、主尾块分离 |

## 3. 关键参数 / 修改点

| 手段 | 关键修改 | 前置与不变量 |
|---|---|---|
| ICache 预取 | 调整预取位置与长度 | PC/timeline 先证实取指停顿；长度按目标 case A/B |
| 静态 LocalTensor | 固定位置、偏移和容量 | buffer 布局编译期已知，覆盖所有 dtype/layout/feature 分支 |
| TilingData 直读 | 入口持有 GM 指针，热点字段转局部 `const` | 不改变结构体 ABI；不把指针变成热路径的二级成员访问 |
| 循环不变量外提 | 预乘 stride/blockSize、缓存 group/head 和 workspace 公共前缀 | 值在循环期间不变，paged/non-paged 分支同时适配 |

## 4. 核心计算循环 / 关键代码

### ICache 预取

FIA arch22 kernel 已在 FlashDecode 路径调用 `AscendC::ICachePreLoad`。仅在 timeline/PC 证据显示大函数或循环回跳存在取指停顿时调整预取位置/长度；同步等待、Scalar GM load 和 ICache miss 必须先区分。

```cpp
// arch22 FIA 现有形态
static constexpr int64_t fdPrefetchLen = 2;
AscendC::ICachePreLoad(fdPrefetchLen);
```

`2` 是当前实现参数，不是 arch22 通用最优值。

### 静态 LocalTensor

L1/L0 buffer 的数量、大小和偏移均为编译期常量时，可直接构造 `LocalTensor<T>(TPosition, offset, size)`，减少通用 `InitBuffer` 头开销。动态 TQue、运行时 tile 或可变 feature 共享 buffer 仍应保留 TPipe 管理。

改造前必须输出完整地址区间表，证明不同 dtype/layout/feature 分支下均不重叠。

```cpp
// 仅固定布局可用；offset/size 由目标 buffer 预算给出。
LocalTensor<T> qLocal(TPosition::A1, qOffset, qSize);
```

### TilingData 读取

arch22 regular FA 已有从 `__gm__ FAInferTilingData*` 直接读取并缓存热点字段的实现。优先级：

1. 不复制整块大 TilingData 到栈；
2. 入口用局部 GM 指针；
3. 热循环字段在入口缓存为局部 `const`；
4. 不把 GM 指针再存成频繁经 `this` 访问的成员。

另一路 FIA v3 使用 `GET_TILING_DATA_WITH_STRUCT`，不能全局机械替换；应按实际结构大小、调用频次和编译结果选择。

```cpp
// 需避免的整结构体拷贝
auto tilingLocal = *reinterpret_cast<__gm__ FAInferTilingData *>(tiling);
// 候选形态：保留 GM 指针，再缓存实际热点字段
auto tilingPtr = reinterpret_cast<__gm__ FAInferTilingData *>(tiling);
```

### 循环不变量外提

可以迁移的 A5 方法仅限编译器通用机制：

- workspace 基址公共前缀外提；
- stride、blockSize 乘积在循环外预乘；
- 不变的 head/group 参数入口计算；
- 热路径成员读一次到局部变量；
- 首块/主循环/尾块分离。

禁止直接使用 A5 的字段名、`SCALAR% >= 80%`、`SCALARLDST% >= 95%` 或预期收益区间。arch22 是否受益以反汇编/trace 与逐 case A/B 为准。

## 5. 效果与适用边界

| 观察指标 | 可接受的结论 |
|---|---|
| 取指/Scalar trace | 与目标机制相符的停顿或指令数下降；排除仅等待转移 |
| 逐 case Task Duration | 与精度同时通过后确认收益；未上板不写固定加速比 |
| buffer 与 ABI | 静态地址区间闭合，TilingData 结构与分支保持一致 |

上述机制有 arch22 实现证据，但本页没有独立的 A2/A3 全集实测收益；A2 与 A3 需分别记录编译、精度、trace 和性能结果。

### 不迁移项

- `__ssbuf__` CV TilingData 转发；
- RegBase/MicroAPI/VF；
- A5 SIMT；
- A5 DataCache line 批量 `GetValue` 的固定 batch size；
- A5 scalar/subbound 的性能数字和判定阈值。

### 验证与自检

- ICache 结论已排除同步等待和 Scalar load。
- 静态 tensor 地址区间覆盖所有模板分支。
- TilingData 的局部指针/局部字段没有改变 ABI。
- 外提变量在循环期间确实不变，paged/non-paged、prefix、mask 分支同步修改。
- A2/A3 分开记录编译、精度、性能状态。

### 静态检视判据

<a id="prc-fa-003"></a>
#### 启动与热循环 Scalar 减载（PRC-FA-003）

先用 PC/timeline 区分 ICache miss、`WaitFlag` 和 Scalar GM load；没有取指证据不报预取缺失。检查大 TilingData 是否整块拷栈、GM 指针是否存为成员后在热循环反复两级解引用，以及 stride、workspace 基址、group/head 参数等循环不变量是否重复计算。布局编译期已知时可考虑静态 LocalTensor；动态 TQue、运行时 tile 或 feature 分支不机械替换。

finding 应给出 PC/timeline 或反汇编证据、热点执行次数、结构体大小与拷贝形态、可外提表达式及其不变量证明。arch22 源码已有 `ICachePreLoad`、静态 L1/L0 tensor 和 GM tiling 指针直读证据，但 A5 的 `SCALAR%`/`SCALARLDST%` 阈值与加速比不能当作 arch22 收益承诺。

### 实现依据

FIA arch22 的启动与 FlashDecode 路径提供预取实例，regular FA 提供 GM TilingData 指针读取实例。具体预取长度、结构体读取形态和收益仍需在目标源码与 case 中取证。
