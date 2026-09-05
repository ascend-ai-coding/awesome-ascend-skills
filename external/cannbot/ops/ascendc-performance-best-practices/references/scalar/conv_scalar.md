# Conv 算子 Scalar 优化

Conv 类算子的 Scalar 优化围绕一个核心思路：**利用场景特化，将运行时可变量尽可能转化为编译期常量**，从而减少 Scalar 侧的地址计算与分支判断，降低 Load/Store 指令占比。

---

## 1. 固定循环轴与泛化范围

### 原理

通用 Conv 实现需要考虑任意 N/H/W/C/K 组合，循环轴排列、tiling 粒度、weight 加载策略都做成参数化的，这导致 Scalar 在每个循环层级都要做动态地址计算和分支。而 **depthwise** 和 **小 case** 场景下，输入规模受限，循环轴是固定的，比如上面2种场景下weight 在单核内是全载（fullload）的 —— 此时可以去除不必要的循环轴，降低scalar开销。

> **小 case** 定义：FMAP 和 Weight 能在一轮搬运中全载到 L1 的场景。此时不再需要逐 tile 切分 weight/fmap，循环轴可大幅精简。

### Depthwise 场景的循环轴

```
for group in [0, groupOpt):           // 组间循环（AIV/AIC 交替）
    for batch in [0, batchCount):     // batch 循环
        for m_AL1 in [0, actualM):    // 输出空间 M 维（ho×wo），按 AL1 tile 切
                for k in [0, kTotal): // 输入通道 K 维（K = cin/groups × kh × kw）
```

Weight 全载：每组 weight 的完整 `[cout/groups, cin/groups, kh, kw]` 在 `SetupGroup` 阶段一次性加载到 L1（B1 buffer），整个 group 内的 batch 迭代和 M 迭代复用它。

### 小 case 场景的循环轴

```
for m in [0, actualM):       // 输出空间 M 维
    for k in [0, kTotal):    // 输入通道 K 维（K = cin × kh × kw）
```

Weight 全载：`LoadWeightL1()` 在 Process 开头一次性将本 core 的 `[singleCoreCo, kTotal]` weight 加载到 L1，之后 M-loop 和 K-loop 中不再搬运 weight。

### 为什么省 Scalar

去除无效的循环轴后：代码段长度减少（少一层循环 = 少一套地址计算逻辑），运行过程中的变量减少（循环变量、临时偏移量不再需要），运行时额外计算减少（不再需要每层循环的动态乘加来推导地址）。指令数显著下降。

---

## 2. 去除 Queue / TBuffer / Tpipe，改用 SetFlag/WaitFlag + LocalTensor

### 原理

Queue / TBuffer / Tpipe 是 Ascend C 的**高级抽象**，它们在底层展开为大量 Scalar 指令：队列状态查询、缓冲区索引更新、自动同步插入。在 hot loop 中使用这些抽象，Scalar 需要：
- 维护 Queue 的读/写指针和状态字
- 为 TBuffer 做地址重映射
- 在 Tpipe 的各阶段之间插入隐式同步

当场景足够简单（固定循环轴、全载 weight）时，可以直接用底层原语替代：

### 替代对照

```
Queue → SetFlag / WaitFlag（显式硬件事件同步）
TBuffer → LocalTensor（直接指定 TPposition + 偏移，手动管理缓冲区）
Tpipe  → 不需要（Queue 和 TBuffer 都去掉了，Tpipe 只剩空壳）
```

### 简要代码模式

**同步：Queue → SetFlag/WaitFlag**

```
// 原模式（Queue）
// EnQue<MTE1_M>(queue, ...);

// 新模式（显式事件同步）
uint16_t pingPong = 0;
SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(pingPong));    // 生产者信号
WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(pingPong));   // 消费者等待
SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(pingPong));    // 消费者完成信号
pingPong ^= 1;
```

**缓冲区：TBuffer → LocalTensor**

```
// 原模式（TBuffer）
// TBuffer al0Buf = pipe.GetBuffer<A2>();

// 新模式（直接指定位置和大小）
constexpr uint32_t L0A_HALF = 32768;
LocalTensor<half> al0(TPosition::A2, pingPong * L0A_HALF, L0A_HALF / sizeof(half));
```

### 优化效果示意

```
优化前（Queue + TBuffer + Tpipe）:
  每条 Load/Copy 指令前 Scalar 需要:
    - 查询 queue 状态（Load）
    - 计算 buffer offset（ALU + Load）
    - 更新 buffer 索引（Store）
  → 典型 Scalar Load/Store 占比 > 30%

优化后（SetFlag/WaitFlag + LocalTensor）:
  每条 Load/Copy 指令前 Scalar 仅需要:
    - WaitFlag（硬件信号量，无 Scalar 开销）
    - offset 来自编译期常量（常量折叠，0 指令）
  → Scalar Load/Store 占比可降到 < 15%
```

---

## 3. TilingData 常量化

### 原理

在**图静态**场景（编译期已知所有维度参数），将 TilingData 声明为 `constexpr`，编译器会对所有 `.convRunInfo.xxx` 和 `.convApiTiling.xxx` 的访问做**常量传播 + 常量折叠**。这意味着：

- `ri.kh * ri.kw`、`GCeilDiv(kTotal_, K0_VAL)` 等表达式在编译期算出结果
- `Load3DBitModeParam` 的 config0/config1 字段直接嵌入立即数
- 不再产生运行时从内存 Load TilingData 的 Scalar 指令

### 声明方式

```cpp
// 编译期常量 TilingData（图静态场景）
constexpr Conv2DTilingData kTiling = {
    .convRunInfo = { /* 所有字段编译期已知 */ },
    .convApiTiling = { /* ... */ },
};
```

kernel 内部通过 `Tiling()` 访问（指针指向 `&kTiling`），编译器看到 `constexpr` 后会将所有成员访问折叠。

### 适用条件

| 场景 | TilingData 来源 | 可否常量 |
|------|----------------|---------|
| 直接调用（demo/验证） | 编译期常量 | 可 `constexpr` |
| ops-nn 集成（框架下沉） | Host 侧运行时计算，通过 workspace 传入 | 不可 |

图静态场景常量化后，TilingData 相关的 Scalar Load 指令从 **数十条 → 0**。

---

## 三者叠加效果

```
                ┌─────────────────────────────┐
                │   固定循环轴（场景特化）      │
                │   消除分支 + 动态地址计算     │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   去除 Queue/TBuffer/Tpipe   │
                │   消除队列维护 + 缓冲区映射   │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   TilingData constexpr      │
                │   常量传播 → 消除参数 Load   │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │  Scalar Load/Store 占比      │
                │  从 90%+ → 10% 以内          │
                └─────────────────────────────┘
```

三者是**递进关系**：固定循环轴是前提（场景特化让循环结构可写死），再去掉高级抽象（Queue/TBuffer/Tpipe 没有存在的必要），最后常量化 TilingData（把仅存的参数 Load 也消除）。单独做一项收效有限，三者叠加才能把 Scalar 开销压到最低。


