# Scalar 通用减载优化设计

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

## 1. 优化目标

针对 `fused_infer_attention_score`（FIA）在 SCALAR_BOUND 场景中的通用 Scalar 开销，优先处理 kernel 启动头、Init 阶段和主循环 I-Cache miss。该类优化不改变 tiling 字段和主数据流，适合作为 FIA SCALAR / SCALARLDST 子类优化前的低风险前置项。

已验证效果：

| 指标 | 优化效果 |
|------|----------|
| Scalar 耗时 | 8 组 case 算术平均下降 33.22% |
| 端到端耗时 | 8 组 case 算术平均提升 14.74% |
| Bound 状态 | 8 组 case 从 SCALAR_BOUND 跳出 |

## 2. 架构概览

FIA 通用 Scalar 开销主要来自 4 个位置：

| 阶段 | 问题场景 | 优化手段 |
|------|----------|----------|
| 主循环 | 热路径函数体大，I-Cache miss 导致 PC 停滞 | I-Cache 预取 |
| Init | `TPipe::InitBuffer` 通用框架带来条件判断和队列操作 | 静态创建 `LocalTensor` |
| Init | `ConstParam` 作为类成员访问，编译器难以寄存器化 | 类成员局部化 |
| Kernel 入口 | TilingData 字段多，整块拷贝到栈形成头开销 | TilingData 指针直读 |

## 3. 关键参数 / 修改点

| 手段 | 关键修改 | 约束 |
|------|----------|------|
| I-Cache 预取 | 在循环外最后一次 MTE 等待之后、第一次 Scalar 计算之前插入预取 | 预取范围只覆盖热循环主体，避免预取本身增压 |
| 静态 `LocalTensor` | 用 `LocalTensor<T>(TPosition, offset, size)` 替代动态 `InitBuffer` | Buffer 偏移必须按 L1 布局精确计算 |
| `ConstParam` 局部化 | 将只在 `Init()` 或单个内联链路使用的 `ConstParam` 从成员改为局部变量 | 跨非内联函数共享的参数不能局部化 |
| TilingData 直读 | 保存 `__gm__ FATilingData*` 指针，避免整块 tiling struct 栈拷贝 | 热路径反复使用字段仍需缓存到局部变量或少量成员 |

## 4. 核心代码骨架

### 静态创建 LocalTensor

```cpp
// 优化前：动态 InitBuffer
pipe->InitBuffer(tmpBufL1Q, Q_L1_SIZE);
L1QTensorPingPong = tmpBufL1Q.template Get<T>();

// 优化后：编译期确定位置和大小后静态创建
L1QTensorPingPong = LocalTensor<T>(TPosition::A1, 0, Q_L1_SIZE);
L1KRopeTensor = LocalTensor<T>(
    TPosition::B1,
    (Q_L1_SIZE + 2 * P_L1_SIZE + 3 * KV_L1_SIZE) * 2,
    K_ROPE_SIZE);
```

### 类成员局部化

```cpp
// 优化前：成员变量，热路径通过 this 指针访问
ConstParam constParam;

// 优化后：只在 Init 使用时改为局部变量
__aicore__ inline void Init(...) {
    ConstParam constParam;
    ...
}
```

### TilingData 直读

```cpp
// 优化前：整块拷贝到栈
auto tilingData = *((__gm__ FATilingData *)tiling);
auto x = tilingData.field1;

// 优化后：保存指针，按需读取
auto tilingPtr = (__gm__ FATilingData *)tiling;
auto x = tilingPtr->field1;
```

## 5. 效果与适用边界

| 场景 | 建议 |
|------|------|
| 启动头开销高，TilingData 大 | 优先使用 TilingData 指针直读，热点字段单独缓存 |
| L1 buffer 布局固定 | 可静态创建 `LocalTensor`，但必须补充 buffer 不重叠检查 |
| 参数只在单函数或内联链路使用 | 可局部化，降低成员 load |
| PC 曲线出现主循环水平段 | 可尝试 I-Cache 预取，验证水平段消失且 Scalar 未增压 |

实测明细：

| case | 总耗时优化前 | 总耗时优化后 | Scalar 优化比例 | 端到端提升 |
|------|-------------:|-------------:|----------------:|-----------:|
| 1 | 24408 | 19890 | 32.92% | 18.51% |
| 2 | 32400 | 27432 | 24.67% | 15.33% |
| 3 | 25182 | 20250 | 28.44% | 19.59% |
| 4 | 24210 | 20268 | 21.92% | 16.28% |
| 5 | 26208 | 22212 | 32.52% | 15.25% |
| 6 | 24354 | 21240 | 30.43% | 12.79% |
| 7 | 32166 | 26820 | 50.53% | 16.62% |
| 8 | 52704 | 50832 | 44.36% | 3.55% |

若某 case Scalar 明显下降但完整任务收益很小，优先检查 CUBE / MTE2 等剩余关键路径，不仅凭这一现象断定瓶颈转移。PipeUtilization 各流水可重叠，busy time 不能相加作为总耗时，占比也不是独占耗时；尤其不能将 `aiv_scalar_ratio` 直接代入其他分析工具的 `SCALAR%` 子类门槛。

### 循环压缩与 I-Cache 的证据边界

将静态展开改为运行时循环会减小代码体积，但可能增加循环控制、动态地址计算，或减少常量折叠与跨迭代调度空间。静态代码字节减少不等于执行指令减少，也不证明 I-Cache miss 减少。没有 I-Cache miss / PC 停滞证据时，可将其作为有待证伪的候选，不能称为已定位的 I-Cache 优化。

对带双缓冲的循环，同时核对每轮槽位序列、ready/release 配对、跨迭代复用，以及生成代码是否新增等待或 spill；源代码中每轮推进偶数个槽只是检查起点。比较完整任务时间、Scalar 时间和 MTE 活动；若代码缩小而 Scalar 不降、总耗时也无稳定改善，不保留该改动。已有 SFA 反例：AIV 代码缩小 41.6%，未证明性能收益。
