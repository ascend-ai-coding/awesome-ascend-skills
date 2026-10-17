# arch22 MatMul 流水、预取与大包搬运设计

> 适用架构：arch22（DAV_2201 / Atlas A2/A3）。结构与 API 支持由 `dav-2201` 样例证明；本文不声明 A3 性能收益已上板。

## 1. 优化目标

把 MatMul 的 `GM -> L1 -> L0 -> Cube -> GM` 串行链改成分层流水，并减少 MTE2/MTE1 的发射次数与等待间隙。验收指标是 Task Duration、各流水时间/占比和 timeline 重叠，不使用 arch35 文档中的固定百分比阈值。

## 2. 架构概览

```text
GM --MTE2--> L1 ping/pong --MTE1--> L0A/L0B ping/pong
                                      --Cube--> L0C --Fixpipe--> GM
         next A/B package  <---- overlaps current baseK compute ---->
```

| 阶段 | 串行风险 | 流水化方向 |
|---|---|---|
| GM -> L1 | 逐 baseK 小包发射、搬运间隙 | 合并 stepKa/stepKb 并按紧迫侧预取 |
| L1 -> L0 | 同一 buffer 反复等待 | A/B 独立 ping-pong 与精确事件 |
| L0 -> Cube -> GM | 累加/写回阻断后续 tile | 保持 L0C 预算并核对 UnitFlag 语义 |

## 3. 关键参数 / 修改点

| 参数或手段 | 修改点 | 前置条件 |
|---|---|---|
| L1/L0 ping-pong | A/B 两侧各保留双缓冲 | 目标 L1/L0 容量和地址区间闭合 |
| `stepKa` / `stepKb` | 多个 baseK 合为一个 GM->L1 包 | K 尾块、DMA 和 L1 预算均满足 |
| `doMTE2Preload` | 提前搬入下一 M/N tile | MDL、K 全载、对应方向 DB |
| `LoadData3DParamsV2` | 一次描述完整 MTE1 切片 | layout、transpose、extension 匹配 |
| UnitFlag / 常量 Tiling | 缩短累加/写回或 Scalar 控制路径 | 累加语义和 shape 分桶保持正确 |

## 4. 核心计算循环 / 关键代码

K 循环的基本顺序是：等待当前 A/B L1 包就绪，按 baseK 装入 L0 并发起 MMAD，同时用另一组 buffer 预取下一包；仅在包的首个 baseK 等待就绪、最后一个 baseK 释放 L1。下面的事件与搬运片段必须在目标实现中按 buffer 生命周期配对。

### L1/L0 ping-pong

L1 的 A/B 各保留 ping/pong，L0A/L0B 也保留 ping/pong。K 循环中当前 buffer 送 MTE1/Cube 时，下一份 GM 数据搬入另一 buffer。

预算必须满足：

```text
2 * (A1_chunk_bytes + B1_chunk_bytes) <= usable_l1_bytes
2 * A2_tile_bytes <= usable_l0a_bytes
2 * B2_tile_bytes <= usable_l0b_bytes
L0C_tile_bytes <= usable_l0c_bytes
```

容量来自目标平台或现有 tiling，不把样例容量写死进通用实现。

### 精确事件模型

arch22 高性能基础 API 样例为 A1/B1 分配独立的 `MTE2_MTE1` 与 `MTE1_MTE2` 事件，并为 L0 ping/pong 使用 `MTE1_M` / `M_MTE1`：

```cpp
SetFlag<HardEvent::MTE1_MTE2>(aPing);
SetFlag<HardEvent::MTE1_MTE2>(aPong);
SetFlag<HardEvent::MTE1_MTE2>(bPing);
SetFlag<HardEvent::MTE1_MTE2>(bPong);
SetFlag<HardEvent::M_MTE1>(l0Ping);
SetFlag<HardEvent::M_MTE1>(l0Pong);
```

反向事件必须在第一次 `WaitFlag` 前预置。大包内只在首个 baseK 等待“数据就绪”，只在最后一个 baseK 发“buffer 已释放”，避免每个 baseK 重复同步。

事件表应在设计阶段列出生产流水、消费流水、buffer 与 ID。arch22 可用事件 ID 数量有限，增加三缓冲或混合 AIC/AIV 同步前必须做冲突检查。

### 大包搬运与发射顺序

用 `stepKa/stepKb` 把多个 baseK 合成一次 GM->L1 搬运：

```cpp
Nd2NzParams p;
p.nValue = curM;
p.dValue = baseK * stepKa;
DataCopy(a1Local, aGm[offset], p);
```

大包大小由 L1 预算、K 尾块和目标 DMA 约束共同决定。`stepKa > stepKb` 时 B 更早耗尽，应先发 B 的下一包；反之先发 A。判据是“距下一次消费还剩多少个 baseK”，不是固定先 A 或先 B。

### M/N 预加载

Matmul API 的 `doMTE2Preload` 在 A2/A3 有样例支持，但必须同时满足：

- MDL 模板；
- K 全载：`singleK <= baseK * stepK`；
- 被预加载的 M 或 N 方向已开 double buffer；
- timeline 确认 MTE2 存在可隐藏间隙。

不满足 K 全载或 DB 条件时，预加载可能增加占用和控制分支而没有收益。

### arch22 的 MTE1 降压

当 LoadData2D 需要循环发多条指令才能覆盖一个 `baseM x baseK`/`baseK x baseN` tile 时，优先评估 `LoadData3DParamsV2`，用一条描述完整切片，降低 MTE1 队列占用。此项是 arch22 路线，不能照搬 A5 的 `LoadData2DParamsV2` 结论。

### UnitFlag 与常量 Tiling

- 开启 UnitFlag 让 MMAD 与 Fixpipe 以更细粒度协同；中间 K 块和最终 K 块的 `unitFlag` 值必须与累加/写回语义匹配。
- shape 固定或可分桶时，用常量 Tiling 消除 kernel 内动态 shape/stride 推导；动态 shape 不能为省 Scalar 而强行套固定参数。

## 5. 效果与适用边界

| 修改 | 预期观察 | 停止条件 |
|---|---|---|
| 双缓冲与事件配对 | MTE2/MTE1/Cube 重叠增加、等待间隙缩短 | 容量或事件生命周期无法闭合 |
| 大包与紧迫侧优先 | DMA 发射次数减少、消费前等待减少 | 尾块/布局不合法或挤掉双缓冲 |
| 预加载与 MTE1 降压 | 可隐藏搬运间隙或减少 MTE1 指令 | K 非全载、无 DB 或 layout 不匹配 |

公开样例证明 arch22 的实现路线，不证明目标 case 的收益。A2/A3 各自以精度、Task Duration 和 timeline 为准，不填写未经实测的固定百分比。

### 自检

- L1/L0 每个 ping/pong 区间不重叠，尾块不会越界。
- 每个反向事件先初始化、后等待；退出前事件配对闭合。
- 大包的首/尾同步只包围同一块真实数据。
- 预加载候选满足 K 全载与 M/N DB 条件。
- `LoadData3D` 的 layout、transpose、extension 与源矩阵物理布局一致。
- A2 与 A3 分别做精度、timeline 和逐 case 性能验证。

### 静态检视判据

<a id="prc-mm-001"></a>
#### 多级 Ping-Pong 与事件闭环（PRC-MM-001）

逐 K 块在同一 L1/L0 buffer 上依次 `DataCopy -> LoadData -> Mmad` 会使 MTE2、MTE1 与 Cube 串行。检视时列出 A/B 的 L1 和 L0 地址区间、buffer 数、K 循环及每个 `WaitFlag` 的同类型 `SetFlag`；确认反向事件在首次等待前预置，大包只在首块等待、末块释放。相同事件 ID 只有在 HardEvent 类型或生命周期真实冲突时才算问题，不能只靠文本命中。

K 仅一轮、容量不允许双缓冲，或高阶 Matmul API 已在内部管理流水时，不报手排 ping-pong 缺失。若建议增缓冲，finding 必须给出 K 循环、地址与事件配对表、目标 L1/L0 预算、可重叠时段及容量闭合证明；若建议改事件，须指出真实生产者和消费者，不能只建议删等待。机制证据见 book-910：0042、0043、0067。

<a id="prc-mm-002"></a>
#### 大包搬运、紧迫侧优先与预加载门禁（PRC-MM-002）

逐 baseK 发起小 DMA 是候选，而不是仅凭 `stepKa/stepKb == 1` 就认定瓶颈。由 DataCopy 参数计算 A/B 单次字节和发起次数，检查 L1 剩余量是否允许在不挤掉 ping-pong 的前提下合包；再比较两侧距下次消费的 baseK 数。固定先搬非紧迫侧且 timeline 显示等待时，才提出调整顺序。

M/N 预加载还须同时满足 MDL、K 全载、对应 M/N 双缓冲和可隐藏的 MTE2 gap，任一不满足就不报“未开启 preload”。finding 应列出当前与候选包字节、发起次数、L1 预算、A/B 消费距离及 timeline 依据；A5 的搬运阈值和收益数字不是 arch22 判据。

### 实现证据

公开 `dav-2201` MatMul 基础 API 高性能样例提供多级流水依据，Matmul API 预加载样例提供 K 全载与 M/N 双缓冲门禁。具体能力、事件预算和收益仍需按目标 SDK、型号与 case 复核。
