# arch22 UB Bank 冲突规避

> 适用架构：arch22（DAV_2201 / A2/A3）。保留本库原有 220x bank 规则和代码示例；精确型号、可用 UB、地址映射及周期数需按目标文档/实测核验。

### 2.2 UB Bank 结构与冲突原理

以 Ascend 910B 为例：UB (192KB) = 48 bank × 4KB = 16 bank group × 3 bank/group。不同芯片的 bank 数可能不同，需查阅对应硬件手册。

#### 2.2.1 Bank 冲突的硬件原理

Vector 计算单元在一个 Repeat 周期内会并行发出多个 DataBlock 的读写请求。理想情况下，这些请求被分发到不同的 bank/bank group 上并发执行，1 个 Repeat 周期完成全部访问。当多个请求因地址映射关系落到**同一 bank 或同一 bank group** 时，硬件仲裁逻辑会将其串行化，导致单 Repeat 从 1-2 拍退化到多拍（具体拍数因芯片和冲突 DataBlock 数量而异）。

> **注意**：不同芯片的 bank 结构和冲突条件不同，必须查阅对应硬件文档。下文只保留 NPU 架构版本 220x（Atlas A2/A3 系列）的记录。

#### 2.2.2 不同芯片的 UB Bank 结构

| 芯片 | UB 大小 | Bank 数 | Bank Group 数 | 每组 Bank 数 | 每 Bank 大小 | 每行大小 |
|------|---------|---------|---------------|-------------|-------------|---------|
| DAV_2201 | 192 KB | 48 | 16 | 3 | 4 KB (128 行) | 32 B |

**Bank Group 组成方式**：group 内各 bank 的编号满足 `bank_id % num_groups` 相同。以 220x 为例：
- group 0: bank 0, 16, 32
- group 1: bank 1, 17, 33
- ...
- group 15: bank 15, 31, 47

**冲突条件差异**（关键区别）：

| 冲突类型 | arch22 / NPU 220x |
|---------|----------|
| 读写冲突 | 同一 **bank** |
| 写写冲突 | 同一 **bank group** |
| 读读冲突 | 同一 **bank group** |


#### 2.2.3 DataBlock 与 Bank 映射

Vector 指令处理的数据被切分为固定大小的 **DataBlock**，每行长度为 32B。一个 Vector 指令在一个 Repeat 内最多处理 **8 个 DataBlock**（block0~block7）。各 DataBlock 根据起始地址映射到某个 bank，进而归属于某个 bank group。

**映射的核心规律**（由官方示例推导，具体公式以芯片手册为准）：
- 地址上相邻的 32B 行通常映射到相邻的 bank（按 bank 总数循环）。
- 因此，**DataBlock 之间的地址差（由 `blk_stride` 决定）直接决定了它们是否落入同一 bank group**。

以 220x 为例：
- `blk_stride = 16`：相邻 DataBlock 地址差 = 16 × 32B = 512B。bank 编号差 = 16（假设按 32B 行轮询），16 % 16 = 0，即**所有 DataBlock 落入同一 bank group**，8 拍完成一个 Repeat。
- `blk_stride = 8`：bank 编号差 = 8，block0 和 block2 的 bank 差 = 16，16 % 16 = 0，落入同一 group，4 拍完成一个 Repeat。

#### 2.2.4 冲突判断方法

**步骤 1：识别并发访问的数据源**

一个 Vector 指令（如 `Add(dst, src0, src1)`）在同一 Repeat 内会同时读取 `src0`、`src1` 并写入 `dst`。这三个操作数的 DataBlock 是并发访问的候选冲突源。

**步骤 2：分析地址间隔规律**

- **多操作数之间**：若 `dst`、`src0`、`src1` 的 buffer 起始地址在 UB 内连续分配，且间隔恰好是 bank size 的整数倍，则它们的对应 DataBlock 会周期性地映射到同一 bank/bank group。
- **单操作数内部**：若 `blk_stride` 使 DataBlock 之间的 bank 差等于 group 数（220x: 16）或其倍数，则多个 DataBlock 会回卷到同一 bank group。

**步骤 3：通过官方工具或实验确认**

- **msProf 资源冲突占比分析**：官方提供的 msProf 工具可以采集资源冲突占比数据，直接定位 bank 冲突。具体使用方法参考《算子开发工具》文档。
- **Profiling 验证**：如果 Vector 计算耗时远高于理论值，且 MTE2 不是瓶颈，则极可能存在 bank 冲突。
- **实验验证**：对疑似冲突的代码添加 256B padding 或调整 `blk_stride` 后重新 profiling，观察周期数是否下降。

#### 2.2.5 常见冲突场景与规避原理

以下场景来自官方文档的典型示例。

**场景 1：读写冲突 —— src 与 dst 落在同一 bank**

```cpp
// 假设 x 起始地址在 bank0，y 起始地址也在 bank0（地址差为 bank size 的整数倍）
Add(dst, src, src2, ...);  // src 读 bank0，dst 写 bank0，读写冲突
```

**规避**：确保 src 和 dst 的起始地址不在同一 bank。对于 220x，通常给中间 buffer 增加 padding 使地址错开至少一个 bank。

**场景 2：写写冲突 —— dst 的多个 DataBlock 落入同一 bank group**

```cpp
// 220x 上 blk_stride=16：8 个 DataBlock 全部落入同一 bank group，8 拍完成
Adds(dst, src, scalar, MASK, 1, {1, 16, 1, 16});

// 220x 上 blk_stride=8：block0 和 block2 落入同一 bank group，4 拍完成
Adds(dst, src, scalar, MASK, 1, {1, 8, 1, 8});
```

**规避**：改为 `blk_stride = 1`（连续读），通过 `dst_gap/src_gap` 控制跨 Repeat 的地址增量。

**场景 3：读读冲突 —— 双 src 落入同一 bank group（220x）**

```cpp
// 220x：x 和 y 起始地址差为 bank size 整数倍，DataBlock 0 同时读同一 bank group
Add(zLocal, xLocal, yLocal, ...);
```

**规避**：在 `xBuf` 后增加 256B padding，打破地址周期性，使 `x` 和 `y` 的 DataBlock 分散到不同 bank group。

**场景 4：连续分配多个 buffer，地址周期性重叠**

```cpp
// 220x 原始实现：x/y/z 连续分配，起始地址差为 0x4000（16KB）
// x: bank0, y: bank0（地址差 16KB = 4×bank size），z: bank0
// 一个 Repeat 内 x 和 y 同时读同一 group，x/y 和 z 同时读写同一 bank
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));
pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```

**规避（220x 官方推荐）**：
```cpp
// x 多申请 256B，避免一个 Repeat 内 x 和 y 同时读同一 bank group
// y 多申请空间，确保 z 不会和 x/y 落入同一个 bank
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float) + 256);
pipe.InitBuffer(inQueueY, 1, 64 * 1024 - (4096 * sizeof(float) + 256));
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```
