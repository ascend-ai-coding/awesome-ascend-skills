# matmul L1 Bank 冲突规避（Half-L1 Ping / Half-L1 Pong 布局）优化设计

## 1. 优化目标

解决 matmul 开启 L1 Double Buffer 后，**MTE2 写 L1 与 MTE1 读 L1 落入同一 Bank** 触发的 Bank 冲突：冲突导致 MTE1 读带宽降低（芯片策略优先写 L1），进而打断 MMAD 指令连续性。通过将 L1 按 Bank 边界切分为前/后两半，Ping 块与 Pong 块分居不同 Bank，从根本上消除冲突条件。

适用 Bound 类型：**MTE1 Bound**（L1→L0 搬运带宽利用率不足、MMAD 断流）。

实测收益：

- 通用 matmul（M=1024, K=2048, N=4096, dav-3510）：开启双缓冲基线 kernel **66.0 μs → 50.87 μs**（-23%）；
- MXFP4（m=256, k=4096, n=256）：`LOAD_2Dv2` 平均耗时 **270 → 233 cycle**、`LOAD_MX_2Dv2` 平均 **442 → 384 cycle**（搬运次数不变，均为 32 次）。

## 2. 架构概览

### 2.1 L1 Bank 物理结构

- L1 以 **256 KB 粒度划分为两个独立 Bank**（L1 总容量 512 KB，`TOTAL_L1_SIZE`）；
- 每次搬运指令的发包大小为 256 B，一次传输必定横跨 8 个 Bank Group——在 Ping/Pong 连续排布下，MTE2 写 Pong 与 MTE1 读 Ping 必然引发同 Bank 读写冲突。

### 2.2 未优化路径（顺序排布，冲突）

```text
L1: [A_ping | A_pong | B_ping | B_pong | scaleA_ping | scaleA_pong | ...]
      ↑ MTE1 读 ping（Bank0）        ↑ MTE2 写 pong（跨 Bank0/Bank1）
冲突：MTE2 写入数据与 MTE1 读取数据落于同一 Bank → 读带宽降低
```

### 2.3 优化路径（Half-L1 Ping / Half-L1 Pong）

```text
L1: [ A_ping | B_ping | scaleA_ping | scaleB_ping ]  ← Bank0（前半，TOTAL_L1_SIZE >> 1）
     [ A_pong | B_pong | scaleA_pong | scaleB_pong ]  ← Bank1（后半）
      ↑ MTE1 读 Bank0                                  ↑ MTE2 写 Bank1，读写分离
```

MTE2 指令与 MTE1 指令同时操作 L1：前者将 pong/ping 块数据写入 L1，后者从 ping/pong 块读取。读写分居不同 Bank 后，冲突的必要条件被消除。

### 2.4 事件同步模型

不变，沿用 pingpong 的事件集（`MTE1_MTE2` / `MTE2_MTE1` / `M_MTE1` / `MTE1_M` 等）；本优化**只改 L1 地址布局，不改同步时序**。

## 3. 关键参数配置

### 3.1 L1 偏移计算

```cpp
// [Ping A][Ping B] | [Pong A][Pong B]：以 L1 总容量一半为 Ping/Pong 隔离边界
uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * l1BufId;   // l1BufId: 0=ping, 1=pong
l1BufferAOffset[l1BufId] = l1Offset;                            // A 矩阵起始偏移
l1BufferBOffset[l1BufId] = l1Offset + baseM * kL1 * sizeof(T);  // B 紧邻 A 存放
```

对比传统写法（顺序排布）：

```cpp
// [Ping A][Pong A][Ping B][Pong B]（存在 Bank 冲突）
l1BufferAOffset[l1BufId] = l1BufId * AOffsetL1;
l1BufferBOffset[l1BufId] = DOUBLE_BUFFER_COUNT * AOffsetL1 + l1BufId * BOffsetL1;
```

### 3.2 容量约束

单个 Bank（半个 L1 = 256 KB）需容纳该侧全部 ping（或 pong）数据：

```text
aL1OneBuffer + bL1OneBuffer + scaleAL1OneBuffer + scaleBL1OneBuffer ≤ TOTAL_L1_SIZE / 2
```

若超限，回退顺序排布并缩小 `kL1`（`L1_BUFFER_NUM`），或参照 [pingpong_design.md](pingpong_design.md) §6 的预算推导。

## 4. 核心计算循环

循环结构与 [pingpong_design.md](pingpong_design.md) §4.1 完全一致，仅 L1 偏移计算改为 §3.1 的 Bank 边界切分形式，此处不重复。

## 5. 优化的关键修改点

| 修改项 | 顺序排布（优化前） | Half-L1 排布（优化后） |
|--------|------------------|----------------------|
| Ping/Pong 边界 | 按各矩阵单份大小顺序排 | `TOTAL_L1_SIZE >> 1`（Bank 物理边界） |
| Ping 区组成 | A_ping（起始偏移 0） | A_ping + B_ping + scaleA_ping + scaleB_ping |
| Pong 区组成 | 各矩阵 pong 紧随其 ping | 对称镜像，位于后半 Bank |
| 同步事件 | 不变 | 不变 |

## 6. 注意事项与约束

1. **容量边界对齐**：切分时确保 Ping/Pong 各占一半（`TOTAL_L1_SIZE >> 1`）；若 L1 存在其他预留空间（如 workspace），需重新计算实际可用容量，避免越界或空间浪费。
2. **对整体 case 的收益差异**：部分 case 中 L1 Bank 冲突对整体性能影响不大（如 MTE1 非主导流水的量化场景）；先确认 MTE1 Bound 或 MMAD 断流证据再实施。
3. **与 3-Buffer 的预算冲突**：Grouped MXFP8 split-M 的 3-Buffer 实现要求 `A1+B1+scaleA+scaleB+A3 ≤ HALF_L1_SIZE` 等约束；若不满足应回退 2-Buffer 并结合本优化 + SWAT 提升吞吐。
4. **验收指标**：流水图中 MTE1 段缩短、`LOAD_2Dv2`/`LOAD_MX_2Dv2` 平均耗时下降、MMAD 断流减少；搬运次数不应变化。

## 7. 实施常见问题与解决方案

| # | 问题 | 根因 | 解决方案 |
|---|------|------|---------|
| 1 | 改后排布越界 | 单侧数据总量 > 半个 L1 | 缩小 kL1 / L1_BUFFER_NUM，或去掉非必需驻留数据 |
| 2 | 收益不明显 | case 非 MTE1 Bound | 用 profiling 确认瓶颈后再实施；转向 MTE2/Cube 侧优化 |
| 3 | 量化场景 scale 区遗漏 | 仅迁移 A/B 未迁移 scale 的偏移 | scaleA/scaleB 的 ping/pong 同样纳入对应半区 |

## 8. 与其他优化的叠加关系

- **前置**：pingpong 双缓冲（本优化是 Double Buffer 的地址布局增强，不开双缓冲无从谈起 Ping/Pong 隔离）。
- **正交可叠加**：SWAT（多核调度层）、UnitFlag（L0C 输出层）、Scale 合并载入（MTE2 层）、FullLoad（A/B 侧驻留层——注意 FullLoad 的 A 全载区也须排入对应半区预算）。
- 在 SWAT 模板与 FullLoad 模板中均作为标配手段。
