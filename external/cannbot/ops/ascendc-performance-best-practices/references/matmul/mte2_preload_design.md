# matmul MTE2 预加载（MTE2 Preload）优化设计

## 1. 优化目标

针对 matmul 族（含 `matmul`、`matmul_mxfp4`、`matmul_mxfp8`、`batch_matmul` 等变体）在 **pingpong 双缓冲已启用但仍存在 MTE2_PONG 发射阻塞** 的场景下，通过在**首轮一次性预加载两份 A/B tile 到 L1**，使 `MTE2_PONG` 指令提前发射，绕开芯片指令队列深度上限导致的被动等待，保持 MTE2 → MTE1 → CUBE 三级流水在指令发射层面的连续性。

典型收益：Task 时间 **−3%–10%**（对应 `matmul` BF16 `M=1024, K=4096, N=2048` 实测 `66.0μs → 63.3μs`，约 **−4%**；MTE2 段 `37.6μs → 32.7μs`，约 **−13%**）。

> **增强层定位**：MTE2 Preload 是 **pingpong 双缓冲之上的 Kernel 指令调度增强**，**不改动 TilingData 字段**（`baseM/baseN/baseK`、`stepK`、`kL1`、`nBufferNum`、`scaleKL1` 等都保持原值），仅修改 Kernel 主循环的 MTE2 发射时序。与 SWAT / A-Full-Load / B-Full-Load / StreamK / 小数据块合并载入等 Tiling 层优化**正交**，可任意叠加（叠加后仍需满足相应层的 L1 预算约束）。

---

## 2. 架构概览

### 2.1 未优化路径（pingpong 已开但 PONG 被阻塞）

双缓冲语义下，`MTE2_PING`（本轮搬 L1[0]）发完之后，**下一轮** `MTE2_PONG`（本轮搬 L1[1]）必须等**所有发射在 PING 之后的指令**都发完才能拿到发射机会。当 PING 与 PONG 之间插入的指令数超过芯片预设指令队列深度（硬件参数，不可软件调大），即使 PING 的数据依赖已解除，PONG 仍会被指令队列阻塞。

```
时间线 →  (pingpong 常规实现)
───────────────────────────────────────────────────────────────
MTE2:  [PING 发射]────────────────────[PONG 被阻塞，被动等待]────[PONG 发射]
MTE1:              [PING→L0]  [MMAD 多轮]  [PONG→L0]
CUBE:                        [MMAD MMAD ...]
```

- 现象：`trace_core*.json` 中 `MTE2_PONG` 事件起点 ≫ 理论起点（≈ `MTE2_PING` 事件终点）；PING ↔ PONG 之间存在**确定性 gap**
- 不是 MTE2 带宽 bound，不是 CUBE bound，常呈 **"pingpong 已启用但各流水 busy 仍 ≤ 60%–70%"** 的"准无 bound"形态
- **KL1 已无法继续缩减**：继续缩减 `kL1` 会减少 PING/PONG 之间插入的指令数（化解阻塞），但单次覆盖 K 减少 → 外层循环次数上升 → 循环起停开销上升 / CUBE busy 下降，**净收益为负**

### 2.2 优化路径（MTE2 Preload：首轮一次发两份，解耦 PONG 发射）

**核心思想**：把 "第 1 轮搬 PING、第 2 轮搬 PONG" 的顺序改为 **"第 1 轮先搬 PING 再紧跟着搬 PONG"**（提前把第二份数据发射进 L1），使 PONG 的发射时机**前移到 PING 之后几条指令内**，完全落在指令队列深度范围内。后续各轮继续用双缓冲轮换，但 **一直保持"当前轮算的是 L1[cur]，MTE2 已经搬好 L1[1-cur]"** 的错拍状态。

```
时间线 →  (MTE2 Preload)
────────────────────────────────────────────────────────────────
MTE2:  [PING 发射][PONG 发射]─────[PING2 发射][PONG2 发射]─── ...
MTE1:                          [PING→L0][PONG→L0][PING2→L0] ...
CUBE:                                   [MMAD MMAD MMAD ...]

关键差异:
- PING/PONG 在首轮被紧邻发射（指令队列深度内，不阻塞）
- 后续各轮的 "PONG → PING_{next}" 也在指令队列深度内（错拍一轮，形成稳态）
```

### 2.3 L1 布局（不变）

MTE2 Preload **不改 L1 布局**，沿用 pingpong 的 `[A_ping | A_pong | B_ping | B_pong | scaleA_ping | scaleA_pong | scaleB_ping | scaleB_pong]` 排布（见 `pingpong_design.md §2.1 / §3.1`）。唯一差异是**首轮的搬运时序**：原本 `[搬 L1_ping → 算 → 搬 L1_pong → 算 → ...]`，改为 `[搬 L1_ping → 搬 L1_pong → 算 L1_ping → 搬 L1_next_ping → 算 L1_pong → ...]`。

### 2.4 事件同步模型（相对 pingpong 的增量）

| 事件类型 | pingpong 基线用途 | MTE2 Preload 增量 |
|---------|-----------------|-----------------|
| `MTE1_MTE2(l1BufId)` | L1 buffer 释放控制 | **首轮两份 L1 buffer 都需要在 Init 时预置 SetFlag**（基线只需按 pingpong 数预置，本优化保持一致，但 WaitFlag 顺序要按"先等 PING 的，再等 PONG 的"排） |
| `MTE2_MTE1(l1BufId)` | L1 数据就绪通知 | **首轮发两次 SetFlag**（PING 搬完发一次、PONG 搬完发一次），内层循环按 `l1BufId` 对应取 |
| `M_MTE1` / `MTE1_M` / `M_FIX` / `FIX_M` | L0 / L0C 同步 | 不变 |

关键约束：**首轮（`tileIdx / blockNum == 0 && iter0 == 0`）必须把两份 tile（PING + PONG）都搬完再进入 MMAD**；后续各轮（`tileIdx / blockNum > 0` 或 `iter0 > 0`）每轮只搬一份对侧 buffer，形成稳态错拍。

---

## 3. 关键参数配置

MTE2 Preload **不引入新的 TilingData 字段**，沿用 pingpong 的参数集即可：

```cpp
constexpr uint32_t BASE_M         = 256;     // L0 tile M 维度
constexpr uint32_t BASE_N         = 256;     // L0 tile N 维度
constexpr uint32_t BASE_K         = 128;     // L0 tile K 维度（按 dtype 折算，本示例 FP16）
constexpr uint32_t PINGPONG_NUM   = 2;       // pingpong 双缓冲（前置条件，必须开）
constexpr uint64_t kL1            = 512 / sizeof(T);  // L1 中单个 buffer 的 K 大小

// 新增常量（Kernel 内部使用，不下发 Tiling）
constexpr uint64_t BUFFER_GAP     = 2;       // 预加载跨度 = 2（PING + PONG 一次性发射）
```

> **没有新的 Host Tiling 字段**：MTE2 Preload 的启用 / 禁用在 Kernel 侧通过"首轮 `iter0 == 0` 的分支判断"实现，Host 不需要下发开关；但**建议建模/策略阶段在《性能优化方案》里显式说明"启用 MTE2 Preload"**，以便实施专家定位到本 design.md。

---

## 4. 核心计算循环（相对 pingpong 的增量改造）

### 4.1 改造前（常规 pingpong）

```cpp
for (tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
    for (iter0 = 0; iter0 < kL1TileNum; ++iter0) {
        l1BufId = l1PingPong & 1;

        // 每轮都搬一份
        WaitFlag<MTE1_MTE2>(l1BufId);
        CopyGM2L1(A[tileIdx, iter0] → L1[l1BufId]);
        CopyGM2L1(B[tileIdx, iter0] → L1[l1BufId]);
        SetFlag<MTE2_MTE1>(l1BufId);

        WaitFlag<MTE2_MTE1>(l1BufId);
        // inner loop: L1 → L0 → MMAD
        ...
        SetFlag<MTE1_MTE2>(l1BufId);
        l1PingPong++;
    }
}
```

### 4.2 改造后（MTE2 Preload）

```cpp
for (tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
    for (iter0 = 0; iter0 < kL1TileNum; ++iter0) {
        l1BufId = l1PingPong & 1;

        // 段 1：首轮首个分片（tileIdx / blockNum == 0 && iter0 == 0）
        //        搬 PING 到 L1[l1BufId]
        if (tileIdx / blockNum == 0 && iter0 == 0) {
            WaitFlag<MTE1_MTE2>(l1BufId);
            CopyGM2L1(A[0, 0] → L1[l1BufId]);
            CopyGM2L1(B[0, 0] → L1[l1BufId]);
            SetFlag<MTE2_MTE1>(l1BufId);
        }

        // 段 2：预取下一分片（首轮后续分片 或 非首轮所有分片）
        //        搬 PONG 到 L1[1 - l1BufId]（首轮后续）/ 对侧 buffer（非首轮）
        //        → 关键：首轮 iter0==0 走完段 1 后紧跟段 2，等效"一次发射两份 MTE2"
        if (tileIdx / blockNum > 0 ||
            (tileIdx / blockNum == 0 && kL1TileNum > 1 && iter0 + 1 < kL1TileNum)) {
            uint64_t curL1BufId = l1BufId;
            uint64_t curOffsetL1 = iter0 * kL1;

            if (tileIdx / blockNum == 0) {
                curOffsetL1 = (iter0 + 1) * kL1;     // 首轮后续：搬下一 K 分片
                curL1BufId = 1 - l1BufId;            // 切到另一块 L1 buffer
                // K 向尾片处理：若 iter0 + 1 是最后一分片，curGmKL1 = k - (iter0+1)*kL1
            }

            WaitFlag<MTE1_MTE2>(curL1BufId);
            CopyGM2L1(A[tileIdx, curOffsetL1] → L1[curL1BufId]);
            CopyGM2L1(B[tileIdx, curOffsetL1] → L1[curL1BufId]);
            SetFlag<MTE2_MTE1>(curL1BufId);
        }

        // 段 3：消费 L1[l1BufId]（读 L0 + MMAD）
        WaitFlag<MTE2_MTE1>(l1BufId);
        for (iter1 = 0; iter1 < kL0IterNum; ++iter1) {
            // inner loop: L1 → L0 → MMAD（不变）
            ...
        }
        SetFlag<MTE1_MTE2>(l1BufId);   // 释放当前 L1 buffer，供下一轮 MTE2 覆写
        l1PingPong++;
    }
}
```

### 4.3 流水时序对照

```
常规 pingpong (PONG 被阻塞)：
MTE2: [MTE2 PING]─────────────────────[MTE2 PONG]──[MTE2 PING2]─
                                       ^阻塞等待^
MTE1:               [MTE1 PING→L0]              [MTE1 PONG→L0]
CUBE:                                [MMAD MMAD MMAD ...]

MTE2 Preload (稳态错拍)：
MTE2: [MTE2 PING][MTE2 PONG]───[MTE2 PING2][MTE2 PONG2]──
                                ^提前发射^
MTE1:                 [MTE1 PING][MTE1 PONG][MTE1 PING2]
CUBE:                         [MMAD MMAD MMAD MMAD ...]
```

---

## 5. 从 pingpong 到 MTE2 Preload 的关键修改点

| 修改项 | pingpong 基线 | MTE2 Preload |
|--------|--------------|---------------|
| 外层循环 MTE2 搬运条件 | 每轮搬一次 | 分两段条件判断（段 1：首轮首分片搬 PING；段 2：后续每轮搬下一分片 PONG） |
| 首轮事件初始化 | Init 中 `SetFlag<MTE1_MTE2>(0/1)` 各一次 | **保持不变**（两份 L1 buffer 都需要初始释放 flag），但首轮 `WaitFlag` 顺序为"先等 PING 的 `MTE1_MTE2(PING_id)`，再等 PONG 的 `MTE1_MTE2(PONG_id)`" |
| 首轮搬运条件 | 无特殊分支，进入循环就搬 | `if (tileIdx / blockNum == 0 && iter0 == 0)` 段 1 只搬 PING；`if (iter0 + 1 < kL1TileNum)` 段 2 搬 PONG |
| 尾片处理 | 循环最后一轮自然退出 | 段 2 条件中必须加 `iter0 + 1 < kL1TileNum` 限制，**避免越界预取**（末轮不再发起下一分片 MTE2） |
| `MTE2_MTE1` 事件 | 每轮 `SetFlag + WaitFlag` 一次 | 首轮 `iter0 == 0` 发**两次 SetFlag**（PING 一次、PONG 一次），后续每轮发一次；消费段按 `l1BufId` 对应 WaitFlag |
| `curL1BufId` 语义 | 固定 `l1PingPong & 1` | 首轮首分片用 `l1BufId`；首轮后续分片切换到 `1 - l1BufId`（提前把下一分片搬到对侧 buffer） |

---

## 6. 注意事项

1. **前置条件：pingpong 必须先启用**。未开 pingpong 时 `l1BufNum = 1`，根本没有 PING/PONG 概念，谈不上"PONG 阻塞"，MTE2 Preload 无意义，此时应先按 `pingpong_design.md` 启用双缓冲
2. **L1 容量**：MTE2 Preload 不额外占 L1（只是提前搬，不是多开一份 buffer），L1 预算与 pingpong 一致；但若同时叠加 A/B-Full-Load 或小数据块合并载入，须按各自 design.md 校验 L1 预算
3. **尾片边界**：首轮 `iter0 + 1 < kL1TileNum` 是**关键判断**，漏掉必越界；末轮不再触发下一分片 MTE2，也不要覆盖已搬好的数据
4. **首轮双搬的两份数据**：段 1 搬的是**本轮** tile 的 PING（`curOffsetL1 = iter0 * kL1 = 0`）；段 2 搬的是**本轮 tile 的下一分片** PONG（`curOffsetL1 = (iter0 + 1) * kL1 = kL1`）。**不要把段 2 误写成"下一 tile 的 PING"**（会导致本轮内层 MMAD 读不到第二分片）
5. **多 tile 场景（非首轮）**：非首轮 tile（`tileIdx / blockNum > 0`）不需要段 1，只需段 2 每轮预取"下一 K 分片到对侧 buffer"。首轮 tile 末段 MTE2 已为下一 tile 的首段 L1 做好预取（跨 tile 的 pingpong 自然衔接）
6. **精度验证**：优化后必须通过现有精度验证（小数据块偏移、flag 配对错误都会导致精度错误）
7. **效果依赖指令队列深度**：本优化收益取决于"PING/PONG 之间插入的指令数"与"芯片指令队列深度"的差距。指令数 ≫ 队列深度时收益最大；指令数 < 队列深度时本来就不阻塞，收益 ≈ 0
8. **代码复杂度**：分支判断和 flag 管理复杂度显著上升；scalar busy 可能从 **~3μs 升到 ~6μs**（见实测对照表）—— 这是已知取舍，正常情况下 MTE2 段节省远大于 scalar 段增量

---

## 7. 实施常见问题与解决方案

### 问题 1：首轮段 1 的 `WaitFlag<MTE1_MTE2>` 漏发

**现象**：kernel 挂起或 L1 数据被脏写，精度错误。

**原因**：pingpong 基线里每轮入口先 `WaitFlag<MTE1_MTE2>(l1BufId)`；段 1 是新加的"首轮专用分支"，容易忘记写 WaitFlag，导致首次 MTE2 时 L1 [l1BufId] 的释放 flag 还没被等到。

**解决方案**：段 1 入口严格保留 `AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);`；Init 中的 `SetFlag<MTE1_MTE2>(0)` 和 `SetFlag<MTE1_MTE2>(1)` 两次预置**都不能省**（MTE2 Preload 在首轮需要两个 L1 buffer 的释放 flag 都到位）。

### 问题 2：首轮段 2 的 `curL1BufId` 未切到对侧

**现象**：首轮 `iter0 == 0` 的段 1 和段 2 都写到了同一个 L1 buffer，第二分片把第一分片覆写了，内层 MMAD 读到错数据。

**原因**：段 2 忘记加 `curL1BufId = 1 - l1BufId` 的切换语句（首轮专用），继续沿用段 1 的 `l1BufId`。

**解决方案**：段 2 入口必须写：
```cpp
if (tileIdx / blockNum == 0) {
    curOffsetL1 = (iter0 + 1) * kL1;
    curL1BufId  = 1 - l1BufId;   // 必须切换到对侧 L1
}
```
非首轮 tile 不需要切换，`curL1BufId = l1BufId` 即可（pingpong 自然轮换）。

### 问题 3：段 2 越界预取

**现象**：末轮 `iter0 = kL1TileNum - 1` 仍触发段 2，搬了 `(iter0 + 1) * kL1` 分片，越过了 K 边界读到非法 GM 地址；报错或精度错误。

**原因**：段 2 条件漏写 `iter0 + 1 < kL1TileNum`。

**解决方案**：段 2 条件必须为 `(tileIdx / blockNum > 0 || (tileIdx / blockNum == 0 && kL1TileNum > 1 && iter0 + 1 < kL1TileNum))`；末轮时段 2 不执行，由段 3 消费最后一份数据后自然退出。

### 问题 4：`MTE2_MTE1` 事件配对错位

**现象**：偶现 kernel 挂起（`WaitFlag<MTE2_MTE1>(l1BufId)` 等不到匹配的 SetFlag）；或数据就绪前就被 MTE1 读到。

**原因**：首轮 `iter0 == 0` 发了两次 `SetFlag<MTE2_MTE1>`（段 1 发 `l1BufId` 一次，段 2 发 `curL1BufId = 1 - l1BufId` 一次）；但后续段 3 里只 `WaitFlag<MTE2_MTE1>(l1BufId)` 等一次。下一轮入口 `iter0 == 1` 时，`l1BufId` 已轮换到 `1 - l1BufId`，此时应该读"首轮段 2 搬的 PONG"，对应的 SetFlag 是首轮段 2 发的那一次，配对成立；**但若工程师误把段 3 的 WaitFlag 放到了段 1 内部**（而不是段 1 / 段 2 之后的公共位置），会导致首轮只等到 PING 就开始读，把未就绪的 PONG 也读进 L0。

**解决方案**：`WaitFlag<MTE2_MTE1>(l1BufId)` 必须放在段 3（消费段）入口，段 1 和段 2 只发 SetFlag，不消费；**两者的 WaitFlag 天然被外层循环按 pingpong 轮换错开**。

### 问题 5：仅测试小 K 场景（`kL1TileNum == 1`）看不到收益

**现象**：小 K 场景（如 `K = 256, kL1 = 512`，`kL1TileNum = 1`）跑出来和 pingpong 基线性能一样。

**原因**：`kL1TileNum == 1` 时段 2 的条件 `iter0 + 1 < kL1TileNum` 永远不成立，首轮段 2 不触发 → 退化为"每 tile 只搬一次"，MTE2 Preload 不生效。这是**预期行为**：小 K 下 PING 与 PONG 之间插入的指令数少，指令队列阻塞本来就不显著。

**解决方案**：收益验证应选 `K ≥ 2 × kL1` 且 `kL1TileNum ≥ 2` 的 shape（如教程例 `M=1024, K=4096, N=2048`，`kL1 = 256`，`kL1TileNum = 16`），才能观察到 MTE2 段 cycle 节省。

### 问题总结

| # | 问题 | 根因 | 解决方案 | 影响 |
|---|------|------|---------|------|
| 1 | 段 1 漏 WaitFlag | 新增分支忘写 `WaitFlag<MTE1_MTE2>` | 段 1 入口必须保留 WaitFlag | 挂起 / 数据脏写 |
| 2 | `curL1BufId` 未切换 | 段 2 沿用段 1 的 `l1BufId` | `if (tileIdx/blockNum==0) curL1BufId = 1 - l1BufId` | 数据覆写 / 精度错 |
| 3 | 段 2 越界预取 | 漏判 `iter0 + 1 < kL1TileNum` | 段 2 条件严格包含末轮边界 | GM 非法访问 |
| 4 | `MTE2_MTE1` 配对错位 | WaitFlag 放错位置 | WaitFlag 只放段 3 入口 | 数据未就绪 / 挂起 |
| 5 | 小 K 场景无收益 | `kL1TileNum == 1` 段 2 不触发 | 选 `kL1TileNum ≥ 2` 的 shape 验证 | 看起来"无效"（实际是场景不适用） |

---

## 8. 实测性能（教程参考）

以 `M=1024, K=4096, N=2048`、BF16、`baseM=baseN=256, baseK=128/sizeof(BF16)=64, kL1=512/sizeof(BF16)=256, kL1TileNum=16, PINGPONG_NUM=2` 为例：

| 版本 | kernel (μs) | mac (μs) | scalar (μs) | mte1 (μs) | **mte2 (μs)** | fixpipe (μs) | icache_miss (%) |
|------|------------|---------|------------|----------|--------------|-------------|----------------|
| pingpong 基线（n_buffer） | 66.000 | 40.810 | 2.558 | 10.659 | **37.595** | 1.980 | 1.200 |
| **MTE2 Preload** | **63.288** | 41.880 | 5.924 | 14.299 | **32.722** | 2.543 | 0.500 |
| Δ | **−2.712（−4.1%）** | +1.070 | +3.366 | +3.640 | **−4.873（−13.0%）** | +0.563 | −0.700 |

**关键观察**：
- **MTE2 段 −13% 是本优化的直接收益**（PONG 发射不再被阻塞）
- scalar / mte1 轻微上升是分支判断和事件处理复杂度的正常代价
- kernel 净收益 **−4.1%**（MTE2 段节省 > 其他段增量）
- icache_miss 下降是循环结构简化的副效应

> 不同 shape 下净收益波动范围 **−3% ~ −10%**。收益上界取决于 MTE2 占比（MTE2 占比越高，节省越显著）和指令队列阻塞程度（PING/PONG 间插入指令数越多，阻塞越严重，收益越大）。

---

## 9. 选型决策（与策略专家共享）

| 场景判定 | 建议 | 原因 |
|---------|------|------|
| pingpong 未启用 | **先开 pingpong，不要直接上 MTE2 Preload** | 本优化的前置条件是 pingpong 已开，未开时"PING/PONG 阻塞"不存在 |
| pingpong 已开；`kL1TileNum < 2` | 不开 MTE2 Preload | 段 2 不触发，等价于退回 pingpong 基线 |
| pingpong 已开；流水图显示 MTE2_PONG 被明显延迟（PING 终点与 PONG 起点 gap 大） | **推荐开 MTE2 Preload** | 本优化的直接信号 |
| pingpong 已开；MTE2 busy > 85% 且带宽利用率 ≥ 85%（真 MTE2 bound HBM 打满型） | 不推荐（收益小） | 瓶颈在 HBM 带宽，不在 MTE2 发射；应先走 A/B-Full-Load 或 baseK 调整 |
| pingpong 已开；MTE2 busy > 60% 且带宽利用率 < 70%（假 MTE2 bound 小数据块密集） | **先上小数据块合并载入，再叠加 MTE2 Preload** | 小数据块合并载入能把单次搬运拉到 ≥ 20 KB，间接减少 PING/PONG 间指令数；若仍有发射阻塞，MTE2 Preload 继续叠加 |
| pingpong + A/B-Full-Load 已开；流水图仍显示全载侧首轮 MTE2 后有 gap | **可叠加 MTE2 Preload** | 全载侧只在首轮搬一次，对侧 pingpong 仍存在阻塞风险；叠加后首轮 MTE2 按"全载侧 + 对侧 PING + 对侧 PONG"依次发射 |
| StreamK 分支 | 可叠加但收益有限 | StreamK 固定 `l1BufferNum=2`、每核 `skSingleCoreK ≥ 2048`，MTE2 段天然大，发射阻塞概率低 |

---

## 10. 与其他优化的叠加关系总表

| 叠加对象 | 兼容性 | 语义与注意事项 |
|---------|-------|---------------|
| **pingpong 双缓冲** | ✅ **前置条件** | MTE2 Preload 本质是 pingpong 的指令级增强，必须先开 pingpong |
| **SWAT 机制 A/B/C** | ✅ 兼容 | SWAT 改的是 `GetTileIdx` 顺序，不影响 MTE2 发射时序 |
| **StreamK（SK / DP+SK）** | ✅ 兼容但收益有限 | StreamK 下 MTE2 段天然够长，阻塞风险低 |
| **A-Full-Load / B-Full-Load（减少重复载入）** | ✅ 兼容 | 全载只改驻留侧布局，对侧仍 pingpong，MTE2 Preload 作用于对侧 |
| **小数据块合并载入（Scale Access Coalescing）** | ✅ 兼容（**推荐先合并再 Preload**） | 合并载入把单次 MTE2 拉大，间接减少 PING/PONG 间指令数；若仍有发射阻塞，叠加 MTE2 Preload |
| **指令队列深度软件调大** | ❌ 不存在（硬件约束） | 指令队列深度是芯片固定值，软件只能绕开，不能调大 |

---

## 11. 自检清单（实施专家在提交代码前完成）

- [ ] **pingpong 双缓冲已启用**（`l1BufNum = 2`，L1 内存布局含 PING/PONG 两份 A/B）
- [ ] `kL1TileNum ≥ 2`（`K / kL1 ≥ 2`），否则段 2 不触发
- [ ] 外层循环引入**段 1**（首轮首分片，`tileIdx/blockNum == 0 && iter0 == 0`）专用分支
- [ ] 外层循环引入**段 2**（预取下一分片）分支，条件含 `iter0 + 1 < kL1TileNum`
- [ ] 段 2 在首轮（`tileIdx/blockNum == 0`）时有 `curL1BufId = 1 - l1BufId` 切换；非首轮保持 `curL1BufId = l1BufId`
- [ ] 段 1 与段 2 都有独立的 `WaitFlag<MTE1_MTE2>` + `SetFlag<MTE2_MTE1>`
- [ ] 段 3（消费段）只在入口 `WaitFlag<MTE2_MTE1>(l1BufId)` 一次，不在段 1 / 段 2 内部 Wait
- [ ] Init 中 `SetFlag<MTE1_MTE2>(0)` 和 `SetFlag<MTE1_MTE2>(1)` 两次预置都保留
- [ ] Kernel 结束前 `WaitFlag<MTE1_MTE2>(0)` 和 `WaitFlag<MTE1_MTE2>(1)` 两次收尾（与 pingpong 基线一致）
- [ ] **精度验证通过**（小数据块偏移、flag 配对错误都会导致精度错误）
- [ ] 性能对比报告已给出 kernel / mac / scalar / mte1 / **mte2** / fixpipe 各段对照；**mte2 段应显著下降**（若未下降，说明场景不适用或代码写错）

若缺任一项，应回到 §7 问题列表排查，或标注"MTE2 Preload 在本场景下不适用，回退 pingpong 基线"。
