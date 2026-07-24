## 公共工具：Cross Core Sync 与 WorkspaceQueue 详细参考

本文档包含 WorkspaceQueue 环形缓冲区、批量同步模式、CrossCore flag 的完整实现细节与代码示例。
概览与判断规则见 `@references/dsl2Ascendc.md`。

---

## 第三章：公共工具

### 0. 同步模式判断规则

| DSL 特征 | 同步模式 | AscendC 实现 |
|:---|:---|:---|
| `T.set_cross_flag` 在 n_tile 循环**内部** | 逐 tile 同步（WorkspaceQueue） | 环形缓冲 + 每 tile Acquire/Release |
| `T.set_cross_flag` 在 n_tile 循环**外部** | 批量同步（Bulk Sync） | 单次 CrossCoreSetFlag/WaitFlag |

### 1. 通过 Workspace Queue 实现跨核同步

使用基于 workspace GM 的环形缓冲区进行 AIC → AIV 数据传输，配合 `CrossCoreSetFlag/WaitFlag` 同步：

```cpp
template <typename T, uint32_t DEPTH>
class WorkspaceQueue {
public:
    // AIV 初始化：将所有槽位标记为空闲（生产者可以写入）
    __aicore__ inline void InitFreeSlots() {
        for (uint32_t i = 0; i < DEPTH; ++i) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);
        }
    }

    // AIC（生产者）：等待空闲槽位，然后通过 Fixpipe 写入
    __aicore__ inline GlobalTensor<T> ProducerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(vecNotifyCubeId_);  // 等待"槽位空闲"
        return workspace_[head_ % DEPTH * slotSize_];
    }
    __aicore__ inline void ProducerRelease() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeNotifyVecId_);  // 发送"数据就绪"信号
        head_++;
    }

    // AIV（消费者）：等待数据就绪，然后通过 MTE2 读取
    __aicore__ inline GlobalTensor<T> ConsumerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(cubeNotifyVecId_);  // 等待"数据就绪"
        return workspace_[tail_ % DEPTH * slotSize_];
    }
    __aicore__ inline void ConsumerRelease() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);  // 发送"槽位空闲"信号
        tail_++;
    }
};
```

**关键同步流程：**
```
初始化阶段：
  AIV: InitFreeSlots() → 设置 DEPTH 个"槽位空闲"标志

循环阶段（每个 tile）：
  AIC: ProducerAcquire() → 等待"槽位空闲"
  AIC: Fixpipe(slot, ...) → 写入 workspace GM
  AIC: ProducerRelease() → 设置"数据就绪"

  AIV: ConsumerAcquire() → 等待"数据就绪"
  AIV: DataCopy(local, slot) → 从 workspace GM 读取
  AIV: Process() → 计算
  AIV: ConsumerRelease() → 设置"槽位空闲"
```

#### A. 批量同步模式（Bulk Sync，无环形缓冲）

当 DSL 中 Cube 需要**先完成所有 n_tile 写入 workspace 后**，Vector 才能开始处理时（如需要全局统计量的两遍处理），不使用 WorkspaceQueue 环形缓冲，而是使用**单次 CrossCoreSetFlag/WaitFlag 批量同步**：

**典型场景**：
```python
# DSL: Cube 完成所有 tile 后一次性通知 Vector
with T.Scope("C"):
    for by in T.serial(n_num):
        # ... matmul all tiles, write to workspace ...
        T.copy(C_L0, workspace[bx * block_M, by * block_N])
    T.set_cross_flag("FIX", 0)  # 所有 tile 完成后才发一次信号

with T.Scope("V"):
    T.wait_cross_flag(0)  # 等待所有 tile 就绪
    # Pass 1: 扫描所有 tile 计算统计量
    for by in T.serial(n_num):
        T.copy(workspace[...], ub)
        # ... 累积 per-row absmax ...
    # Pass 2: 利用统计量处理
    for by in T.serial(n_num):
        T.copy(workspace[...], ub)
        # ... quantize ...
```

**AscendC 翻译**：
```cpp
// Cube 侧：循环所有 n_tile，全部写完后一次性信号
if ASCEND_IS_AIC {
    for (int by = 0; by < nTiles; by++) {
        auto wsBlock = wsGM_[bx * baseM * N + by * baseN];
        mm_.ComputeBlock(aBlock, bBlock, wsBlock, H_K, N);  // dstStride=N
    }
    CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_NOTIFY_VECTOR_ID);  // 只发一次信号
}

// Vector 侧：等待一次信号，然后两遍处理
if ASCEND_IS_AIV {
    CrossCoreWaitFlag<0x2>(CUBE_NOTIFY_VECTOR_ID);  // 只等一次

    // Pass 1: 全局扫描
    for (int by = 0; by < nTiles; by++) { /* ... accumulate stats ... */ }
    // Pass 2: 利用统计量处理
    for (int by = 0; by < nTiles; by++) { /* ... quantize ... */ }
}
```

### 2. WorkspaceQueue vs 批量同步对比

| 特性 | WorkspaceQueue（逐 tile 同步） | 批量同步（Bulk Sync） |
|:---|:---|:---|
| **信号次数** | 每个 tile 一次 Acquire/Release | Cube 全部完成后一次 |
| **Workspace 大小** | DEPTH × baseM × baseN × sizeof(T) | M × N × sizeof(T)（全输出） |
| **Vector 启动时机** | Cube 写完一个 tile 即可开始 | 必须等 Cube 全部写完 |
| **适用场景** | 逐 tile 独立处理（如 LeakyReLU、Scale） | 需要全局统计量（如 ReduceMax + 量化） |
| **DSL 特征** | `T.set_cross_flag` 在循环内 | `T.set_cross_flag` 在循环外 |
| **核分配** | BlockScheduler 分配 mBlocks×nBlocks | 每核一个 m_block，Cube 内循环 n_tiles |

### 3. CrossCore flag 规则

#### 规则：`T.set_cross_flag` → 单个 CrossCoreSetFlag

TileLang 中 `T.set_cross_flag("FIX", 0)` / `T.wait_cross_flag(0)` 翻译为 **单个 flag ID**，对所有 AIV 子块广播：

```cpp
// ✅ 正确：单 flag 广播给所有 AIV 子块（KERNEL_TYPE_MIX_AIC_1_2）
#define CUBE_NOTIFY_VECTOR_ID 0x8

if ASCEND_IS_AIC {
    // AIC 完成所有 tile 后发一次信号
    CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_NOTIFY_VECTOR_ID);
}

if ASCEND_IS_AIV {
    // 所有 AIV 子块（vid=0, vid=1）都等待同一个 flag
    CrossCoreWaitFlag<0x2>(CUBE_NOTIFY_VECTOR_ID);
    // vid 由 GetSubBlockIdx() 区分各自的数据偏移
    int rowOffset = AscendC::GetSubBlockIdx() * subTileM;
}
```

**❌ 错误写法**（逐 AIV 子块发送不同 flag）：

```cpp
// 错误：AIC 发送 0x8 给 vid=0，发送 0x9 给 vid=1
for (int i = 0; i < VEC_NUM; i++) {
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x8 + i);
}
// AIV: CrossCoreWaitFlag<0x2>(0x8 + vid_);
```

> **为什么会出错**：per-subblock flag 是逐 tile 同步（ring buffer）模式的写法，适用于 `matmul_leakyrelu` 中的 WorkspaceQueue。批量同步场景（two-pass 量化）只需一次信号，用单 flag 广播即可。

**识别口诀**：
- `T.set_cross_flag("FIX", idx)` → `CrossCoreSetFlag<0x2, PIPE_FIX>(0x8 + idx)`，只调用一次
- `T.wait_cross_flag(idx)` → `CrossCoreWaitFlag<0x2>(0x8 + idx)`，所有 AIV 子块共享同一 flag
- 批量同步场景用单 flag 广播，**不要**逐 AIV 子块发送不同 flag
