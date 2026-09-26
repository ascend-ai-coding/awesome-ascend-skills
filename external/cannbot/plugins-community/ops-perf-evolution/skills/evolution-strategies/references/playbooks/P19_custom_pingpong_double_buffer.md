# P19 Playbook: 自定义多级 PingPong 双缓冲

> 本 Playbook 为**强制流程**。采纳 P19 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P19 的核心是**绕过标准 TQue 队列，用手写 ping/pong 轮转实现 L0A/L0B/L1 等多级存储的自定义双缓冲**，支持 `GetPre`（取前一轮）和 `GetReused`（复用模式），使搬运与计算在多级别上完全重叠。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p19_locations.txt`：

```bash
# 1. 多级存储流水（L1→L0→Cube）
grep -n "L0A\|L0B\|L0C\|L1\|L1A\|L1B\|GM.*L1\|L1.*L0" \
    shared/original/op_kernel/*.cpp > /tmp/p19_locations.txt
# 2. 标准 TQue / TBuf 使用
grep -n "TQue\|TBuf\|BUFFER_NUM\|InitBuffer" \
    shared/original/op_kernel/*.cpp >> /tmp/p19_locations.txt
# 3. 复用模式（resident / preload / reuse）
grep -n "reuse\|reused\|preLoad\|resident\|Q.*reuse\|KV.*reuse" \
    shared/original/op_kernel/*.cpp >> /tmp/p19_locations.txt
# 4. 已有的自定义 PingPong
grep -n "BuffersPolicyDB\|pingpong\|ping_pong\|GetPre\|GetReused\|flag1_\|flag2_" \
    shared/original/op_kernel/*.cpp >> /tmp/p19_locations.txt
# 5. 同步事件
grep -n "SetFlag\|WaitFlag\|HardEvent\|SyncAll\|PipeBarrier" \
    shared/original/op_kernel/*.cpp >> /tmp/p19_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **多级存储**：L1→L0A/L0B→Mmad 的代码段位置
- **标准队列**：当前 TQue/TBuf 声明和使用位置
- **复用需求**：哪些数据需要复用（Q 常驻、KV 旋转等）
- **已有 PingPong**：是否已有自定义实现
- **同步现状**：事件 ID 分配、SetFlag/WaitFlag 使用

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 存储级别 | `?` (L1/L0A/L0B/L0C) | 需要自定义的级别 | `?_kernel.cpp:L?` |
| 当前机制 | `?` (TQue/TBuf) | BuffersPolicyDB | `?_kernel.cpp:L?` |
| 复用模式 | `?` (无/Get/GetPre/GetReused) | `alpha/beta/gamma` 见 3A | `?_kernel.cpp:L?` |
| Buffer 大小 | `? bytes × ?` | `? bytes × 2` 每级 | `?_kernel.cpp:L?` |
| 事件 ID | `?` (当前) | 2× 级别数 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的存储级别和复用需求，判断你的代码属于以下哪种形态：

- **形态 α — 单级自定义 PingPong（替换单级 TQue）**：仅一个级别（如 L1 或 L0A）需要自定义双缓冲。
- **形态 β — 多级 PingPong（L1 + L0A + L0B 同时自定义）**：多级存储都需要自定义控制，实现 GM→L1→L0→Cube 全流水重叠。
- **形态 γ — Q/KV 复用模式（GetPre + GetReused）**：Attention 场景下 Q 需要 `GetPre`（前一轮复用），KV 需要 `GetReused`（交替复用）。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — 多级 PingPong，最常见）

```cpp
// === 改造前（标准 TQue，无 fine-grained 复用）===
TQue<TPosition::L1, BUFFER_NUM> l1Queue;
TQue<TPosition::L0A, BUFFER_NUM> l0aQueue;
TQue<TPosition::L0B, BUFFER_NUM> l0bQueue;

for (uint32_t i = 0; i < loopCount; i++) {
    auto l1Buf = l1Queue.AllocTensor<T>();
    DataCopy(l1Buf, gmData, count);
    l1Queue.EnQue(l1Buf);
    // ... TQue 自动管理，无法做 GetPre/GetReused
}

// === 改造后（自定义 BuffersPolicyDB，多级控制）===
template<typename T, uint32_t Size>
class BuffersPolicyDB {
public:
    __aicore__ inline void Init(TPipe& pipe) {
        pipe.InitBuffer(ping_, Size);
        pipe.InitBuffer(pong_, Size);
    }
    
    // Get: 标准 ping-pong，返回当前轮 buffer
    __aicore__ inline LocalTensor<T>& Get() {
        flag1_ = !flag1_;
        return flag1_ ? ping_.Get<T>() : pong_.Get<T>();
    }
    
    // GetPre: 返回前一轮的 buffer（与 Get 相反）
    __aicore__ inline LocalTensor<T>& GetPre() {
        return flag1_ ? pong_.Get<T>() : ping_.Get<T>();
    }
    
    // GetReused: 用独立 flag2 做交替复用
    __aicore__ inline LocalTensor<T>& GetReused() {
        flag2_ = !flag2_;
        return flag2_ ? pong_.Get<T>() : ping_.Get<T>();
    }
    
private:
    TBuf<QuePosition::VECCALC> ping_, pong_;
    bool flag1_ = false;
    bool flag2_ = false;
};

// 多级实例化
static constexpr uint32_t L1_SIZE = 128 * 1024;
static constexpr uint32_t L0_SIZE = 64 * 1024;

BuffersPolicyDB<float, L1_SIZE> l1Buffers;
BuffersPolicyDB<float, L0_SIZE> l0aBuffers;
BuffersPolicyDB<float, L0_SIZE> l0bBuffers;

l1Buffers.Init(pipe);
l0aBuffers.Init(pipe);
l0bBuffers.Init(pipe);

// 事件 ID（每级 2 个）
static constexpr int EV_L1[2] = {0, 1};
static constexpr int EV_L0A[2] = {2, 3};
static constexpr int EV_L0B[2] = {4, 5};

for (uint32_t i = 0; i < loopCount; i++) {
    uint32_t bufIdx = i % 2;
    
    // Stage 1: GM→L1（ping）
    auto& l1Current = l1Buffers.Get();
    DataCopy(l1Current, gmData + i * tileSize, count);
    SetFlag<HardEvent::MTE2_MTE1>(EV_L1[bufIdx]);
    
    // Stage 2: L1→L0（consume previous L1 = GetPre）
    WaitFlag<HardEvent::MTE2_MTE1>(EV_L1[(bufIdx + 1) % 2]);
    auto& l1Prev = l1Buffers.GetPre();
    auto& l0aCurrent = l0aBuffers.Get();
    auto& l0bCurrent = l0bBuffers.Get();
    CopyL1ToL0(l1Prev, l0aCurrent, l0bCurrent);
    SetFlag<HardEvent::MTE1_M>(EV_L0A[bufIdx]);
    SetFlag<HardEvent::MTE1_M>(EV_L0B[bufIdx]);
    
    // Stage 3: Cube Compute（consume previous L0 = GetPre）
    WaitFlag<HardEvent::MTE1_M>(EV_L0A[(bufIdx + 1) % 2]);
    WaitFlag<HardEvent::MTE1_M>(EV_L0B[(bufIdx + 1) % 2]);
    auto& l0aPrev = l0aBuffers.GetPre();
    auto& l0bPrev = l0bBuffers.GetPre();
    Mmad(cTensor, l0aPrev, l0bPrev, ...);
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（单级自定义 PingPong）**：
  仅替换一个级别的 TQue：
  ```cpp
  BuffersPolicyDB<float, BUF_SIZE> l1Buffers;
  l1Buffers.Init(pipe);
  // 其他级别仍用标准 TQue
  ```
  形态 α 改动最小，适合只有一个级别需要 fine-grained 控制的场景。

- **形态 γ（Q/KV 复用模式）**：
  Attention 场景下 Q 常驻、KV 旋转：
  ```cpp
  // Q: GetPre 模式（前一轮 Q 直接复用）
  auto& qCurrent = qBuffers.Get();
  auto& qPrev = qBuffers.GetPre();  // 前一轮的 Q
  // 若 Q 不变，直接用 qPrev，无需重新加载
  
  // KV: GetReused 模式（交替复用）
  auto& kvCurrent = kvBuffers.GetReused();
  auto& kvPrev = kvBuffers.GetReused();  // 上一轮交替的 KV
  ```
  形态 γ 需配合 P14（FA tiling）和 P18（L1 分区）使用。

- **与 P1 的冲突**：P1 是标准双缓冲（TQue BUFFER_NUM=2），P19 是自定义双缓冲（BuffersPolicyDB）。同一数据流不能同时用两者。若算子有多个数据流，部分可用 P1，部分可用 P19。

- **与 P20 的冲突**：P20 是三缓冲，P19 是双缓冲。同一数据流不能同时用两者。性能瓶颈流选 P20，其他流选 P19。

- **与 P28 的协同**：P28 的 HardEvent 同步是 P19 多级 PingPong 的必需组件。P19 提供 buffer 轮转，P28 提供跨级事件同步。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: 每级 buffer 大小 × 2 ≤ 该级别可用空间
约束 2: 事件 ID 总数 = 2 × 级别数 ≤ 16
约束 3: GetPre() 返回的 buffer 必须已完成消费（不能还在被上一阶段使用）
约束 4: GetReused() 的 flag2_ 与 Get() 的 flag1_ 独立，不能互相干扰
约束 5: 自定义 PingPong 失去 TQue 的自动流控，所有同步必须显式管理
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `级别数 = ?`, `每级大小 = ?`, `总占用 = ?`
- `事件 ID = [?, ?, ?]`, `总数 = ?`
- `flag1_ 初始值 = ?`, `flag2_ 初始值 = ?`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 存在 BuffersPolicyDB 或等效自定义类
grep -cE "BuffersPolicyDB|class.*ping.*pong|class.*PingPong" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 Get / GetPre / GetReused 方法
grep -cE "Get\s*\(\s*\)|GetPre\s*\(\s*\)|GetReused\s*\(\s*\)" modified_files/op_kernel/*.cpp
# 期望: >= 2

# 检查 3: 有 2× InitBuffer（ping + pong）
grep -cE "InitBuffer.*ping|InitBuffer.*pong" modified_files/op_kernel/*.cpp
# 期望: >= 2

# 检查 4: 有 flag1_ / flag2_ 切换逻辑
grep -cE "flag1_|flag2_" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 有被替换级别的 TQue 已删除
grep -cE "TQue.*L1.*BUFFER|TQue.*L0A.*BUFFER|TQue.*L0B.*BUFFER" modified_files/op_kernel/*.cpp
# 期望: == 0（被替换的级别）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：TBuf 位置参数不匹配 | `BuffersPolicyDB` 的 `TBuf` 位置必须与替换的 `TQue` 一致。`QuePosition::VECCALC` 不能替换 `QuePosition::A1` |
| 运行时：数据竞争（读到未完成的 buffer） | GetPre() 返回的 buffer 可能还在被上一阶段使用。确保事件同步完整，WaitFlag 后再 GetPre |
| 运行时：flag1_/flag2_ 初始值错误 | 两个 flag 必须从 `false` 开始。若初始值不一致，Get 和 GetPre 的对应关系会错位 |
| 事件 ID 超过 16 | 多级 PingPong 每级需 2 个事件。3 级 = 6 个事件，通常安全。若级别 > 8，考虑合并事件或减级 |
| 多级事件同步遗漏 | 常见遗漏：L1→L0 的 SetFlag 与 L0→Cube 的 WaitFlag 不配对。每级必须独立管理事件 |
| TQue 残留导致冲突 | 改造后必须删除被替换级别的 TQue 声明和 InitBuffer。残留会导致同一级别有两套 buffer 管理 |
| GetReused 与 Get 混淆 | GetReused 用独立 flag2_，与 Get 的 flag1_ 不共享。不要混用两个方法于同一数据流 |
| 多核场景事件 ID 冲突 | 若多核共享事件 ID，需按 coreId 偏移。例如 `eventId = baseEventId + blockIdx * eventsPerCore` |
| 循环结束后 buffer 未释放 | TQue 自动释放，自定义 PingPong 不自动释放。循环结束后显式标记所有 buffer 为空闲 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P19 Playbook Completion]
Step 1: done (/tmp/p19_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: levels=? size=? events=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
