# P20 Playbook: 三缓冲轮转（BuffersPolicy3buff）

> 本 Playbook 为**强制流程**。采纳 P20 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P20 的核心是**用三套独立 buffer 和三个独立轮转指针，让 MTE2 搬运、Cube 计算、Vector 计算三个阶段完全并行**，形成三级流水。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p20_locations.txt`：

```bash
# 1. Cube/Vector 混合架构
 grep -n "Cube|Vector|AIC|AIV|mmad|Mmad|VEC" \
    shared/original/op_kernel/*.cpp > /tmp/p20_locations.txt
# 2. 三级流水阶段
 grep -n "CopyIn|DataCopy|CopyOut|Compute|EnQue|DeQue" \
    shared/original/op_kernel/*.cpp >> /tmp/p20_locations.txt
# 3. 当前 buffer 策略
 grep -n "BUFFER_NUM|InitBuffer|TBuf|TQue|double.*buffer|ping.*pong" \
    shared/original/op_kernel/*.cpp >> /tmp/p20_locations.txt
# 4. 同步机制
 grep -n "SyncAll|PipeBarrier|SetFlag|WaitFlag|HardEvent" \
    shared/original/op_kernel/*.cpp >> /tmp/p20_locations.txt
# 5. 已有的三缓冲
 grep -n "BuffersPolicy3buff|triple.*buffer|三缓冲" \
    shared/original/op_kernel/*.cpp >> /tmp/p20_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **混合架构**：Cube / Vector / Mmad 使用位置
- **流水阶段**：Load / Cube / Vector 的代码段划分
- **Buffer 现状**：当前 buffer 数量、大小、策略（单/双缓冲）
- **同步现状**：SyncAll / PipeBarrier / SetFlag 使用情况
- **已有三缓冲**：是否已使用

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Buffer 策略 | `?` (单/双缓冲) | 三缓冲 | `?_kernel.cpp:L?` |
| 流水阶段 | `?` (Load/Cube/Vector) | 三级完全重叠 | `?_kernel.cpp:L?` |
| Buffer 大小 | `? bytes × ?` | `? bytes × 3` | `?_kernel.cpp:L?` |
| 同步方式 | `?` (SyncAll / PipeBarrier) | HardEvent 分段同步 | `?_kernel.cpp:L?` |
| 事件 ID | `?` (当前分配) | 3 组 × 3 阶段 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的流水结构和 buffer 空间，判断你的代码属于以下哪种形态：

- **形态 α — 基础三缓冲（3 个 VEC buffer，手动轮转）**：UB 空间紧张，仅用 3 个 VECCALC buffer 做简单轮转。
- **形态 β — BuffersPolicy3buff 模板类（标准模式，最常见）**：用模板类封装三套 buffer 和三个独立指针，代码可维护性高。
- **形态 γ — 多路三缓冲（多个独立数据流各用一套三缓冲）**：算子有多个并行数据流（如 Q/K/V 各自独立），每路一套三缓冲。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — BuffersPolicy3buff 模板类）

```cpp
// === 改造前（双缓冲，仅 2 阶段重叠）===
TBuf<QuePosition::VECCALC> bufA;
TBuf<QuePosition::VECCALC> bufB;
pipe.InitBuffer(bufA, tileSize * sizeof(T));
pipe.InitBuffer(bufB, tileSize * sizeof(T));

for (uint32_t i = 0; i < loopCount; i++) {
    auto& inBuf = (i % 2 == 0) ? bufA : bufB;
    auto& outBuf = inBuf;
    
    // Load 和 Compute 部分重叠，但 Cube/Vector 串行
    DataCopy(inBuf.Get<T>(), inGm, count);
    SyncAll();  // 粗粒度等待
    Mmad(...);
    Compute(...);
    DataCopy(outGm, outBuf.Get<T>(), count);
}

// === 改造后（三缓冲，三级完全重叠）===
template<typename T, uint32_t Size>
class BuffersPolicy3buff {
public:
    __aicore__ inline void Init(TPipe& pipe) {
        pipe.InitBuffer(a_, Size);
        pipe.InitBuffer(b_, Size);
        pipe.InitBuffer(c_, Size);
    }
    
    __aicore__ inline TBuf<QuePosition::VECCALC>& GetLoad() {
        uint32_t idx = loadFlag_ % 3;
        loadFlag_++;
        return GetByIdx(idx);
    }
    __aicore__ inline TBuf<QuePosition::VECCALC>& GetCube() {
        uint32_t idx = cubeFlag_ % 3;
        cubeFlag_++;
        return GetByIdx(idx);
    }
    __aicore__ inline TBuf<QuePosition::VECCALC>& GetVec() {
        uint32_t idx = vecFlag_ % 3;
        vecFlag_++;
        return GetByIdx(idx);
    }
    
private:
    __aicore__ inline TBuf<QuePosition::VECCALC>& GetByIdx(uint32_t idx) {
        return (idx == 0) ? a_ : (idx == 1) ? b_ : c_;
    }
    
    TBuf<QuePosition::VECCALC> a_, b_, c_;
    uint32_t loadFlag_ = 0;
    uint32_t cubeFlag_ = 0;
    uint32_t vecFlag_ = 0;
};

// 使用
static constexpr uint32_t BUF_SIZE = tileSize * sizeof(T);
BuffersPolicy3buff<T, BUF_SIZE> buffers;
buffers.Init(pipe);

// 事件 ID 分配（每阶段 3 个，共 9 个，需合理复用或限制）
static constexpr int EV_LOAD[3] = {0, 1, 2};
static constexpr int EV_CUBE[3] = {3, 4, 5};
static constexpr int EV_VEC[3]  = {6, 7, 8};

for (uint32_t i = 0; i < loopCount; i++) {
    uint32_t bufIdx = i % 3;
    
    // Stage 1: MTE2 Load（与上一轮 Cube/Vector 并行）
    auto& loadBuf = buffers.GetLoad();
    WaitFlag<HardEvent::MTE2_MTE1>(EV_LOAD[bufIdx]);
    DataCopy(loadBuf.Get<T>(), inGm + i * tileSize, count);
    SetFlag<HardEvent::MTE2_MTE1>(EV_LOAD[bufIdx]);
    
    // Stage 2: Cube Compute（与本轮 Load / 上轮 Vector 并行）
    auto& cubeBuf = buffers.GetCube();
    WaitFlag<HardEvent::MTE1_M>(EV_CUBE[bufIdx]);
    Mmad(cubeBuf.Get<T>(), ...);
    SetFlag<HardEvent::MTE1_M>(EV_CUBE[bufIdx]);
    
    // Stage 3: Vector Compute（与本轮 Load/Cube 并行）
    auto& vecBuf = buffers.GetVec();
    WaitFlag<HardEvent::M_FIX>(EV_VEC[bufIdx]);
    Compute(vecBuf.Get<T>(), ...);
    SetFlag<HardEvent::M_FIX>(EV_VEC[bufIdx]);
    
    // CopyOut（可并入 Stage 3 或独立为 Stage 4）
    DataCopy(outGm + i * tileSize, vecBuf.Get<T>(), count);
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（基础三缓冲，手动轮转）**：
  当不想引入模板类时，直接手动管理 3 个 buffer：
  ```cpp
  TBuf<QuePosition::VECCALC> buf[3];
  for (uint32_t i = 0; i < 3; i++) {
      pipe.InitBuffer(buf[i], BUF_SIZE);
  }
  
  for (uint32_t i = 0; i < loopCount; i++) {
      uint32_t loadIdx = i % 3;
      uint32_t cubeIdx = (i + 2) % 3;  // Cube 落后 Load 2 轮
      uint32_t vecIdx  = (i + 1) % 3;  // Vector 落后 Load 1 轮
      // ... 使用 buf[loadIdx], buf[cubeIdx], buf[vecIdx]
  }
  ```
  形态 α 代码直观但易出错，flag 计算需仔细验证。

- **形态 γ（多路三缓冲）**：
  若算子有 Q/K/V 三路独立数据流，每路一套三缓冲：
  ```cpp
  BuffersPolicy3buff<T, BUF_SIZE> qBuffers, kBuffers, vBuffers;
  qBuffers.Init(pipe);
  kBuffers.Init(pipe);
  vBuffers.Init(pipe);
  // 三路各自独立轮转，事件 ID 需分组（每路 3 个，共 9 个）
  ```
  形态 γ 的 UB 占用是单路的 3 倍，需严格复核容量。

- **与 P1 的冲突**：P1 是双缓冲，P20 是三缓冲。两者互斥——同一数据流不能同时用双缓冲和三缓冲。若算子有多个数据流，次要流可用 P1 双缓冲，主性能瓶颈流用 P20 三缓冲。

- **与 P19 的冲突**：P19（Custom PingPong）也是 buffer 轮转策略。P20 的三缓冲可视为 P19 的扩展（从 2 套到 3 套）。若算子已有 P19，升级 P20 只需增加第三套 buffer 和指针。

- **与 P5 的协同**：P5 解决 tiling 分块，P20 解决单块内的三级流水。P5 决定 `tileSize`，P20 在 `tileSize` 内做三级并行。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: 3 × BUF_SIZE ≤ UB 可用空间（或 L1，取决于 buffer 位置）
约束 2: 事件 ID 总数 ≤ 16。三缓冲通常需 3-9 个事件 ID，若超 16 需复用
约束 3: loadFlag_ / cubeFlag_ / vecFlag_ 为 uint32_t，循环次数 < 2^32 不会溢出
约束 4: 每轮迭代中，同一 buffer 不能同时被 Load 和 Cube 使用（时序由事件保证）
约束 5: 循环结束后需 Wait 所有未完成阶段，不能提前退出
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `BUF_SIZE = ?`, `3 × BUF_SIZE = ?`
- `UB 可用 = ?`, `占用率 = ?%`
- `事件 ID = [?, ?, ?]`, `总数 = ?`
- `循环次数 = ?`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 存在 3 个 InitBuffer（同一组三缓冲）
 grep -cE "InitBuffer.*a_|InitBuffer.*b_|InitBuffer.*c_|InitBuffer.*buf\[0\]|InitBuffer.*buf\[1\]|InitBuffer.*buf\[2\]" modified_files/op_kernel/*.cpp
# 期望: >= 3

# 检查 2: 有三套独立轮转指针或 % 3 索引
grep -cE "loadFlag_|cubeFlag_|vecFlag_|%\s*3|GetLoad|GetCube|GetVec" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 存在分段 HardEvent 同步（不是全局 SyncAll）
grep -cE "SetFlag|WaitFlag" modified_files/op_kernel/*.cpp
# 期望: >= 6（每轮至少 3 Set + 3 Wait）

# 检查 4: 循环内无 SyncAll（已被细粒度同步替代）
grep -cE "for.*\{[^}]*SyncAll" modified_files/op_kernel/*.cpp
# 期望: == 0

# 检查 5: 循环结束后有最终 Wait（所有阶段完成）
grep -cE "WaitFlag.*EV_|WaitFlag.*event|for.*end.*WaitFlag" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：TBuf 模板参数不匹配 | `BuffersPolicy3buff` 的 `TBuf` 位置参数（`QuePosition::VECCALC` / `A1` / `B1`）必须与算子实际使用的位置一致 |
| 运行时：UB 越界 | 三缓冲占用是单缓冲的 3 倍。复核约束 1，若超 UB 容量，退化为 P1 双缓冲或减少 tileSize |
| 运行时：数据竞争（读到未完成的 buffer） | 检查事件同步是否完整。常见遗漏：Cube 阶段未 Wait Load 完成就开始计算 |
| 运行时：最后一轮结果错误 | 循环结束后必须 Wait 所有未完成阶段。若直接退出，最后几轮的 CopyOut 可能未完成 |
| 事件 ID 超过 16 | 三缓冲通常需 3-9 个事件。若多路共用时超 16，将相邻阶段的事件合并（如 Load→Cube 和 Cube→Vector 用同一组事件 ID，按奇偶区分） |
| 指针轮转不同步 | `loadFlag_` / `cubeFlag_` / `vecFlag_` 必须从 0 开始且严格每轮 +1。若某分支跳过一轮，会导致 buffer 冲突 |
| 与 P1 的 buffer 冲突 | 同一算子内不要混合双缓冲和三缓冲于同一数据流。若有多流，明确标注哪流用 P1、哪流用 P20 |
| 三缓冲但无实际重叠 | 若 Cube 计算时间极短（< MTE2 搬运时间），三缓冲的收益可能不明显。先 profiling 确认瓶颈 |
| flag 溢出 | `uint32_t` flag 在循环次数 > 2^32 时溢出。但通常 loopCount < 1M，安全。若超长序列，用 `loopIdx % 3` 替代 flag 自增 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P20 Playbook Completion]
Step 1: done (/tmp/p20_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: buf_size=? total_3x=? ub_avail=? events=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
