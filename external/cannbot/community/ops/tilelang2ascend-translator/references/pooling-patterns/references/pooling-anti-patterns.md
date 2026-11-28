# Anti-Patterns: Common Pooling Kernel Mistakes

## AP-1: TQue Overhead in Small-Window Pooling

**症状**: 小窗口 pooling（如 KH=KW=1）性能远低于预期，kernel 时间中队列操作占比过高。

**错误模式**:
```cpp
// 每个 kd 位置都走完整的 Alloc→EnQue→DeQue→Free 循环
// 但窗口只有 KD 个数据行，TQue 开销无法被 KH×KW 摊薄
for (int kd = 0; kd < KD_; ++kd) {
    inLocal_ = inQueue_.AllocTensor<dataType>();   // 每次分配
    AscendC::DataCopy(inLocal_, xGM_[addr], ...);
    inQueue_.EnQue(inLocal_);                      // 入队
    inLocal_ = inQueue_.DeQue<dataType>();          // 出队
    // ... 很少的计算 ...
    inQueue_.FreeTensor(inLocal_);                 // 释放
}
```

**修复**: 当 KH==1 && KW==1 时，使用 direct TBuf 快路径（实现见 [reduce-d-fastpath-implementation.md](reduce-d-fastpath-implementation.md)，设计语义见 `ops/tilelang-op-design/references/pooling/reduce-d-fastpath.md`）。

---

## AP-2: Per-ow PipeBarrier<PIPE_V>

**症状**: ow 循环内每个位置都有 `PipeBarrier<PIPE_V>()`，kernel 性能异常低。

**错误模式**:
```cpp
for (int ow = 0; ow < OW_; ++ow) {
    AscendC::Add(acc[ow*C], acc[ow*C], in[iw*C], C);
    AscendC::PipeBarrier<PIPE_V>();  // ← OW 个冗余 barrier!
}
```

**修复**: 将 barrier 移到循环外，所有 ow 累加完成后只需一次 barrier（在下个 DataCopy 覆盖 buffer 前）。

```cpp
for (int ow = 0; ow < OW_; ++ow) {
    AscendC::Add(acc[ow*C], acc[ow*C], in[iw*C], C);
}
AscendC::PipeBarrier<PIPE_V>();  // 只一次
```

**影响**: c2 (OW=32) 下修复前 209.68us → 修复后 145.78us（1.44x 加速）。

---

## AP-3: Direct DataCopy Without Dual Barrier

**症状**: 使用 direct DataCopy（不用 TQue）但只有单个 barrier 或无 barrier，导致间歇性数据损坏或精度不一致。

**错误模式**:
```cpp
// ❌ 无 barrier：V 核可能在 M2E 完成前读取 UB
AscendC::DataCopy(buf, xGM_[addr], count);
PrepareInputTensorHelper(inFp32, buf, ...);  // 读到脏数据

// ❌ 单 barrier：只等 M2E 完成，但跨-pipe 同步不足
AscendC::DataCopy(buf, xGM_[addr], count);
AscendC::PipeBarrier<PIPE_MTE2>();            // 不够
PrepareInputTensorHelper(inFp32, buf, ...);
```

**修复**: 使用 PIPE_MTE2 + PIPE_ALL 双重屏障。

```cpp
AscendC::DataCopy(buf, xGM_[addr], count);
AscendC::PipeBarrier<PIPE_MTE2>();   // M2E 搬运完成
AscendC::PipeBarrier<PIPE_ALL>();    // 所有 pipe 同步
PrepareInputTensorHelper(inFp32, buf, ...);  // 数据安全
```

---

## AP-4: DeQue In-Place With BUFFER_NUM=1

**症状**: 编译错误 `static_assert "can not DeQue tensor in place while tque's depth is non zero"`。

**错误模式**:
```cpp
constexpr int32_t BUFFER_NUM = 1;
// ...
inLocal_ = inQueue_.DeQue<dataType>(inLocal_);  // ❌ depth≠0 时 in-place 不合法
```

**修复**: BUFFER_NUM=1 时使用无参 DeQue。
```cpp
inLocal_ = inQueue_.DeQue<dataType>();  // ✅ 无参形式
```

**原理**: `DeQue<T>(tensor)` 是 in-place dequeue（将新数据放入已有 tensor），要求 queue depth=0（队列为空，无旧数据）。`DeQue<T>()` 是无参形式（返回队列中的 tensor），要求 queue depth>0。

---

## AP-5: ceil_mode + count_include_pad 只用固定 divisor

**症状**: ceil_mode=True 场景下右边界输出值与 PyTorch 不匹配。

**错误模式**:
```cpp
// 假设所有位置的 divisor 都是 KD*KH*KW
const float invDiv = 1.0f / static_cast<float>(KD_ * KH_ * KW_);
AscendC::Muls(outFp32_, acc_, invDiv, OW_ * C_);
```

**修复**: ceil_mode=True 时逐位置计算 padded divisor（见 [precision-patterns.md](precision-patterns.md)）。

---

## AP-6: 缺少对齐守卫

**症状**: 运行时莫名崩溃（"UB address not aligned"）或输出值异常（512.0/nan），但标准测试全部通过。

**错误模式**: 标准场景的 C/W/OW 都碰巧满足对齐条件，守卫缺失只在边界 case 暴露。

**修复**: Host 侧显式 TORCH_CHECK（见 [alignment-guards.md](alignment-guards.md)）。

---

## AP-7: TileLang 性能迭代无上限分析即放弃

**症状**: TileLang 层几轮 p_retry 后直接认为"无法达标"并跳过。

**正确模式**: 
1. 3 轮 p_retry 用完后必须产出 `final_report.md`（上限分析）
2. 分析必须基于证据（编译错误日志、探针测量数据），不能主观断言
3. TileLang 层最优设计结构（即使不达标）必须记录到 PERF_DESIGN.md 作为 AscendC 蓝图
4. TileLang 设计蓝图中的关键技术决策（row-granularity、NDHWC 等）必须在 AscendC 层保留

---

## AP-8: 用固定 core 数而非 Platform API

**症状**: `usedCoreNum = 24` 硬编码。

**修复**:
```cpp
int32_t aivCoreNum = static_cast<int32_t>(
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
int32_t usedCoreNum = static_cast<int32_t>(std::min<int64_t>(totalRows, aivCoreNum));
```
