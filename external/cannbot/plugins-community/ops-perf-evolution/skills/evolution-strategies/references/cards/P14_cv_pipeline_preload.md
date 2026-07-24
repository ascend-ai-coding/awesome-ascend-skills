---
id: P14
bottlenecks: [no_overlap]
op_families: [attention, cv_fusion, flash_attention, matmul]
complexity: L2
conflicts_with: []
synergizes_with: [P17, P18, P29, P38, P53, P60, P80]
has_preconditions: true
has_playbook: true
---

# P14: CV 流水预发射 (CV Pipeline Preload)

## 核心思想
Cube 核和 Vector 核交替执行不同迭代阶段的任务，用环形任务缓存交错编排，消除跨核空等。

## 代码骨架
```cpp
// === 改造前（顺序执行，全局同步）===
for (int loop = 0; loop < totalLoops; loop++) {
    if (AIC) ComputeMm1(info);   // Cube: Q×K
    SyncAll();                    // 全局等待 — 浪费！
    if (AIV) ComputeVec1(info);   // Vector: softmax
    SyncAll();
    if (AIC) ComputeMm2(info);   // Cube: S×V
    SyncAll();
    if (AIV) ComputeVec2(info);  // Vector: output
    SyncAll();
}

// === 改造后（3阶段交错，无全局同步）===
// 用 3 槽环形缓存，不同 loop 的 Cube/Vector 阶段交错
for (uint64_t loop = 0; ; loop++) {
    auto &e0 = extraInfo[loop % 3];      // 当前轮
    auto &e1 = extraInfo[(loop + 2) % 3]; // 上一轮
    auto &e2 = extraInfo[(loop + 1) % 3]; // 上两轮

    // Stage 0: Cube 预发射当前轮 MM1
    if (e0.isValid && AIC) ComputeMm1(e0);

    // Stage 1: Vector 处理上一轮 softmax + Cube 执行上一轮 MM2
    if (e1.isValid) {
        if (AIV) ComputeVec1(e1);
        if (AIC) ComputeMm2(e1);
    }

    // Stage 2: Vector 完成上两轮输出累加
    if (e2.isValid && AIV) {
        ComputeVec2(e2);
        e2.isValid = false;  // 标记完成
    }

    // 流水排空: 所有 isValid 任务完成后退出
    if (!shouldDispatch && !hasValidTask) break;
}
```

## 关键修改点
1. **引入环形任务缓存**: `extraInfo[PRELOAD_TASK_CACHE_SIZE]`，3 槽（或算子对应的槽数）
2. **重写执行循环**: 从顺序 for 改为交错执行（Stage 0/1/2）
3. **移除全局 SyncAll**: 用 `isValid` 标志管理任务生命周期
4. **增加流水排空**: 任务分发结束后继续执行直到所有阶段完成
5. **跨核同步**: 用 `CrossCoreSetFlag/WaitFlag` 替代 `SyncAll`

## 常见陷阱
- ❌ **只改 PRELOAD_TASK_CACHE_SIZE 不改循环结构** → 无流水效果
- ❌ **遗漏流水排空** → 最后几轮任务未完成，精度错误
- ❌ **isValid 标志管理错误** → 任务被重复执行或遗漏
- ❌ **未移除 SyncAll** → Cube 和 Vector 仍在空等

## 代码搜索关键词
```bash
grep -n "SyncAll\|PipeBarrier.*ALL" *.cpp *.h        # 找全局同步点
grep -n "for.*loop.*Compute\|ExecuteTask\|Process" *.cpp  # 找执行循环
grep -n "PRELOAD.*CACHE\|TASK_CACHE" *.cpp *.h      # 找任务缓存
grep -n "CrossCore.*Flag\|SetFlag\|WaitFlag" *.cpp   # 找跨核同步
grep -n "AIC\|AIV\|ASCEND_IS_AIC\|ASCEND_IS_AIV" *.cpp  # 找核类型判断
```

## 与其他策略的关系
- 通常配合 **P18 (L1 7-buffer 常驻分区)** 使用
- 需要 **workspace 双缓冲** 支持（PRELOAD_NUM=2）
