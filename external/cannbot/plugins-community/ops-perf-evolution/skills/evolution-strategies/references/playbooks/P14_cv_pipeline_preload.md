# P14 Playbook: CV 流水预发射实操流程

> **强制流程**。P14 把 Cube+Vector 串行 `MM1 → Vec1 → MM2 → Vec2` 改造为 3 阶段交错流水，Cube 和 Vector 交替执行不同迭代的不同阶段。**架构级重构**，风险高收益也高（典型 30-50% 提速）。

## Step 1: 定位 CV 执行结构

执行下面的 grep，把结果写入 `/tmp/p14_locations.txt`：

```bash
# 1.1 Cube/Vector 分支
grep -n "ASCEND_IS_AIC\|ASCEND_IS_AIV" shared/original/op_kernel/*.cpp > /tmp/p14_locations.txt

# 1.2 主执行循环
grep -nE "for\s*\(.*loop|while.*loop|void\s+Process" shared/original/op_kernel/*.cpp >> /tmp/p14_locations.txt

# 1.3 SyncAll / CrossCore 同步
grep -n "SyncAll\|CrossCoreSetFlag\|CrossCoreWaitFlag" shared/original/op_kernel/*.cpp >> /tmp/p14_locations.txt

# 1.4 MatmulImpl 调用 / Vec 计算
grep -n "mm\.Iterate\|mm\.GetTensorC\|mm\.SetTensor\|Matmul<" shared/original/op_kernel/*.cpp >> /tmp/p14_locations.txt

# 1.5 Workspace 使用
grep -n "workspace\|Workspace\|GM_ADDR.*ws" shared/original/op_kernel/*.cpp >> /tmp/p14_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **Cube 分支**：处理什么（通常 MM1, MM2）+ 代码位置
- **Vector 分支**：处理什么（通常 Softmax, OutputUpdate）+ 代码位置
- **主循环**：文件 + 行范围
- **现有 SyncAll 数量** 和位置
- **Workspace 占用情况**：当前 workspace 大小，是否能扩展到 PRELOAD_NUM=2 的 2 倍

## Step 2: 3 阶段流水计划表（强制）

**不填完不得进入 Step 3**：

| Stage | Cube 行为 | Vector 行为 | 读 / 写的数据 |
|---|---|---|---|
| Stage 0 | 迭代 i 的 `MM1(i)`（预发射） | - | 写 workspace[i % 2].mm1_out |
| Stage 1 | 迭代 i-1 的 `MM2(i-1)` | 迭代 i-1 的 `Vec1(i-1)` | Vec1 读 mm1_out[(i-1)%2]，写 softmax_out；MM2 读 softmax_out，写 mm2_out |
| Stage 2 | - | 迭代 i-2 的 `Vec2(i-2)`（输出累加） | 读 mm2_out[(i-2)%2]，写 final_output |

**关键参数**：
- `PRELOAD_TASK_CACHE_SIZE = 3`（环形缓存槽数，覆盖 3 个连续迭代）
- `PRELOAD_NUM = 2`（workspace 双缓冲，让 Cube 和 Vector 不撞车）
- `totalIter = totalLoops + 2`（多 2 轮用于 Epilogue 排空流水）

## Step 3: 代码重构

### 3A. 形态识别

你的算子主循环是哪种：

- **形态 α — 标准 FA 2-stage**：MM1 → Softmax → MM2 → OutputUpdate（Flash Attention 类）
- **形态 β — 1-stage**：只有 MM + Vec 各一次（P14 收益有限，考虑改 P1）
- **形态 γ — N-stage（N≥3）**：超过 2 对 MM/Vec（需要更大的 cache，超出 P14 范围；联系 P87 手动 Mmad 流水）


**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

**P14 主要针对形态 α**。非 α 建议换策略。### 3B. Canonical Template（形态 α）

```cpp
// === 改造前（串行 + SyncAll）===
for (uint64_t loop = 0; loop < totalLoops; loop++) {
    RunInfo info;
    GetInfo(loop, info);
    if ASCEND_IS_AIC { ComputeMm1(info); }
    SyncAll();
    if ASCEND_IS_AIV { ComputeVec1(info); }
    SyncAll();
    if ASCEND_IS_AIC { ComputeMm2(info); }
    SyncAll();
    if ASCEND_IS_AIV { ComputeVec2(info); }
    SyncAll();
}

// === 改造后（3 阶段环形缓存 + CrossCore 细粒度同步）===
constexpr int32_t PRELOAD_TASK_CACHE_SIZE = 3;
RunInfo extraInfo[PRELOAD_TASK_CACHE_SIZE];  // 3 槽环形缓存

for (uint64_t loop = 0; loop < totalLoops + 2; loop++) {  // +2 用于 Epilogue 排空
    // 计算任务槽位
    uint64_t slotCur  = loop       % PRELOAD_TASK_CACHE_SIZE;
    uint64_t slotPrev = (loop + 2) % PRELOAD_TASK_CACHE_SIZE;  // 相当于 (loop-1)%3
    uint64_t slotPP   = (loop + 1) % PRELOAD_TASK_CACHE_SIZE;  // 相当于 (loop-2)%3

    // Stage 0: Cube 预发射当前迭代 MM1
    if (loop < totalLoops) {
        GetInfo(loop, extraInfo[slotCur]);
        extraInfo[slotCur].isValid = true;
        if ASCEND_IS_AIC { ComputeMm1(extraInfo[slotCur]); }
    }

    // Stage 1: 处理上一迭代（Vec1 + MM2）
    if (loop >= 1 && extraInfo[slotPrev].isValid) {
        if ASCEND_IS_AIV { ComputeVec1(extraInfo[slotPrev]); }
        if ASCEND_IS_AIC { ComputeMm2(extraInfo[slotPrev]); }
    }

    // Stage 2: 处理上上迭代（Vec2，输出累加）
    if (loop >= 2 && extraInfo[slotPP].isValid) {
        if ASCEND_IS_AIV { ComputeVec2(extraInfo[slotPP]); }
        extraInfo[slotPP].isValid = false;  // 标记完成，防重入
    }
}
```

### 3C. Variant Notes

- **如果 workspace 不够** 支持 `PRELOAD_NUM=2`：降级到 `PRELOAD_NUM=1`（不做 workspace 双缓冲），性能收益减半但仍有效。
- **如果算子有 atomic accumulation**（OutputUpdate 用 SetAtomicAdd）：**必须**在 Vec2 末尾调 `SetAtomicNone()` 清状态，否则下次会撞车。参考源策略文件 line 14。
- **跨核同步**：严格说 P14 需要 `CrossCoreSetFlag<PIPE_FIX>` 让 Cube 通知 Vector "MM1 好了"，以及 `CrossCoreSetFlag<PIPE_MTE3>` 让 Vector 通知 Cube "Vec1 好了"。若不加，严格说在某些 shape 下会有时序错乱。**初版可以先不加**，等精度测试失败时再补。

## Step 4: Workspace 容量复核

**公式**：
```
新 workspace = 原 workspace × PRELOAD_NUM (=2) × coreNum
```

若 `新 workspace > 可用 workspace`，走 3C 降级到 PRELOAD_NUM=1。在 `implementation_note.txt` "Playbook Step 4" 报告具体数值。

## Step 5: 编码后自检（6 条 grep，全部必须过）
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
# 检查 1: PRELOAD_TASK_CACHE_SIZE 已声明为 3（或至少 >=2）
grep -cE "PRELOAD_TASK_CACHE_SIZE\s*=\s*[2-9]" modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: >= 1

# 检查 2: 3 槽环形缓存（extraInfo[slot] 使用 % 取模索引）
grep -cE "%\s*PRELOAD_TASK_CACHE_SIZE|%\s*3\b" modified_files/op_kernel/*.cpp
# 期望: >= 2（至少 slotCur 和 slotPrev 用了取模）

# 检查 3: 主循环上限已 +2（Epilogue 排空）
grep -cE "loop\s*<\s*totalLoops\s*\+\s*[12]|for.*loops\s*\+\s*2" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 原粗粒度 SyncAll 已被大幅减少 / 替换
grep -c "SyncAll()" modified_files/op_kernel/*.cpp
# 期望: 比原始 SyncAll 数至少少 2/3

# 检查 5: Stage 条件判断（loop >= 1 / loop >= 2 / loop < totalLoops）
grep -cE "loop\s*>=?\s*[12]|loop\s*<\s*totalLoops" modified_files/op_kernel/*.cpp
# 期望: >= 3（3 个 stage 各 1 条）

# 检查 6: isValid 标志管理
grep -cE "isValid\s*=\s*(true|false)" modified_files/op_kernel/*.cpp
# 期望: >= 2（至少一次 = true 一次 = false）
```

**在 implementation_note.txt "Playbook Step 5" 列出每条实际输出**。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 死锁 / 超时 | 检查 `isValid` 标志：Stage 2 完成后必须置 false；Stage 0 初始化时必须置 true |
| Workspace 地址冲突 | `PRELOAD_NUM=2` 时，每核的 workspace offset 必须按 core_idx × PRELOAD_NUM × buffer_size 计算，防止核间重叠 |
| 精度错（output 累加异常） | Vec2 里的 SetAtomicAdd 用完必须 SetAtomicNone 清状态；参考源策略 Variant A 结尾 |
| 首 2 轮输出错 | Stage 1 / Stage 2 的 `if (loop >= 1)` / `if (loop >= 2)` 条件必须严格 |
| 末 2 轮漏处理 | 主循环上限必须是 `totalLoops + 2`，让 Epilogue 阶段清理剩余 isValid=true 的槽 |
| 性能提升 < 20% | 检查 Cube / Vector 的实际时长：若其中一个远大于另一个，3 阶段没用（瓶颈在单一核）；考虑回退到 P1 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P14 Playbook Completion]
Step 1: done
Step 2: 3-stage pipeline plan filled
Step 3: form = alpha, canonical applied (or beta/gamma not applicable - reason: ...)
Step 4: workspace calc: 新=原×PRELOAD_NUM×coreNum ≤ 可用: yes/no (PRELOAD_NUM=2 or 1)
Step 5: all 6 grep checks passed
Step 6: no pitfalls / {列出触发的}
```
