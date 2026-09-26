# P90 Playbook: 条件性 PipeBarrier 基于矩阵规模

> 本 Playbook 为**强制流程**。采纳 P90 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P90 的核心是**在手动 Matmul 的 L0C 累加模式中，根据 (M/16)*(N/16) 的 fractal 数量动态决定是否插入 PipeBarrier<PIPE_M>()**。小矩阵（< 10 fractal）必须同步保证正确性，大矩阵跳过同步提升性能。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p90_locations.txt`：

```bash
# 1. 手动 Mmad 调用
grep -n "Mmad\|mmad" shared/original/op_kernel/*.cpp > /tmp/p90_locations.txt
# 2. PipeBarrier<PIPE_M> 或等效同步
grep -n "PipeBarrier.*PIPE_M\|PipeBarrier<PIPE_M>" shared/original/op_kernel/*.cpp >> /tmp/p90_locations.txt
# 3. L0C 累加模式标志
grep -n "isL0CAccum\|L0CAccum\|l0cAccum\|L0C.*accum" shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h >> /tmp/p90_locations.txt
# 4. MmadParams 或 M/N 计算（fractal 数来源）
grep -n "MmadParams\|\.m\s*=\|\.n\s*=\|baseM\|baseN" shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h >> /tmp/p90_locations.txt
# 5. LoadData / L0A/L0B 搬运（确认手动 Matmul 流水线）
grep -n "LoadData\|l0a\|l0b\|l0c" shared/original/op_kernel/*.cpp >> /tmp/p90_locations.txt
```

**交付物**（记录到 `implementation_note.txt` "Playbook Step 1"）：
- **Mmad 调用列表**：每个 Mmad 的位置 + M/N 参数来源
- **PipeBarrier 列表**：每个 PIPE_M 同步的位置
- **L0C 累加标志**：isL0CAccum 的定义和使用位置
- **是否为手动 Matmul 流水线**：有 LoadData + Mmad + 手动同步 → 是; 只有 MatmulImpl → 不适用 P90

## Step 2: 改造计划表（强制填写）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Mmad 点的 M 值 | `?` | — | `?_kernel.cpp:L?` |
| Mmad 点的 N 值 | `?` | — | `?_kernel.cpp:L?` |
| 当前 fractal 数 | `(M/16)*(N/16) = ?` | — | — |
| 当前 PipeBarrier 行为 | 无条件 | 条件性（fractal < 10） | `?_kernel.cpp:L?` |
| isL0CAccum 状态 | `true/false` | 不变 | — |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 手动 Matmul + L0C 累加**：代码中存在 `SetPixelFormat(isL0CAccum=true)` 或手动设置 L0C 累加 + Mmad + PipeBarrier。适合 P90。
- **形态 β — 仅 MatmulImpl 高阶 API**：无手动 Mmad + PipeBarrier。不适用 P90，检查 P46（MatmulImpl 高阶 API）或 P87（手动 Mmad 流水线）。
- **形态 γ — 非累加模式**：isL0CAccum=false。不适用 P90（非累加模式下的 PipeBarrier 不可省略）。

### 3B. Canonical Template（形态 α）

```cpp
// === 改造前（无条件 PipeBarrier）===
void ManualMmad(const MmParam& mmParam, const MmadParams& mmadParams) {
    LoadData(l0a, l1Src, loadParams);
    Mmad(l0c, l0a, l0b, mmadParams);
    PipeBarrier<PIPE_M>();  // 无条件等待，大矩阵浪费性能
}

// === 改造后（条件性 PipeBarrier）===
void ManualMmad(const MmParam& mmParam, const MmadParams& mmadParams) {
    LoadData(l0a, l1Src, loadParams);
    Mmad(l0c, l0a, l0b, mmadParams);

    // 小矩阵（< 10 fractal）：等待 Mmad 完成
    // 大矩阵（≥ 10 fractal）：跳过，L0 搬运与 Mmad 自动重叠
    if (mmParam.isL0CAccum &&
        ((mmadParams.m / 16) * (mmadParams.n / 16) < 10)) {
        PipeBarrier<PIPE_M>();
    }
}
```

## Step 4: 阈值验证

- 阈值 10 是来自 SLI/LI 训练算子的经验值
- **A3 (910B)**：推荐阈值 10，可尝试 8-12
- **A5 (950)**：若 Cube 计算周期不同，阈值可能需要调整
- 在 `implementation_note.txt` "Playbook Step 4" 中记录当前 fractal 数及判断结果

## Step 5: 编码后自检

```bash
# 检查 1: 条件性判断存在（fractal 计算 + 比较）
grep -c "m / 16\|n / 16\|fractal\|< 10\|< 8\|< 12" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: isL0CAccum 标志检查存在
grep -c "isL0CAccum\|L0CAccum" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: Mmad 调用存在
grep -c "Mmad" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: PipeBarrier<PIPE_M> 仍在条件分支内
grep -c "PipeBarrier.*PIPE_M" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 非累加模式下 PipeBarrier 保留（未误删）
# 手动检查：确认 isL0CAccum=false 的路径中 PipeBarrier 仍在
grep -B5 "PipeBarrier.*PIPE_M" modified_files/op_kernel/*.cpp
# 人工检查条件分支
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 小矩阵数据错乱 | 阈值设太大，跳过 PipeBarrier 导致 Mmad 未完成就搬运 L0。降低阈值或恢复无条件同步 |
| 大矩阵性能无提升 | 瓶颈不在 scalar_loading，fractal ≥ 10 时已无同步开销。检查其他瓶颈 |
| 编译失败 | 缺少 `#include` 或 MmadParams 结构体未定义 m/n 字段。确认字段名正确 |
| 非累加模式误删 PipeBarrier | 确认只对 `isL0CAccum` 路径做条件性判断，非累加路径保留原同步 |
| A5 架构上阈值不匹配 | Cube 周期不同。Profiling 大矩阵 Mmad 耗时，若 > L0 搬运 2 倍，阈值可降至 5 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P90 Playbook Completion]
Step 1: done (/tmp/p90_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical applied
Step 4: fractal = (M/16)*(N/16) = ?, threshold verified
Step 5: all 5 grep checks passed
Step 6: no pitfalls triggered / {列出触发的}
```
