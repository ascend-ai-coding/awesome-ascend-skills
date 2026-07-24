# P67 Playbook: Vector Counter 模式改造流程

> **强制流程**。P67 把 Normal 模式下的"手算 repeatTimes + 主尾块分支"替换为 Counter 模式下的"总元素数 + 一次 Vector 调用"，消除每次 tile 迭代中 ~20 次 scalar div/mod/比较，直接缩短 scalar_time。**必须**按本 Playbook 逐步执行。

## Step 1: 定位 Vector 调用与 mask 模式

执行下面的 grep，把结果写入 `/tmp/p67_locations.txt`：

```bash
# 1.1 当前 Vector API 调用（找改造对象）
grep -nE "\b(Add|Sub|Mul|Div|Cast|Exp|Log|Sqrt|Rsqrt|Abs|Max|Min|Compare|Select|Mla)\s*<?" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null > /tmp/p67_locations.txt

# 1.2 当前 mask 设置位置（判断 Normal vs Counter）
grep -nE "SetMaskNorm|SetMaskCount|SetVectorMask|ResetMask" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p67_locations.txt

# 1.3 手算 repeatTimes / tail 块的证据（P67 最关键的优化目标）
grep -nE "repeatTimes?\s*=|%\s*(ONE_REPEAT|FULL_MASK|MASK_NUM|FP_ONE_BLOCK)|tail\s*=|if\s*\(\s*tail" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p67_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **命中的 Vector 调用行号**：列出循环内每个 `Add/Mul/...` 的文件+行号
- **mask 模式现状**：当前用的是 `SetMaskNorm()`（或默认 Normal）还是已经有 `SetMaskCount()`
- **手算 repeatTimes 位置**：标记所有需要消除的 scalar 计算行

## Step 2: Counter 改造计划表（强制）

**不填完此表不得进入 Step 3**：

| 调用点 (file:line) | Vector 指令 | 当前 repeat 计算 | 当前 tail 分支 | 总元素数表达式 | 改造后模式 |
|---|---|---|---|---|---|
| kernel.h:123 | Add | `repeat = n / 64` | `if (tail) Add(...)` | `n` | COUNTER |
| kernel.h:145 | Mul | `repeat = col / 64` | 无 | `col` | COUNTER |
| ... | ... | ... | ... | ... | ... |

**排除规则**（下列情况**不改**）：
- **该 Vector 指令不支持 Counter 模式**（极少数高级归约 API 如 BlockReduceMax 仍需 Normal + 显式 repeat）。查卡片或 API 文档确认
- **数据量固定且 < ONE_REPEAT 元素**（如固定 64 fp16 = 1 repeat）：Counter 无收益，跳过
- **已有 SetMaskCount 的代码段**：已是 Counter，不重复改

## Step 3: 改造实施

### 3A. 形态识别

判断你的算子属于以下哪种主导形态：

- **形态 α — 单次无尾块**：`Add(dst, a, b, mask=FULL, repeat=N, ...)`，N 固定
- **形态 β — 主+尾双路径**（最典型）：主块循环 + `if (tail > 0) { SetVectorMask(tailMask); Add(...) }`
- **形态 γ — 多指令链**：同一元素数下多条 Vector 指令连续执行（Cast→Add→Mul）

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template（形态 β，最常见）

```cpp
// === 改造前（Normal + 手算 tail）===
const uint32_t dataSize = tile.numCol;
const uint32_t FULL_MASK = 64;  // fp32: 64, fp16: 128
uint32_t repeatTimes = dataSize / FULL_MASK;
uint32_t tailSize   = dataSize % FULL_MASK;

Add(dst, src1, src2, FULL_MASK, repeatTimes, {1,1,1,8,8,8});
if (tailSize > 0) {
    uint64_t tailMask[2] = {((uint64_t)1 << tailSize) - 1, 0};
    SetVectorMask(tailMask[1], tailMask[0]);
    Add(dst[repeatTimes * FULL_MASK], src1[repeatTimes * FULL_MASK],
        src2[repeatTimes * FULL_MASK], tailMask, 1, {1,1,1,8,8,8});
    ResetMask();
}

// === 改造后（Counter 模式）===
SetMaskCount();                                           // 进入 Counter 模式
SetVectorMask<uint32_t, MaskMode::COUNTER>(tile.numCol);  // 直接传总元素数
Add(dst, src1, src2, MASK_PLACEHOLDER, 1, {1,1,1,8,8,8}); // repeat=1, 硬件内部展开
ResetMask();                                               // 退出 Counter，恢复 Normal
```

**关键替换要点**：
1. `SetMaskCount()` 开启 Counter（作用域内所有 Vector API 都按总元素数派发）
2. `SetVectorMask<..., MaskMode::COUNTER>(total_elem)` 替代 `SetVectorMask(mask_per_repeat)`
3. Vector 调用中 mask 参数填 `MASK_PLACEHOLDER`（实际 mask 由 Counter 托管），`repeatTimes=1`
4. **结束时必须** `ResetMask()` 回到 Normal，否则后续 Vector API 行为未定义

### 3C. Variant Notes

- **形态 γ（多指令链）**：整段链外面加一对 `SetMaskCount() ... ResetMask()`，中间所有指令共享同一 Counter 设置，无需每条指令重设
- **循环内每迭代元素数不同**（rare，如 sparse tile）：Counter 设置需放入循环内，每次迭代重设 `SetVectorMask<MaskMode::COUNTER>(cur_elem)`
- **元素数 = 编译期常量**：Counter 收益退化为 0~10%，仍建议改造（代码简化）；若算子极敏感于指令数，可保留 Normal

## Step 4: 约束复核

**必须**在 `implementation_note.txt` "Playbook Step 4" 报告：

```
Counter 改造约束核查：
  总元素数上限: 每次 SetVectorMask<COUNTER> 的 N 不超过 2^32-1（当前最大 tile: ___）  ✓/✗
  SetMaskCount/ResetMask 配对: 新增 M 处 SetMaskCount，M 处 ResetMask             ✓/✗
  排除的 Vector 指令: BlockReduceMax / PairReduceMax / Brcb 等不支持 Counter 的调用仍保留 Normal ✓/✗
  作用域隔离: Counter 设置不跨函数 / 不跨 subTaskIdx 边界                           ✓/✗
```

**规则**：
- SetMaskCount 和 ResetMask **必须在同一函数内**成对出现（跨函数 / 跨子任务会污染外部状态）
- Counter 模式**不适用**的 API 列表（保留 Normal）：`BlockReduceMax/Min/Sum`, `PairReduceMax/Min/Sum`, `WholeReduceMax/Min/Sum`, `Brcb`, `Transpose`, `GatherMask/GatherScatter`
- Counter 模式下 Vector API 的 `repeatTimes` 参数**必须填 1**

## Step 5: 编码后自检（5 条 grep，全部必须过）
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
# 检查 1: 新增的 SetMaskCount 出现次数
grep -c "SetMaskCount()" modified_files/op_kernel/*.{cpp,h} 2>/dev/null
# 期望：>= 1（若 Step 2 计划表有 N 处改造，这里至少应 >= N 或 >= 1 合并形式）

# 检查 2: SetMaskCount 与 ResetMask 必须配对
PB_SETCOUNT=$(grep -c "SetMaskCount()" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s}')
PB_RESET=$(grep -c "ResetMask()" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s}')
echo "SetMaskCount=$PB_SETCOUNT, ResetMask=$PB_RESET"
# 期望：SetMaskCount == ResetMask（不等则 Step 3C 有漏配对）

# 检查 3: COUNTER mask 模式显式使用
grep -En "SetVectorMask<[^>]*MaskMode::COUNTER" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | wc -l
# 期望：>= 1

# 检查 4: 手算 repeatTimes 减少（改造前后对比）
BEFORE=$(grep -cE "repeatTimes?\s*=\s*\w+\s*/" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s}')
AFTER=$(grep -cE "repeatTimes?\s*=\s*\w+\s*/" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s}')
echo "before=$BEFORE after=$AFTER"
# 期望：AFTER < BEFORE（至少减少 1 处；若改造计划表覆盖全部则应减少 M 处）

# 检查 5: 禁止在 Counter 作用域内调用不支持的 API（严重错误会编译失败或结果错误）
grep -En "BlockReduceMax|BlockReduceMin|PairReduceMax|WholeReduceMax|Brcb|GatherMask" modified_files/op_kernel/*.{cpp,h} 2>/dev/null > /tmp/p67_check5.txt
# 检查这些调用是否都在 ResetMask 之后、下一个 SetMaskCount 之前（agent 需人工比对 /tmp/p67_check5.txt 与 SetMaskCount 分布）
# 期望：若文件包含这些调用，agent 在 implementation_note 列出每个调用的作用域（在或不在 Counter 段内），违规则回 Step 3 修正
```

**在 implementation_note.txt "Playbook Step 5" 列出每条实际输出**。任一失败 → 回 Step 3 重做。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译错 `no matching function SetVectorMask` | 模板参数错：正确是 `SetVectorMask<uint32_t, MaskMode::COUNTER>(n)`（dtype 在前，模式在后）|
| 运行结果全 0 或 NaN | 漏了 `SetMaskCount()`，Counter 模式未激活；或 Vector 调用的 repeatTimes 仍填了 > 1|
| 精度大幅飘移（max_abs_diff > 0.01）| 在 Counter 作用域内调用了不支持的 API（如 BlockReduceMax）；保留该 API 但用 ResetMask 先退回 Normal |
| 性能反而变差 | 元素数太小（< 2 × ONE_REPEAT）时 Counter 收益不明显，额外的 SetMaskCount 反而是开销；该段退回 Normal |
| 后续模块 Vector 行为异常 | 漏了 `ResetMask()`，污染了后续模块的 mask 状态 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P67 Playbook Completion]
Step 1: done, identified N Vector calls in main loop
Step 2: refactor plan filled (M call sites selected)
Step 3: form = alpha/beta/gamma, template applied
Step 4: constraints checked (setcount=N_sites, reset=N_sites, excluded_apis=list_unsupported): yes/no
Step 5: all 5 grep checks passed (before=X after=Y repeat reductions)
Step 6: no pitfalls / {列出触发的}
```
