# P10 Playbook: 向量化数据拷贝与合并短搬运

> **强制流程**。P10 把"多次小粒度 DMA"或"CompareScalar + 手动循环"替换为"一次向量化搬运 / Vector 指令链"，减少 MTE2 / MTE3 短搬运占比与 scalar 循环开销。核心思想：**让硬件以 Vector 并行粒度工作，而不是以 Scalar 串行粒度工作**。**必须**按本 Playbook 逐步执行。

## Step 1: 定位向量化机会

执行下面的 grep，把结果写入 `/tmp/p10_locations.txt`：

```bash
# 1.1 短搬运（undersize_transfer）证据：blockLen < 32B 的 DataCopy
grep -nE "DataCopyPad|DataCopyExtParams" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null > /tmp/p10_locations.txt

# 1.2 CompareScalar + 手动循环 AND 证据（典型的 scalar-style mask 生成）
grep -nE "CompareScalar|\\bmask\\s*&=|for.*mask|\\bAnd\\b" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p10_locations.txt

# 1.3 Scalar 循环内 GM/UB 地址计算（典型 scalar bottleneck）
grep -nE "for\\s*\\([^)]*\\).*(GetPhyAddr|SetValue|GetValue)" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p10_locations.txt

# 1.4 已有向量化指令（参考基线）
grep -nE "\\b(Compare|Select|TransDataTo5HD|TransDataTo4HD|Transpose|Mul|Brcb)\\b" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p10_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **短搬运清单**：每个 `blockLen < 32B` 的 DataCopy 位置
- **标量化代码清单**：`CompareScalar` / 手动循环 / Scalar 地址计算的位置
- **可向量化的语义**：每个点对应哪个向量化替代（Compare+Select / TransDataTo5HD / Mul-as-AND / batched DataCopy）

## Step 2: 向量化改造计划表（强制）

**不填完此表不得进入 Step 3**：

| 调用点 (file:line) | 现状 | 替代手段 | 目标 API | 预计节省 |
|---|---|---|---|---|
| kernel.h:145 | 16 次 2B DataCopy | 拼接后一次 DataCopy | `DataCopy(dst, src, 16)` | 16 次 MTE3 → 1 次 |
| kernel.h:234 | `CompareScalar + for + And` | Compare + Select | `Compare + Select` 组合 | N 次 scalar → 1 次 Vector |
| kernel.h:289 | `for: dst[i] = src[i*8]` | 转置 API | `TransDataTo5HD<T>` | N 次 scalar addr → 1 次 |
| ... | ... | ... | ... | ... |

**排除规则**（下列情况**不改**）：
- **DataCopy 已经是连续大块**（blockLen ≥ 256B）：MTE 已高效，P10 无收益
- **scalar 循环被编译期展开**（constexpr loop + 小 N）：编译器已优化，手动向量化可能反而增加指令数
- **需要非对齐 stride / 稀疏访问**：无现成向量化 API，强改会引入 Copy + Pad 反而变慢
- **dtype 不在 Vector 指令支持列表**：先 Cast 的开销需与节省对比

## Step 3: 改造实施

### 3A. 形态识别

判断你的算子的主导标量化形态：

- **形态 α — 多次短 DataCopy（undersize_transfer）**：MTE3 短搬运占比高，P10 合并 batched copy
- **形态 β — CompareScalar + 手动 AND**：mask 生成类标量循环，P10 换 Compare+Select+Mul
- **形态 γ — 标量转置 / 标量地址计算**：P10 换 TransDataTo5HD 或 Brcb

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template（形态 α — 最高 ROI 且最通用）

```cpp
// === 改造前（多次 2B 短搬运）===
for (int i = 0; i < 16; ++i) {
    DataCopyPad(dstGm[i * STRIDE], srcLocal[i * 2], {1, sizeof(half), 0, 0, 0});
}
// MTE3 issued 16 transactions, each 2B → underutilized bandwidth

// === 改造后：先在 UB 内拼接到 aligned buffer，然后一次 DataCopy ===
LocalTensor<half> packedUb = outputQue.template AllocTensor<half>();
// Option A: Gather + Copy（若 UB 源地址非连续）
for (int i = 0; i < 16; ++i) {
    // 用 DataCopy intra-UB 而非 for + SetValue（保持 Vector 带宽）
    DataCopy(packedUb[i * ELEM_PER_ROW], srcLocal[i * 2], ELEM_PER_ROW);
}
pipe_barrier(PIPE_V);
// 一次大块 DataCopy 写出
DataCopy(dstGm, packedUb, 16 * ELEM_PER_ROW);   // 单次连续 MTE3
outputQue.FreeTensor(packedUb);
```

**关键替换要点**：
1. **目标**：把 N 次 < 32B 的短搬运折叠成 1 次 ≥ 256B 的连续搬运
2. **拼接工具**：优先 `DataCopy` intra-UB（走 Vector 流水）；避免 `SetValue` / `GetValue`（走 Scalar）
3. **必须** `pipe_barrier(PIPE_V)` 或 `SetFlag<HardEvent::V_MTE3>` 隔离拼接与最终搬运
4. **UB 额外开销**：`16 × ELEM_PER_ROW × sizeof(dtype)` 字节，需纳入 tile 预算

### 3C. Variant Notes

**形态 β — Compare+Select 取代 CompareScalar 循环**：
```cpp
// 改造前：CompareScalar + 标量 AND
for (int i = 0; i < n; ++i) {
    mask[i] = (x[i] > lo) & (x[i] < hi);  // 标量化
}

// 改造后：两次 Compare + 一次 Mul（Mul 用 fp16 数值代替位 AND）
Compare(selectTemp1, curX, loBuf, CMPMODE::GT, calCount);   // x > lo → bitmask 1
Compare(selectTemp2, curX, hiBuf, CMPMODE::LT, calCount);   // x < hi → bitmask 2
Select(maskTemp, selectTemp1, oneBuf, zeroBuf, ...);        // bitmask → fp16
Select(maskTemp2, selectTemp2, oneBuf, zeroBuf, ...);
Mul(maskTemp, maskTemp, maskTemp2, calCount);               // AND = 乘法
```

**形态 γ — TransDataTo5HD 取代标量转置**：
```cpp
// 改造前：标量循环转置（scalar_time 高）
for (int r = 0; r < rowNum; ++r)
    for (int c = 0; c < colNum; ++c)
        dstUb[c * rowNum + r] = srcUb[r * colNum + c];

// 改造后：TransDataTo5HD Vector 转置 API
uint64_t srcAddrList[TRANS_ADDR_LEN], dstAddrList[TRANS_ADDR_LEN];
for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
    for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
        srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
        dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i / 2 * rowNum + i % 2 * BLOCK_NUM_32].GetPhyAddr());
    }
    TransDataTo5HDParams params;
    params.repeatTimes = colNum / BLOCK_NUM_32;
    TransDataTo5HD<float>(dstAddrList, srcAddrList, params);
}
```

**与其他策略协同**：
- + **P1 双缓冲**：合并后的大块 DataCopy 是 P1 的典型受益场景，建议组合
- + **P66 GM 512B 对齐**：若合并后搬运仍 < 512B，可搭配 P66 进一步对齐
- 与 **P67** 正交：P67 管 Vector 参数，P10 管 DMA 与 Vector 的替换

## Step 4: UB + 对齐约束复核

**必须**在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P10 改造约束核查：
  1. UB 开销:
     - 新增 pack buffer 大小 = ___ bytes
     - 现有 tile 预算剩余 = ___ bytes
     - 剩余 >= 新增  ✓/✗
  2. 对齐:
     - 合并后的 DataCopy blockLen ≥ 32B（否则仍是短搬运）  ✓/✗
     - 建议 ≥ 256B 以获得峰值带宽
  3. 形态 β 特有:
     - Compare 输出类型（uint8_t bitmask）与 Select 输入一致  ✓/✗
     - Mul 代替 AND 时，fp16 的 0/1 数值精度足够（rmse << 阈值）
  4. 形态 γ 特有:
     - rowNum 必须是 TRANS_ADDR_LEN (通常 16) 的整数倍
     - colNum 必须是 BLOCK_NUM_32 (8 for fp32, 16 for fp16) 的整数倍
```

## Step 5: 编码后自检（5 条 grep，全部必须过）
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
# 检查 1: 短搬运数量减少（改造前后对比）
BEFORE=$(grep -cE "DataCopyPad.*sizeof\\([a-zA-Z0-9_]+\\)\\s*," shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
AFTER=$(grep -cE "DataCopyPad.*sizeof\\([a-zA-Z0-9_]+\\)\\s*," modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "DataCopyPad sites: before=$BEFORE after=$AFTER"
# 期望：AFTER < BEFORE，或 AFTER == BEFORE 但每次 blockLen 增大（需 Step 2 计划表佐证）

# 检查 2: SetValue / GetValue 使用减少（避免 scalar 拼接）
GETVAL=$(grep -cE "\\b(SetValue|GetValue)\\s*\\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
GETVAL_BEFORE=$(grep -cE "\\b(SetValue|GetValue)\\s*\\(" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "SetValue/GetValue: before=$GETVAL_BEFORE after=$GETVAL"
# 期望：GETVAL <= GETVAL_BEFORE（不得新增 scalar 拼接）

# 检查 3: 形态 β — 新增的 Compare+Select
grep -cE "\\bCompare\\s*\\(|\\bSelect\\s*\\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}'
# 期望：若 Step 2 选了形态 β，此值 >= 改造前 + 2（Compare 和 Select 各至少 +1）

# 检查 4: CompareScalar 使用减少（形态 β 改造证据）
CSCALAR_B=$(grep -cE "CompareScalar\\s*\\(" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
CSCALAR_A=$(grep -cE "CompareScalar\\s*\\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "CompareScalar: before=$CSCALAR_B after=$CSCALAR_A"
# 期望：若 Step 2 选了形态 β，AFTER < BEFORE；其他形态则相等

# 检查 5: pipe_barrier 在拼接→搬运之间存在
grep -nE "DataCopy\\s*\\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | head -20
# agent 人工检查：每次形态 α 合并后的大块 DataCopy 前是否有 pipe_barrier(PIPE_V) 或 SetFlag/WaitFlag；
# 在 implementation_note.txt Step 5 列出每个合并点的同步证据（file:line）
```

**在 implementation_note.txt "Playbook Step 5" 列出每条实际输出**。任一失败 → 回 Step 3 重做。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 改造后精度不过（max_abs_diff 大）| 漏了 pipe_barrier(PIPE_V)；拼接写入与最终 DataCopy 重叠；按 Step 5 check 5 补同步 |
| 形态 α 改造后性能反而变差 | 拼接用了 SetValue（scalar 写），带宽没起来；换 DataCopy intra-UB 或 Copy |
| 形态 β Compare 输出全 0 | Compare 的 CMPMODE 写反（GT 写成 LT）；或 Select 的 "true/false" buffer 初始化时没写 1/0 |
| 形态 γ TransDataTo5HD 编译失败 | `srcAddrList` / `dstAddrList` 必须是 `uint64_t[TRANS_ADDR_LEN]`，不能用 `LocalTensor` |
| UB 溢出（新增 pack buffer）| Step 4 UB 预算不足；减小 tile 或用 Que 分批 |
| 某些 DataCopy 无法合并 | 跨 tile 的数据依赖使得合并需要额外同步，收益为负；保留原短搬运，不改 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P10 Playbook Completion]
Step 1: done, N vectorization opportunities identified
Step 2: refactor plan filled (M sites, form={alpha|beta|gamma})
Step 3: template applied, key replacement = {short DMA merge / Compare+Select / TransDataTo5HD}
Step 4: UB delta=__, alignment=__, dtype=__ ✓
Step 5: all 5 grep checks passed (DataCopyPad: X→Y, SetValue: X→Y, CompareScalar: X→Y)
Step 6: no pitfalls / {列出触发的}
```
