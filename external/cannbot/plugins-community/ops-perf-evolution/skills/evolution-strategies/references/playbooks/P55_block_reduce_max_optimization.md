# P55 Playbook: BlockReduceMax 替代 DataCopy stride slice

> **强制流程**。P55 把"stride 模式 DataCopyPad 抽取元素"替换为"BlockReduceMax + 连续 DataCopy"，利用 UB 内归约消除跨 block stride 传输，显著减少 MTE3 短搬运 + scalar 地址计算。**适用场景**：Attention / Normalization 中按固定间隔（8/16）从 UB 提取 max/min/sum 并写 GM 的循环。误用会引入额外 UB 开销且收益为负，**必须**按本 Playbook 逐步执行。

## Step 1: 定位 stride slice 传输点

执行下面的 grep，把结果写入 `/tmp/p55_locations.txt`：

```bash
# 1.1 当前 DataCopyPad 的 stride 模式（最典型的 P55 改造对象）
grep -nE "DataCopyPad|DataCopyExtParams" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null > /tmp/p55_locations.txt

# 1.2 找 srcStride / dstStride 非零的 DataCopy 配置点
grep -nE "srcStride\s*=\s*\(?[^0)]|dstStride\s*=\s*\(?[^0)]" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p55_locations.txt

# 1.3 找现有的 BlockReduceMax/Min/Sum（若已有，该段大概率不需要 P55）
grep -nE "BlockReduceMax|BlockReduceMin|BlockReduceSum|WholeReduceMax" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p55_locations.txt

# 1.4 找"每 N 个元素取 1"的显式 stride slice 循环（P55 改造对象）
grep -nEB2 "for.*stride|step\s*=\s*8|step\s*=\s*16" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p55_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **命中的 stride DataCopyPad 位置**（file:line + blockCount/blockLen/srcStride）
- **归约语义推断**：该段是取 max / min / sum / mean 还是纯下采样（决定用哪种 BlockReduce 变体）
- **dtype 确认**：BlockReduceMax 仅支持 **half / float**，其他 dtype 需先 Cast

## Step 2: 改造计划表（强制）

**不填完此表不得进入 Step 3**：

| 调用点 (file:line) | 当前模式 | blockCount | blockLen | srcStride | 每块元素数 | 归约算子 | dtype | 改造后 API |
|---|---|---|---|---|---|---|---|---|
| kernel.h:234 | DataCopyPad stride | 16 | 4B | 0（32B实际）| 8 | max | float | BlockReduceMax + DataCopy |
| kernel.h:289 | DataCopyPad stride | 32 | 4B | 1 | 8 | sum | half | BlockReduceSum + DataCopy |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**排除规则**（下列情况**不改**）：
- **归约语义不是 max/min/sum**（如取中位数、采样第 k 个）：BlockReduce 无对应语义，跳过
- **每块元素数不是 {2, 4, 8, 16, 32, 64} 之一**：BlockReduceMax 的 mask 参数必须对齐到 block size，不规则间隔需先 Rearrange
- **源数据在 GM 而非 UB**：P55 依赖"UB 内归约"，GM→GM 的 stride slice 改不了
- **dtype 是 int32 / int8 / bfloat16 等**：先 Cast 到 float，改造收益要减去 Cast 开销，小块可能得不偿失

## Step 3: 改造实施

### 3A. 形态识别

判断你的算子属于以下哪种主导形态：

- **形态 α — UB 内按 8/16 对齐间隔取一（最典型）**：DataCopyPad 带 srcStride，每 `blockElementNum` 取 1 个元素
- **形态 β — 跨 tile 累积归约**：stride DataCopyPad 后还做一次 Reduce，P55 可以合并 UB 归约 + 跨 tile 归约
- **形态 γ — stride=0 但 blockLen < 32B 短搬运**：形式上不是 stride slice，但 MTE3 短搬运占比高，也可改为 UB 拼接 + 一次性连续 DataCopy（借助 P55 思路）

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template（形态 α，BlockReduceMax）

```cpp
// === 改造前（stride DataCopyPad 短搬运）===
constexpr uint32_t BLOCK_ELEM_FP32 = 8;   // 32B / 4B
LocalTensor<int32_t> nInt32Out = outputQue2.template AllocTensor<int32_t>();
DataCopyExtParams params;
params.blockCount = dealRowCount;                         // 16 次短搬运
params.blockLen   = 1 * sizeof(int32_t);                  // 4B
params.srcStride  = (BLOCK_ELEM_FP32 - 1) / BLOCK_ELEM_FP32; // 实际 = 32B stride
params.dstStride  = 0;
DataCopyPad(nUpdateGm[offset], nInt32Out, params);        // 16 次 4B 短传

// === 改造后（BlockReduceMax + 连续 DataCopy）===
constexpr uint32_t BLOCK_ELEM_FP32 = 8;
constexpr uint32_t FP32_MAX_MASK   = 64;
// dealRowCount 个 element × 每块 8 个 fp32 = dealRowCount * 8 个源元素
int32_t repeatTime   = (dealRowCount * BLOCK_ELEM_FP32) / FP32_MAX_MASK;  // 每 repeat 处理 64 elem
int32_t srcRepStride = FP32_MAX_MASK / BLOCK_ELEM_FP32;                    // 8 blocks/repeat

// UB 内按 block 归约（每 8 个 fp32 取 max）→ 连续存放 dealRowCount 个结果
BlockReduceMax<float>(nUpdateTmp, nUpdateTmp, repeatTime, FP32_MAX_MASK, 1, 1, srcRepStride);
pipe_barrier(PIPE_V);

// 后续处理保持不变（Cast/ShiftLeft 等）
Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
pipe_barrier(PIPE_V);

// 一次连续 DataCopy（dealRowCount 个 int32 = dealRowCount × 4B 连续）
DataCopy(nUpdateGm[offset], nInt32Out, dealRowCount);
outputQue2.FreeTensor(nInt32Out);
```

**关键替换要点**：
1. stride 模式的 `DataCopyPad` → UB 内 `BlockReduce{Max|Min|Sum}` → 连续 `DataCopy`
2. `repeatTime` = 总源元素数 / 每 repeat mask（fp32 用 64，fp16 用 128）
3. `srcRepStride` = 每 repeat 跨越多少个 block（block=32B）
4. BlockReduce 后**必须** `pipe_barrier(PIPE_V)`，因为后续 Cast/Shift 读 `nUpdateTmp` 是同 pipe 依赖
5. 归约后的元素数 = 原 blockCount；后续 DataCopy 的 count = dealRowCount（连续，无 stride）

### 3C. Variant Notes

- **形态 β（跨 tile 累积）**：本地 BlockReduceMax 完成 → PipeBarrier → 与累积寄存器做一次 `Max(accum, accum, local)` → 合并 P55 + 累积逻辑为单次 UB 操作
- **形态 γ（短搬运拼接）**：UB 内用 `DataCopy` 或 `Copy` 把多条短数据拼成连续段，然后一次性 DataCopy 到 GM。此变体不用 BlockReduce，但同样消除 MTE3 短搬运
- **half dtype**：`FP16_MAX_MASK = 128`，`BlockReduceMax<half>(...)`，repeat 计算相应改成 /128
- **与 P67 协同**：若后续 Cast/ShiftLeft 在 scalar_bound 路径上，P55+P67 组合可再降低 scalar_time（Counter 模式处理后续 elementwise）

## Step 4: UB + 精度双约束复核

**必须**在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P55 改造约束核查：
  1. UB 开销:
     - 改造前 DataCopyPad 仅消耗 output TQue（N 个 element 的 UB 空间）
     - 改造后 BlockReduceMax 需保留 src UB 直到归约完成，不得释放
     - 新增 UB 使用 = 0（原地归约，dst=src 允许）
  2. repeat / mask 参数:
     - repeatTime = (dealRowCount × blockElemNum) / maxMaskElem = ___
     - 必须满足 repeatTime ≥ 1 且 ≤ 255（硬件 repeat 上限）
     - 若 repeatTime > 255 → 拆成多次调用或回退 Normal
  3. dtype 合法性:
     - BlockReduceMax/Min 仅支持 half / float。当前 dtype = ___  ✓/✗
  4. 归约语义匹配:
     - 原 stride slice 的语义是 {取第 1 个 / 取 max / 取 min / 取 sum}
     - 替换成 Block{Reduce 语义} 不改变最终结果  ✓/✗
```

**特别注意**：若原代码是"按 stride 取第 1 个元素"（纯下采样，非 max），P55 的 BlockReduceMax 会改变结果（取 block 内最大值而非首元素）。此时**不适用** P55，应换 `Copy` 或 `Adds(dst, src, 0)` 配合 stride 实现下采样。

## Step 5: 编码后自检（5 条 grep，全部必须过）
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
# 检查 1: 新增的 BlockReduce 调用
grep -cE "BlockReduceMax|BlockReduceMin|BlockReduceSum" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s}'
# 期望：>= 1

# 检查 2: stride DataCopyPad 减少（改造前后对比）
BEFORE=$(grep -cE "srcStride\s*=\s*[^0)]" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s}')
AFTER=$(grep -cE "srcStride\s*=\s*[^0)]" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s}')
echo "srcStride before=$BEFORE after=$AFTER"
# 期望：AFTER < BEFORE（至少减少 1 处，对应 Step 2 计划表中改造的调用点）

# 检查 3: BlockReduce 后的 PipeBarrier（防同 pipe 数据竞争）
python3 - <<'PY'
import re, glob
for f in glob.glob('modified_files/op_kernel/*.*'):
    lines = open(f).readlines()
    for i, l in enumerate(lines):
        if re.search(r'BlockReduce(Max|Min|Sum)', l):
            next5 = ''.join(lines[i+1:i+6])
            if 'pipe_barrier' not in next5 and 'PipeBarrier' not in next5:
                print(f'{f}:{i+1} BlockReduce 后 5 行内无 pipe_barrier — 高风险')
PY
# 期望：无输出（每个 BlockReduce 后有 pipe_barrier）

# 检查 4: 改造后 DataCopy 参数应是连续模式（无 stride）
grep -cE "DataCopy\s*\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s}'
# 期望：>= 1（至少一处新增的连续 DataCopy 替代原 stride DataCopyPad）

# 检查 5: 归约 API dtype 合法（half/float）
grep -nE "BlockReduce(Max|Min|Sum)<\s*(int32_t|int8_t|uint8_t|bfloat16_t|double)" modified_files/op_kernel/*.{cpp,h} 2>/dev/null
# 期望：无输出（空）。若有输出说明用了不支持的 dtype，回 Step 3 加 Cast
```

**在 implementation_note.txt "Playbook Step 5" 列出每条实际输出**。任一失败 → 回 Step 3 重做。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 结果错（max_abs_diff 很大，数值像混叠）| 原语义是"取第 1 个"而非 max；BlockReduce 改变了下采样语义。回 Step 4 确认归约语义；若确为 stride pick，换 P55 不适用，用 `Copy` 或 `Adds(dst, src, 0, {..., stride, ...})` |
| 编译错 `BlockReduceMax no matching function` | dtype 不支持，需先 Cast：`Cast(fpBuf, int32Buf, fp32, n)` → `BlockReduceMax<float>(...)` → `Cast(int32Buf, fpBuf, int32, n/8)` |
| 精度微飘移（max_abs_diff 在阈值边缘）| BlockReduce 在同一 block 内累积顺序可能与原 DataCopyPad 预读顺序不同；对 fp16 极敏感。考虑临时 Cast 到 fp32 做 Reduce 再 Cast 回 fp16 |
| 性能反而变差 | 原 blockCount 太小（< 4），BlockReduce 的 repeat 开销占主导；退回 stride DataCopyPad，或合并多次归约为一次大归约 |
| repeatTime > 255 断言失败 | 一次归约太大，拆分：外层循环 `while (remaining > 0) { BlockReduceMax(..., min(remaining, 255), ...); remaining -= 255; }` |
| BlockReduce 与后续 Cast 之间结果不对 | 漏了 pipe_barrier(PIPE_V)，Cast 在 BlockReduce 未完成前读数据 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P55 Playbook Completion]
Step 1: done, N DataCopyPad stride sites identified
Step 2: refactor plan filled (M sites, algo={max|min|sum}, dtype={half|float})
Step 3: form = alpha/beta/gamma, template applied
Step 4: repeat=__ ≤ 255, UB delta=0, dtype=__ ✓
Step 5: all 5 grep checks passed (stride reduced from X to Y)
Step 6: no pitfalls / {列出触发的}
```
