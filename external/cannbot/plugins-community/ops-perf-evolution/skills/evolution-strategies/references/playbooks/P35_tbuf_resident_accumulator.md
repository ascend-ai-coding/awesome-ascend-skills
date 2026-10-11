# P35 Playbook: TBuf 常驻中间累加器

> 本 Playbook 为**强制流程**。采纳 P35 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P35 的核心是**使用 TBuf<VECCALC> 分配的 buffer 不参与队列管理，Init 中分配后整个 Process 生命周期内常驻，用于存储跨迭代的递推状态或梯度累加器**。

## Step 1: 定位关键结构

```bash
grep -n "TBuf|TPosition::VECCALC|VECCALC" \
    shared/original/op_kernel/*.cpp > /tmp/p35_locations.txt
grep -n "UB|ub|LocalTensor|AllocTensor|FreeTensor" \
    shared/original/op_kernel/*.cpp >> /tmp/p35_locations.txt
grep -n "累加|accumul|Add|Sum|累加器|accumulator" \
    shared/original/op_kernel/*.cpp >> /tmp/p35_locations.txt
grep -n "PipeBarrier|SyncAll|SetFlag|WaitFlag" \
    shared/original/op_kernel/*.cpp >> /tmp/p35_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前累加器实现方式（队列/TBuf**：文件 + 行号
- **同步方式**：文件 + 行号
- **跨迭代状态需求**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| buffer 类型 | `?` (TQue) | TBuf<VECCALC> | `op_kernel/*.cpp:L?` |
| 队列管理 | `?` (EnQue/DeQue) | 无（手动同步） | `op_kernel/*.cpp:L?` |
| 生命周期 | `?` (单次迭代) | 整个 Process | `op_kernel/*.cpp:L?` |
| 同步 | `?` (队列隐式) | PipeBarrier 显式 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 TBuf 累加器（跨迭代状态传递）**。
- **形态 β — 仅替换单个 buffer**：部分使用 TBuf。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
TBuf<TPosition::VECCALC> y0Buf_;
TBuf<TPosition::VECCALC> y1Buf_;
TBuf<TPosition::VECCALC> y2Buf_;

void ComputeAndUpdate() {
    LocalTensor<float> y0BufLocal = y0Buf_.Get<float>();
    LocalTensor<float> y1BufLocal = y1Buf_.Get<float>();
    Mul(y2BufLocal, x32BufLocal, weight2, curDim_);
    Add(y2BufLocal, y1BufLocal, y2BufLocal, curDim_);
    Mul(y1BufLocal, x32BufLocal, weight1, curDim_);
    Add(y1BufLocal, y0BufLocal, y1BufLocal, curDim_);
    Mul(y0BufLocal, x32BufLocal, weight0, curDim_);
}
```

### 3C. Variant Notes

- TBuf 不受队列同步保护，需程序员自行 PipeBarrier 保证一致性。
- 常驻 buffer 在整个 Process 期间占用 UB，减少可用于数据 tile 的空间。

## Step 4: 约束复核

- 手动同步责任
- UB 空间占用
- 生命周期管理

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "TBuf.*VECCALC|VECCALC" modified_files/op_kernel/*.cpp  # >=1
grep -cE "y0Buf_|y1Buf_|y2Buf_" modified_files/op_kernel/*.cpp  # >=1
grep -cE "\.Get<|Get<float>" modified_files/op_kernel/*.cpp  # >=1
grep -cE "PipeBarrier" modified_files/op_kernel/*.cpp  # >=1
grep -cE "EnQue.*y0|DeQue.*y0" modified_files/op_kernel/*.cpp  # ==0（无队列管理）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 同步遗漏 | 加 PipeBarrier |
| UB 不足 | 减 tile |
| 生命周期错 | 严格阶段边界 |
| 队列混用 | 不混用 TQue/TBuf |

---

**完成清单**：
```
[P35 Playbook Completion]
Step 1: done (/tmp/p35_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 手动同步责任; UB 空间占用; 生命周期管理: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
