# P21 Playbook: 二维矩阵缓冲策略 (Matrix2x2BufferPolicy)

> 本 Playbook 为**强制流程**。采纳 P21 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P21 的核心是**用 2x2=4 个 buffer 组成二维矩阵，支持 M/K 双维度同时流水，最大化 Cube 单元利用率**。

## Step 1: 定位关键结构

```bash
grep -n "Matmul|matmul|Cube|MLA|nope|rope" \
    shared/original/op_kernel/*.cpp > /tmp/p21_locations.txt
grep -n "BUFFER_NUM|InitBuffer|TQue|AllocTensor|FreeTensor" \
    shared/original/op_kernel/*.cpp >> /tmp/p21_locations.txt
grep -n "切M|切K|M方向|K方向|双维度|mBaseSize|kBaseSize" \
    shared/original/op_kernel/*.cpp >> /tmp/p21_locations.txt
grep -n "pipe.*InitBuffer|pipe-.*>InitBuffer" \
    shared/original/op_kernel/*.cpp >> /tmp/p21_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 buffer 数量（1D/2D**：文件 + 行号
- **M/K 切分方式**：文件 + 行号
- **流水模式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| buffer 维度 | `?` (1D) | 2x2 矩阵 | `op_kernel/*.cpp:L?` |
| buffer 数量 | `?` (2) | 4 | `op_kernel/*.cpp:L?` |
| 索引管理 | `?` (单索引) | Alloc/Reuse/Free 三套 | `op_kernel/*.cpp:L?` |
| 遍历顺序 | `?` (行优先) | 列优先 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 2x2 Buffer Policy（M/K 双维度流水）**。
- **形态 β — 仅扩 buffer 数**：不做二维索引管理。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
template<BufferType bufferType, SyncType syncType>
class Matrix2x2BufferPolicy {
    Buffer<bufferType, syncType> bufferM0k0_, bufferM0k1_, bufferM1k0_, bufferM1k1_;
    int32_t mExtent_ = 0;
    int32_t aIdx_ = -1, uIdx_ = -1, fIdx_ = -1;
    int32_t amIdx_ = 0, akIdx_ = 0, umIdx_ = 0, ukIdx_ = 0, fmIdx_ = 0, fkIdx_ = 0;
    static constexpr int32_t kSize_ = 2;

    __aicore__ inline Buffer<bufferType, syncType> &GetBuffer(
        int32_t xIdx, int32_t &mIdx, int32_t &kIdx) {
        mIdx = (mIdx + mExtent_ - 1) % mExtent_;
        kIdx = (xIdx / mExtent_) % kSize_;
        return buffers_[mIdx * kSize_ + kIdx];
    }
    __aicore__ inline Buffer<bufferType, syncType> &AllocNext() {
        aIdx_++; return GetBuffer(aIdx_, amIdx_, akIdx_);
    }
    __aicore__ inline Buffer<bufferType, syncType> &ReuseNext() {
        uIdx_++; return GetBuffer(uIdx_, umIdx_, ukIdx_);
    }
    __aicore__ inline Buffer<bufferType, syncType> &FreeNext() {
        fIdx_++; return GetBuffer(fIdx_, fmIdx_, fkIdx_);
    }
};
```

### 3C. Variant Notes

- 4 个 buffer 占用大量片上存储。
- 列优先遍历与常规行优先直觉不同。
- 适用于 MLA headDim=576 的 nope/rope 分离等场景。

## Step 4: 约束复核

- 4 buffer 存储占用
- 三套独立索引管理复杂
- 列优先易出错

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "Matrix2x2BufferPolicy|2x2" modified_files/op_kernel/*.cpp  # >=1
grep -cE "AllocNext|ReuseNext|FreeNext" modified_files/op_kernel/*.cpp  # >=1
grep -cE "mIdx.*kIdx|buffers_\[" modified_files/op_kernel/*.cpp  # >=1
grep -cE "kSize_|mExtent_" modified_files/op_kernel/*.cpp  # >=1
grep -cE "BUFFER_NUM.*=.*2|InitBuffer.*2" modified_files/op_kernel/*.cpp  # ==0（或注释）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 存储不足 | 减 tile 或回退 1D |
| 索引混乱 | 严格列优先测试 |
| 管理复杂 | 封装为类 |
| 非 MLA | 评估收益 |

---

**完成清单**：
```
[P21 Playbook Completion]
Step 1: done (/tmp/p21_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 4 buffer 存储占用; 三套独立索引管理复杂; 列优先易出错: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
