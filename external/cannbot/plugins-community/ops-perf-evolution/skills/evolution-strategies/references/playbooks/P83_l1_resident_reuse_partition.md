# P83 Playbook: L1 常驻复用与多 Buffer 分区

> 本 Playbook 为**强制流程**。采纳 P83 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P83 的核心是**在 Cube 密集型算子中，将 Q/左矩阵搬入 L1 后常驻，结合多 buffer 分区（常驻区+轮转区），跨迭代复用并支持预取/消费流水**。

## Step 1: 定位关键结构

```bash
grep -n "Cube|cube|Matmul|matmul|FlashAttention|IFA|MLA" \
    shared/original/op_kernel/*.cpp > /tmp/p83_locations.txt
grep -n "Q|query|left.*matrix|CopyQ|左矩阵" \
    shared/original/op_kernel/*.cpp >> /tmp/p83_locations.txt
grep -n "L1|l1|CubeInBuffer|AllocTensor|GetBuffer" \
    shared/original/op_kernel/*.cpp >> /tmp/p83_locations.txt
grep -n "nloops|loop|迭代|callTimes" \
    shared/original/op_kernel/*.cpp >> /tmp/p83_locations.txt
grep -n "BUFFER_NUM|InitBuffer|buffer.*分区" \
    shared/original/op_kernel/*.cpp >> /tmp/p83_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Cube 密集型确认**：文件 + 行号
- **左矩阵类型**：文件 + 行号
- **L1 容量**：文件 + 行号
- **迭代次数**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 左矩阵 | `?` (Q/其他) | 常驻 L1 | `?_kernel.cpp:L?` |
| L1 容量 | `?` | 分区 | `?_kernel.cpp:L?` |
| 迭代次数 | `?` | 跨迭代复用 | `?_kernel.cpp:L?` |
| Buffer 数 | `?` (2) | 4+7 | `?_kernel.cpp:L?` |
| 常驻占比 | `?` | <20% | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 常驻区+轮转区（最常见）**。
- **形态 β — MLA 7-buffer 分区（极致）**：4 QP + 3 KV buffer。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
__aicore__ inline LocalTensor<TRANS_T> LoadData(int curRow, int curCol,
    int tileHeight, int tileWidth, int batchNum = -1) {
    LocalTensor<TRANS_T> l1;
    UserDefDataType flag = MATMUL_PARAM_VAR.dataPtr_;
    LocalTensor<TRANS_T> dst;
    if (flag.reuseLeft) {
        // 非首轮：复用 L1 已有数据
        l1 = MATMUL_MODULE(CubeInBuffer)->GetBuffer(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        ++callTimes_;
        return dst;
    } else {
        // 首轮：从 GM 加载
        l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        // DataCopy nd2nz...
    }
    ++callTimes_;
    return dst;
}
```

### 3C. Variant Notes

- 常驻区占 L1 空间，需控制 <20%。
- 7-buffer 分区约 504KB/512KB，迁移需重新计算。

## Step 4: 约束复核

- 常驻区 <20% L1
- 多 buffer 需独立索引和同步
- 代码复杂度高于双缓冲

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "reuseLeft|flag\.reuseLeft" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CubeInBuffer|GetBuffer|AllocTensor" modified_files/op_kernel/*.cpp  # >=1
grep -cE "callTimes|callTimes_" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CopyQ|LoadData.*Q|Q.*L1" modified_files/op_kernel/*.cpp  # >=1
grep -cE "for.*nloops|nloops" modified_files/op_kernel/*.cpp  # ==0（或条件复用）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| L1 溢出 | 常驻 <20%，或减小 tile |
| 索引冲突 | 常驻区与轮转区独立索引 |
| 同步遗漏 | 事件同步保证一致性 |
| 7-buffer 迁移 | 重新计算切分 |

---

**完成清单**：
```
[P83 Playbook Completion]
Step 1: done (/tmp/p83_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 常驻区 <20% L1; 多 buffer 需独立索引和同步; 代码复杂度高于双缓冲: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
