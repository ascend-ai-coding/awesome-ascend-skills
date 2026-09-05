# P78 Playbook: 片上缓存加速 Matmul

> 本 Playbook 为**强制流程**。采纳 P78 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P78 的核心是**利用 TSCM 或 L1 Carry 路径预加载 Matmul 输入矩阵，避免 GM→L1 搬运延迟**。

## Step 1: 定位关键结构

```bash
grep -n "Matmul|matmul|SetAType|SetBType|TPosition::GM" \
    shared/original/op_kernel/*.cpp > /tmp/p78_locations.txt
grep -n "TSCM|tscm|L1.*Carry|Carry|OnChip" \
    shared/original/op_kernel/*.cpp >> /tmp/p78_locations.txt
grep -n "headDim|M|N|K|矩阵规模" \
    shared/original/op_kernel/*.cpp >> /tmp/p78_locations.txt
grep -n "InitTscmBuffer|tscmBuf|TBuf.*TSCM" \
    shared/original/op_kernel/*.cpp >> /tmp/p78_locations.txt
grep -n "IterateAll|Iterate|LoadData" \
    shared/original/op_kernel/*.cpp >> /tmp/p78_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Matmul 输入位置**：文件 + 行号
- **矩阵规模**：文件 + 行号
- **TSCM/L1 Carry 使用情况**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 输入位置 | `?` (GM) | TSCM/L1 Carry | `?_kernel.cpp:L?` |
| 矩阵规模 | `?` | 中等/小 M 大 N | `?_kernel.cpp:L?` |
| 预加载 | `?` (无) | 有 | `?_kernel.cpp:L?` |
| 格式 | `?` (ND) | NZ | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — TSCM 路径（训练，中等规模）**。
- **形态 β — L1 Carry 路径（推理 decode，小 M 大 N）**。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// TSCM 路径
struct FaTscmArray {
    TBuf<TPosition::TSCM> tscmBuf[TSCM_BUF_NUM];
    void InitTscmBuffer(TPipe* pipe) {
        pipe->InitBuffer(tscmBuf[Q_VEC1_INDEX], tscmSize);
        pipe->InitBuffer(tscmBuf[K_V_INDEX], tscmSize);
    }
};

if (mmPolicyType == MmPolicyType::UNSPLITK) {
    mm1.SetAType(TPosition::TSCM, CubeFormat::NZ);
    mm1.SetBType(TPosition::TSCM, CubeFormat::NZ);
    mm1LoadData(tscmBuf[Q_VEC1_INDEX], qGm);
}
```

### 3C. Variant Notes

- TSCM 带宽远高于 GM→L1。
- 需额外 Nd2Nz 预处理。

## Step 4: 约束复核

- TSCM/L1 容量有限
- 大 headDim 可能放不下
- 需 Nd2Nz 预处理

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "TSCM|tscm|TPosition::TSCM" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SetAType.*TSCM|SetBType.*TSCM" modified_files/op_kernel/*.cpp  # >=1
grep -cE "InitTscmBuffer|tscmBuf" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CubeFormat::NZ|NZ" modified_files/op_kernel/*.cpp  # >=1
grep -cE "TPosition::GM|GM" modified_files/op_kernel/*.cpp  # ==0（或 note）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 容量不足 | 检查 headDim，回退 GM |
| Nd2Nz 开销 | 预加载时完成 |
| 与 GM 共存 | 条件分支 |

---

**完成清单**：
```
[P78 Playbook Completion]
Step 1: done (/tmp/p78_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: TSCM/L1 容量有限; 大 headDim 可能放不下; 需 Nd2Nz 预处理: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
