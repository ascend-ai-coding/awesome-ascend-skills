# P60 Playbook: KV NZ 格式优化

> 本 Playbook 为**强制流程**。采纳 P60 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P60 的核心是**将 KV 矩阵从 ND 格式改为 NZ（Channel-first）格式，使数据布局与 Cube 计算的 L0 输入格式一致，消除 MM1/MM2 阶段的 nd2nz 格式转换开销**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p60_locations.txt`：

```bash
# 1. KV 数据
grep -n "KV|kv|key.*value|KeyValue|LoadDataToL1|LoadB|L0B" \
    shared/original/op_kernel/*.cpp > /tmp/p60_locations.txt
# 2. ND 格式与转换
grep -n "ND|nd2nz|ndToNz|format.*convert|DataCopy.*nd|LoadData.*nd" \
    shared/original/op_kernel/*.cpp >> /tmp/p60_locations.txt
# 3. NZ 格式
grep -n "NZ|nzFormat|Channel.*first|channel.*first|NZ.*layout" \
    shared/original/op_kernel/*.cpp >> /tmp/p60_locations.txt
# 4. Cube 计算
grep -n "Cube|cube|MM1|MM2|Matmul|matmul|LoadA|LoadB" \
    shared/original/op_kernel/*.cpp >> /tmp/p60_locations.txt
# 5. 当前搬运路径
grep -n "DataCopy|CopyIn|CopyOut|Fixpipe" \
    shared/original/op_kernel/*.cpp >> /tmp/p60_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **KV 数据位置**：KV tensor, LoadDataToL1 等
- **ND 格式**：当前 ND 布局、nd2nz 转换代码
- **NZ 格式**：是否有 NZ 引用
- **Cube 计算**：MM1/MM2, LoadA/LoadB
- **搬运路径**：DataCopy, 格式转换步骤

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| KV 格式 | `?` (ND/NZ) | NZ | `?_kernel.cpp:L?` |
| 转换开销 | `?` (有/无) | 无 | `?_kernel.cpp:L?` |
| Cube 输入 | `?` (L0A/L0B) | 与 NZ 对齐 | `?_kernel.cpp:L?` |
| 搬运步骤 | `?` (nd2nz/连续) | 连续搬运 | `?_kernel.cpp:L?` |
| 对齐要求 | `?` (有/无) | 16 对齐 | `?_kernel.cpp:L?` |
| 性能基线 | `?` (us) | 提升 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的当前格式和 Cube 输入要求，判断你的代码属于以下哪种形态：

- **形态 α — KV 存储改为 NZ（最常见）**：上游数据直接以 NZ 格式存储，kernel 内无需 nd2nz 转换。
- **形态 β — MM 阶段格式对齐**：保持 ND 存储，但 LoadB 阶段做零开销格式对齐。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — KV 存储改为 NZ）

```cpp
// === 改造前（ND 格式，需要 nd2nz 转换）===
// ND 格式: [N, D]
// LoadB 阶段: DataCopy(nd2nz) -> L1 -> LoadB -> L0B ❌
__aicore__ inline void LoadKV_ND(LocalTensor<half> kvGm,
                                   LocalTensor<half> kvL1Tensor,
                                   uint32_t n, uint32_t d) {
    // nd2nz 转换参数
    Nd2nzParams nd2nzParams;
    nd2nzParams.srcND = {n, d};
    nd2nzParams.dstNZ = {d / 16, n / 16, 16, 16};
    
    // 格式转换搬运
    LoadDataToL1(kvL1Tensor, kvGm, nd2nzParams);  // ❌ 转换开销
    
    // 从 L1 加载到 L0B
    LoadB(l0BTensor, kvL1Tensor, ...);
}

// === 改造后（NZ 格式，直接加载）===
// NZ 格式: [N, D/16, S/16, 16, 16] — 与 Cube L0B 一致
// LoadB 阶段: DataCopy(连续) -> L1 -> LoadB -> L0B ✅
__aicore__ inline void LoadKV_NZ(LocalTensor<half> kvGmNZ,
                                   LocalTensor<half> kvL1Tensor,
                                   uint32_t n, uint32_t d) {
    // 连续搬运（无格式转换）
    DataCopy(kvL1Tensor, kvGmNZ, n * d);
    
    // 从 L1 直接加载到 L0B（格式已对齐）
    LoadB(l0BTensor, kvL1Tensor, ...);
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（MM 阶段格式对齐）**：
  若上游无法改为 NZ 存储，在 LoadB 阶段做对齐：
  ```cpp
  // ND 存储，但 LoadB 参数与 NZ 对齐
  DataCopy(kvL1Tensor, kvGm, n * d);  // 连续搬运
  // LoadB 时指定 NZ 格式的参数，硬件自动处理
  LoadB(l0BTensor, kvL1Tensor, {d / 16, n / 16, 16, 16});
  ```

- **与 P14 的协同**：P14（DataCopy 对齐优化）处理搬运对齐，P60 处理格式对齐。两者可同时存在。
- **与 P76 的边界**：P76（V_TEMPLATE KV 预合并 Workspace）处理 KV 的 workspace 合并，P60 处理 KV 的格式。两者可同时存在。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: NZ 格式的维度必须是 16 的倍数（Cube 要求）
约束 2: 上游数据必须以 NZ 格式存储，否则需要额外转换
约束 3: DataCopy 的 src 和 dst 地址必须 512B 对齐
约束 4: 非对齐场景需 padding 处理，不能直接用 NZ
约束 5: 与现有 ND 格式算子不兼容，需全链路改造
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `当前格式 = ?`, `目标格式 = ?`
- `维度 = ?`, `16 对齐 = ?` (yes/no)
- `转换步骤 = ?` (改造前/后)
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 NZ 格式引用
grep -cE "NZ|nzFormat|Channel.*first" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 无 nd2nz 转换
grep -cE "nd2nz|ndToNz|format.*convert" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "形态 beta"）

# 检查 3: 有连续 DataCopy
grep -cE "DataCopy.*kv|DataCopy.*KV" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有 LoadB 或 Cube 输入
grep -cE "LoadB|L0B|Cube|cube" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 有 KV 数据引用
grep -cE "KV|kv|KeyValue" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：NZ 格式维度不对 | 检查维度是否为 16 的倍数。NZ: [N, D/16, S/16, 16, 16] |
| 运行时：数据错乱 | 确认上游数据确实以 NZ 格式存储。若上游仍为 ND，需先转换 |
| 非对齐场景错误 | 若维度非 16 倍数，需 padding 或回退到 ND+nd2nz |
| 与 ND 算子不兼容 | NZ 格式需全链路改造。不能部分 NZ 部分 ND |
| DataCopy 未对齐 | 确认 src/dst 地址 512B 对齐。未对齐需调整 |
| 性能不如预期 | 确认 nd2nz 转换已完全消除。检查是否仍有隐式转换 |
| 内存布局错误 | NZ 的内存排布与 ND 不同。确认索引计算正确 |
| 与 P14 冲突 | P14 也修改 DataCopy。两者同时存在时，先 P60（格式），再 P14（对齐） |
| 上游改造困难 | 若上游无法改为 NZ，考虑形态 β（MM 阶段对齐） |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P60 Playbook Completion]
Step 1: done (/tmp/p60_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: format=? alignment=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
