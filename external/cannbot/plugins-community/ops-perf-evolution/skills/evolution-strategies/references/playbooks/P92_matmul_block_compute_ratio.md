# P92 Playbook: Matmul 基本块计算访存比优化

> 本 Playbook 为**强制流程**。采纳 P92 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P92 的核心是**优化 Matmul 基本块参数 [baseM, baseN, baseK]，最大化计算访存比 = Cube cycle / 搬运数据量**。增大 baseM/baseN 可提高计算密度，配合 MDL 大包搬运提升 MTE2 带宽利用率。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p92_locations.txt`：

```bash
# 1. SetFixSplit / baseM/baseN 设置
grep -n "SetFixSplit\|baseM\|baseN\|baseK\|FixSplit" shared/original/op_host/*_tiling.cpp shared/original/op_kernel/*.cpp > /tmp/p92_locations.txt
# 2. Matmul 配置模板
grep -n "CFG_MDL\|CFG_NORM\|CFG_NBuffer33\|MatmulConfig\|Matmul<" shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h >> /tmp/p92_locations.txt
# 3. M/N/K shape 参数（用于匹配基本块大小）
grep -n "m\s*=\|n\s*=\|k\s*=\|M\s*=\|N\s*=\|K\s*=" shared/original/op_host/*_tiling.cpp >> /tmp/p92_locations.txt
# 4. profiling 数据（若存在）
grep -rn "MTE2.*us\|mte2_time\|bandwidth\|Cube.*util" shared/original/ --include="*.csv" --include="*.txt" --include="*.json" >> /tmp/p92_locations.txt
```

**交付物**（记录到 `implementation_note.txt` "Playbook Step 1"）：
- **当前 baseM/baseN/baseK**：值 + 定义位置
- **M/N/K 范围**：算子支持的最大/最小 shape
- **当前 Matmul 配置**：CFG_NORM / CFG_MDL / 其他

## Step 2: 改造计划表（强制填写）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| baseM | `?` | `128`（或 shape 适配值） | `?_tiling.cpp:L?` |
| baseN | `?` | `256`（确保 512B 对齐） | `?_tiling.cpp:L?` |
| baseK | `?` | 由 tiling 自动推导 | — |
| Matmul 模板 | `CFG_NORM` | `CFG_MDL` | `?_kernel.cpp:L?` |
| 搬运数据量 | `?` KB | `?` KB（期望降低） | — |
| 搬出偏移对齐 | `?B` | `≥ 512B` | — |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 大 shape + 小基本块**：M/N/K 较大（如 M≥256, N≥512）但 baseM=64, baseN=64。适合 P92，增大基本块。
- **形态 β — 小 shape**：M/N/K 较小（如 M≤64 或 N≤128）。无法增大基本块，P92 不适用。
- **形态 γ — 已优化**：baseM ≥ 128 且 baseN ≥ 256。已接近最优，检查 MDL 模板是否已使能。

### 3B. 搬出对齐检查

`baseN * sizeof(T)` 必须为 512B 整数倍：
- fp16: `baseN * 2 = 512` → baseN = 256 ✓
- fp32: `baseN * 4 = 512` → baseN = 128 ✓
- bf16: `baseN * 2 = 512` → baseN = 256 ✓

### 3C. Canonical Template（形态 α）

```cpp
// === 改造前（小基本块）===
int32_t baseM = 64;
int32_t baseN = 64;   // 搬出偏移 64*2=128B，非 512B 对齐
tilingApi.SetFixSplit(baseM, baseN, -1);
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> mm;  // CFG_NORM

// === 改造后（大基本块 + MDL）===
int32_t baseM = 128;  // Cube cycle = 512, 搬运 = 48KB
int32_t baseN = 256;  // 搬出偏移 256*2=512B，对齐
tilingApi.SetFixSplit(baseM, baseN, -1);
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm;
```

## Step 4: 计算访存比验证

搬运总量公式：`totalCnt = (N/baseN)*M*K + (M/baseM)*K*N`

```cpp
// 示例：M=1024, N=2048, K=4096
// 旧: baseM=64, baseN=64
// totalCnt = (2048/64)*1024*4096 + (1024/64)*4096*2048 = 32*4M + 16*8M = 128M + 128M = 256M
// 新: baseM=128, baseN=256
// totalCnt = (2048/256)*1024*4096 + (1024/128)*4096*2048 = 8*4M + 8*8M = 32M + 64M = 96M
// 搬运减少 = (256-96)/256 = 62.5%
```

在 `implementation_note.txt` "Playbook Step 4" 中写入当前 shape 的实际计算结果。

## Step 5: 编码后自检

```bash
# 检查 1: baseM ≥ 128（若 shape 允许）
grep -c "baseM\s*=\s*128\|baseM\s*=\s*256" modified_files/op_host/*_tiling.cpp
# 期望: >= 1

# 检查 2: baseN 满足 512B 对齐
grep -c "baseN\s*=\s*256\|baseN\s*=\s*128" modified_files/op_host/*_tiling.cpp
# 期望: >= 1

# 检查 3: CFG_MDL 已使能
grep -c "CFG_MDL" modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: >= 1

# 检查 4: SetFixSplit 调用
grep -c "SetFixSplit" modified_files/op_host/*_tiling.cpp
# 期望: >= 1

# 检查 5: baseN * sizeof(T) 为 512 的倍数（人工验证）
grep "baseN" modified_files/op_host/*_tiling.cpp
# 人工计算：baseN * sizeof(dtype) % 512 == 0 ?
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 小 shape 编译失败 | 基本块 > M/N，框架无法切分。降低 baseM/baseN 到 shape 允许上限 |
| MTE2 不降反升 | 基本块太大，L1 无法容纳，导致多次搬运。适度降低 baseK 或回退 |
| 512B 对齐检查失败 | baseN 为奇数或 sizeof(T)=2 时 baseN 不是 256 的倍数。改用 128 或 256 |
| CFG_MDL 不适用 | 小 shape 场景 MDL 头开销大于收益。回退到 CFG_NORM |
| Cube 利用率不升 | 瓶颈不在 mte2_stall。检查是否 compute_bound（Cube 本身负载不足） |
| 与 P93 同时使用时 MDL + K 轴错峰交互异常 | CFG 顺序：先设置 enableKdimReorderLoad，再设置其他 MDL 参数 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P92 Playbook Completion]
Step 1: done (/tmp/p92_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical applied
Step 4: totalCnt reduced from ? to ? = ?% reduction, 512B aligned: yes/no
Step 5: all 5 grep checks passed
Step 6: no pitfalls triggered / {列出触发的}
```
