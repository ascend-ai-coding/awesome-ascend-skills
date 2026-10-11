# P93 Playbook: Matmul K 轴错峰访问内存

> 本 Playbook 为**强制流程**。采纳 P93 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P93 的核心是**在多核 Matmul 场景中，通过 `enableKdimReorderLoad = true` 使不同核从 K 方向不同偏移开始搬运，缓解多核同时访问同一 GM 地址的冲突**。仅支持 MDL 模板 + K 轴非全载 + 多核场景。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p93_locations.txt`：

```bash
# 1. CFG_MDL 使用（K 轴错峰仅支持 MDL 模板）
grep -n "CFG_MDL\|MatmulConfig\s*{" shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h > /tmp/p93_locations.txt
# 2. 多核分核逻辑
grep -n "block_dim\|blockIdx\|coreNum\|GetBlockNum\|SplitCore\|GetBlockIdx" shared/original/op_host/*_tiling.cpp shared/original/op_kernel/*.cpp >> /tmp/p93_locations.txt
# 3. K 轴大小 + 全载判断
grep -n "k\s*=\|K\s*=\|kSize\|k_axis\|kAxis\|k_split" shared/original/op_host/*_tiling.cpp >> /tmp/p93_locations.txt
# 4. MTE2 搬运效率（profiling 证据）
grep -rn "GM_to_L1\|bandwidth.*util\|MTE2.*BW\|mte2_bw" shared/original/ --include="*.csv" --include="*.json" >> /tmp/p93_locations.txt
# 5. enableKdimReorderLoad（检查是否已使能）
grep -rn "enableKdimReorderLoad" shared/original/ >> /tmp/p93_locations.txt
```

**交付物**（记录到 `implementation_note.txt` "Playbook Step 1"）：
- **MDL 模板使用**：是/否 + 配置位置
- **多核分核**：coreNum + 分核方式
- **K 轴信息**：K 大小 + 是否全载（K ≤ L1 可容纳量 → 全载，P93 不适用）
- **GM_to_L1 带宽利用率**：profiling 中的当前值

## Step 2: 改造计划表（强制填写）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Matmul 配置 | `CFG_MDL` | `CFG_MDL + enableKdimReorderLoad` | `?_kernel.cpp` |
| 核数 | `?` | 不变 | — |
| K 轴大小 | `?` | 不变 | — |
| K 轴全载？ | `yes/no` | 必须为 no | — |
| GM_to_L1 带宽利用率 | `?%` | 期望 +5~20% | — |

## Step 3: 代码改造

### 3A. 前置条件检查（全部通过才能继续）

- [ ] 使用 CFG_MDL 模板
- [ ] 多核并行（coreNum ≥ 2）
- [ ] K 轴非全载（数据无法一次搬入 L1）
- [ ] 尚未使能 enableKdimReorderLoad

**任一不满足 → 不适用 P93**，在 `implementation_note.txt` 中明确记录不适用原因。

### 3B. Canonical Template

```cpp
// === 改造前 ===
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj;

// === 改造后 ===
constexpr MatmulConfig GetMDLKDimReorderConfig() {
    auto CFG = CFG_MDL;
    CFG.enableKdimReorderLoad = true;
    return CFG;
}
constexpr static MatmulConfig MM_CFG = GetMDLKDimReorderConfig();

AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> matmulObj;
```

## Step 4: 性能验证

- 期望 MTE2 平均耗时降低 15-25%
- 期望 GM_to_L1 带宽利用率提升 5-20%
- 若无提升，确认：K 轴是否非全载？核数是否 ≥ 2？瓶颈是否在 MTE2？
- 在 `implementation_note.txt` "Playbook Step 4" 中记录 profiling 对比

## Step 5: 编码后自检

```bash
# 检查 1: enableKdimReorderLoad 已使能
grep -c "enableKdimReorderLoad\s*=\s*true" modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: >= 1

# 检查 2: CFG_MDL 仍在使用
grep -c "CFG_MDL" modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: >= 1

# 检查 3: MatmulConfig 自定义函数存在
grep -c "GetMDLKDimReorder\|MatmulConfig\s*{" modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: >= 1

# 检查 4: 多核分核逻辑保留（未误删）
grep -c "block_dim\|coreNum\|GetBlockNum" modified_files/op_host/*_tiling.cpp
# 期望: >= 1

# 检查 5: 未引入 Norm 模板（确认 K 轴错峰未被错误模板覆盖）
grep -c "CFG_NORM" modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: == 0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 编译失败：不支持的模板 | enableKdimReorderLoad 仅支持 MDL。检查是否误用了 CFG_NORM 或 CFG_NBuffer33 |
| 性能无变化 | K 轴全载（数据一次搬入）。P93 不适用，移除 enableKdimReorderLoad |
| 性能退化 | 单核场景（coreNum=1）。K 轴错峰无意义，移除 |
| 精度退化 | K 轴错峰不改变计算逻辑。若精度变了，检查其他修改 |
| 与 P92 同时使能时异常 | 先设置 enableKdimReorderLoad，再用该 config 作为 baseM/baseN 优化的基础 |
| 小 K 轴场景无效 | K ≤ 4096 时冲突不严重。可跳过 P93 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P93 Playbook Completion]
Step 1: done (/tmp/p93_locations.txt written)
Step 2: plan table filled
Step 3: all 4 preconditions met: yes/no (if no, reason)
Step 4: GM_to_L1 BW: before=?%, after=?%
Step 5: all 5 grep checks passed
Step 6: no pitfalls triggered / {列出触发的}
```
