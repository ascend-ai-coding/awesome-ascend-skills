# P61 Playbook: L2 Cache 优化

> 本 Playbook 为**强制流程**。采纳 P61 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P61 的核心是**在 PageAttention 场景下，关闭双页表功能并开启 KV 的 L2 Cache，减少 GM 访问延迟，提升数据复用效率**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p61_locations.txt`：

```bash
# 1. L2 Cache 配置
grep -n "l2Cache|L2Cache|L2.*cache|cache|Cache" \
    shared/original/op_kernel/*.cpp > /tmp/p61_locations.txt
# 2. 双页表
grep -n "page.*table|pageTable|double.*page|双页表" \
    shared/original/op_kernel/*.cpp >> /tmp/p61_locations.txt
# 3. PageAttention
grep -n "PageAttention|page.*attention|paged|KV.*page" \
    shared/original/op_kernel/*.cpp >> /tmp/p61_locations.txt
# 4. 当前缓存配置
grep -n "l2CacheOffFlag|cacheOff|cache.*off|disable.*cache" \
    shared/original/op_kernel/*.cpp >> /tmp/p61_locations.txt
# 5. KV 访问
grep -n "KV|kv|key.*value|LoadData" \
    shared/original/op_kernel/*.cpp >> /tmp/p61_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **L2 Cache 配置**：当前 l2CacheOffFlag 值
- **双页表**：是否开启双页表
- **PageAttention**：算子是否为 PageAttention 场景
- **当前缓存**：cache 相关配置
- **KV 访问**：KV 数据访问模式

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| L2 Cache | `?` (开/关) | 开 | `?_kernel.cpp:L?` |
| 双页表 | `?` (开/关) | 关 | `?_kernel.cpp:L?` |
| 场景 | `?` (PageAttention/其他) | PageAttention | `?_kernel.cpp:L?` |
| KV 复用 | `?` (高/低) | 高 | `?_kernel.cpp:L?` |
| 硬件支持 | `?` (有/无) | 有 | `?_kernel.cpp:L?` |
| 性能基线 | `?` (us) | 提升 ~16% | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的缓存配置和场景，判断你的代码属于以下哪种形态：

- **形态 α — 关闭双页表 + 开启 L2 Cache（最常见）**：直接修改配置标志位。
- **形态 β — 条件化 L2 Cache**：根据输入大小动态决定是否开启 L2 Cache。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — 关闭双页表 + 开启 L2 Cache）

```cpp
// === 改造前（双页表开启，L2 Cache 关闭）===
// 基线：l2CacheOffFlag = true
// KV 数据直接访问 GM，无 L2 缓存
__aicore__ inline void ConfigNaive() {
    l2CacheOffFlag = true;  // ❌ L2 Cache 关闭
    // 每次 KV 访问都需要 GM 读取
}

// === 改造后（关闭双页表，开启 L2 Cache）===
__aicore__ inline void ConfigOptimized() {
    // Step 1: 关闭双页表
    l2CacheOffFlag = false;  // ✅ 开启 L2 Cache
    
    // Step 2: KV 数据进入 L2 Cache
    // 后续访问命中缓存，减少 GM 延迟
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（条件化 L2 Cache）**：
  根据输入大小动态开关：
  ```cpp
  if (kvSize > L2_CACHE_THRESHOLD) {
      l2CacheOffFlag = false;  // 大 KV 开启 L2 Cache
  } else {
      l2CacheOffFlag = true;   // 小 KV 关闭，避免缓存污染
  }
  ```

- **与 P52 的冲突**：P52（L2 Cache Hint 优化）与 P61 都修改 L2 Cache。P52 处理 Hint 优化，P61 处理开关配置。两者可同时存在：P61 开启 L2，P52 优化 Hint。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: 硬件必须支持 L2 Cache 功能
约束 2: 开启 L2 Cache 可能影响其他算子的缓存使用，需权衡
约束 3: 双页表关闭后，页表管理需改用其他方式
约束 4: 仅在 PageAttention 场景下有收益，其他场景可能无益
约束 5: l2CacheOffFlag = false 表示开启，= true 表示关闭（注意语义）
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `l2CacheOffFlag = ?` (改造前/后)
- `场景 = ?`, `KV 大小 = ?`
- `性能 = ?` (改造前/后)
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 l2CacheOffFlag 修改
grep -cE "l2CacheOffFlag.*false|l2CacheOffFlag.*=.*false" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 无 l2CacheOffFlag = true
grep -cE "l2CacheOffFlag.*true|l2CacheOffFlag.*=.*true" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "回退路径"）

# 检查 3: 有 PageAttention 或 KV 引用
grep -cE "PageAttention|KV|kv" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有 cache 相关配置
grep -cE "cache|Cache" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 有性能注释或基线
grep -cE "us|性能|performance|baseline" modified_files/op_kernel/*.cpp
# 期望: >= 1（或 note 中说明 "未记录"）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：l2CacheOffFlag 未定义 | 确认硬件配置结构体中包含该字段 |
| 运行时：性能下降 | L2 Cache 可能与其他算子竞争。检查缓存命中率和污染情况 |
| 硬件不支持 L2 Cache | 确认硬件版本支持。不支持则跳过 |
| 双页表关闭后出错 | 检查页表管理逻辑是否依赖双页表。需改用单页表 |
| 非 PageAttention 场景误触发 | 添加场景判断：`if (isPageAttention) { l2CacheOffFlag = false; }` |
| l2CacheOffFlag 语义混淆 | false = 开启，true = 关闭。注意与直觉相反 |
| 与 P52 冲突 | P52 和 P61 都修改 L2。P61 开关，P52 Hint。不冲突 |
| 缓存污染 | 若 KV 数据量过大，可能挤出其他有用数据。考虑条件化开启 |
| 其他算子性能下降 | L2 Cache 是全局资源。监控其他算子性能变化 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P61 Playbook Completion]
Step 1: done (/tmp/p61_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: l2CacheOffFlag=? scene=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
