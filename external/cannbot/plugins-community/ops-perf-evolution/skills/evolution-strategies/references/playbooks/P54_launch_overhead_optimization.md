# P54 Playbook: 头开销优化策略

> 本 Playbook 为**强制流程**。采纳 P54 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P54 的核心是**减少算子启动到开始搬运 Tensor 数据之间的 Scalar 流水初始化耗时，通过减少栈变量访问、合并写操作、利用 STB 加速等手段降低头开销**。

## Step 1: 定位关键结构

```bash
grep -n "Scalar|scalar|标量|Init|初始化|head|overhead" \
    shared/original/op_kernel/*.cpp > /tmp/p54_locations.txt
grep -n "struct.*LocalVars|栈变量|stack|local.*var" \
    shared/original/op_kernel/*.cpp >> /tmp/p54_locations.txt
grep -n "class.*Kernel|成员变量|member|var1|var2" \
    shared/original/op_kernel/*.cpp >> /tmp/p54_locations.txt
grep -n "for.*nloops|loop|迭代|Process" \
    shared/original/op_kernel/*.cpp >> /tmp/p54_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前变量类型（栈/成员**：文件 + 行号
- **初始化代码段**：文件 + 行号
- **Cache miss 热点**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 变量 | `?` (栈变量) | 成员变量 | `op_kernel/*.cpp:L?` |
| 访问 | `?` (LD/ST) | 寄存器访问 | `op_kernel/*.cpp:L?` |
| 写操作 | `?` (分散) | 合并 | `op_kernel/*.cpp:L?` |
| STB | `?` (无) | 使能 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整优化（成员变量 + 合并写 + STB）**。
- **形态 β — 仅成员变量**：不做 STB 和合并写。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 优化前：频繁访问栈变量
struct LocalVars {
    uint32_t var1;
    uint32_t var2;
    // ... 大量栈变量
};

void Process() {
    LocalVars vars;
    for (int i = 0; i < nloops; i++) {
        vars.var1 = ...;  // 每次访问都是 LD/ST
        vars.var2 = ...;
    }
}

// 优化后：使用成员变量减少栈访问
class Kernel {
    uint32_t var1_;  // 成员变量，编译器优化为寄存器访问
    uint32_t var2_;
};
```

### 3C. Variant Notes

- 需要深入理解 Scalar 流水线和 Cache 结构。
- 过度优化可能降低代码可读性。
- 部分优化手段需要硬件特性支持。

## Step 4: 约束复核

- 代码可读性下降
- 硬件特性依赖
- 收益需 profiling 验证

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "class.*Kernel|成员变量" modified_files/op_kernel/*.cpp  # >=1
grep -cE "uint32_t var1_|uint32_t var2_" modified_files/op_kernel/*.cpp  # >=1
grep -cE "struct.*LocalVars|LocalVars" modified_files/op_kernel/*.cpp  # ==0（或注释）
grep -cE "STB|StoreBuffer|合并写" modified_files/op_kernel/*.cpp  # >=0
grep -cE "for.*i.*vars\." modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 可读性下降 | 加注释说明 |
| 硬件不支持 | 条件编译 |
| 收益不明显 | profiling 验证后回退 |
| 过度优化 | 保持核心路径简洁 |

---

**完成清单**：
```
[P54 Playbook Completion]
Step 1: done (/tmp/p54_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 代码可读性下降; 硬件特性依赖; 收益需 profiling 验证: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
