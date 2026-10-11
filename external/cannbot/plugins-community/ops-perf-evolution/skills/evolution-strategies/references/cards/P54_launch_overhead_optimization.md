---
id: P54
bottlenecks: [scalar_loading]
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P54: 头开销优化策略 (Launch Overhead Optimization)

## 核心思想
头开销定义为算子启动到开始搬运 Tensor 数据（MTE2）的耗时，这段代码主要由 Scalar 流水执行，完成计算初始化工作。优化重点在于减少 Cache miss 导致的 LD/ST 延迟，通过减少栈变量访问、合并写操作、利用 STB 加速等手段降低头开销。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：大量栈变量，频繁 LD/ST
// Cache miss 时 LD/ST 延迟可达数百 cycle
```

// === 改造后（专家模式）===
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

## 关键修改点

1. 预期收益: 减少 Cache miss 导致的 LD/ST 延迟，降低头开销

## 常见陷阱

⚠️ 需要深入理解 Scalar 流水线和 Cache 结构
⚠️ 过度优化可能降低代码可读性
⚠️ 部分优化手段需要硬件特性支持

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
