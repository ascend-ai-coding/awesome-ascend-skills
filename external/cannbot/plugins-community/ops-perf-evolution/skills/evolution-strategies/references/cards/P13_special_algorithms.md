---
id: P13
bottlenecks: [compute_bound]
op_families: [special]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Title is vague 'Special Algorithms'. Content is actually FakeQuant DAG scheduling (very specific, not L0 universal)."
---

# P13: Special Algorithms & High-Level AscendC APIs (特殊算法与高阶API)

## 核心思想
专家实现采用了面向对象的设计模式，定义了FakeQuantAffineCachemaskBase基类和FakeQuantAffineCachemaskFp32/FakeQuantAffineCachemaskFp16派生类。基类封装了通用的成员变量（如headNum, calcLength, tileLength等）和通用方法（如CommonCopyIn, CommonCopyOut, CommonBufferGet），实现了代码的高度复用。派生类则专注于特定数据类型的计算逻辑实现。这种架构的优势在于：1) 减少了代码冗余，提高了维护性；2) 通过模板参数yType实现了编译期多态，零运行时开销；3) 便于后续扩展新的数据类型（如BF16）。

## 代码骨架

// === 改造前（基线）===
```cpp
// lingxi-code手动队列管理
__aicore__ inline void CopyIn(uint32_t idx) {
    AscendC::LocalTensor<float> varLocal = varQueue.AllocTensor<float>();
    AscendC::DataCopyPad(varLocal, varGm[tileStart], ...);
    varQueue.EnQue(varLocal);
}

__aicore__ inline void Compute(uint32_t idx) {
    AscendC::LocalTensor<float> varLocal = varQueue.DeQue<float>();
    AscendC::Mul(gradPowerLocal, gradLocal, gradLocal, this->tileSize);
    AscendC::Add(accumOutLocal, accumLocal, gradPowerLocal, this->tileSize);
    // ... 手动管理所有操作
}
```

// === 改造后（专家模式）===
```cpp
// 专家实现 - DAG定义
using OpGradPower = Bind<Vec::Mul<T>, OpGradCast, OpGradCast>;
using OpAccumOut = Bind<Vec::Add<T>, OpAccumCast, OpGradPower>;
using OpAccumSqrt = Bind<Vec::Sqrt<T>, OpAccumOut>;
using OpLrMulGrad = Bind<Vec::Mul<T>, OpGradCast, OpLrCast>;
using OpVarT = Bind<Vec::DivHighPrecision<T>, OpLrMulGrad, OpAccumSqrt>;
using OpVarOut = Bind<Vec::Sub<T>, OpVarCast, OpVarT>;

using Outputs = Elems<OpCopyOutVar, OpCopyOutAccum>;
using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
using OpDag = DAGSch<Outputs, void, MemCfg>;

// Kernel中执行
ElementwiseSch<schMode, ApplyAdagradDOp::ApplyAdagradDUpdateSlots<DTYPE_VAR>::OpDag> sch(...);
sch.Init(var, accum, lr, grad, var_out, accum_out);
sch.Process();
```

## 关键修改点

1. 预期收益: 自动内存访问融合、计算流水线优化、最优寄存器分配、自动向量化，显著提升性能

## 常见陷阱

⚠️ 需要学习和理解DAG抽象概念，调试相对复杂，对调度器行为需要一定信任
⚠️ 需要额外的buffer存储索引序列
⚠️ 需要更多的向量指令

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
