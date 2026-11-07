# 技术卡片 vec-10：小定长循环全展开

- **适用代际**：arch35（A5 / DAV_3510）独占；本卡依赖 RegBase / MicroAPI / VF，不用于 arch22 实现。
- **适用 bound**：VEC / 无 bound
- **一句话**：VF 内编译期已知的极小循环全展开，消除循环控制开销、暴露更多指令级并行。

## 何时用（诊断特征）

- VF 内层是**次数很小且编译期已知**的循环（如卷积核 K=3、序列 2 个位置）。
- VECTOR 泳道有可被 ILP 填满的空泡，循环控制（计数/分支）占比不低。

## 原理

把小定长循环完全展开，消除循环控制开销（计数、分支判断），把各迭代的独立指令暴露给调度器，增大指令级并行。

## 怎么改（before / after）

```cpp
// before：编译期常量小循环，仍带循环控制
for (uint32_t k = 0; k < 3; ++k) {          // K=3，编译期已知
    Mul(acc, in[k], weight[k], m);
    Add(sum, sum, acc, m);
}

// after：全展开，无循环控制，指令可并行调度
Mul(a0, in[0], weight[0], m);  Mul(a1, in[1], weight[1], m);  Mul(a2, in[2], weight[2], m);
Add(sum, a0, a1, m);           Add(sum, sum, a2, m);
```

## 性能数据

msprof `aiv_time`：消除循环控制 + ILP 提升。**待实测**（预期：仅对极小定长循环有正收益；循环越大展开收益越可能被指令膨胀抵消）。

## 红线（Stop rules）

- 仅对**编译期常量、次数很小**的循环做；大循环全展开会爆指令 cache。
- 展开后注意寄存器数量是否超限（超了反而 spill，得不偿失——可配合 vec-12 拆分 / 复用寄存器）。
- 运行期 trip count（随 shape 变化）**不属**本卡适用范围，那类归约链请用 vec-08 的「常量步长 + 运行期循环界」。

## 关联

- 源：外部 VF 策略 7（小定长循环全展开）；案例：fused-causal-conv1d VF kernel unroll（K=3 / 2 seq 位置全展开）。
- 相邻卡片：vec-08（运行期归约的多路展开）、vec-09（外层 for 下沉后配合）。
