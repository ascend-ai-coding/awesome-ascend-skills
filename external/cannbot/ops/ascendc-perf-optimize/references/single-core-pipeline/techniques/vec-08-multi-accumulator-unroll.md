# 技术卡片 vec-08：归约累加链多路展开（打破 loop-carried dependency）

- **适用代际**：arch35（A5 / DAV_3510）独占；本卡依赖 RegBase / MicroAPI / VF，不用于 arch22 实现。
- **适用 bound**：VEC / 无 bound
- **一句话**：单累加器跨迭代串行的归约，改成 N 路独立累加器 + 树形合并，把依赖链长从 N 降到常数级。

## 何时用（诊断特征）

- VF 内层是「单寄存器跨迭代累加」的归约（sum / dot-product）；VECTOR 泳道因等待累加器写回出现**周期性空泡**、依赖链长，且此时访存不忙。
- 区别于寄存器压力型空泡：这里瓶颈是**归约累加器的 RAW 串行链**本身，不是寄存器/依赖过多。

## 原理

单一累加器 `acc = acc + tmp_i` 的相邻迭代是真依赖，硬件无法乱序/展开，链长 ≈ 归约次数。改成 N 路独立累加器 + 树形归约，链长从 N 降到常数级，N 个 Mul / 前几个 Add 可并行填满 VF 流水。

## 怎么改（before / after）

```cpp
// before：唯一累加器，链长 = 归约次数
for (uint16_t i = 0; i < aTimes; i++) { Mul(tmp, a_i, b_i, m); Add(acc, acc, tmp, m); }  // acc 跨 i 串行

// after：4 路独立累加，主循环步长 4（跑 aTimes/4 次，运行期）+ 运行期尾段 + 树形合并
Duplicate(acc0,0); Duplicate(acc1,0); Duplicate(acc2,0); Duplicate(acc3,0);
for (uint16_t i = 0; i + 4 <= aTimes; i += 4) {                 // 循环界是运行期变量，不是常量
    Mul(t0,a[i],b[i],m);   Mul(t1,a[i+1],b[i+1],m);
    Mul(t2,a[i+2],b[i+2],m); Mul(t3,a[i+3],b[i+3],m);           // 互不依赖
    Add(acc0,acc0,t0,m); Add(acc1,acc1,t1,m); Add(acc2,acc2,t2,m); Add(acc3,acc3,t3,m);
}
for (uint16_t r = aTimes / 4 * 4; r < aTimes; r++) {            // 运行期尾段
    Mul(t0,a[r],b[r],m); Add(acc0,acc0,t0,m);
}
Add(acc0,acc0,acc1,m); Add(acc2,acc2,acc3,m); Add(acc0,acc0,acc2,m);   // 树形合并 4 路
```

配套：4 路源数据（a/b）按 `{0,1,2,3}*stride` 一次性载入独立寄存器。

## 性能数据

msprof `aiv_time` + VECTOR 泳道空泡：多路展开填满流水后周期性空泡消失。**待实测**（预期：归约维越长、原串行链越长，收益越大；4 路是常见折中）。

## 红线（Stop rules）

- **循环次数绝不写死成常量**：VF 要对不同 shape 通用，trip count 必须由 shape/tiling 运行期推出。N 路展开 = 「常量步长 N + 运行期主循环界（`i+N<=loops`）+ 运行期尾段」，**不是**把循环展平成 N 条定长直线指令。
- 展开倍数与归约维耦合：**必须处理归约维尾段**（归约次数 % N ≠ 0 会漏算/越界），或在 tiling 侧约束整除。别把「数据维尾段」和「归约维尾段」混为一谈。
- 多路 + 树形归约**改变浮点累加次序**，与原串行结果不 bit 一致——用 fp32 中间累加缓解，仍需过 `func_cases` 精度回归。
- 展开倍数过大 → 寄存器翻倍 → 可能 spill 反劣化；寄存器压力大时配合 vec-12。

## 关联

- 源：外部 VF 策略 8（multi-accumulator unroll）；案例：hc-post A5 多路累加破归约依赖链。
- 相邻卡片：vec-07（硬件树形归约）、vec-12（preg 溢出拆分）、vec-10（小定长循环展开）。
