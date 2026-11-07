# 技术卡片 vec-12：preg 溢出按条件拆分 VF（尤其 int64）

- **适用代际**：arch35（A5 / DAV_3510）独占；本卡依赖 RegBase / MicroAPI / VF，不用于 arch22 实现。
- **适用 bound**：VEC / 无 bound
- **一句话**：VF 段出现异常 preg spill/reload 时，按运行期条件把功能拆到两个 VF，降低单流同时存活寄存器数。

## 何时用（诊断特征）

- 仿真/msprof trace 里 VF 段出现**异常的 preg spill/reload**（成对出现、打断计算流），即便源码里寄存器变量个数看着没超。
- 常见于 int64：arch3510 无原生 int64 计算单元，int64 用 `RegTensor<T, RegTraitNumTwo>` 表达（一个逻辑 int64 = 2 个物理 32-bit 子寄存器 `reg[0]/reg[1]`），**实际 preg 占用 ×2**，源码寄存器数看不出已超。

## 原理

减少单个 VF（`__VEC_SCOPE__`）内**同时存活**的（双寄存器）逻辑寄存器数。按运行期条件把功能拆到两个 VF，让每条计算流的活跃寄存器更少、preg 峰值下降，spill/reload 随之消失。

## 怎么改（before / after）

```cpp
// after：选一个"能消掉某段逻辑、从而减少活跃寄存器"的拆分维度，入口按条件分流
if (dimR < VL) {          // 小 R：不需要跨块两两比较，砍掉 value1/level1 两个 int64 逻辑寄存器 = 省 4 个 preg
    VF_small(...);        // 精简版：只保留小 R 需要的寄存器
    return;
}
VF_full(...);             // 完整版
```

before 形态：单一 VF 无条件走完整路径，`value1`/`level1` 等大 R 才需要的 int64 逻辑寄存器全程占用，小 R case 也白占 preg 导致溢出。
辅助降 preg：用 `DeInterleave` 合并 int64 高低位，替代 `Mul(twoMask)+ReduceSum(twoMask)`，减少中间寄存器/掩码。

## 性能数据

msprof / 仿真 trace：**验收必须回图看 preg spill/reload 是否消失**，不能只数源码寄存器个数（int64 的 ×2 占用看不出）。**待实测**（预期：spill/reload 消失后计算流连续、`aiv_time` 下降）。

## 红线（Stop rules）

- **验收看 trace 的 spill/reload 是否消失**，不能只数源码寄存器个数。
- 拆分维度要保证两路**功能等价**；选错维度（活跃寄存器没真正减少）会两路都仍溢出。
- 拆分引入分支/代码膨胀，收益不抵成本时换维度或改用其他降 preg 手段（复用寄存器、缩短活跃区间）。

## 关联

- 源：外部 VF 策略 10（preg 溢出按条件拆分）；案例：argmax-with-value（int64 preg split by dimR）。
- 相邻卡片：vec-08（多路展开导致寄存器翻倍时用本卡降压）、vec-09（下沉引入 int64 压力）。
