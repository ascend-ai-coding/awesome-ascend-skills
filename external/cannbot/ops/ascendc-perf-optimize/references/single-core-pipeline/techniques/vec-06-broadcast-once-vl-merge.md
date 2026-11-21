# 技术卡片 vec-06：循环不变量广播装载一次 + 多行合并进单寄存器（VL merge）

- **适用代际**：arch35（A5 / DAV_3510）独占；本卡依赖 RegBase / MicroAPI / VF，不用于 arch22 实现。
- **适用 bound**：VEC / 访存
- **一句话**：a 轴常量循环外广播 load 一次复用；单行 ≪ 一拍 lane 时把多行 packed 进一个 VL，一条指令算多行。

## 何时用（诊断特征）

- MTE2（载入）占满、循环内**反复 load 同一段 a 轴常量**（mean/rstd/gamma…）。
- 单行元素数远小于一拍 lane 数（小 A），寄存器利用率低；或归约维 R ≪ 一拍 lane、`ReduceMax/ReduceSum` 按小 mask（宽度 ≪ VL）反复发射——一拍只点亮几条 lane（lane 利用率低、VECTOR 指令数偏高）。

## 原理

- a 轴常量跨行共享 → 用 `DataCopy<..., DIST_BLK>` 广播到整个 VL，**循环外只 load 一次**，循环内复用不 reload。
- 单行太短填不满一拍 → 把多行 packed 进一个 VL（`mergeNum = VL_FP32 / aFactorAlign`），用 `UpdateMask` 在一条 vec 指令里同时算多行。
- 行长 R ≪ VL 且按行归约 → **pack = VL/R 行 gather 进一个满 lane 寄存器**一起算；行内归约用**蝶式 `DeInterleave`×log2(R) + `Max`/`Add` 组内规约、再 `Interleave` 广播回本行**，替代逐行低利用率的 `ReduceMax/ReduceSum`。

## 怎么改（before / after）

```cpp
// after：广播 helper —— DIST_BLK 把 1 个 block(8 fp32) 分发到整 VL
__aicore__ inline void LoadOneTensorBrcVL(const __ubuf__ void* in,
        Reg::RegTensor<float>& dst, uint32_t offset) {
    LoadAlign<float, Reg::LoadDist::DIST_BLK>(dst, (__ubuf__ float*)in + offset);
}
// 入口对 mean/rstd/gamma/dbeta/dgamma 各只调一次 LoadOneTensorBrcVL(addr, reg, 0)，
// 整个 R 循环不再 reload；内层按 mergeNum 行打包：loopTimes = CeilDiv(rLength, mergeNum)
```
before 形态：循环内每次迭代都 `DataCopy` 重新载入同一段常量、且每行单独发一条窄 mask 的 `ReduceMax`。

## 性能数据

msprof `aiv_time` + MTE2 泳道 / lane 利用率：广播一次消除重复 load、pack 多行提升 lane 利用率。**待实测**（预期：小 A / R≪VL 形状收益最大）。

## 红线（Stop rules）

- packed layout 要求 UB 内数据已按 `aFactorAlign_` 对齐连续摆放，装载阶段需保证该布局。
- 该策略常需**同时放宽 host tiling 的 dispatch 阈值**，让"小 A"形状真正走到这个 kernel 分支，否则改了 kernel 也接不到流量（属 tiling 联动，见 `../../tiling/`）。
- 只对"a 轴常量跨行共享"成立；逐行变化的数据不能广播复用。
- pack 多行 + 蝶式归约需 **R 为 2 的幂**（蝶式级数 = log2(R)）；数据须紧凑连续（`R*R`），buffer/CopyIn/CopyOut 须成套改；尾段用 `act=dim0*R` 收窄 mask，勿越界。

## 关联

- 源：外部 VF 策略 5（广播一次 + VL merge）；案例：batch-norm-grad 小 A add-broadcast VL merge、hc-pre 小行 pack + 蝶式归约。
- 相邻卡片：vec-05（寄存器内广播）、vec-07（归约）、vec-09（外层 for 下沉）。
