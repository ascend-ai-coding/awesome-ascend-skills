# GEMM 与 CV 融合编码规范

## 目录

- [1. GEMM 编码规范](#1-gemm-编码规范)
  - [1.1 gemm_v0 初始化](#11-gemm_v0-初始化)
  - [1.2 NPU 分形限制](#12-npu-分形限制)
- [2. CV 融合 pass_configs](#2-cv-融合-pass_configs)
- [3. FA 反向（FAG）CV 融合实例](#3-fa-反向fagcv-融合实例)

---

## 1. GEMM 编码规范

### 1.1 gemm_v0 初始化

第一次调用必须清零 C_L0：

```python
for k_chunk in T.serial(k_num):
    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k_chunk == 0))
```

### 1.2 NPU 分形限制

GEMM 的 block size 必须满足 L0A/L0B/L0C 分形限制（详见 [api-compute.md](../../../tilelang-api-best-practices/Ascend910/references/api-compute.md)）：

- int8 GEMM：`block_M ≥ 16`, `block_N ≥ 16`, `block_K ≥ 32`
- float16 GEMM：`block_M ≥ 16`, `block_N ≥ 16`, `block_K ≥ 16`

---

## 2. CV 融合 pass_configs

CV 融合算子必须开启全部 4 个 pass_configs：

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动分离 Cube/Vector
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

---

## 3. FA 反向（FAG）CV 融合实例

模式判定见 tilelang-op-design 的 `attention-patterns/references/fa_backward_pattern.md`）。

### 3.1 多 matmul 的 init 语义

FAG 主计算含 5 个 matmul（S=Q@K^T、dP=dy@V^T、dq+=dS@K、dk+=dS^T@Q、dv+=P^T@dy）。
`init` 的语义是"是否清零累加器"：

- **S/dP**：每个 (s1o, s2o) tile 独立产出，`init=True`（L0C 不复用）。
- **dq**：沿 s2o（内层 KV 循环，及 GQA 的 g）方向**跨 tile 累加**，首个 tile
  `init=True`、后续 `init=False`——即 `init=(s2o == 0)`。
- **dk/dv**：沿 s1o（外层 Q 循环）方向累加——内层循环对每个 s2o 只经过一次，
  `init=(s2o == 0)` 对 s2o>0 恒为 False，会让 L0C 复用上一 kv 块结果造成跨块
  污染。合法写法二选一：① 每 tile `init=True` 独立产出、经 workspace `+=`
  累加进 GM；② L0C 跨 s1o 累加 `init=(s1o == 0)` 且仅在最后一个 s1o 写回。
  GM 累加与 L0C 累加不可混用，否则重复计数。写错极性会丢累加或残留脏数据。
- 所有转置（K^T/V^T/dS^T/P^T）由 mma transpose 标志消化，不物化转置。

### 3.2 CV 双发与 ping-pong

- **Cube 双发（CV preload）**：C1 同一窗口连发 mm(S) 与 mm(dP)，使 Cube 搬运
  覆盖 V1（P 重建 + sfmg + dS 合成）的 Vector 时长；相邻 s2o tile 以
  SubGrapA/B 双通道 ping-pong（ring slot = s2o % 2，eventIdList 独立）。
- **CV 数据配比 1:16**（A2：Cube 128×1024 fp16 vs Vector 8×1024 fp32，
  S1 nRatio=1、S2 ratio=8）。配比是可调 tiling 参数，不是常数。

### 3.3 fp32 workspace 三段式与 host 契约

```text
Pre : AIV 清零 fp32 workspace（dq/dk/dv 累积区）
Main: CV 融合主计算，Cube 结果以 fp32 累加进 workspace
Post: workspace → 仅 dq/dk Muls(scale)，dv 只 Cast 不乘 → 按 numel 全量写回 dq/dk/dv
      （needMuls = type != TYPE_V：dv 数学上无 scale）
```

- Post 写回范围恰好铺满输出 numel（tiling 保证 blockTotal == numel）时，
  **host 侧输出用 `at::empty` 而非 `at::zeros`**——zeros 产生的 ZerosLike
  kernel 会额外占用 device 时间。fp32 dtype 例外路径：dv 由 Pre 清 dvGm +
  Main 直写，Post 跳过（对应生产 CAST_DV 条件）。
- sfmg（FlashSoftmaxGrad = rowsum(dy∘attention_in)）的 D 方向按
  sfmgd(outer/inner/tail) 切分，大 value_d 时多轮累加。

### 3.4 小 shape 结构成本（设计决策依据）

小 shape（S≤512）泛化模板瓶颈为 CV 同步/标量控制（标量管占主导、
mac 低载），非算力/带宽；核间不均严重时（尾核效应）为主要矛盾之一。
设计决策：小 shape 应按 shape 路由专用模板（见 fa_backward_pattern.md
模板路由表），而非套用泛化模板。