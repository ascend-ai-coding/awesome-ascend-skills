# Softmax — Tiling 实践案例

> 以 Softmax 为例，展示如何将算子数学流程映射到 [通用五模板](../../tiling-flow.md)。
> Agent 推导 LayerNorm、RMSNorm 等算子时，可复用相同结构。

---

## 1. 算子概述

Softmax 沿 R 轴计算：

```
output[i] = exp(input[i] - max(R)) / sum(exp(input[R] - max(R)))
```

等价于四步流水：

| 步骤 | 操作 | 沿 R 轴类型 |
|------|------|-----------|
| ① | ReduceMax → max | 归约 |
| ② | Sub + Exp → exp_shifted | 逐元素 |
| ③ | ReduceSum → sum | 归约 |
| ④ | Div → output | 逐元素 |

**关键特征**：步骤 ②③ 需要步骤 ① 的结果（max），步骤 ④ 需要步骤 ③ 的结果（sum）。
Recompute 模式下 R 无法一次载入，需**分轮重读原数据**。

---

## 2. 形状准备

Softmax 的归约轴为 R，合轴后得到 `(A1, R, A0)`：

| 原始 shape | axes | 合轴结果 |
|-----------|------|---------|
| `[batch, seq, dim]` axis=1 | axis=1 | A1=batch, R=seq, A0=dim |
| `[batch, seq]` axis=1 | axis=1 | A1=batch, R=seq, A0=1 → AR 族 |
| `[N]` axis=0 | axis=0 | A1=1, R=N, A0=1 → AR 族 |

---

## 3. 模板选型（Softmax 视角）

Softmax 使用通用决策树，无额外分支。以下说明各模板下 Softmax 的具体行为。

### AR 族（A0=1，如 `[batch, seq]`）

```
A0 == 1
│
├─ R 极小（≤ 约 16~32）
│     → AR-SmallR
│       转置为 (R, batch)，沿 batch 切 tile
│       每个 tile 内完成完整 Softmax 四步
│
├─ R 可整段载入 UB
│     → AR-FullLoad
│       R 行一次载入，batch 方向多核切分
│       max / exp / sum / div 均在 UB 内完成，数据只读一次
│
└─ R 超出 UB
      → AR-Recompute
        R 方向分块：每块做 max + exp + partial sum
        块间合并 max 和 sum，最后分轮重读原数据做 div
```

### ARA 族（A0>1，如 `[batch, seq, dim]`）

```
A0 > 1
│
├─ R 可整段载入 UB
│     → ARA-FullLoad
│       沿 dim(A0) 切 tile，每个 tile 内 R 行全载
│       R ≤ 8: 直接多行累加求 max/sum
│       R > 8: 用二分累加做行归约
│
└─ R 超出 UB
      → ARA-Recompute
        沿 dim(A0) 切 tile，R 按 128 行分 bin
        每个 bin 内做 partial max/sum，跨 bin 二分合并
```

---

## 4. UB 预算修正（相对标准归约）

Softmax 在通用公式基础上，UB 内同时存在多份 buffer：

### AR-FullLoad

```
每行 UB 开销 = r_align × (输入×2 + 输出×2 + FP32中间量×2)
固定预留 = 1024 + 512

rows_per_ub = (ub_size - 1536) / (r_align × (4 + 4×dtype_bytes))
rows_per_core = ceil(A1 / core_num)
rows_per_ub = min(rows_per_ub, rows_per_core)

FullLoad 条件: rows_per_ub ≥ 1
```

对比标准 ReduceSum：分母多了 FP32 中间量和双倍 buffer（输入/输出各 2 份）。

### AR-Recompute

```
可用 UB = ub_size - 2112   // max_buf(32) + sum_buf(32) + binary_cache(2048)

每元素开销 = 输入×3 + 输出×2 + FP32×1
  // 3 份输入: 原数据 + exp 结果 + 重读缓冲
  // Recompute 需分轮重读原数据计算 exp

r_chunk = floor(可用UB / 每元素开销)，32B 对齐
```

### AR-SmallR

```
r_tile_unit = 8(FP32) / 16(FP16)
r_align = ceil(R, r_tile_unit) × r_tile_unit

per_tile = r_align × (输入×2 + 输出×2 + FP32×2)
max_a1_tiles = ub_size / (64 × (per_tile + 4))
```

### ARA-FullLoad / ARA-Recompute

```
ARA-FullLoad 每 tile:
  per_tile = R × (输入×2 + FP32×2 + FP32) + 8

ARA-Recompute 每 tile:
  r_bin_size = 128
  per_tile = 128 × (输入×2 + FP32) + FP32 × (11 + cache_layers)
```

---

## 5. 典型案例

### 案例 A：`[32, 512]` FP32，axis=1

- 合轴: A1=32, R=512, A0=1 → AR 族
- R=512 超出 SmallR 阈值，检查 FullLoad 预算
- 假设 ub_size=248KB：rows_per_ub ≥ 1 → **AR-FullLoad**
- 多核: rows_per_core = ceil(32/64) = 1，used_core_num = 32

### 案例 B：`[8, 65536]` FP32，axis=1

- 合轴: A1=8, R=65536, A0=1 → AR 族
- R 远超 UB 容量 → **AR-Recompute**
- r_chunk 由可用 UB 和每元素开销决定
- 块间用二分累加合并 partial max 和 partial sum

### 案例 C：`[2, 64, 128]` FP16，axis=1

- 合轴: A1=2, R=64, A0=128 → ARA 族
- R=64 可全载 → **ARA-FullLoad**
- R > 8，启用 BinaryAdd 做行归约
- a0_tile_len 由 UB 容量和多核均衡共同决定

---

## 6. 推导 Norm 类算子的提示

LayerNorm / RMSNorm 与 Softmax 结构类似，沿 R 轴做 mean → var → normalize：

| 对比项 | Softmax | LayerNorm |
|--------|---------|-----------|
| 归约次数 | max + sum | mean + var（两个关联统计量） |
| 中间 buffer | exp 缓冲 | 去均值缓冲、方差缓冲 |
| Recompute 增强 | 重读原数据 | 建议启用 **Welford Online** |
| 模板选择 | 通用五模板 | 相同决策树，修正 UB 预算 |

**Agent 步骤**：

1. 列出 Norm 算子沿 R 轴的计算步骤
2. 数清 UB 内同时需要的 buffer 份数
3. 套用 [通用 UB 预算公式](../../tiling-flow.md#7-算子差异如何修正-ub-预算)
4. 若有两个关联统计量且走 Recompute，启用 Welford

---

## 7. 选型自检

- [ ] 输入已合轴为 (A1, R, A0)
- [ ] dtype 已确定（FP32/FP16/BF16）
- [ ] 平台参数 ub_size / core_num 已获取
- [ ] AR 族: 按 R 从小到大尝试 SmallR → FullLoad → Recompute
- [ ] ARA 族: 按 UB 预算选 FullLoad → Recompute
- [ ] Softmax Recompute 模式确认需重读原数据
- [ ] 尾块: 检查 a1_tile_tail / r_chunk_tail / a0_tile_tail
- [ ] used_core_num ≥ 1

---

## 8. 关联文档

- [通用五模板决策树](../../tiling-flow.md)
- [TilingData 字段语义](../../tiling-fields.md)
- [参考实现（简化预算）](../../script/reduction_tiling.py)
