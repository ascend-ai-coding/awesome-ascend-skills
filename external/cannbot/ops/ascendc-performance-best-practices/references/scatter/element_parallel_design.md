# 实现②③④：元素级并行（打包键 / 两趟胜者 / 直接原子）

> 总览与选型见 [scatter_design.md](scatter_design.md)。本文讲三条元素级并行路径，它们共享同一套线程划分与坐标分解。
> 一句话：**纤维路径饥饿时，把并行宽度从 `numFibers` 换成 `numel(indices)`，用原子或仲裁把丢掉的免原子买回来。**

## 1. 共同骨架

### 1.1 为什么切过来

纤维划分的并行宽度只有 `numFibers` 条，且线程号必须绑在最内坐标上（绑别处会拆碎读侧的合并访问）——
**想靠缩小 block 换核数是死路**（实测 `inner` 大时净收益 ≈ 0）。改成"线程 = 一个更新元素"后两个矛盾同时消失：

- 读 `indices[i]` / `updates[i]` 恒定完全合并（`i` 就是线性下标）
- 并行宽度 = `numel(indices)`，可达千万级，铺满 `coreNum × 1024` 线程

代价是丢掉纤维独占这个免费的冲突消解。三条路径就是三种买回来的方式。

### 1.2 坐标分解（三条共用）

```cpp
// i（indices 线性下标）-> (o, j, c)；outer==1 时省掉一次除法
COMP_T o, j, c;
if (cOuter == 1) { o = 0; j = i / cInner;    c = i - j * cInner; }
else             { o = i / idxPlane; const COMP_T rem = i - o * idxPlane;
                   j = rem / cInner; c = rem - j * cInner; }

COMP_T ix = static_cast<COMP_T>(indices[i]);
if (ix < 0) ix += cSVar;
if (ix < 0 || ix >= cSVar) continue;
const COMP_T dst = o * cSVar * cInner + ix * cInner + c;
```

`idxPlane = sIdx * inner`。`outer == 1` 的特判值得保留——大量真实 shape 是 `dim=0` 的二维/三维张量。

> 注意这里**没有**用 quick-div（`GetUintDivMagicAndShift` + `Simt::UintDiv`）。
> CANN 内置用了，本族实测标量优化零收益（瓶颈在访存延迟），保持直白的除法即可。

## 2. 打包键单趟仲裁

**适用**：`reduce=none` + `esize == 2`（fp16/bf16/int16）+ `sIdx + 1 ≤ 65535`。**六条路径里最优的一条。**

### 原理

把"谁最后写"和"写什么"打进同一个 32 位键，一次 `atomic_max` 同时完成仲裁与传值：

```
key = (j + 1) << 16 | bits16(value)
       ↑ 高 16 位：沿 dim 的下标，atomic_max 选出 j 最大者 = 后写覆盖
                  ↑ 低 16 位：值的原始比特（none 只搬比特，不做算术）
```

**低位为什么不会干扰胜负**：由核心不变量（[scatter_design.md §1](scatter_design.md#1-核心不变量冲突只发生在纤维内)），
同一个 `dst` 只可能来自同一条纤维，而纤维内 `j` 互不相同 ⇒ 高 16 位永远能分出胜负，**比不到低位**。

`j+1` 而不是 `j`：留出 `0` 给初值。init pass 写 `ws[i] = data[i]`（高位为 0），任何真实更新（`j+1 ≥ 1`）都能压过它。

### 三个 pass

| pass | 做什么 | 用什么 |
|---|---|---|
| init | `ws[i] = zero_extend(data[i])` | **SIMD**：MTE 搬运 + `Cast` |
| arb | `atomic_max(ws + dst, key)` | SIMT，256 线程/block |
| unpack | `y[i] = ws[i] & 0xffff` | **SIMD**：MTE + `And` + `Cast` |

init / unpack 是顺序流式 pass，**必须走 SIMD**，SIMT 标量循环慢得多。unpack 的掩码用
`Duplicate` 铺一次常量再 `And`：

```cpp
auto maskLocal = maskBuf.Get<KEY>();
AscendC::Duplicate(maskLocal, static_cast<KEY>((KEY{1} << JSHIFT) - 1), TILE);
AscendC::PipeBarrier<PIPE_V>();
/* 循环内： */
AscendC::And(inLocal, inLocal, maskLocal, static_cast<int32_t>(n));
AscendC::Cast(outLocal, inLocal, AscendC::RoundMode::CAST_NONE, n);
```

> ⚠ **unpack 这一趟省不掉。** 曾经把 `ws` 的低 16 位用 `as_strided(sizes, stride×2)` 直接当输出返回，
> 本地全绿、分数还更高，**平台上 7/20 直接崩** —— 见 [precision_and_traps.md 陷阱 1](precision_and_traps.md#3-陷阱-1输出必须是连续张量)。

### 为什么不用 64 位键

`key = (j+1) << 32 | bits32(value)` 对 4 字节值也成立，但 **64 位原子只有 32 位原子约 1/3 的吞吐**
（4.7~5.0 vs 11~18 G/s）。4 字节值改走③两趟 32 位反而更快，顺序流量还少一半。

## 3. 两趟胜者

**适用**：`reduce=none` + 键装不进 32 位（`esize` 4/8 字节、或 `sIdx` 超 16 位容量）。是 `none` 的兜底路径。

拆成两趟，每趟都只用 32 位原子：

```cpp
// 趟 1：只裁决"谁最后写"，不搬数据
asc_atomic_max(win + dst, static_cast<uint32_t>(i + 1));

// 趟 2：比中才写（胜者唯一，普通存储即可）
if (win[dst] == static_cast<uint32_t>(i + 1)) y[dst] = updates[i];
```

三个设计点：

1. **记 `i+1` 而不是 `j+1`**。对固定 `(o,c,dst)`，线性下标 `i` 随 `j` 严格递增，所以 `i` 同样能裁决最后写；
   而它额外带来一个好处：**趟 2 可以直接用 `updates[winner-1]` 取值，不必重算 `dst`、也不必再读 `indices`。**
2. **`win` 需要预置 0**，表示"无人写"，保留 `y` 里已 clone 好的 `data`。用一个 SIMT fill-zero pass。
3. **趟 2 有两种遍历方式**，按密度选：

   ```cpp
   const bool denseWinResolve = epTotal * 4 >= numelData;   // 更新足够密
   ```

   - **密**（`outer == 0` 传入作标记）：按 `dst` 遍历 `numelData` 个位置，`winner != 0` 就写。省掉坐标分解。
   - **疏**：按 `i` 遍历 `epTotal` 个更新，重算 `dst` 后比对。避免扫过大量空位置。

散射趟数 2~3（含 fill-zero），是六条路径里最贵的 `none` 实现 —— 但对 4/8 字节值它是唯一正确且够快的选择。

## 4. 直接原子（需满足数值契约的 reduce）

**适用**：`add` / `amin` / `amax` / `multiply`。

最值归约在满足 NaN、符号零等算子语义前提下通常无需裁决写入先后；浮点 `add`/`multiply` 的执行顺序会影响舍入结果。只有精度和确定性契约允许时，才直接铺满线程、把冲突交给硬件原子：

```cpp
if constexpr      (RED == RED_ADD)  asc_atomic_add(y + dst, v);
else if constexpr (RED == RED_AMIN) asc_atomic_min(y + dst, v);
else if constexpr (RED == RED_MUL)  ScatterAtomicMul<ACC_T>(y + dst, v);   // §5
else                                asc_atomic_max(y + dst, v);
```

原子比普通存储慢 2~5 倍，但**比线程饥饿划算得多** —— 前提仍然是纤维路径本来就铺不满核。

`asc_atomic_*` 家族（950 实测可用）：`add / sub / max / min / and / or / xor / cas / exch / inc / dec`。
`max`/`min` 覆盖 int32/uint32/float/int64/uint64/half/bf16；`cas` 覆盖 int32/uint32/int64/uint64/float/half2。

> `add` 这一路实测用 **256 线程/block** 比 1024 好，其余 reduce 用 1024。原因未逐条定位（推测是原子冲突下的
> 重试放大），**这是实测结论不是推导结论**，换硬件/换 CANN 版本后应重测。

### fp16/bf16 的分岔

| reduce | 走法 | 原因 |
|---|---|---|
| `amin` / `amax` | **直接用原生 half/bf16 原子** | 不产生舍入，无精度问题。省掉 workspace 的两趟 cast（实测那两趟能占某用例约一半时间） |
| `add` / `multiply` | **必须走 fp32 累加器** | 硬件的 `asc_atomic_add(__gm__ half*)` 每次操作都舍回 fp16，累加误差与软件版逐位相同，过不了精度。详见 [precision_and_traps.md](precision_and_traps.md#1-半精度的累积型-reduce-必须在-fp32-域做) |

## 5. multiply：用 CAS 循环补原子乘

硬件**没有原子乘**。有两个选择：退回纤维路径（串行但免原子），或用 `asc_atomic_cas` 循环重试。

```cpp
template <typename ACC_T>
__simt_callee__ __aicore__ inline void ScatterAtomicMul(__gm__ ACC_T* addr, ACC_T v)
{
    if constexpr (AscendC::IsSameType<ACC_T, float>::value) {
        __gm__ uint32_t* p = (__gm__ uint32_t*)addr;
        uint32_t assumed = *p, old;
        do {
            old = assumed;
            const uint32_t desired = __float_as_uint(__uint_as_float(old) * v);
            assumed = asc_atomic_cas(p, old, desired);
        } while (assumed != old);
    } else {
        ACC_T assumed = *addr, old;                 // int32 / int64 直接 CAS
        do { old = assumed; assumed = asc_atomic_cas(addr, old, static_cast<ACC_T>(old * v)); }
        while (assumed != old);
    }
}
```

> ⚠ **实测结论：不要因为"CAS 循环可能活锁"就把 multiply 踢出元素级路径。**
> 曾在 `useAtomicEP` 判据上多挂一个 `&& red != SCATTER_RED_MUL`，让 multiply 一律退回纤维路径，
> 结果某 fp32 multiply 用例（16.7M 元素）**从 3767us 劣化到 12602us**
> （"改好"这一侧另一轮测得 5051us —— runner 噪声，量级一致，结论不变）。
> 随机索引下同一输出位置平均只被更新 ~1 次，竞争很轻，循环几乎总是一次成功。
>
> 反过来也要注意：**改之前先量该用例的实际冲突度**。冲突密集（同一 `dst` 被更新几十次）时 CAS 会退化，
> 那时纤维路径才是对的。判据应该是冲突度，不是"是不是 multiply"。

half/bf16 **没有标量 CAS**（只有 `half2`），调用侧要用 `if constexpr (!ScatterIsHalfLike<ACC_T>())` 挡掉实例化，
让它们走 fp32 累加器路径。

## 6. 三条路径的成本对照

| 路径 | 散射趟数 | 顺序流量 | workspace | 原子宽度 |
|---|---|---|---|---|
| ② 打包键 | 1 | init + unpack（2× numel） | `int32[numel]` | 32 位 |
| ③ 两趟胜者 | 2 | clone + fill-zero（2× numel） | `int32[numel]` | 32 位 |
| ④ 直接原子 | 1 | clone（1× numel），半精度累加类 +2 趟 cast | 半精度累加类需 `float[numel]` | 32 位 |
| ⑤ warp 段末 | 1 | clone（1× numel） | **无** | **无** |

⑤ 在成本上全面占优，但适用面最窄（见 [warp_last_design.md](warp_last_design.md)）。

## 7. 范式代码

见 [code/scatter_simt_common.h](code/scatter_simt_common.h)：
`ScatterPackArbVF`（②）、`ScatterWinVF` + `ScatterApplyVF`（③）、`ScatterAtomicVF` + `ScatterAtomicMul`（④）。
Host 侧判据见 [code/scatter_host_routing.cpp](code/scatter_host_routing.cpp)。
