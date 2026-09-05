# Dequant 类算子 AscendC 开发知识库

> 适用范围：在 AscendC 上实现 **dequant 类**（dequant、dequant + activation +
> quant、dynamic quant 收尾、smooth quant、group quant 等）算子，且 reference
> 是 PyTorch CPU fp32 实现的场景。

## 1. 为什么需要这份知识库

dequant 类算子的核心难点不在算法本身，而在 **NPU vs CPU 精度档不匹配**：
NPU 端的 hw 超越函数（`Exp`/`Reciprocal`/`Sigmoid`/`Tanh`/`Sqrt`/`Rsqrt`/
`Log`/`GeluV2`/`Div`）在 910B Atlas A2 AI Vector Core 上是 **piecewise
polynomial 近似，~fp16 mantissa 精度**；而 PyTorch CPU 用 libm `expf` 等
是 ~0.5 ulp fp32 精度。

下面这些事实必须先理解清楚，再决定算子怎么写：

- **dtype 描述存储格式，不描述计算精度**：fp32 tensor 占 32 bit，但里面
  数值的 *信息内容* 取决于产生它的整条计算链。如果链上某一步只输出
  ~10 bit mantissa 的结果，下游 fp32 存储里 11–23 位就是物理噪声。
- **`.claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py` 当前的阈值表是按"输出 dtype"取的**，
  fp32 用 MERE = 2⁻¹³ / MARE = 2⁻¹²；fp16 用 2⁻¹⁰ / 2⁻⁹。这套阈值是按
  "reference 和 candidate 在同一精度档" 校准的。
- **PR #139 把 `11_DequantSwigluQuant` 的 reference 从 NPU vendor kernel
  改成 CPU fp32**——同一个 op 的 oracle 精度档跳了一档（NPU ~fp16 → CPU
  ~fp32），但阈值表没动。结果是隐藏的"事实上严苛了 16×"的状态。

## 2. 三档精度分类

按 **NPU 计算链上最差精度环节** 划分：

| 档 | 计算链特征 | NPU-vs-CPU 实测 MARE | 现有阈值表行为 |
|---|---|---|---|
| **Tier 1** | 只用 bit-exact op（Mul/Add/Sub/Cast/ShiftLeft/ReduceMax/Mins/Maxs/Duplicate/ReinterpretCast/Compare-Select） | 接近 fp32 ulp，~2⁻²² | 过 fp32 阈值 ✓ |
| **Tier 2** | 含 hw 超越函数 | 锁在 ~2⁻⁹ 到 2⁻¹⁰（fp16 量级） | fp32 阈值不可达；fp16 阈值通过 |
| **Tier 3** | 输出是 int8/int16 量化整数 | max_abs_diff = 1 LSB（量化阶梯本身的离散性） | 严格相等 fail；±1 LSB 容忍可过 |

判定方法：**机械扫 kernel 源码**，看是否调用任一 hw 超越函数。这是客观的、
不依赖人工标注的判据。

## 3. 决策流程：怎么选实现路径

```
                   输出 dtype 是 int8 / int16 ？
                          │
              ┌───── yes ──┴──── no
              │                  │
   Tier 3 (±1 LSB)        reference 是什么？
   ─────────────────       │
   用 hw 超越函数最合理     ├── NPU vendor kernel
   （量化阶梯吸收 fp16 噪声）│   ↓
                            │   两端 NPU activation 互相 cancel
                            │   用 hw 超越函数（Tier 2 阈值过得去）
                            │
                            └── CPU fp32 / libm
                                ↓
                         算子输出含 fp32 (如 dynamic quant 的 scale)？
                                │
                    ┌──── yes ──┴──── no（输出全部已被 quant 吸收）
                    │                  │
                    ▼                  ▼
              **必须走 Tier 1**     用 hw 超越函数
              **软件 fp32 路径**    （Tier 2 阈值用 fp16 标准评判）
              全部 bit-exact op
              手写 sigmoid / exp / 1/x
```

## 4. 软件 fp32 sigmoid（Tier 1 路径标准实现）

`sigmoid(x) = 1 / (1 + exp(-x))`，全部用 bit-exact 基础 op 实现。

### 4.1 算法分解

```
1) clamp x to [-50, 50]                          // sigmoid 在两端饱和到 fp32 ulp
2) software exp(-x):
     k       = round(-x · log2 e)                // Cast<fp32→int32, RINT>
     r       = -x - k · ln 2                     // r ∈ [-ln2/2, ln2/2]
     poly(r) = 7 阶 Horner: c7·r⁷ + ... + c0     // 14× Mul/Adds
     2^k     = (k + 127) << 23 reinterpret fp32  // ShiftLeft + ReinterpretCast
     exp(-x) = poly(r) · 2^k
3) Adds 1.0  →  1 + exp(-x)
4) software 1/(1+exp(-x)):
     // 通过 fp32 bit 拆出 mantissa m ∈ [1,2) 和 exponent e
     m_bits   = ((src_bits << 9) >> 9) | 0x3F800000   // (Adds 0x3F800000 等价于 OR)
     m        = reinterpret(m_bits) ∈ [1, 2)
     y0_mant  = 1.5 - 0.5 · m                          // 端点精确，最差 ~12.5%
     2^(-e)   = (254 - exp_bits) << 23 reinterpret fp32
     y0       = y0_mant · 2^(-e)
     // Newton 3 步：y ← y · (2 - src · y)
     for _ in range(3):
         y = y * (2 - src * y)
```

### 4.2 精度分析

- 7 阶 Horner exp(r) on r ∈ [-ln2/2, ln2/2]: ~2⁻²¹ relative
- 1/m 初值: ~3 bit (worst case 12.5% relative)
- Newton 收敛: 3 → 6 → 12 → 24 bit（饱和 fp32）
- 整体 sigmoid 输出: ~2⁻²⁰ relative on x ∈ [-50, 50]
- 远过 fp32 MERE 阈值 2⁻¹³

### 4.3 允许使用的 op（全部 bit-exact 或离散选择）

`Mul`、`Muls`、`Add`、`Adds`、`Sub`、`Mins`、`Maxs`、`Cast<fp32↔int32>`、
`ShiftLeft`、`ShiftRight`（uint32 逻辑 / int32 算术）、`Duplicate`、
`ReinterpretCast`、`ReduceMax`、`Abs`、`DataCopy/DataCopyPad`。

### 4.4 禁止使用的 op

`Exp`、`Log`、`Sigmoid`、`Tanh`、`Reciprocal`、`Sqrt`、`Rsqrt`、`GeluV2`、
`Div`，以及 PTA 高级激活 API。这些在 910B 上都是 piecewise polynomial
近似，**带进任何 fp32 输出路径都会让 MARE 锁死在 ~2⁻⁹ 量级**。

## 5. UB Buffer 预算

每行处理时，软件 sigmoid helper 需要 4 个 LocalTensor：
- `dst`：H fp32（输出 sigmoid 值，也作为 k_int / 2^k 的 int32 view）
- `src`：H fp32（输入，过 helper 后会被覆盖；调用端要先用完）
- `scratchA`：H fp32（capture 后保存 r）
- `scratchB`：H fp32（k_float、polynomial 累加器，最后变成 2^(-e) bits 等）

mode 0 的典型复用（实例见 `dequant_swiglu_quant_mode0_kernel.h`）：
- `dst = midLocal_`，`scratchA = swLocal_`（sw 在 sigmoid 之后才计算），
  `scratchB = reduceTmpBuf_.Get<float>()`（ReduceMax 在后面才用到）。
- `src = xF32Local_[gluOff]`：sub-tensor，调用端**必须**用 `auto` 或
  显式 `LocalTensor<float>` 命名为 lvalue 后传入（否则 ref 绑定到临时 fail）。

mode 1 in-place 调用（输入是 `linLocal_` = α·glu_c，输出也覆盖 `linLocal_`）：
- helper 第一步 `Mins(scratchA, src, 50.0f)` 已经把 src 内容捕获到
  `scratchA`，之后再写 `dst` 安全。
- `scratchA = gluLocal_`（α·glu_c 计算后 gluLocal_ 不再使用），
  `scratchB = reduceTmpBuf_.Get<float>()`。

## 6. dequant 类算子的标准骨架

按"行级并行 + 单核处理 blockM 行"的模式：

```
per row m:
  // ---- dequant ----
  load x[m, :] int32 [N] from GM → UB
  cast int32 → fp32 (CAST_NONE)
  Mul by weight_scale fp32 [N] (resident)
  Muls by activation_scale[m] (per-row scalar)

  // ---- activation (SwiGLU / GeLU / SiLU / ...) ----
  split halves   (mode 0 / standard)         OR
  host-permute even/odd → halves in Python wrapper (mode 1 / gpt-oss)
                  ┊
                  ├── 用 software fp32 sigmoid (本文档第 4 节)
                  │
                  └── 普通 fp32 算术 (Mul/Mins/Maxs/Adds 等)

  // ---- smooth quant ----
  Mul sw by quant_scale fp32 [H]

  // ---- per-row dynamic int8 quant ----
  Abs sw → tmp
  ReduceMax(absmax, tmp, work, H)
  scale = absmax / 127.0  (host-side scalar)
  if absmax == 0:
      duplicate 0 to int8 output
      scale = 0
  else:
      Muls sw by 1.0/scale
      Cast<fp32 → fp16, RINT>
      Cast<fp16 → int8, RINT>

  // ---- write outputs ----
  DataCopy y_i8 [H] to GM
  Per-row scale → 同 GM 区段（每核累加，DataCopyPad 落盘 / 或 SetValue）
```

## 7. 几个一定会踩的坑

### 7.1 同 buffer 在 Cast 时别名（in-place dtype conversion）

❌ **错**：
```cpp
LocalTensor<int32_t> kInt = scratchB.ReinterpretCast<int32_t>();
Cast(kInt, scratchB, RoundMode::CAST_RINT, count);   // src 与 dst 同一物理 buffer
```

AscendC 不保证支持 src/dst 是 *同一物理 buffer 但不同元素类型* 的 Cast。
观察上：在 fp32 → int32 这种宽度相同的 cast 里偶尔像是 work，但稳定性
依赖编译器和指令调度，会在某些行上给出 garbage。

✅ **对**：让 `kInt` 落到一个**当前未被使用**的 fp32 buffer 上的 int32
view（比如 helper 里把 `kInt` 放到 `dst` 上而不是 `scratchB` 上），让 src 和
dst 走不同物理 buffer。

### 7.2 Reciprocal Newton 初值范围错配

教科书 Newton-Raphson 1/D 的初值 `48/17 - 32/17·D` 是给
**D ∈ [0.5, 1]** 用的。

❌ **错**：把这个公式直接套在我们 mantissa 提取后的 D ∈ [1, 2) 上——
当 D > 1.5 时输出**负数**，Newton 会从负数初值发散到错误根（fixed point
y=0）或负无穷。**这是 pre-fix 时看到 max_abs_diff = 130 的根因**。

✅ **对**：D ∈ [1, 2) 时用 `y0 = 1.5 - 0.5·D`。endpoints 精确（D=1→1，
D=2→0.5），最差 12.5% 误差，3 步 Newton 收敛到 fp32 ulp。

### 7.3 mode-1 even/odd permute 漏了 weight_scale

mode 1 的 SwiGLU 是 even/odd 列拆分，host wrapper 上做的
`x.view(M,H,2).permute(0,2,1).contiguous().view(M,N)` 把 x 列重排成
"前半 = glu，后半 = lin"——**weight_scale 是 per-column 的，必须用同一个
permute 重排**，否则 dequant 时第 j 列乘到错误的 ws 上。

### 7.4 quant_scale dtype 没在 host 端归一化

input 里 `quant_scale` 可能是 fp16 / bf16 / fp32（取决于 case）。kernel
通常假设单一 dtype。host wrapper 端用 `.contiguous().to(torch.float32)`
统一——这是 SKILL.md 允许的 tensor transform 操作，不算 hacking。

### 7.5 GlobalTensor::SetValue 性能 vs DataCopyPad

每行写一个 fp32 scale，两条路：
- `scaleGm_.SetValue(row, scale)`：scalar 路径，**每行触发一次硬件标量
  写**，多核并行多行时性能差（每核独立写多个不连续位置）。
- 每核一个 H_per_core 大小的 UB scale 累加 buffer，循环结束后 1 次
  `DataCopyPad` 写 GM。性能好，但要管 byte-level 长度对齐。

两个都算合法，按 rowsPerCore 大小选择即可。

## 8. 不应该做的事（反 hacking 清单）

下面这些是 **不能进 PR 的妥协路径**，因为它们破坏了"算子按硬件事实正确
实现 / 验证按硬件事实合理判定"的双向契约：

- ❌ 改 `.claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py` 让 fp32 阈值放宽 / 加 op 白名单
- ❌ 改 `model.py`（reference 实现）让它适应 NPU 精度
- ❌ 用形状特化、magic constant、benchmark-aware 输入路径让 50 个 case 过
- ❌ 用 `torch_npu.npu_*` vendor kernel 当 oracle 让两端互相 cancel——
  PR #139 已经明确 vendor kernel 自身有 bug（mode 1 实现错误）
- ❌ 在 kernel 里用 hw 超越函数后**声称** op 是 Tier 1 用 fp32 阈值评判

诚实的路径是：**output dtype + reference 精度档 + 计算链分类** 三者一致。
带 fp32 输出 + CPU oracle 的 op，唯一干净的实现就是软件 fp32 路径
（本文档第 4 节）。

## 9. 参考实现指针

- 当前 task：`output/11_DequantSwigluQuant/`
  - mode 0 / mode 1 拆分到独立 kernel
  - software sigmoid helper 在 `kernel/kernel_common.h`
  - host wrapper 端做 mode-1 even/odd permute（含 ws）和 quant_scale dtype 归一化
  - 全 50 个 benchmark shape 通过 `Status: pass / Result: pass`

- 相关 archive_tasks：
  - `archive_tasks/quant_matmul/` — int8 matmul + dequant scale
  - `archive_tasks/reshape_matmul_rowwise_quant_int8/` — rowwise dynamic int8
    quant 收尾（**注意**：这个 task 的 reference 是 NPU vendor kernel，
    不是 CPU fp32，所以它用 hw 超越函数完全合规——属于 Tier 2 路径而非
    Tier 1。**新算子开发不要直接照抄它的 sigmoid 部分**到 CPU oracle 场景）

## 10. 常用 AscendC API 速查（Tier 1 实现需要的）

| 类别 | API | 说明 |
|---|---|---|
| 算术 | `Mul / Muls / Add / Adds / Sub` | bit-exact fp32 |
| 比较选择 | `Mins / Maxs` | bit-exact, 用作 clamp |
| 类型转换 | `Cast<T1, T2, RoundMode>` | RINT = round-half-to-even |
| 位操作 | `ShiftLeft / ShiftRight` | int32 算术移 / uint32 逻辑移 |
| 视图 | `LocalTensor::ReinterpretCast<T>()` | bit reinterpret，不改 buffer |
| 填充 | `Duplicate(dst, scalar, count)` | 写常量，但常量需匹配 T |
| 归约 | `ReduceMax<T, Pattern::AR>` | bit-exact，需 work buffer |
| 绝对值 | `Abs` | bit-exact |
| Memcpy | `DataCopy / DataCopyPad` | DataCopyPad 支持 byte-level len |
| 同步 | `PipeBarrier<PIPE_V>` / `SetFlag / WaitFlag<HardEvent>` | 跨 pipe 用后者 |

文档主索引在
`asc-devkit/docs/api/README.md`。
