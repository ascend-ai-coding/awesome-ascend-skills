# A2/A3(`arch22`)FlashAttention 实现手册 — Cube 最小验证

> **适用范围**:Ascend 910B3(A2/A3 代际,`arch22` / `DAV_2201`);`dav-2201` 编译目标;fp16 / bf16。
> **不适用**:Ascend 950(A5,`arch35`);causal / GQA / MLA / 量化 / 多卡;跨代际不继承本文取值。
> **性质**:单实现实证手册,不承担精度或性能的验收口径。
> **术语**:C1 / C2、S / P / PV、`Bq / Bk / D` 等以 [`patterns.md`](../patterns.md) 与 [`fundamentals.md §0`](../foundation/fundamentals.md) 为准;符号名一律视为角色名。
> **最后核对**:2026-11-10。

---

## §1 最小目标

先只写一个 **64×64×64 的 Q·Kᵀ 矩阵乘**:

```
C[64,64] = A[64,64] @ B[64,64]      # fp16 输入,fp32 输出
```

通过标准(三条都要):

1. 输出与 CPU 或朴素实现对得上,误差在 [`/ops-precision-standard`](/ops-precision-standard) 对应 dtype 的容差内;
2. **cube 真的执行**:实测计数里 cube 忙时非零,或"AIC 单跑"与"纯 AIV 模拟"同形状耗时差出一个数量级;
3. 混合核(`__mix__(1, 1)` 或 `__mix__(1, 2)`)能正常结束,不报 `AICORE_EXCEPTION`,不静默全零。

只过第一条不算过:纯向量实现也能算对,但会把后面的性能全部带偏。

**cube smoke 是架构门槛,不是完成标准**;通过后还要按 [`bringup_crosscore.md`](./bringup_crosscore.md) 做跨核排布,再回到 [`architecture.md`](./architecture.md) 补 L0C 常驻与同步拆分。

---

## §2 入口与任务类型

```cpp
extern "C" __global__ __aicore__ void cube_smoke(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);   // 先单子核,避免信用问题
    // ...
    if (ASCEND_IS_AIC) { /* Mmad / Fixpipe */ }
    if (ASCEND_IS_AIV) { /* 可空;或做 softmax / 搬运 */ }
}
```

- AIC 的核号:`GetBlockIdx()`;AIV 的核号:`GetBlockIdx() / GetSubBlockNum()`。
- **不要先用高层 `Matmul` API 试**:本环境 `__cube__` + 高层 Matmul 会编译通过但 kernel 静默不启动;`__aicore__` 路径又要求 `MatmulApiStaticTiling`。最小路径直接用 **`Mmad` + `Fixpipe`**。
- 先 `__mix__(1, 1)`(对应 `KERNEL_TYPE_MIX_AIC_1_1`),跨核跑通后再切 `__mix__(1, 2)`;不要一上来就双子核。

---

## §3 最小数据通路

```
GM(A/B) ──Nd2Nz──► L1(Nz) ──LoadData──► L0A/L0B ──Mmad──► L0C ──Fixpipe──► GM(C)
```

### 3.1 Buffer

```cpp
TBuf<TPosition::A1> aL1, bL1;   // L1
TBuf<TPosition::A2> aL0;        // L0A
TBuf<TPosition::B2> bL0;        // L0B
TBuf<TPosition::CO1> cL0;       // L0C
```

申请字节(fp16 输入、fp32 输出):`aL1 / bL1 = 64*64*2 = 8 KiB`,`aL0 / bL0` 同上,`cL0 = 64*64*4 = 16 KiB`。

### 3.2 GM(ND)→ L1(Nz):必须用 `Nd2NzParams`

```cpp
__aicore__ inline void Load2Nz(const LocalTensor<half>& l1, const GlobalTensor<half>& gm,
                               uint32_t rows, uint32_t cols) {
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = (uint16_t)rows;
    p.dValue = (uint32_t)cols;
    p.srcNdMatrixStride = 0;
    p.srcDValue = (uint64_t)cols;
    p.dstNzC0Stride = (uint16_t)((rows + 15u) / 16u * 16u);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    DataCopy(l1, gm, p);
}
```

- **不要**直接 `DataCopy(l1, gm, 64*64)`:ND 直拷不会变成 Nz,后续 `LoadData` 会读错。
- `dstNzC0Stride` 必须是 16 行对齐后的行数;这是最常见的错位来源。

### 3.3 L1 → L0A:A 操作数(源 ND `[M,K]`)

```cpp
__aicore__ inline void LoadA3D(const LocalTensor<half>& aL0, const LocalTensor<half>& aL1,
                               uint32_t m, uint32_t k) {
    LoadData3DParamsV2<half> p;
    p.padList[0] = 0; p.padList[1] = 0; p.padList[2] = 0; p.padList[3] = 255;
    p.l1H = (uint16_t)(m / 16u);
    p.l1W = 16;
    p.channelSize = (uint16_t)k;
    p.kExtension = (uint16_t)k;
    p.mExtension = (uint16_t)m;
    p.kStartPt = 0; p.mStartPt = 0;
    p.strideW = 1; p.strideH = 1;
    p.filterW = 1; p.filterH = 1;
    p.dilationFilterW = 1; p.dilationFilterH = 1;
    p.enTranspose = 0; p.fMatrixCtrl = 0;
    LoadData(aL0, aL1, p);
}
```

### 3.4 L1 → L0B:B 操作数有两种源,不能混用一套参数

**Kᵀ 风格(源 ND `[N,K]`,K 已转置)**——直接 2D 拷贝:

```cpp
__aicore__ inline void LoadB2DNk(const LocalTensor<half>& bL0, const LocalTensor<half>& bL1,
                                 uint32_t k, uint32_t n) {
    LoadData2DParams p;
    p.startIndex = 0;
    p.repeatTimes = (uint8_t)((n / 16u) * (k / 16u));
    p.srcStride = 1;
    p.dstGap = 0;
    p.ifTranspose = false;
    LoadData(bL0, bL1, p);
}
```

**V 风格(源 ND `[K,N]`)**——块转置 + 分形转置:

```cpp
__aicore__ inline void LoadB3DKn(const LocalTensor<half>& bL0, const LocalTensor<half>& bL1,
                                 uint32_t k, uint32_t n) {
    LoadData3DParamsV2<half> p;
    p.padList[0] = 0; p.padList[1] = 0; p.padList[2] = 0; p.padList[3] = 255;
    p.l1H = (uint16_t)(k / 16u);
    p.l1W = 16;
    p.channelSize = (uint16_t)n;
    p.kExtension = (uint16_t)n;
    p.mExtension = (uint16_t)k;
    p.kStartPt = 0; p.mStartPt = 0;
    p.strideW = 1; p.strideH = 1;
    p.filterW = 1; p.filterH = 1;
    p.dilationFilterW = 1; p.dilationFilterH = 1;
    p.enTranspose = 1; p.fMatrixCtrl = 0;
    LoadData(bL0, bL1, p);
}
```

> 实测:B 的 3D 装载在块数较少且带源偏移时容易错乱或报 `AICORE_EXCEPTION`;最小 smoke 用单块 `n=64` 时,必须确认 `l1H / l1W / channelSize` 与 Nz 块序一致。
> 如果 B 装载结果不对,先回到 64×64 单块,不要加偏移与尾块。

### 3.5 Mmad

```cpp
MmadParams mp;
mp.m = 64;
mp.n = 64;
mp.k = 64;
mp.cmatrixSource = false;
mp.cmatrixInitVal = true;      // 第一块覆盖写;累加时置 false
mp.unitFlag = 0;
Mmad(cL0, aL0, bL0, mp);
```

### 3.6 L0C → GM(Fixpipe)

```cpp
FixpipeParams<float> p;
p.cburstNum = (uint16_t)(n / 16u);
p.burstLen  = (uint16_t)(2u * m);      // 32B 块:m*16*4/32
p.srcStride = (uint16_t)m;      // 本 tile 的源行跨度,不要写 0
p.dstStride = n;
p.reluEn    = false;
p.quantParams.quantPre = QuantMode_t::NoQuant;
p.nz2ndParams = Nz2NdParams(true, 1, 0, 0, (uint16_t)n);
p.unitFlag  = 0;
Fixpipe(cGm, cL0, p);
```

**两套参数组都在本代架构上实现,不要混组、不要照抄对方的默认值**:老式一组(`cburstNum` / `burstLen` / `nz2ndParams`)与 `FixpipeParamsV220` 一组(`nSize` / `mSize` / `srcStride` / `dstStride` / `ndNum` / `srcNdStride` / `dstNdStride`)都能编过;两组 `srcStride` 的单位相同(C0 分形块,fp16 即 16 个元素),但字段语义与默认值不同。

| 项 | 要求 | 违反后果 |
|---|---|---|
| `srcStride` | **本 tile 的源行跨度**(= 行数 m),不是 0 | 写 0 只在源恰好连续(跨度等于标称宽度)时正确:单块自测通过,**换 tile / 尾块才错位** |
| `dstStride` | 目标行 pitch(元素) | 与 GM 视图的行 pitch 不一致 ⇒ 输出错位或半块为零 |
| 随路 NZ→ND | 需要时显式打开并给原始 `N` | 不打开 ⇒ 按 nz 解释,形状对、元素序错 |
| 单位 | `srcStride` 按 C0 分形块计 | 按元素填 ⇒ 跨度放大一个分形块,越界或错位 |

> 两套参数组来自不同代的接口约定:**沿用哪一组,就把该组整组重验**(默认值、单位、字段组合一起),不要只改一个字段。

### 3.7 同步最小集

| 点 | 事件 |
|---|---|
| `DataCopy` 之后、`LoadData` 之前 | `MTE2_MTE1` |
| `LoadData` 之后、`Mmad` 之前 | `MTE1_M` + `PipeBarrier<PIPE_M>()` |
| `Mmad` 之后、`Fixpipe` 之前 | `M_FIX` |
| `Fixpipe` 之后、下一次 `Mmad` 覆盖同一块 L0C 之前 | `FIX_M`(`M_FIX` 的反向;`Fixpipe` 是异步的,只 `PipeBarrier` 不够) |
| `Fixpipe` 之后、AIV 读 GM 之前 | 跨核 flag(见 [`bringup_crosscore.md`](./bringup_crosscore.md)) |

TQue 路径由 `EnQue / DeQue` 自动带事件;只有 `TBuf` 路径需要手写上面这些。

---

## §4 编译 / 链接

```bash
source <CANN 安装目录>/set_env.sh

export CPLUS_INCLUDE_PATH=<标准库头文件目录>:${CPLUS_INCLUDE_PATH}
export LIBRARY_PATH=<标准库库目录>:${LIBRARY_PATH}

bisheng --npu-arch=dav-2201 -O2 -std=c++17 \
    -I op_kernel -I op_host \
    -I "${ASCEND_HOME_PATH}/aarch64-linux/include" \
    op_host/<your_host>.cpp -o <your_binary> \
    -L"${ASCEND_HOME_PATH}/aarch64-linux/lib64" -L"${ASCEND_HOME_PATH}/lib64" \
    -lascendcl -lascendc_runtime -lgraph_base -ltiling_api -lregister -lplatform -lunified_dlog -ldl -lm
```

坑(实测):

1. **头文件版本**:驱动可能按另一版标准库目录找头文件,而本机只装了其一;用 `CPLUS_INCLUDE_PATH` 注入,或补齐版本。
2. **eager-link stub 不继承 `-I`**:链接阶段仍要显式给 kernel 头文件路径。
3. **`-lascendcl` 不能省**:host 侧的 `aclrtMalloc` / `aclrtLaunchKernel` 需要它;只给 `ascendc_runtime` 会缺符号。
4. **单 TU 优先**:host + kernel 放一个源文件里,一次编译;按路径分离编译产物是后续优化,不是最小路径的前提。
5. 以上是特定工具链组合下验证过的写法;换版本要重验。

---

## §5 静默排查:症状 → 先查什么

| 症状 | 最可能 | 先查 |
|---|---|---|
| 正确性集全过,但性能差一个数量级 | cube 没跑 | cube 忙时计数;AIC 单跑与纯向量模拟的耗时对比;任务类型 |
| `__cube__` 编译过、kernel 不执行 | `__cube__` 与高层 Matmul 组合会静默不启动 | 改 `__aicore__` + `__mix__(1, 1)`;直接 `Mmad` |
| 高层 Matmul 输出全零 | 缺静态 tiling 参数 | 补 tiling / 模板;或改用 `Mmad` |
| 报 `AICORE_EXCEPTION` | L1/L0 越界、对齐、TQue 槽数 | `TQue::InitBuffer` 的槽数必须是编译期常量;L1/L0 字节复算 |
| 输出错位 / 半块为零 | `Nd2Nz` / `Fixpipe` 的 stride | `dstNzC0Stride`;`srcStride / dstStride`;**若"单块对、换 tile 才错",先查源侧行跨度是否写成了 0** |
| AIC 不启动 | 核类型或分支写错 | `ASCEND_IS_AIC` 是否进入;`GetBlockIdx`;block dim 设置 |
| NaN | L0C 未初始化 / mask 取值 | 首块 `cmatrixInitVal = true`;mask 用有限大负数 |

**确认 cube 真的跑了**(至少一种):cube 忙时非零;AIC 单跑显著快于纯向量模拟;把 `Mmad` 换成 `Add` 的对照版本明显变慢(说明原来在走标量)。

---

## §6 从 smoke 到 QK / PV

1. **Q·Kᵀ**:`A=Q[Bq,D]`、`B=Kᵀ[D,KV]` → `S[Bq,KV]`;Kᵀ 在输入准备期物化,或在 kernel 内转置。
2. **P·V**:`A=P[Bq,KV]`、`B=V[KV,D]` → `O[Bq,D]`;注意 V 的源布局是 `[K,N]`,走 3D 装载。
3. 先固定一个 KV 分块、一个 task,把两次 `Mmad` 串起来;数值对之后再引入跨核。
4. 每一步都保留"关掉新特性即回默认路径"的开关,便于定位。

---

## §7 通过清单

- [ ] 64×64×64 单块:数值正确、cube 确实执行、混合核正常结束
- [ ] Nd2Nz / 两种 B 装载 / Fixpipe 参数逐项核对,无错位
- [ ] 同步最小四件套齐全(`MTE2_MTE1`、`MTE1_M`、`M_FIX`、跨核 flag)
- [ ] QK 与 PV 串起来后仍保持确定性
- [ ] 已确认不是"纯向量也能算对"的假通过
