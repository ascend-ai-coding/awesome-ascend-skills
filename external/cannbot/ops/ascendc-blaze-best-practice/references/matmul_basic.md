# Matmul NO_FULL_LOAD_MODE 深度

> **适用架构**：DAV_3510（CANN 9.0.0-beta.2）。L1=512KB / L0A=L0B=64KB / L0C=256KB / BT=4KB。其他架构需另行适配。
>
> 前置阅读：[`matmul_pattern.md`](matmul_pattern.md)（模板复制 + 共享基础）。本文聚焦 NO_FULL_LOAD_MODE 的 SWAT Tiling + 常见改造。A 全载请读 [`matmul_full_load.md`](matmul_full_load.md)。
>
> **Developer 提示**：改 dtype 时最容易漏的是 `DATA_SIZE_FP16`（tiling 常量）和 `BLOCK_CUBE`（fp8=32, int8=32, 其他=16）。改完跑一次 `bash run.sh 256 256 256` 快速验证。

## 1. Tiling 策略：SWAT

`MatmulTilingSwat` 主流程（`include/tiling/matmul_tiling.h`）：

| 步骤 | 作用 |
|---|---|
| `FormulateBasicBlock` | 以 256×256 起步，按核数做 M/N 轴分裂，得 `baseM/baseN` |
| `OptimizeEdgeBasicBlock` | 合并尾块，消除尾行/列负载不均 |
| `CalcTailBasicBlock` | 尾块切分提高占核率 |
| `CalL1Tiling` | 按 L1 容量反推 `stepKa/stepKb`，确保 `(baseM+baseN)*stepK*sizeof(dtype) ≤ L1_SIZE/2` |
| `FormulateLoadBalanceBlock` | 256×256 不能填满核时，参考 `BLOCK_TABLE` 重选均衡参数 |

输出 `MatmulTilingData` 字段：

| 字段 | 含义 | Kernel 端 |
|---|---|---|
| `m/n/k` | 问题规模 | `ProblemShape` |
| `baseM/baseN/baseK` | L0 切分颗粒 | `MatmulTiling` qbmmParams |
| `kL1` | L1 K 方向窗口 = baseK×stepK | `L1Params{kL1}` |
| `mTailCnt/nTailCnt` | 尾块 split 次数 | `BlockSchedulerParams` |
| `usedCoreNum` | 启动核数 | `<<<usedCoreNum, ...>>>` |
| `l0cDB` | L0C ping-pong 级数 | `enableL0cPingPong` |

> 切 dtype 时同步改 `DATA_SIZE_FP16`（`matmul_tiling_constant.h`）：fp8→1, bf16/fp16→2, fp32→4, fp4×2→0.5。

## 2. 常见改造

### 2.1 切换 dtype

| 位置 | 改动 |
|---|---|
| Launcher | `AType/BType/CType`、`sizeA/B/C` |
| BlockMmad | `BLOCK_CUBE`（bf16/fp16=16, int8/fp8=32, fp4=64）；`L0CType`（int8→int32，其他→fp32） |
| tiling | `DATA_SIZE_FP16` 改对应字节数 |
| gen_data.py | 输入 dtype + golden cast |
| verify_result.py | 读取 dtype + 容差 |

dtype 速查：

| dtype | C++ 类型 | sizeof | BLOCK_CUBE | L0CType |
|---|---|---|---|---|
| bf16 | `bfloat16_t` | 2 | 16 | float |
| fp16 | `half` | 2 | 16 | float |
| fp32 | `float` | 4 | 16 | float |
| fp8_e4m3 | `fp8_e4m3fn_t` | 1 | **32** | float |
| int8 | `int8_t` | 1 | 32 | **int32** |
| fp4 | `fp4x2_e2m1_t` | 0.5 | 64 | float |

> fp8 落盘：`torch.tensor(...).to(torch.float8_e4m3fn).view(torch.uint8).numpy().tofile()`。golden 用 `A.float() @ B.float()` 匹配 fp32 L0C 累加精度。`BLOCK_CUBE_L0C` **恒为 16**，不用 `32/sizeof` 推导。

### 2.2 增加 Bias 输入

**核心原则**：bias 数据移动合并到已有 A/B 双缓冲流水线中（跟随 BL1→BL0），不做独立 `PipeBarrier` 前置阶段。

| 步骤 | 位置 | 操作 |
|---|---|---|
| ① 声明 | `operator()` 内 K-loop 之前 | 构造 `gmBias`/`gmBiasBlock`/`tensorBiasL1`/`tensorBiasBT`，**不做 Copy** |
| ② L1 bias 跟 BL1 | K-loop 内 B 的 Copy 之后 | `if (iter0==0) Copy(copyGM2L1AB, tensorBiasL1, gmBiasBlock);` |
| ③ L0 bias 跟 BL0 | L0 子循环内 B 的 L1→L0B 之后 | `if (iter0==0 && iter1==0) Copy(copyL12BT, tensorBiasBT, tensorBiasL1);` |
| ④ MMAD 首轮带 bias | mmad 处 | `cmatrixInitVal=true` 时传 `tensorBiasBT` 作为第 4 参数 |

**同步保证**：`WaitFlag<MTE2_MTE1>` 等 A/B/bias 三个 GM→L1 全部完成；`WaitFlag<MTE1_M>` 等 L1→L0 全部完成。无需额外 event slot。

**关键约束**：
- Bias dtype = fp32，shape `(1, N)`，ND 行主序
- L1 staging 区在 L1 末尾：`biasL1Offset_ = TOTAL_L1_SIZE - Align(baseN*sizeof(BiasType)+32, 32)`（32B margin 防止越界写；**不能硬编码偏移值**）
- BT 双缓冲：`l0cDB==2` 时 BT 也切两半，`biasBtOffset = (l0cPingPong_ & 1) * 256 * sizeof(BiasType)`（dav-3510 BT=4KB ÷ 2 = 256 fp32/半）

完整参考实现见 `matmul_block_mmad.h`。

### 2.3 新增 MX/量化 Trait

三步并存多条计算路径：

1. `policy/dispatch_policy.h` 加新 dispatch tag（如 `QuantMatmulMxPolicy<...>`）
2. `matmul_block_mmad.h` 复制一份 SFINAE 特化，`enable_if_t` base 类换成新 tag
3. launcher 切换 `using DispatchPolicy = ...`

### 2.4 调大 L1 stage 数（2 → 4）

1. `dispatch_policy.h` 加 `STAGES_` 模板参数
2. BlockMmad 把 `L1_BUFFER_NUM` 改为 `DispatchPolicy::stages`
3. 重算 `l1BufferAOffset_/l1BufferBOffset_`（4 半区 interleave）
4. Tiling 缩小 `kL1`（半区容量减半）

> A 全载模板无此概念（A 端单缓冲驻留，B 端默认双缓冲）。
