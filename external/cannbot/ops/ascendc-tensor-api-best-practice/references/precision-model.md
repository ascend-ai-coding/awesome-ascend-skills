---
type: reference
status: stable
platforms:
- '950'
sources:
- resource: project execution records
  role: primary
- resource: asc-devkit master
  role: supporting
updated_at: '2026-10-29'
---

# 精度模型与排障方法论（A5 Cube matmul）

本参考在测试设计、联调和功能精度验收节点按 implementation_route=tensor_api_custom 条件加载。

> 实证载体：GroupedMatmul bf16 大 K case；官方 torch_npu matmul 作对照；419 万元素 bit 级对比。

## 1. A5 bf16 Cube matmul 精度模型

### σ 模型（核心结论）

```
σ ≈ 2.6e-6 × sqrt(K 段数)        # K 段数 = ceil(K / 128)
```

- 每段 `mmad(k=128)` 内部是**顺序逐项 fp32 舍入**，每段贡献 ~2.5e-6 的独立误差
- 段间 L0C 累加舍入仅 ~0.4e-6（次要项）
- **mmad k=256 会被编译器/硬件拆成 2×128**（软件层 `mmad_instr → asc_mmad → 内建 mad` 直传无拆分，拆分发生在更低层）——所以 **K 链长度下限 = ceil(K/128)，物理不可减**
- bf16 大 K（≥7 段）× 大 M×N 的 case 必然出现小值域超标（相对误差门 2 的逐点判定）——这是 A5 硬件极限，官方 torch_npu matmul 同数据同样超标且输出与自研实现 bit 级一致

### 位置基数效应

小值域超标数 = 位置数 × 超标率。大 M×N case（百万级输出位置）即使误差率与其他 case 相同，绝对超标数也更容易撞"≤2 个超标点"这类逐点门限。**对比不同 shape 的精度结论时必须区分"率"和"数"。**

## 2. 纯 Cube 精度修复死路表

以下路径均经实验或架构论证排除，勿重复尝试：

| 路径 | 不可行原因 |
|---|---|
| 树状（两两）累加 | L0C 64KB 仅容一个 128×128 fp32 tile，无 L0C→L0C 加法指令 |
| split-K 部分和经 GM 中转再累加 | GM 中转的 fp32 走 hf32（19 位尾数）丢 5 位，部分和 ~32 量级时引入 ~6e-5 新误差，比待修误差更大 |
| 拆双链再合并 | 总段数不变 → 独立误差平方和不变 → σ 不减 |
| `HF32_MODE_BIT`(bit46) | 控制的是 **fp32 输入转换**精度，与 bf16 累加误差无关 |
| fixpipe 舍入模式（RTZ 嫌疑） | 超标点偏差方向双向混合，非系统性单向舍入，可排除 |
| Vector 辅助 fp32 累加 | 纯 Vector matmul 性能远低于 Cube（数量级差距），性能分崩 |

## 3. 精度排障方法论（按证据强度排序）

### 3.1 官方 kernel 对照实验（决定性，第一步就做）

官方对应算子（如 torch_npu matmul）跑同数据 + 同判定：
- 官方也 FAIL → 倾向硬件极限；再做 **bit 级对比**（输出逐位比较），bit 一致 = 自己的实现与官方数值等价
- 一步区分"我的实现差"还是"硬件极限"，避免在不可修问题上空耗

### 3.2 控制变量变体实验

从现有 case 数据**派生真子集**（切 K/M/N 维度），一次只变一个因素（段数/尾块/对齐）：

- 段数单调性实验：0 → 4 → 6 段，超标数随段数单调递增 → 锁定段数为误差源
- ⚠️ **无效切片坑**：K=1023 的数据切 `[:, :1024]` 返回原数据（变体 ≡ 原版，白跑一轮）——**变体生成后必须先验证 shape 真的变了**

### 3.3 三重证据法（定案"拆分在硬件层"）

1. **输出 bit 级一致**：配置改变但输出逐位不变 = 舍入序列不变
2. **性能持平**：耗时不变 = 指令数未变
3. **软件层源码直传**：反查编译链软件层无拆分逻辑

三者同时成立 → 拆分发生在编译器/硬件更底层，软件侧不可控。

### 3.4 σ(段数) 拟合

超标数随段数变化的数据点拟合 σ 模型，反推单段误差贡献，与理论值（~2.5e-6）对照吻合即定案。

### 3.5 bit 级一致性作为通用工具

- 证明优化不改数值（性能优化的回归门禁）
- 证明配置参数无效（如 base_k=256 vs 128 输出 bit 一致 = k=256 被拆）
- 排除构建 bug（demo 链路 vs torch 链路输出一致 = 构建无污染）

## 4. 实现级 bug 模式（数值正确性，区别于硬件极限）

| 缺陷模式 | 现象 | 根因 |
|---|---|---|
| B 复用跨 expert 错配 | E>1 大 case MERE ~3.9 | 复用槽未区分 expert 维度 |
| L1 滚动 buffer 与跨 tile 复用冲突 | K>256 场景 MERE ~3.0 | 两种 buffer 策略状态互踩 |
| B 槽位超 L1 容量 | 有 bias case 全挂 + 305ms 超时 | **容量预算未门禁**（576KB > 512KB） |

判定线索：**误差量级**。实现 bug 的 MERE 通常是 1e-1~1e1 量级（数据错），硬件舍入极限是 1e-6~1e-5 量级（末位抖动）。先看量级再选排障路径。

## 5. 低精度数据类型与随路量化精度模型

§1-§4 覆盖 bf16 Cube matmul 的累加舍入误差；本节覆盖**设计时选型**视角——低精度数据类型（hifloat8、MXFP4/MXFP8）与随路量化。两者同为 A5 Cube matmul 精度主题，选型错误会直接决定精度上限，先于排障。

### 5.1 低精度数据类型速查（A5 = Ascend 950PR&950DT 专属）

| 类型 | 私有元素位宽 | 块大小 k | 共享缩放因子 | 定位 |
|---|---|---|---|---|
| hifloat8_t | 8 bit（单元素，非块格式） | — | 无（Fixpipe 统一 scalar） | Matmul 输入/输出低精度 |
| MXFP8 | 8 bit（fp8_e5m2 / fp8_e4m3fn） | 32 | fp8_e8m0_t（8 bit） | MxMatmul 输入 |
| MXFP4 | 4 bit（fp4x2_e1m2 / fp4x2_e2m1） | 32 | fp8_e8m0_t（8 bit） | MxMatmul 输入 |

### 5.2 hifloat8 数据通路

hifloat8_t 占 1 字节。数据量随 C 类型变化（M×K、K×N、M×N）：

| 方案 | A/B 输入 | C 输出 | 理论 GM 数据量 |
|---|---|---|---|
| 基线 | half | float | 2MK + 2KN + 4MN |
| 仅输入量化 | hifloat8_t | float | MK + KN + 4MN |
| 输入输出量化 | hifloat8_t | hifloat8_t | MK + KN + MN |

输出量化走 **Fixpipe 随路量化**：C 输出 Tensor dtype 配 `hifloat8_t`，结果从 L0C 搬出到 GM 时通过 `copy(copy_l0c2gm_atom, dst, src, quant)` 的第 4 参数量化 scale tensor 直接完成 `float→hifloat8_t`，不产生中间 float 往返（官方文档用传统 AscendC 表达为 `DT_HIFLOAT8` + `SetDequantType(SCALAR)` + `SetQuantScalar` + `Iterate`）。舍入模式 `QF322HIF8_PRE`，**Half to Away Round**。量化系数不能为 Inf/NaN/非规格化数。

### 5.3 MX 格式（MxMatmul）

MX 是块格式：k 个私有元素共享一个缩放因子 X，块内元素同类型。公式 `C = (scaleA ⊗ A) * (scaleB ⊗ B) + Bias`，**K 方向每 32 元素共享一个量化因子**（scaleA shape [M, K/32]，scaleB shape [K/32, N]）。

精度损失有两个来源，选型时要分别评估：
1. **私有元素低比特尾数**：MXFP4 只有 4 bit 尾数/指数，单元素相对误差远大于 fp16。
2. **块级共享缩放的量化误差**：一个缩放因子代表 32 个元素，块内动态范围不一致时，小值被大值「淹没」，产生块级量化误差（与 §1 的「位置基数效应」同理，需区分率与数）。

### 5.4 随路量化的精度约束（坑）

| 约束 | 内容 |
|---|---|
| 量化输出 float 精度上限 | 达不到**双万分之一（1e-4）**，只能到**双千分之一（1e-3）**；要 1e-4 精度改用高阶反量化 API（传统 AscendC 为 `AscendDequant`） |
| 反量化 INF_NAN | int8/int4b→half 反量化**不支持 INF_NAN 模式**；需 INF_NAN 时走独立反量化路径（传统 AscendC 表达为输出到 VECIN + int32 + `AscendDequant`） |
| 舍入模式 | hifloat8 输出固定 Half to Away Round（非 RTZ），与 §2 的「RTZ 嫌疑」区分开 |
| 接口时序 | 随路量化的 `quant` 参数须在 `mmad`/`copy` 调用前备好；量化粒度与 Tiling 侧配置必须一致 |

两种量化模式（按 quant tensor 粒度区分）：**PER_TENSOR**（全 C 共享一个系数，quant shape `[1]`）与 **PER_CHANNEL**（C 每列一个系数，quant shape `[1, n]`，见 `mmad_tensor_api` 样例），按量化粒度选择。

### 5.5 平台差异

- **FP Buffer 随路量化**（量化参数放 Fixpipe Buffer C2PIPE2GM，一次 Fixpipe 完成量化）官方标注**仅 Atlas A2 系列生效**，A5 不适用——A5 的低精度输出走 §5.2 的 C 输出 dtype 配低精度类型 + `copy` 的 `quant` 参数，勿混用。
- hifloat8、MXFP4/MXFP8、int8→bf16 等低精度组合均为 **Ascend 950PR&950DT（A5）专属**，选型时先查 §5.1/§5.4 的平台列。

## 6. 边界语义与数值合同冻结

§1-§5 覆盖「舍入误差」；本节覆盖「边界语义」——设计时必须在 DESIGN §8 Golden 合同里**显式写死**的边界决策。边界不清是功能 bug 高发区，且常在尾块/极端输入才显形。

### 6.1 dtype 数值范围速查（IEEE 754 标准，选型与测试边界用）

| dtype | 指数位 | 尾数位 | 最大有限值 | 最小正规数 | 最小非规格化 |
|---|---|---|---|---|---|
| fp16 (half) | 5 | 10 | 65504 | 6.10e-5 | 5.96e-8 |
| bf16 (bfloat16) | 8 | 7 | 3.39e38 | 1.18e-38 | ~9.2e-41 |
| fp32 (float) | 8 | 23 | 3.40e38 | 1.18e-38 | 1.40e-45 |
| fp64 (double) | 11 | 52 | 1.80e308 | 2.23e-308 | 4.94e-324 |

整数类型：int8 `[-128, 127]`、uint8 `[0, 255]`、int16 `[-32768, 32767]`、uint16 `[0, 65535]`、int32 `[-2^31, 2^31-1]`。

**选型判据**：输入动态范围超过 fp16 上限（65504）→ 不能选 fp16；bf16 指数范围同 fp32 但尾数仅 7 位（累加误差大于 fp16，见 §1 场景）；hifloat8/MX 的私有位宽与块共享缩放见 §5.1。

### 6.2 溢出 / 下溢 / 饱和（必须冻结的三问）

| 决策项 | 冻结内容 | 备注 |
|---|---|---|
| 上溢（overflow） | 超最大有限值 → 转 Inf？还是饱和到最大有限值？ | fp32 累加中途不溢出、最终 cast 到 fp16 时可能溢出；语义以当前手册/实测为准 |
| 下溢（underflow） | 非规格化数 → 保留？还是 flush-to-zero（FTZ）？ | FTZ 会引入单向误差，与 §2 的「RTZ 嫌疑」区分 |
| 饱和（saturation） | float→int/低精度 cast 溢出 → 饱和到边界？还是环绕（wrap）？ | 量化/反量化链路见 §5.2/§5.4；`int8` 溢出是环绕还是饱和以手册为准 |

**纪律**：这三问的答案写进 DESIGN §8「非有限值」栏，且 golden 与 kernel 用同一语义（golden 里显式写 clamp/FTZ，见 [golden-generation.md](golden-generation.md) §5），否则边界 case 全挂。

### 6.3 非有限值（NaN / Inf）

- **equal_nan 语义**：NaN 算不算相等，按算子语义冻结（`np.isclose(..., equal_nan=True/False)` 二选一，[golden-generation.md](golden-generation.md) §5）。
- **传播**：分母为 0、`sqrt(负)`、`0/0` 产生 NaN/Inf 的传播路径是否接受，写进支持域（§1）。
- **量化系数禁区**：量化 scale 不能为 Inf/NaN/非规格化数（§5.2 原文），golden 生成时过滤。

### 6.4 零维 / 标量 shape

- 零维 tensor（scalar shape `[]`）与 shape 含 0（空 tensor）是边界，kernel 侧 `length=0` 的循环/`update_mask` 是否显式短路，设计时写清；默认**不支持零维**则必须在 §11 阻塞点登记。
- 单元素 tile（`length=1`）走尾块逻辑还是专门标量路径，避免与 `vector_len` 整除性冲突。

### 6.5 尾块 padding 边界

- 尾块 padding 区填什么值（0/复制边界值/任意）**必须写死**，且 padding 不参与 golden（[api-dataflow-layout.md](api-dataflow-layout.md) mmad 门禁）。
- padding 值若参与后续 element-wise（如残差加），会污染有效区——设计 §5 尾块 policy 与 §3 数据流同时核对。

## 事实源

- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/simd_operator_optimization/matrix_compute/hifloat8_matmul_optimization.md`
- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/simd_operator_optimization/matrix_compute/fp_buffer_quantization.md`
- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/{mx_matmul_scenario,matmul_output_quant_dequant}.md`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时在 skill 根目录运行 `scripts/ensure_docs.py` 自动获取。
