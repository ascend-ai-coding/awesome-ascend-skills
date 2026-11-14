# Conv 类算子设计模板（img2col + GEMM）

> 本文是 conv2d / conv3d / 类卷积算子的 TileLang-Ascend **专用设计模板**。通用骨架见 [`design-template.md`](design-template.md)，本节只填 conv 特有结论。

---

## 0. 结构选型（Conv 强制分支，先于一切）

Conv 类算子（`nn.Conv*` / `F.conv*`）必须显式三选一，并记录结论：

> ⚠️ **fp32 输入前置判断（先于三选一）**：输入 dtype == fp32 且主计算走 Cube 时，按 [§12](#12-fp32-输入专项精度适配路线演进) 的演进定稿选精度适配方案——**默认原生 fp32 cube 单 pass**（落地三件套见 §12.3），平台受限或几何不支持时回退 bf16 hi/mid/lo 六项拆分（§12.4），小 shape 走纯 AIV 直卷。**fp16 hi/lo 3-chain 已证精度不达标**（残差解析上界约 2⁻²²），禁止采用。

| 模式 | 适用条件 | 省去 |
|------|---------|------|
| direct 1x1 flipped GEMM | `NCHW && kH=kW=1 && s=1 && p=0 && d=1 && Ci%C0==0 && (H*W)%C0==0` | wrepack + im2col 两级发射 |
| fused img2col（`T.tile.im2col` 随路展开） | `Ci%C0==0`（whole-fmap 路径用真实 Ci 不吃 pad 通道）且 `xNz+B 面板（物理分配）≤ 480KB` | im2col GM 物化（最大收益） |
| normal（materialized im2col + K-major wrepack） | 通用回退 | — |

> **C0 = 32B / 元素宽**：fp16/bf16 → 16，fp32 → 8（NZ/ZN 分形 `[16 行, C0 列]`）。
> 表中 %C0 约束在 fp16/bf16 下即 %16（现行实现 gate 硬编码 16，因 fused/direct
> 只路由 fp16/bf16）；窗口化 band 路径经 xtrans_pad 上片到 CiPad，不受 Ci 对齐
> 限制。fp32 输入先按 §12 前置路由，不进本表三模式。

路由逻辑（完整代码见附录 §1 `select_mode`）：先判 direct → 再判 fused（L1 预算用**物理分配**而非逻辑 Co）→ 否则 normal。

约束：
- 满足 fused/direct 条件却退回 normal 时，必须在「风险点」写理由。
- **禁止默认标量逐元素 gather 的 im2col**：k'=(ci,kh,kw) 解码按段连续块/向量化表达（附录 §4 的 `unrolled_k` C0 分组转置分支是正确范式）；标量豁免仅限占比 <5% 的偶发小尾块。
- 权重 repack 是（weight 内容, shape, dtype）的确定函数，设计中标注可 host 侧 LRU 缓存。

---

## 1. 概述

### 1.1 算子名称 / 1.2 功能描述

{算子名}；把卷积映射为 GEMM（im2col），按 §0 三模式路由。

### 1.3 数学公式

$$
\text{out}[n,co,oh,ow] = \text{bias}[co] + \sum_{ci,kh,kw} x[n,ci, oh\cdot s_h-p_h+kh\cdot d_h,\ ow\cdot s_w-p_w+kw\cdot d_w] \cdot w[co,ci,kh,kw]
$$

GEMM 视角：`A[M,K] = im2col(x)`，`B[K,N] = w`（K-major），`C[M,N] = acc`（fp32），其中 `M=Ho*Wo`，`K=Ci*kH*kW`，`N=Co`。

### 1.4 算法描述（三级流水）

```
wrepack (AIV) : 权重 [Co,Ci,kH,kW] -> K-major ws [Kpad,CoPad]
im2col  (AIV) : 特征图 -> im2col [totalRows,K]（fused/direct 可省）
conv    (AIC) : GEMM  GM->L1->L0A/L0B->MMA(fp32)->L0C->acc
epilogue(AIV) : acc + bias -> cast -> 目标布局输出
```

### 1.5 数据流图

```text
direct:  raw OIHW + raw NCHW view -> flipped GEMM [Co,Ci]@[Ci,HW] -> common epilogue
fused:   NCHW xtrans（NHWC 免）+ wrepack(fused k 序) -> L1 常驻 im2col GEMM -> epilogue
normal:  wrepack + software im2col -> GM [N,M,K] -> materialized GEMM -> epilogue
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Expert**（所有 AIC GEMM 主计算）；辅助 AIV kernel（xtrans/wrepack/im2col/epilogue）同为 Expert（显式 `alloc_ub` + `T.Scope("V")`）。

### 2.2 选型理由

主计算含 `T.mma`（Cube）+ 显式 `alloc_L1/alloc_L0A/L0B/L0C` + `T.Scope("C"/"V")` 核分离；fused 路径需要 L1 常驻控制与 `T.tile.im2col` 的显式参数，均需 Expert 的手动内存/同步控制。

### 2.3 模式影响

| 维度 | 选择 |
|------|------|
| 内存分配 | `T.alloc_L1 / T.alloc_L0A / T.alloc_L0B / T.alloc_L0C / T.alloc_ub` 显式 |
| 计算方式 | `T.mma`（Cube）+ `T.tile.*` / 索引赋值（Vector） |
| 作用域 | `T.Scope("C")` / `T.Scope("V")` 显式核分离 |
| 同步 | `pass_configs` 的 `AUTO_SYNC` + `MEMORY_PLANNING` 自动；跨核交接见 §8 |

---

## 3. API 映射设计

### 3.1 / 3.2 公式拆解与 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 模式 |
|------|----------|-------------|------|
| 权重重排 | `w[Co,Ci,kH,kW] -> ws[Kpad,CoPad]` | `T.copy`（repack kernel 内索引赋值） | Expert(AIV) |
| x 归一化 | `x NCHW -> xT [HW,Ci]`（fused） | `T.tile.transpose` + `T.copy`（xtrans） | Expert(AIV) |
| im2col 展开 | `x -> im2col [M,K]` | `T.tile.im2col`（fused，L1→L0A）或软件 producer（normal） | Expert |
| GM→L1 | 搬运 A/B 面板 | `T.copy` + `T.annotate_layout(make_zn_layout)` | Expert |
| L1→L0 | SplitA/SplitB | `T.copy` | Expert |
| GEMM | `C += A@B` | `T.mma(a_l0, b_l0, c_l0, init=(panel==0))` | Expert(Cube) |
| L0C→acc | 写 fp32 累加 | `T.copy(c_l0, acc[...])` | Expert |
| epilogue | `y = cast(acc + bias)` | `T.tile.add` + `T.tile.cast(CAST_RINT)` | Expert(AIV) |

### 3.3 计算伪代码（direct 1x1，完整骨架见附录 §2）

```python
with T.Kernel(cores, is_npu=True) as (cid, _):
    a_l1 = T.alloc_L1((BASE_M, BASE_K), dtype); b_l1 = T.alloc_L1((BASE_K, BASE_N), dtype)
    a_l0 = T.alloc_L0A((BASE_M, BASE_K), dtype); b_l0 = T.alloc_L0B((BASE_K, BASE_N), dtype)
    c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")
    for task in T.serial(tasks_per_core):
        block = cid + task * cores
        if block < total_blocks:
            with T.Scope("C"):
                for panel in T.Pipelined(k_panels, num_stages=2):
                    T.copy(weight[co_start, ci_start, 0, 0], a_l1)   # LoadA
                    T.copy(x[image, ci_start, hw_start//w, hw_start%w], b_l1)  # LoadB
                    T.copy(a_l1, a_l0); T.copy(b_l1, b_l0)           # Split
                    T.mma(a_l0, b_l0, c_l0, init=(panel == 0))       # Compute
                T.copy(c_l0, acc[image, co_start, hw_start])         # CopyOut
```

### 3.4 API 可行性确认

- `T.tile.im2col`：本项目新增原语（L1 NC1HWC0 → L0A 滑窗展开，对应 AscendC `LoadData3DParamsV2`），接口与语义见文末「附录 · T.tile.im2col 语义」。
- `T.mma`、`T.copy`、`T.alloc_L1/L0A/L0B/L0C`、`T.Scope("C"/"V")`、`T.Pipelined`、`make_zn_layout`：均已验证（见文末附录完整代码）。

### 3.5 技术约束确认

| 约束 | 是否涉及 | 处理方案 |
|------|---------|---------|
| 三维 Kernel 不支持 | 是（N/C/H/W 多维） | 一维 `T.Kernel(cores)` + 手动 block 索引分解（`image/co_start/hw_start`） |
| threads 限制 | 否 | 单 AIC + 单 AIV 核分离（`T.Scope`） |
| 动态循环边界 | 是（尾块） | 静态 `T.Pipelined(k_panels)` + `if panel * BASE_K < k_size` 条件 |
| 流水线动态边界 | 否 | K 面板数静态化（`k_storage // BASE_K`） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| x | NCHW `[N,Ci,H,W]` / NHWC `[N,H,W,Ci]` | fp16/bf16 | 原始输入 |
| weight | `[Co,Ci,kH,kW]` | fp16/bf16 | 原始权重 |

### 4.2 输出张量

| 参数 | Shape | dtype |
|------|-------|-------|
| y | NCHW `[N,Co,Ho,Wo]` / NHWC `[N,Ho,Wo,Co]` | fp16/bf16 |

### 4.3 中间缓冲区

| Buffer | Shape | 层级 | 用途 |
|--------|-------|------|------|
| a_l1 / b_l1 | `[BASE_M,BASE_K]` / `[BASE_K,BASE_N]` | L1 | A/B 面板（NZ 布局） |
| a_l0 / b_l0 | `[BASE_M,BASE_K]` / `[BASE_K,BASE_N]` | L0A/L0B | Cube 操作数 |
| c_l0 | `[BASE_M,BASE_N]` | L0C | fp32 累加 |
| acc | `[N,M_storage,Co_storage]` | GM | fp32 accumulator workspace |
| im2col ws | `[N,M_storage,K_storage]` | GM | normal 路径专用 |

### 4.4 内存搬运路径

```text
direct:  GM[w/x] --T.copy--> L1 --T.copy--> L0A/L0B --T.mma--> L0C --T.copy--> acc(GM)
fused:   GM[xT] --T.copy--> L1(NC1HWC0 常驻) --T.tile.im2col--> L0A
         GM[ws] --T.copy--> L1(K-major 常驻) --T.copy--> L0B --> T.mma --> L0C --T.copy--> acc
normal:  GM[x] --software im2col--> GM[N,M,K] --T.copy--> L1 --T.copy--> L0A --> T.mma --> acc
```

### 4.5 UB 内存预算（辅助 AIV kernel）

- xtrans：slab `[Ci][T]` + stage `[T][Ci]` ≤ ~48KB，T=128/64/32 按 Ci 折半。
- epilogue：`[64,64]` fp32 acc + `[64,64]` 输出 tile ≈ 32KB。

### 4.6 动态轴

无（conv 输入静态 shape）；尾块用 `T.min(BASE_M, m_size - m_start)` 处理。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
})
```

---

## 5. Tiling 策略

### 5.1 计算类型

**混合**（Cube `T.mma` + Vector epilogue / im2col / repack）。

### 5.2 Block 划分

| 常量 | 值 | 依据 |
|------|----|------|
| BASE_M | 128 | dav-c220 L0C = 128KB，与 BASE_N 配对占满 |
| BASE_N | 256 | L0C 单 tile `128×256×4B = 128KB` 恰好占满 L0C（fp32 cube 的 CO2 双 slot 变体用 BM=64 折半） |
| BASE_K | 64 | fp16/bf16 L0 load 分块；**fp32 cube 用 32**（L0A/L0B 按字节容量折半，见 §12.3） |
| C0 | 16（fp16/bf16）；8（fp32） | 分形 C0 = 32B / 元素宽 |

`block_num = batch × m_blocks × n_blocks`（`m_blocks=M_storage/BASE_M`，`n_blocks=Co_storage/BASE_N`）。

### 5.3 约束分析

- L0A: `BASE_M≥16, BASE_K≥32` ✓；L0B: `BASE_K≥32, BASE_N≥16` ✓；L0C: `BASE_M≥16, BASE_N≥16` ✓。
- L1 预算：fused gate 按 **物理分配** 计算 `xNz + 常驻 B 面板` ≤ 480KB（dav-c220 L1=512KB 留 headroom）；A1/B1 depth-2 双面板预算 448KB。两者口径不同，前者只算常驻对，后者算流水双缓冲。
- UB：epilogue 32KB < 192KB ✓。

### 5.4 注意事项

- m/n 尾块按满 baseM/baseN 计算，CopyOut 只写实际范围（stale 行列丢弃），**不要为尾块缩小 Mmad**。
- n 自适应：`nActual≥64` 按实际列加载；`nActual<64` 保持满 baseN（nActual=16 在 dav-c220 触发 L0C 读写冲突）。

---

## 6. 循环与调度结构

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| (image, m, n) block | block 级并行 | `T.Kernel(cores)` | 一维核 + 手动索引分解 |
| K 面板 | 流水 | `T.Pipelined(k_panels, num_stages=2)` | 深度 2 预取重叠 |
| 尾块条件 | 串行 + if | `T.serial` + `if ...` | 边界裁剪 |

### 6.3 流水线优化

K 面板用 `T.Pipelined(num_stages=2)`；fused 路径 x/weight 在 L1 **常驻**（按 (image,n_block) 分组），K 循环只生成 L0A 窗口块 + 切 L0B 面板。

### 6.4 尾块处理

`m_storage/n_storage/k_storage` 全部 round_up 到 BASE，CopyOut/epilogue 按 `valid_m/valid_n/valid_k` 裁剪真实区。

---

## 7. 同步策略

**模式**：自动同步（`AUTO_SYNC` + `MEMORY_PLANNING` pass），跨核 producer→consumer 依赖用 launch 顺序 + GM workspace 衔接（无跨核 flag）。

---

## 8. 融合算子设计（GEMM + epilogue，是）

conv 主计算是 Cube，epilogue 是 Vector，属 CV 融合。本模板按 **Expert 模式**（显式 `T.Scope("C"/"V")` 核分离 + acc(GM) 中转），与 Developer 模式「消 workspace/vid」的默认写法不同：

- Cube 核（`T.Scope("C")`）：`T.mma` → `T.copy(c_l0, acc)` 写 GM fp32 累加。
- Vector 核（`T.Scope("V")`，独立 epilogue kernel）：`T.copy(acc)` → `T.tile.add(bias)` → `T.tile.cast(CAST_RINT)` → 写 y。
- 跨核经 GM accumulator workspace 衔接，无 `alloc_shared` 直连。

> 若需 Developer 模式（消 workspace），参考 [design-template.md](design-template.md) §8.2；conv 因需 L1 常驻 + `T.tile.im2col` 显式控制，仍推荐 Expert。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
import torch.nn.functional as F
def golden(x, weight, bias, stride, padding, dilation, groups):
    return F.conv2d(x.float(), weight.float(), bias.float() if bias is not None else None,
                    stride=stride, padding=padding, dilation=dilation, groups=groups)
```

### 9.2 L0 门槛测试计划（规则 shape，block 整除）

| 用例名 | Shape 约束 | dtype | 说明 |
|--------|-----------|-------|------|
| l0_basic_1x1 | 满足 §0 direct 条件（fp16 下即 `Ci%16==0, (H*W)%16==0`） | fp16 | direct 1x1 路径 |
| l0_fused_3x3 | 满足 §0 fused 条件（fp16 下即 `Ci%16==0` 且 xNz+B 面板 ≤ 480KB） | fp16 | fused img2col 路径 |
| l0_normal_5x5 | 不满足 direct/fused 条件的通用 shape（含 pad/dilation/NHWC） | bf16 | normal 回退 |

> 具体 shape/dtype 组合由测试设计环节按覆盖门禁生成，不在设计模板枚举。

### 9.3 精度标准

按测试设计环节确定的分层精度标准执行（atol/rtol/max_abs_error/required_matched_ratio，含特殊值位置敏感比较）；fp32 中间 acc 后 cast 回输出 dtype。

---

## 10. 风险点与注意事项

- **K 顺序契约**：fused img2col 的硬件展开 k 序是 `(ci1,kh,kw,ci0)`（C0 块最外层），software im2col 是 `(kh,kw,ci1,ci0)`；两者仅在 `Ci==16` 或 `kH*kW==1` 时重合。weight repack 必须按所选模式用同一 k 序，**不得让未标记的 K-major ws 跨路径复用**。
- **LoadData3D 几何 gate（现行实测）**：`kExtension > 64` 的 img2col LoadData3D 挂死；`channelSize=16`（=单 C0 块，fp16/bf16 实测）走慢路径，须 pad 到 ≥32；窗口化 band 路径仅验证「方形核 ≤5×5、sh==sw、dh==dw、CiPadWin≤768」的几何 envelope，出 envelope 一律回退 normal。早期"通道块索引 C1≥15（Ci≥240）产垃圾"的记录来自 whole-fmap 常驻时代，现行窗口化实现已无此 gate，勿照搬。
- **Nd2Nz 作用域**：Nd2Nz 仅当 GM 侧为 ND（raw 输入 / 自产行主序 workspace）时使用——本包全部 GM workspace 即此类。上游已给 NZ（FE 场景 FMAP/Weight 直接 NZ 落 GM，A2/A3 常见）时走 fractal 保形拷贝，禁止二次转换；Nd2Nz 有随路格式转换开销，设计期优先让生产者直接产 NZ、全链路避免 ND。
- **fused L1 预算用物理分配**：B 面板按 BASE_N 完整分配（非逻辑 Co），否则预算误判。
- **满足 fused/direct 却回退 normal 必须写理由**（[decision-tree.md](../references/decision-tree.md) §2「Conv 类算子结构选型」强制）。
- **禁止默认标量 gather im2col**：主数据路径必须向量化，标量豁免仅 <5% 尾块。
- **fp32 cube 不是免费的**：C0=8 分形对齐陷阱 + M-pipe→Fixpipe 同步要求——可以落地（§12.3 三件套），但漏任一项即报错或挂死；窄小图 fp32 cube 链有 aicore crash 史，小 shape 走纯 AIV 直卷兜底（§12.3 风险节）。

---

## 11. 交付清单

### 11.1 目录结构（完整代码见文末「附录」）

```text
design/
├── README.md                 # 本设计的结论 + 契约（可精简为 design.md）
├── block_level/
│   ├── conv2d_base.py        # 共享常量、select_mode 路由、resolve_design、auxiliary schedules
│   ├── direct_1x1.py / fused_im2col.py / materialized_im2col.py
└── tile_level/
    ├── conv2d_base.py        # 统一导出 + xtrans/wrepack/epilogue 辅助 kernel
    └── direct_1x1.py / fused_im2col.py / materialized_im2col.py
```

### 11.4 实现顺序

1. `select_mode` 路由 + `resolve_design`（先定模式）→ 2. block 级 grid/contract → 3. tile 级端到端 contract → 4. 主计算 kernel → 5. 辅助 AIV kernel → 6. Golden + L0 验证。

---

## 12. fp32 输入专项：精度适配路线演进

> 本节是 fp32 输入 conv 的专用设计结论（§0 已前置路由）。**演进定稿：原生
> fp32 cube 单 pass 为默认**。

### 12.1 设计决策（演进定稿）

| 输入 dtype | 主计算类型 | 设计决策 |
|-----------|-----------|---------|
| fp16 / bf16 | GEMM/Conv | 单条 `T.mma`，标准 fp16 cube（C0=16） |
| **fp32** | GEMM/Conv | **原生 fp32 cube 单 pass**（默认；三件套见 §12.3） |
| fp32（平台/几何不支持时） | GEMM/Conv | bf16 hi/mid/lo 六项拆分（回退方案，见 §12.4） |
| fp32 小 shape | 任意 | 纯 AIV 直卷（免 cube，见 §12.3 风险节） |
| 任意 | 纯 Vector（无 `T.mma`） | 可直接 fp32 Axpy/向量化，无 cube 无此约束 |

### 12.2 路线演进与退役原因

fp32 conv 的精度适配经两代演进定稿，**禁止把已淘汰方案当定稿**：

1. **bf16 hi/mid/lo 六项拆分**：`x ≈ hi + mid + lo`，保留
   `hi·hi + hi·mid + hi·lo + mid·hi + mid·mid + lo·hi` 六项（丢弃
   mid·lo / lo·mid / lo·lo），解析残差低于 fp32 eps。代价是 6 条 GEMM
   的固定成本与三套操作数。
2. **原生 fp32 cube 单 pass**（**定稿默认**）：严格 fp32 Mmad，无拆分误差、
   无额外 GEMM。早期"fp32 cube 必挂死"的结论已被三件套解法推翻（§12.3）。

> 已淘汰：**fp16 hi/lo 拆分 + 3-chain**——残差解析上界约 2⁻²²，实测达不到
> fp32 精度门禁（相消 case），禁止采用（演进记录见
> `dsl2Ascendc_compute_conv.md` 第八章）。

### 12.3 原生 fp32 cube 落地要点（三件套 + 两条禁令）

fp32 分形 `C0=8`（fp16/bf16 为 16），由此引出与 fp16 路径的全部差异：

| 项 | 取值 / 做法 | 原因 |
|----|------------|------|
| 分形 C0 | 8 | 32B 分形行 / 4B 元素 |
| BASE_K | 32 | L0A/L0B 按**字节**容量折半（2×[128,32]×4B = 32KB/queue） |
| Mmad 对齐 | m/n/k 均 16 对齐 + `kDirectionAlign = true` | fp32 硬约束 |
| 舍入模式 | `SetHF32Mode(false)` | 关 CANN 默认 HF32 舍入，保严格 fp32 |
| **L0A Fmatrix（禁令一）** | `SetFmatrix(1, m, PAD, FMATRIX_LEFT)` 须在**每次** LoadData3D 前重设 | Fmatrix 是 sticky 状态，上一次 RIGHT 会污染本次加载 |
| **L0B Fmatrix** | B 按 [k,n] 存储需 fractal 转置路径：`SetFmatrix(1, l1W, PAD, FMATRIX_RIGHT)` + LoadData3D `fMatrixCtrl=true` | fp32 B 操作数转置 |
| **M-pipe→Fixpipe（禁令二）** | 用 `TQue<CO1>` 的 EnQue/DeQue 串行 | 可靠；**禁止手搓 barrier**，漏补即 Cube/Fixpipe 竞态死锁、挂死污染设备 |
| 长 K 链误差 | accLanes 拆多条独立累加链 | 精度手段（缩短单链长度），**不是性能 split-K**；单链长度有实测安全界 |
| L0C 排空 | CO2 双 CO1 slot 乒乓 | Fixpipe drain 时不 stall M-pipe |

**已知风险与兜底**：

- **窄小图（窄 W、低行数）fp32 cube 链有 aicore crash 史**（MTE read
  on-chip buffer out of bounds）——小 shape envelope 内改走**纯 AIV 直卷**
  （单发、无 LoadData3D、无 L0/L1 queue、无跨核同步，按构造避挂）。
- **fp32 链式 kernel 有 case 顺序触发的 aicore-timeout 死锁史**——验证必须
  覆盖 case 顺序排列，单 case 验证通过不等于链安全。
- **±inf 语义**：fp32→bf16 拆分时 inf 的低位分量为 NaN，含 inf 的接收域
  累积为 NaN，与参考 fp32 conv 的 NaN 模式一致；host 侧**禁止**用 device
  reduction 做有限性探测（评测对 aclrtMemcpy 计数有约束），有限性问题靠
  路由兜底而非 host 探测。

### 12.4 bf16 hi/mid/lo 六项拆分（回退方案）

平台不支持 fp32 cube 或几何落入 §12.3 风险域时的回退：

- 拆分链：`x → hi/mid/lo` 三套 bf16（AIV 辅助 kernel）。
- 权重按 **split 布局** repack（hi/mid/lo 三段），与 fused 序、normal 序
  并列的第三种 layout，**缓存键必须带布局标志，三种 layout 不可混用**。
- GEMM：6 条 `T.mma` 共享同一 fp32 L0C，仅首条首 panel `init` 清零
  （不变式：同一累加器、全程只清零一次）。
- fp16 hi/lo 2-chain 只在**白名单几何**（逐个验证过零相消误差）上启用，
  非白名单 fp32 一律走 bf16x6 或原生 fp32 cube。

---

## 附录：骨干代码 + 注解（block_level + tile_level）

### 1. 路由与共享配置（block_level 骨干）

```python
BASE_M = 128; BASE_N = 256; BASE_K = 64   # L0C 128×256×4B = 128KB 占满（fp32 cube: BASE_K=32, C0=8）
C0 = 16                                    # fp16/bf16 分形 C0 = 32B/2B（fp32 为 8）
FUSED_L1_BUDGET = 480 * 1024               # fused gate：xNz(真实Ci) + 常驻 B 面板（物理分配）

@dataclass(frozen=True)
class Conv2dConfig:
    """算子配置：batch/ci/h/w/co/kernel_h/kernel_w + stride/pad/dilation/data_format/dtype/dtype_size。"""

def select_mode(cfg) -> str:
    """三模式路由（先于一切）。direct 1x1 → fused img2col → materialized。
    （fp32 输入先按 §12 前置路由到 fp32 专项，不进本函数的三模式。）"""
    direct = (cfg.data_format == "NCHW" and cfg.kernel_h == cfg.kernel_w == 1
              and cfg.stride_h == cfg.stride_w == 1 and cfg.pad_h == cfg.pad_w == 0
              and cfg.dilation_h == cfg.dilation_w == 1 and cfg.ci % C0 == 0
              and (cfg.height * cfg.width) % C0 == 0)   # raw x [Ci, H*W] Nd2Nz 需 32B 行对齐
    if direct:
        return "direct_1x1"
    ci_pad = round_up(cfg.ci, C0)
    k_storage = round_up(ci_pad * cfg.kernel_h * cfg.kernel_w, BASE_K)
    # fused gate 按【物理分配】——xNz 用真实 Ci（非 CiPad），B 面板按实际列数
    # min(round_up(co, C0), BASE_N)；两者之和（乘 dtype 字节）≤ 480KB
    b_cols = min(round_up(cfg.co, C0), BASE_N)
    l1_bytes = (cfg.ci * cfg.height * cfg.width + k_storage * b_cols) * cfg.dtype_size
    fused = (cfg.ci % C0 == 0 and l1_bytes <= FUSED_L1_BUDGET)
    return "fused_im2col" if fused else "materialized_im2col"

def resolve_design(cfg) -> DesignRoute:
    """每模式的 stage 链（launch 顺序）。"""
    # direct_1x1:      (process_direct, epilogue)
    # fused_im2col:    (xtrans?, weight_repack, process_fused, epilogue)  # NCHW 才需要 xtrans
    # materialized:    (weight_repack, software_im2col, materialized_gemm, epilogue)
```

### 2. direct 1x1 flipped GEMM（block_level 骨干）

> 注解：1x1 的 im2col 是恒等变换，x NCHW 本身就是 `[K=Ci, M=H*W]` 的 B 操作数，权重做 A 即可，零 transpose、零 wrepack/im2col。

```python
def direct_1x1_block_contract(cfg):
    """flipped GEMM [Co,Ci] @ [Ci,H*W] 的 grid + 布局契约。"""
    ci_storage = round_up(cfg.ci, BASE_K); co_storage = round_up(cfg.co, BASE_M)
    hw_storage = round_up(cfg.height * cfg.width, BASE_N)
    total_blocks = cfg.batch * (co_storage//BASE_M) * (hw_storage//BASE_N)
    return {"logical_gemm": "[Co,Ci] @ [Ci,H*W]", "k_panels": ci_storage // BASE_K,
            "pipeline_stages": 2, "dataflow": "GM->L1->L0->MMA->L0C->GM"}

@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process_direct(cfg):
    with T.Kernel(cores, is_npu=True) as (cid, _):
        a_l1 = T.alloc_L1((BASE_M, BASE_K), cfg.dtype)   # A 面板（NZ，make_zn_layout）
        b_l1 = T.alloc_L1((BASE_K, BASE_N), cfg.dtype)   # B 面板（NZ）
        a_l0 = T.alloc_L0A((BASE_M, BASE_K), cfg.dtype)  # Cube 操作数
        b_l0 = T.alloc_L0B((BASE_K, BASE_N), cfg.dtype)
        c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")  # fp32 累加
        for task in T.serial(tasks_per_core):
            block = cid + task * cores
            if block < total_blocks:
                with T.Scope("C"):
                    for panel in T.Pipelined(k_panels, num_stages=2):
                        T.copy(weight[co_start, ci_start, 0, 0], a_l1)  # LoadA: GM ND→L1 NZ
                        T.copy(x[image, ci_start, ...], b_l1)           # LoadB
                        T.copy(a_l1, a_l0); T.copy(b_l1, b_l0)          # Split: L1→L0A/L0B
                        T.mma(a_l0, b_l0, c_l0, init=(panel == 0))      # Compute: 首 panel 清零
                    T.copy(c_l0, acc[image, co_start, hw_start])        # CopyOut: L0C→acc(GM)
```

### 3. fused img2col（block_level 骨干）

> 注解：A 侧不落地——x 在 L1 常驻 NC1HWC0，`T.tile.im2col` 随路展开到 L0A；B 侧 K-major 权重按 `(ci1,kh,kw,ci0)` 序常驻。按 `(image, n_block)` 分组，使 x 与 B 同时常驻。

```python
def fused_im2col_block_contract(cfg):
    """fused grid、常驻范围、L1 足迹。"""
    m_storage = round_up(ho * wo, BASE_M); k_storage = round_up(Ci*kH*kW, BASE_K)
    n_storage = round_up(round_up(co, C0), BASE_N)
    resident_groups = batch * n_blocks   # x 常驻 (image)，B 常驻 (n_block)
    return {"weight_k_order": ("ci1", "kh", "kw", "ci0"),      # LoadData3D 实测 k 序
            "residency_scope": "(image, n_block) across all m_blocks"}

@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process_fused(cfg):
    with T.Kernel(cores, is_npu=True) as (cid, _):
        x_l1 = T.alloc_L1((hw, ci_storage), cfg.dtype)      # 常驻 image（NC1HWC0）
        b_l1 = T.alloc_L1((k_storage, BASE_N), cfg.dtype)   # 常驻 B 面板
        for group in T.serial(groups_per_core):
            with T.Scope("C"):
                T.copy(x[image, 0, 0], x_l1); T.copy(weight[0, n_start], b_l1)  # 常驻
                for m_block in T.serial(m_blocks):
                    for k_panel in T.Pipelined(k_panels, num_stages=2):
                        T.tile.im2col(a_l0, x_l1, image_shape=(H,W,Ci),
                                      kernel=(kH,kW), stride=(sH,sW), dilation=(dH,dW),
                                      padding=(pW,pW,pH,pH),
                                      pos_m=m_start, pos_k=k_start,
                                      valid_m=..., valid_k=..., pad_value=0.0)  # LoadA3D
                        T.copy(b_l1[k_start, 0], b_l0)   # SplitB
                        T.mma(a_l0, b_l0, c_l0, init=(k_panel == 0))
                    T.copy(c_l0, acc[image, m_start, n_start])
```

### 4. materialized im2col（block_level 骨干）

> 注解：normal 回退 = AIV software im2col producer 物化 `[N,M,K]` GM，AIC GEMM consumer 消费。producer 唯一正确范式 `unrolled_k`（BASE_K 编译期固定 + C0 分组转置）；~~`serial_k` 逐项标量索引解码~~已按 0.5.3 禁令废弃（主数据路径禁止标量 gather，标量豁免仅 <5% 偶发小尾块）。

```python
def materialized_launch_pipeline(cfg):
    """producer→consumer 依赖 = im2col workspace [N,M_storage,K_storage]。"""
    return {"launch_order": ("software_im2col", "materialized_gemm", "epilogue"),
            "producer_core": "AIV", "consumer_core": "AIC",
            "k_order": ("kh", "kw", "ci1", "ci0"),
            "producer_strategy": "unrolled_k"}   # 唯一范式；标量索引解码（serial_k）已废弃（0.5.3 禁令）

# producer（AIV, T.Scope("V")）: T.alloc_ub([BASE_M,BASE_K]) + T.tile.fill 零填充
#   unrolled_k: NCHW 按 C0 分组载入 + 16x16 transpose；NHWC 按 C0 向量拷（唯一范式）
#   -> T.copy(tile_ub, workspace[image, m_start, k_start])
# consumer（AIC, T.Scope("C")）: 与 direct 的 GEMM 结构完全一致（LoadA/LoadB→Split→T.mma→CopyOut）
```

### 5. tile 级统一导出 + 辅助 kernel（tile_level 骨干）

> 注解：tile 级负责「端到端 launch + 中间布局契约」，并把 xtrans / weight_repack / epilogue 三个 AIV 辅助 kernel 统一导出。

```python
@dataclass(frozen=True)
class TilePipelineContract:
    mode, stages, x_layout, weight_layout, accumulator_layout, output_layout, workspace_layout

def resolve_tile_pipeline(cfg, mode):
    """每模式的 launch 顺序 + 布局契约。"""
    # direct:       stages=(process_direct, epilogue);            acc=[N,Co_storage,HW_storage]
    # fused:        stages=(xtrans?, weight_repack, process_fused, epilogue); acc=[N,M_storage,Co_storage]
    # materialized: stages=(weight_repack, software_im2col, materialized_gemm, epilogue); ws=[N,M_storage,K_storage]

# 三个 AIV 辅助 kernel（均为 T.Kernel + T.alloc_ub + T.Scope("V")）:
#   xtrans        : NCHW x[img,Ci,H,W] --T.tile.transpose(16x16)--> xT[img,H*W,Ci]
#   weight_repack : OIHW --索引赋值(k 序解码)--> ws[K_storage,Co_storage]（fused=(ci1,kh,kw,ci0) / normal=(kh,kw,ci1,ci0)）
#   epilogue      : acc(fp32) --T.tile.add(bias)--> T.tile.cast(CAST_RINT)--> y（NCHW/NHWC）
```

### 6. 三模式 tile 级 plan（tile_level 骨干）

> 注解：每个 mode 一个 `*Plan` dataclass（静态 storage/grid/layout 契约）+ `make_*_plan(cfg)`（校验 + 构造）+ `@tilelang.jit` 主 kernel（镜像 block_level 的 Process，用 plan 驱动）。计划先行、kernel 后行是「设计即文档」的关键。

```python
@dataclass(frozen=True)
class Direct1x1Plan:      # 同 FusedIm2colPlan / MaterializedIm2colPlan
    hw, ci_storage, co_storage, hw_storage, m_blocks, n_blocks, k_panels, ...
    a_gm_layout = "raw OIHW viewed as [Co,Ci]"
    b_gm_layout = "raw NCHW viewed as [N,Ci,HW]"
    c_gm_layout = "[N,Co_storage,HW_storage] FP32"

def make_direct_1x1_plan(cfg): ...   # 校验 routing 条件 + 算 storage/grid
def make_fused_im2col_plan(cfg): ... # 校验 C0 完整组 + L1 预算
def make_materialized_im2col_plan(cfg): ...  # producer/consumer 契约 + strategy
```

### T.tile.im2col 语义

> 源是物理 NC1HWC0 的 L1 tile，目标是 L0A 矩阵 tile。该操作同时包含滑窗采样、重叠元素复用、stride、dilation、padding、M/K 子块选择和 L0A 布局重排。

```python
T.tile.im2col(
    dst_l0a, src_l1,
    image_shape=(H, W, Ci),
    kernel=(KH, KW),
    stride=(SH, SW),
    dilation=(DH, DW),
    padding=(left, right, top, bottom),
    pos_m=m_start, pos_k=k_start,
    valid_m=m_actual, valid_k=k_actual,
    pad_value=0.0,
)
```

> Ascend lowering 对应 `AscendC::LoadData3DParamsV2<T>` + `AscendC::LoadData(dst, src, params)`。
> K 顺序契约：fused 用 `(ci1,kh,kw,ci0)`（LoadData3D 实测 k 序，C0 块最外层），materialized 用 `(kh,kw,ci1,ci0)`；两者仅 `Ci==16` 或 `kH*kW==1` 时重合。
