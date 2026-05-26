# Ascend C Matmul 单算子生成（分支入口 + 共享基础）

> **适用架构**：DAV_3510（CANN 9.0.0-beta.2）
>
> **统一模板 + 多份文档**：一份代码 `matmul_custom/` 同时承载**纯AIC**（`matmul_custom.cpp`；NO_FULL_LOAD / A_FULL_LOAD 两种 mode，切换 3 行 diff）和 **FixpOpti**（`matmul_fixpopti.cpp` + `include/kernel/matmul_kernel_fused.h` + `include/epilogue/`）两条模板路径。B_FULL_LOAD 与 StreamK 仍为设计草案。
> - `matmul_pattern.md`（本文）— 入口、模式总览、模板复制、共享基础：四层抽象 / Cube 内存 / 流水 / Layout / Launcher / 排错 / 三模板体系
> - [`matmul_basic.md`](matmul_basic.md) — NO_FULL_LOAD_MODE 深度（SWAT Tiling + dtype/Bias/量化/stage 改造）
> - [`matmul_full_load.md`](matmul_full_load.md) — A_FULL_LOAD_MODE 深度（A 常驻 L1 跨 N-tile 复用）；B_FULL_LOAD 对称设计
> - [`matmul_fixpopti.md`](matmul_fixpopti.md) — FixpOpti 改造食谱（启用 launcher + `[MODIFY] N/C/A/E` 标记）
>
> **模板覆盖范围**：`matmul_custom/` 同时承载纯 AIC 模板（`__aicore__ __cube__`，AIV 直接返回）与 FixpOpti 模板（`__mix__(1, 2)`，AIC+AIV 混合 + CV 同步）。StreamK（AIC+AIV 混合 K 切分）仍为设计文档，选择决策见 §10。
>
> 阅读顺序：先看一遍 `tensor_api_user_guide.md` 了解 API → §10 选模板（三模板体系）→ 选定纯AIC 后 §0 选 mode → §0.5 复制+改造；选 FixpOpti 跳到 [`matmul_fixpopti.md`](matmul_fixpopti.md) §3 改造 → 共享基础 §1–§9 按需查阅。

### Quickstart（5 分钟最快上手）

```bash
# 1. 拉取 tensor_api 依赖（首次必做）
git clone --depth 1 https://gitcode.com/cann/ops-tensor.git /tmp/ops-tensor
cp -r /tmp/ops-tensor/include/tensor_api references/matmul_custom/third_party/
rm -rf /tmp/ops-tensor

# 2. 复制模板 + 运行默认样例验证环境
cp -r references/matmul_custom /tmp/my_matmul && cd /tmp/my_matmul
bash run.sh
# 预期输出：verify_result.py PASS（max relative diff ≈ 0.002 for bf16）
```

> 模板仅验证 `DAV_3510`。其他架构需先确认 `tensor_api` 和 L1/L0 容量约束。

## 0. 模式总览

`matmul_custom/` 模板当前**已交付两种** dispatch mode（NO_FULL_LOAD / A_FULL_LOAD），通过 `MatmulMultiBlockPolicy<MODE>` 区分。B_FULL_LOAD 见表后 NOTE，仅有设计草案，工程代码未交付：

| 维度 | NO_FULL_LOAD_MODE（默认） | A_FULL_LOAD_MODE |
|---|---|---|
| Dispatch tag | `MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>` | `MatmulMultiBlockPolicy<A_FULL_LOAD_MODE>` |
| 数据流 | A、B 每轮 GM→L1 | A 首轮搬入驻留，B 每轮搬 |
| L1 布局 | `[A0\|B0] / [A1\|B1]` | A 驻留 offset=0，B 双缓冲紧跟 |
| L1 容量假设 | `(baseM+baseN)·kL1·dtype·2 ≤ L1` | `Align(m,16)·Align(k,16)·dtype + 2·baseN·kL1·dtype ≤ L1` |
| 跨核切分 | M、N 双向 serpentine | 仅 N 切；mCnt=1 |
| 尾块约束 | 无 | `mTailTile=1`（= `tilingData.mTailCnt=1`，Tiling 引擎自动锁） |
| 适用 shape | 通用 | M·K·sizeof(A) ≤ ~256KB，N ≫ M |
| 关键文件 | `matmul_block_mmad.h` | `matmul_block_mmad_a_full_load.h` + `MatmulTilingAFullLoad` |
| 交付状态 | **已交付** | **已交付** |

**选 A 全载**：当 `Align(m,16)·Align(k,16)·sizeof(A) ≤ ~256KB`（bf16，dav-3510 L1=512KB）且 `N ≥ 1024`。

> ⚠️ **B_FULL_LOAD_MODE [未实现]**：本仓库中**不存在**以下任一符号——
> - 常量 `B_FULL_LOAD_MODE`（`common_utils.h` 仅定义 NO_FULL_LOAD=0 / A_FULL_LOAD=1）
> - `MatmulMultiBlockPolicy<B_FULL_LOAD_MODE>` SFINAE 特化
> - `MatmulSwatScheduler<B_FULL_LOAD_MODE>` selector（`matmul_block_scheduler.h` 只注册 NO/A）
> - `MatmulTilingBFullLoad` 类（`matmul_tiling.h` 仅有 SWAT + AFullLoad）
> - 头文件 `matmul_block_mmad_b_full_load.h`
>
> 对称设计（M ≫ N 场景，B 驻留 L1）仅在 [`matmul_full_load.md`](matmul_full_load.md) §3.2 / §4.2 / §4.3 以伪代码+对称差异表呈现。需要时按该文档镜像 A 全载实现一遍（约半天～1 天，含编译验证）。**不要尝试套用本节后续的 3 行切换 diff——所引用的所有符号都不存在，会立刻编译失败。**

不满足则保持 NO_FULL_LOAD_MODE。

## 0.5 模板复制与改造

### 0.5.0 拉取 tensor_api 依赖（首次必做）

```bash
git clone --depth 1 https://gitcode.com/cann/ops-tensor.git /tmp/ops-tensor
cp -r /tmp/ops-tensor/include/tensor_api references/matmul_custom/third_party/
rm -rf /tmp/ops-tensor
# 就位后：references/matmul_custom/third_party/tensor_api/include/tensor_api/tensor.h 应存在
```

> CMakeLists.txt 已把 `third_party/tensor_api` 加入 include path。忘拉这一步编译直接报 `file not found`。

### 0.5.1 复制 + [MODIFY] 三档改造

```bash
cp -r references/matmul_custom <your_project_name> && cd <your_project_name>
```

搜索 `[MODIFY]`，按档位逐一改：

| 档位 | 标记 | 内容 |
|---|---|---|
| **必改** | `[MODIFY N1]` | 函数名 / CMake 目标名 / `run.sh::OP_NAME` 三处同步 |
| **必改** | `[MODIFY N2]` | `AType/BType/CType` + `sizeA/B/C` + `matmul_tiling_constant.h::DATA_SIZE_FP16` |
| **必改** | `[MODIFY N3]` | `gen_data.py` 输入分布 + `verify_result.py` dtype/容差；A 全载改 `trans_b=False` + MERE/MARE 双门 |
| 常改 | `[MODIFY C1]` | `layoutA/layoutB` 与 `transA/transB` 模板实例化 |
| 常改 | `[MODIFY C2]` | TilingData / BlockMmadParams 增删字段（bias/scale） |
| 选改 | `[MODIFY A1]` | 切到 A 全载或量化/MX 变种的 3 行替换（见 §0.5.3） |
| 选改 | `[MODIFY A2]` | `L1_BUFFER_NUM = 4` 等更深 stage |

### 0.5.2 关键文件速查

| 文件 | 说明 |
|---|---|
| `matmul_custom.cpp` | Launcher，[MODIFY] 标记按 N1-3/C1-2/A1-2 分级 |
| `include/kernel/matmul_kernel.h` | Kernel 层：GM Tensor 构造 + scheduler/mmad 驱动 |
| `include/block/matmul_block_mmad.h` | NO_FULL_LOAD_MODE 特化 + 末尾注入全载特化 |
| `include/block/matmul_block_mmad_a_full_load.h` | A_FULL_LOAD_MODE 特化（A 单缓冲 + abL1LoopCnt_） |
| `include/block/matmul_block_scheduler.h` | Serpentine 调度器，注册 NO_FULL_LOAD / A_FULL_LOAD 两个 mode（B_FULL_LOAD 未注册） |
| `include/tiling/matmul_tiling.h` | `MatmulTilingSwat` + `MatmulTilingAFullLoad`（`MatmulTilingBFullLoad` 未实现） |
| `include/policy/dispatch_policy.h` | `MatmulMultiBlockPolicy<MODE>` dispatch tag（`common_utils.h` 仅定义 NO_FULL_LOAD=0 / A_FULL_LOAD=1） |
| `run.sh` | 一键编译+生成数据+运行+校验；`--skip-build` 跳过编译 |

### 0.5.3 mode 切换：3 行 diff

**切到 A_FULL_LOAD_MODE**：

```diff
 // ① launcher 体内（搜 [MODIFY A1]）
-    using BlockScheduler = MatmulSwatScheduler<NO_FULL_LOAD_MODE>;
-    using DispatchPolicy = MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>;
+    using BlockScheduler = MatmulSwatScheduler<A_FULL_LOAD_MODE>;
+    using DispatchPolicy = MatmulMultiBlockPolicy<A_FULL_LOAD_MODE>;

 // ② host main 内
-    MatmulTilingSwat tilingEngine;
+    MatmulTilingAFullLoad tilingEngine;
```

> ⚠️ **B_FULL_LOAD_MODE 切换 diff 不提供**：B 全载所需的 4 个符号（`B_FULL_LOAD_MODE` 常量、`MatmulMultiBlockPolicy<B_FULL_LOAD_MODE>` 特化、`MatmulSwatScheduler<B_FULL_LOAD_MODE>` selector、`MatmulTilingBFullLoad` 类）**均不在仓库中**——任何"对称 3 行 diff"照搬都会编译失败。需 B 全载时，先按 [`matmul_full_load.md`](matmul_full_load.md) §3.2 / §4.2 / §4.3 镜像 A 全载补齐这些符号，再写 launcher diff。

**A 全载切换共同要求**：
- 锁 `transA=transB=false`
- `run.sh` 改为 `M=128 K=256 N=4096 TRANS_B=false`，锁 `mTailTile=1`（Tiling 引擎自动设 `tilingData.mTailCnt=1`）
- 不满足 L1 容量时 Tiling 直接 throw，提示退回 NO_FULL_LOAD_MODE

## 1. 四层抽象

```
matmul_custom.cpp (Launcher)      ← 选 DispatchPolicy / Scheduler / LayoutB
        │
MatmulKernel<..., BlockMmad, BlockScheduler>  ← 构造 GM Tensor，驱动循环
        │
BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC>
        │                            ← L1 双缓冲 + L0 ping-pong + MMAD 流水
tensor_api (Mmad/CopyGM2L1/CopyL12L0A/CopyL12L0B/CopyL0C2GM)
                                     ← 硬件指令
```

| 层 | 典型改点 |
|---|---|
| Launcher | dtype/byte size、输入输出个数 |
| MatmulKernel | 新增 bias/scale 时追加 GM 视图 |
| BlockMmad | Buffer 容量、stage 数、MMAD Trait |
| tensor_api | 一般不动；切换架构改 `--npu-arch` |

## 2. Cube 内存层次

dav-3510 容量：**L1=512KB / L0A=L0B=64KB / L0C=256KB / BT=4KB**。

```
GM ─CopyGM2L1─▶ L1 ─CopyL12L0A/B─▶ L0A/L0B ─Mmad─▶ L0C ─CopyL0C2GM(fixpipe)─▶ GM
                                           └── L1 ─CopyL12BT─▶ BIAS (可选)
```

关键约束：
- **L0C 累加 dtype**：bf16/fp16/fp8 → fp32 L0C；**int8 → int32 L0C**。模板用 `L0CType = conditional_t<is_same_v<AType, int8_t>, int32_t, float>` 派生
- **BLOCK_CUBE**（C0 维度）：bf16/fp16 = 16，int8/fp8 = 32，fp4 = 64
- **BLOCK_CUBE_L0C**：dav-3510 上**恒为 16**，不要用 `32/sizeof(L0CType)` 推导（fp32 时 = 8，导致 fixpipe stride 减半）

## 3. BlockScheduler：Serpentine 遍历

`mCoreNum_ = min(WINDOW_LEN=4, mCnt_)`。蛇形遍历：偶数行 N 正向，奇数行 N 反向，让相邻行同一核心复用 B 侧 L1。A 全载时 tiling 设 `mCnt=1`，自动退化为 N 顺序遍历。

## 4. 三级 ping-pong 流水

| 级别 | Buffer 数 | 事件类型 | 切换变量 |
|---|---|---|---|
| L1 | `L1_BUFFER_NUM = 2` | `MTE1↔MTE2` | `abL1LoopCnt_ & L1_BUFFER_MASK` |
| L0A/L0B | 2（隐式 ping-pong） | `M↔MTE1` | `l0PingPong_ & 0x1` |
| L0C | `l0cDB ∈ {1, 2}` | fixpipe 内部串行 | `l0cPingPong_ & 1` |

要点：
- **Bias 搬运已融入流水线**：bias GM→L1 跟 BL1、bias L1→BIAS 跟 BL0，不用独立 `PipeBarrier`
- **`mmadCmatrixInitVal`**：仅在 `(iter0==0 && iter1==0)` 为 true
- **L0C ping-pong 两套同步机制不能混用**：`l0cDB==1` 用 `FINAL_ACCUMULATION` unitFlag + 硬件串行；`l0cDB==2` 用 `M_FIX/FIX_M` 三段显式握手。混用 → 多核死锁
- **L0C 半区偏移单位是字节**：`HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT`，**不能再除 sizeof(L0CType)**

### 4.1 `cmatrixInitVal` 时序总览（关键，所有模板共同适用）

`cmatrixInitVal` 控制 L0C 是否在 MMAD 前清零，决定本 tile 的累加是从 0 开始还是叠加在已有 L0C 上。**单 tile 的 K 循环内只有第一拍 = true，其余都 = false**：

| 场景 | 首拍取值 | 后续 K 迭代 | 末次 K 迭代 unitFlag |
|---|---|---|---|
| **纯AIC 单 tile** | `(iter0==0 && iter1==0)` → true | false（累加） | `FINAL_ACCUMULATION` |
| **纯AIC 多 tile（每 tile 独立写回）** | 每个 tile 的 `(iter0==0 && iter1==0)` → true | false | `FINAL_ACCUMULATION` |
| **A/B 全载多 tile** | 同上，**每个对侧 tile 独立累加**，不要因驻留 A/B 而把后续 tile 的首拍写成 false | false | `FINAL_ACCUMULATION` |
| **StreamK K 切分段** | 每段首拍 → true（段间不累加，AIV 端归约） | false | 按 `iter0+iter1` 判末次 |
| **FixpOpti** | 同纯AIC | false | `FINAL_ACCUMULATION` |

**判定公式**：`cmatrixInitVal = (iter0 == 0) && (iter1 == 0)`。`iter0` 是 K-L1 外层循环索引、`iter1` 是 K-L0 内层循环索引。

**常见错误**：
- 全载模式下误以为"第二次对侧 tile 应为 false"（A/B 驻留 L1 不代表 L0C 也驻留）→ 多 tile 大面积 mismatch
- StreamK 段间忘记重置 → L0C 跨段污染
- 首拍漏置 true（默认 false）→ Fixpipe 写回全 0 或叠加上一拍残留

## 5. LayoutA / LayoutB 运行时选择

Launcher 直接传 tensor_api pattern 作为模板参数（不经过 RowMajor/ColumnMajor 中间层）：

| Pattern | 含义 | 构成 GM shape |
|---|---|---|
| `AscendC::Te::NDExtLayoutPtn` | 行主序（ND） | A: (M,K); B: (K,N); C: (M,N) |
| `AscendC::Te::DNExtLayoutPtn` | 列主序（DN） | A: (K,M); B: (N,K) |

`TagToTrans<Pattern>` 在 `layout_utils.h` 派生 transA/transB：NDExt→false，DNExt→true。

**L1 layout 必须与 trans 标志同步**：`transA=false` → AL1=NZ，`transA=true` → AL1=ZN；B 同理。必须用 `conditional_t<transA/ZN/NZ>`，不能硬编码。

**常见错误**：新增 transA=true 但 launcher 里 layoutA 仍硬编码 NDExtLayoutPtn → 编译过但 ≈100% mismatch（K 维和 M 维错位）。

### 5.4 B-NZ 输入变体

B 离线重排为 NZ 分形格式，节省 GM→L1 的 ND→NZ 随路转换带宽。适用：`transB=true` + fp8 + 32/16 对齐。

**两步预防**：
1. `layout_utils.h` 加 `TagToTrans<NZLayoutPtn> { value = true }` 特化
2. baseN 必须 32 对齐（fp8 NZ fractal C0=32），双层保险：host 断言 + tiling `BASIC_BLOCK_SIZE_32`

```python
# gen_data: B (K,N) → NZ (N_blocks, K_blocks, 16, 32)
K0, N0 = 16, 32
b_4d = b.reshape(K//K0, K0, N//N0, N0)
b_nz = b_4d.transpose(2, 0, 1, 3).copy()  # 必须 .copy() 保证物理连续
```

Kernel 端：`MakeLayoutB` / `MakeLayoutBL1` 用 `conditional_t<is_same_v<LayoutB, NZ>, FrameLayoutFormat<NZ, Int<32>>, ...>` 分支。

| b_layout | transB | 实例化 LayoutB |
|---|---|---|
| `nd` | true | `DNExtLayoutPtn` |
| `nd` | false | `NDExtLayoutPtn` |
| `nz` | true | `NZLayoutPtn` |
| `nz` | false | `ZNLayoutPtn` |

## 6. Launcher `<<<>>>` 启动

```cpp
matmul_custom<LayoutB><<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
```

`Params` 聚合初始化字段顺序必须与声明完全一致（错位 → `excess elements in scalar initializer`）：

```cpp
struct Params {
    ProblemShape         problemShape;  // {m, n, k, b=1}
    BlockMmadParams      mmadParams;    // {aGmAddr, bGmAddr, cGmAddr}
    L1Params             l1Params;      // {kL1}
    BlockSchedulerParams schParams;     // {baseM, baseN, mTailTile, nTailTile, mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain} 共 8 字段
    MatmulTiling         qbmmParams;    // {baseM, baseN, baseK, dbL0C}
};
```

新增 tiling 字段时，**同时**在 ① `matmul_tiling_data.h` 加字段 ② `BuildTilingData` 写入 ③ launcher 取出填入聚合结构体。

## 7. 对齐约束速查

| 维度 | 约束 |
|---|---|
| M / N | baseM/baseN 是 16 倍数（tiling 自动兜底） |
| K | `kL1` 是 `baseK` 整数倍；`baseK` ≤ 256（L0A 限制） |
| fp8 BLOCK_CUBE | C0 = **32**（与 bf16/fp16=16 不同） |
| B-NZ | baseN 必须 32 对齐 + K%16==0 + N%32==0 |
| GM 地址 | 1B 对齐即可 |

## 8. 排障速查

> **Developer**：编译不过先检查 §8.1；精度不过先跑单 tile 32×32×32 验证计算逻辑（§8.2），再跑多 tile 暴露同步问题。
> **Reviewer**：审查时重点关注 (1) Params 字段顺序与 kernel 声明是否一致，(2) 全载 mode 下 event 配对是否平衡，(3) `BLOCK_CUBE_L0C` 是否硬编码为 16。

### 8.1 编译期错误

| 报错 | 原因 |
|---|---|
| `excess elements in scalar initializer` | Params 字段顺序/个数与 Kernel 声明不一致 |
| `no matching function for call to MatmulKernelImpl` | 上一条的连锁反应；先修字段顺序 |
| `cannot bind non-const lvalue reference to ... rvalue` | `MakeMemPtr<L1,T>(offset)` 的 offset 未乘 `sizeof(T)`（必须是字节偏移） |

### 8.2 精度问题（matmul 专属）

| 现象 | 根因 |
|---|---|
| fixpipe 写回全 0 | `cmatrixInitVal` 第二次 K 迭代仍为 true；或 unitFlag 未设 `FINAL_ACCUMULATION` |
| ≈100% mismatch，仅 transA/B 某方向触发 | launcher 里 layout 硬编码未跟 trans 标志同步 |
| 精度偏大（rtol 略超 1e-3） | `kL1` 非 `baseK` 整数倍；或首次 `cmatrixInitVal=false` |
| 大 K（>2K）退化 | baseK 过小，bf16 累加误差累积；可 `dbL0c=2` |
| 每个 tile 前 16 列对，后续错 | `BLOCK_CUBE_L0C` 错写成 `32/sizeof(L0CType)` |
| 单 tile PASS，多 tile FAIL | L0C ping-pong 半区重叠（`HALF_L0C_SIZE` 多除了 `sizeof`） |
| B-NZ 路径全错 | gen_data transpose 维度错位；或 baseN 未 32 对齐；或 `TagToTrans<NZ>` 漏特化 |

定位流程：先单 tile（32×32×32）验证计算逻辑，再多 tile（256×256×256）暴露同步/半区/fixpipe 问题。

### 8.3 跑通后快速自检

```bash
bash run.sh 256 256 256          # 默认，流水跑通
bash run.sh 1024 1024 1024       # 大开 K，验证累加
bash run.sh 100 100 100 false false  # 非对齐 + 切 transB
# 检查 output.bin 大小 = m*n*sizeof(CType)
```

## 9. 编译运行

```bash
bash run.sh                              # 默认 M=K=N=256, transA=false, transB=true
bash run.sh 1024 1024 1024               # 指定 M K N
bash run.sh 256 256 256 false false      # 指定 M K N transA transB
bash run.sh --skip-build 256 256 256     # 跳过编译，复用 build/ 产物
```

脚本流程：环境校验 → cmake 编译 → gen_data.py → 上板运行 → verify_result.py。改算子名时 `run.sh::OP_NAME`、kernel 函数名、`CMakeLists.txt::project()/add_executable()` 三处同步。

## 10. 三模板体系与选择

`matmul_custom/` 是本 skill 的**纯 AIC 模板**。Matmul 单算子生成共有 3 种模板，分别对应不同的 AIC/AIV 执行模式：

### 10.1 三模板总览

| 模板 | AIC 职责 | AIV 职责 | 执行模式 | 交付状态 |
|------|---------|---------|---------|---------|
| **纯AIC** (`matmul_custom.cpp`) | 搬运+计算+写回 全流程 | 直接 `return`（不参与） | `__aicore__ __cube__`，AIV 空转 | **已交付** |
| **StreamK** | K-split MMA + workspace 写入 | workspace 归约 + cast + 写回 | AIC+AIV 混合，CrossCore 同步 | **设计文档** |
| **FixpOpti** (`matmul_fixpopti.cpp`) | MMA + Fixpipe L0C→UB | epilogue 处理 + MTE3 写回 | `__mix__(1, 2)`，AIC+AIV + CV 同步 | **已交付** |

```
纯AIC:                       StreamK:                       FixpOpti:
┌──────┐                    ┌──────┐   ┌──────┐           ┌──────┐   ┌──────┐
│ AIC  │ GM→L1→L0→MMAD      │ AIC  │   │ AIV  │           │ AIC  │   │ AIV  │
│      │ →L0C→Fixpipe→GM    │MMA(K │   │ work │           │MMA+  │   │ MT3  │
│      │                    │段)+  │──▶│space │           │Fixpipe│──▶│epilog│
├──────┤                    │ws写  │   │归约+ │           │L0C→UB│   │→GM   │
│ AIV  │ return             │入    │   │cast  │           │      │   │      │
└──────┘                    └──────┘   └──────┘           └──────┘   └──────┘
```

> **注意**：StreamK 和 FixpOpti 的 "混合" 指 **Matmul 计算本身由 AIC 和 AIV 协作完成**（K 分段归约、Fixpipe→MT3 分流），而非 Cube 之后追加激活/残差/scale 等 Vector 后处理。激活融合等场景需要在相应模板的 epilogue 中自行扩展。

### 10.2 模板选择决策

```
Matmul 单算子需求
    │
    ├── 通用场景（默认）？
    │       └── 纯AIC (matmul_custom/) — 适用大多数 shape
    │
    ├── MN 欠并行（mCnt·nCnt < aicNum）+ 长 K（≥4096）？
    │       └── StreamK — K 维切分填满空闲核
    │
    └── 需要 AIC/AIV 流水重叠、epilogue 可定制？
            └── FixpOpti — Fixpipe→UB 后由 AIV MT3 写回
```

### 10.3 模板核心差异

| 维度 | 纯AIC | StreamK | FixpOpti |
|------|-------|---------|----------|
| **Kernel 属性** | `__aicore__ __cube__` | `__aicore__` (MIX) | `__mix__(1, 2)` |
| **Launcher** | `matmul_custom.cpp` | (未交付) | `matmul_fixpopti.cpp` |
| **AIV 行为** | `if ASCEND_IS_AIV return` | 归约循环 | epilogue + MTE3 |
| **调度器** | `MatmulSwatScheduler` | `BlockSchedulerStreamK` | `MatmulSwatScheduler` |
| **同步类型** | MTE1/MTE2/M/Fixpipe 事件 | + `CrossCoreSetFlag(AIC→AIV)` | + `CrossCoreSetFlag(AIC→AIV) + Wait(AIV→AIC)` |
| **输出路径** | AIC Fixpipe→GM | AIC→workspace(FP32)→AIV 归约→GM | AIC Fixpipe→UB→AIV MT3→GM |
| **Workspace** | 不需要 | **必需**（~3MB+） | 不需要（直接用 UB 做中间层） |
| **基础代码量** | ~250 行 launcher + ~600 行 kernel | + scheduler + epilogue + tiling 变更 | + epilogue 类 + CV sync（已交付） |
| **交付状态** | **已交付** | **设计文档** | **已交付** |
| **参考文档** | 本文 + `matmul_basic.md` | `matmul_streamk.md` + `streamk_design.md` | `matmul_fixpopti.md` |

> StreamK 详细设计见 `references/matmul_streamk.md` 以及 skill `ascendc-performance-best-practices` 的 `reference/matmul/streamk_design.md`。
> FixpOpti 详细设计见 `references/matmul_fixpopti.md`。

### 10.5 性能预期速查

| 优化 | 预期收益 | 关键前提 |
|------|---------|---------|
| A_FULL_LOAD / B_FULL_LOAD | Task 总时间 **−5%~−15%**（典型 −7.5%，实测 `[128,4096,81920]` MXFP4） | 对侧循环 T≥2 + 真 MTE2 bound |
| StreamK | 核利用率从 `mCnt·nCnt/aicNum` 提升到接近 100% | K≥4096 + workspace 开销可接受 |
| FixpOpti | MTE3 写回延迟被 AIC 计算隐藏 | 多 tile 场景 + MTE3 是瓶颈 |
| L1 stage 2→4 | MTE2/CUBE overlap 提升 | L1 容量充足（半区缩半） |

> 详细数据见 `matmul_full_load.md` 和 skill `ascendc-performance-best-practices` 的 `reference/matmul/fullload_design.md` §7。

### 10.6 模板间迁移对照

| 迁出 → 迁入 | 关键改造 | 工作量估计 |
|-------------|---------|----------|
| 纯AIC → A_FULL_LOAD | launcher 3 行 diff + 锁 transA/B + 锁 mTailTile=1 + run.sh 改默认 shape | 5 分钟 |
| 纯AIC → B_FULL_LOAD | **代码全未交付**：先补 `B_FULL_LOAD_MODE` 常量 + dispatch/scheduler 特化 + `MatmulTilingBFullLoad` + `matmul_block_mmad_b_full_load.h`（~300 行对称代码，按 `matmul_full_load.md` §3.2/§4.2/§4.3 镜像），再做 launcher 切换 | 半天～1 天 |
| 纯AIC → StreamK | 替换 scheduler + tiling 引擎 + Kernel 改 AIC/AIV 统一循环 + 新增 epilogue + host 侧分配 workspace | 数小时 |
| 纯AIC → FixpOpti | 切换 launcher 至 `matmul_fixpopti.cpp`（`MatmulKernelFused` + `IdentityEpilogue` + CV 同步均已交付），按 `matmul_fixpopti.md` §3 改造 | 5 分钟（基底切换）+ 自定义 Epilogue 视复杂度而定 |
| 纯AIC → L1 stage 4 | 改 `L1_BUFFER_NUM` + dispatch_policy 新增 STAGES_ 模板参数 + 重算 L1 偏移 + tiling 缩小 kL1 | 30 分钟 |

> StreamK 当前仅有设计文档（FixpOpti 已交付），迁移前先通读 `matmul_streamk.md` 全部陷阱表。

## 参考

- 通用模板深度（SWAT Tiling + 改造）：[`matmul_basic.md`](matmul_basic.md)
- A 全载深度：[`matmul_full_load.md`](matmul_full_load.md)
- tensor_api 手册：`tensor_api_user_guide.md`
