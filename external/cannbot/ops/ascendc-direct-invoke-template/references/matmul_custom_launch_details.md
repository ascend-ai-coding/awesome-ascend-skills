# Ascend C Matmul Kernel 直调进阶

> 基础用法和修改指南见 `matmul_kernel/matmul_custom.cpp` 中的 `[MODIFY]` 注释。本文档介绍 Matmul 直调工程特有的分层架构、Tiling/调度机制与常见改造点。与 `kernel_launch_details.md` 的通用内容互为补充，本文只聚焦 Cube/Matmul 相关内容。

## 1. Matmul Kernel 的四层抽象

相比逐元素 vector 算子，matmul 直调工程采用**分层模板 + SFINAE dispatch** 的结构，自顶向下 4 层：

```
matmul_custom.cpp (Launcher)
        │  ① 选择 DispatchPolicy / Scheduler / LayoutB
        ▼
Kernel::MatmulKernel<ProblemShape, BlockMmad, BlockScheduler>
        │  ② 组装 GM Tensor，驱动 scheduler 循环
        ▼
Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC>
        │  ③ 单 block 的 L1 双缓冲 + L0 ping-pong + MMAD 流水
        ▼
tensor_api::Mad / CopyGM2L1 / CopyL12L0 / CopyL0C2GM
            ④ 硬件指令封装（mmad / load_data / fixpipe）
```

| 层 | 职责 | 典型改点 |
|----|------|---------|
| Launcher | 运行时选 `LayoutB`、分配 HBM、准备 TilingData、`<<<>>>` 启动 | dtype / byte size、输入输出个数 |
| MatmulKernel | 构造 `gmA/gmB/gmC` Tensor 视图、驱动 scheduler 与 mmad | 新增 bias/scale 时在此处追加 GM 视图 |
| BlockMmad | L1/L0 buffer 布局、双缓冲事件、K 循环 | Buffer 容量、stage 数、MMAD Trait |
| tensor_api | 架构相关硬件 API（dav-3510 等） | 一般不动；切换架构改 `--npu-arch` |

## 2. Cube 专属内存层次

Matmul 使用 **L1/L0A/L0B/L0C** 四级 Cube 缓存，与 vector 算子的 UB 完全独立：

```
┌───────────────────────────────────────────────┐
│  Global Memory (HBM)        A / B / C          │  大容量，高延迟
├───────────────────────────────────────────────┤
│  L1  (Cube L1)              512 KB (dav-3510)  │  GM→L1 搬运 buffer（ping-pong）
├───────────────────────────────────────────────┤
│  L0A / L0B                  64 KB  / 64 KB     │  MMAD 的矩阵输入（Zn/Nz 格式）
├───────────────────────────────────────────────┤
│  L0C                        128 KB             │  fp32 / int32 累加寄存器
└───────────────────────────────────────────────┘
```

数据流：

```
GM ──CopyGM2L1──▶ L1 ──CopyL12L0──▶ L0A/L0B ──Mad──▶ L0C ──CopyL0C2GM──▶ GM
                                               (fp32/int32 累加) (fixpipe 自动 cast)
```

关键事实：
- **L0C 累加 dtype 由输入决定**：bf16/fp16/fp8 → fp32 L0C；**int8 → int32 L0C**。硬件静态检查（`check_data_type_3510.h`）拒绝错误组合。模板里用 `L0CType` 通过 `conditional_t<is_same_v<AType, int8_t>, int32_t, float>` 派生。
- C 侧的 dtype 只影响 fixpipe 写回 GM 时的 `quantPre`（bf16/fp16/fp32/int32 自动选择）。
- **BLOCK_CUBE**（C0 维度颗粒度）随 dtype 变化：bf16/fp16 = 16，int8/fp8 = 32，fp4 = 64。
- **L1 半区**：`L1_SIZE/2` 每半区放一组 `A|B`，两组构成 `L1_BUFFER_NUM = 2` 的 ping-pong。

## 3. Tiling 策略：SWAT

`MatmulTilingSwat`（Streaming With A/B Tile，non-full-load）是 bf16/fp16 matmul 的默认 tiling 引擎。执行 4 步：

| 步骤 | 作用 |
|------|------|
| `FormulateBasicBlock` | 以 256×256 基块起步，按 `mCore*nCore ≥ aicNum` 做 N/M 轴分裂，得到初版 `baseM / baseN` |
| `OptimizeEdgeBasicBlock` | 合并尾块，消除最后一行/列负载不均 |
| `CalcTailBasicBlock` | 剩余块数 < AIC 数时把尾块 `mTailTile / nTailTile` 切分，提高占核率 |
| `CalL1Tiling` | 按 L1 容量反推 `kL1 / depthA1 / depthB1`，保证 `baseM*kL1 + baseN*kL1 ≤ L1_SIZE/2` |

输出落入 `MatmulTilingData`（见 `include/tiling/matmul_tiling_data.h`），关键字段：

```
baseM / baseN / baseK         —— L0 切分颗粒度
kL1                           —— L1 上 K 方向窗口
mTailCnt / nTailCnt           —— 尾块 split 次数
mBaseTailSplitCnt / nBaseTailSplitCnt / mTailMain / nTailMain
                              —— 尾块合并/均衡参数
usedCoreNum                   —— <<< blockDim, ... >>> 的第一参数
l1BufferNum / l0cDB           —— L1/L0C 流水级数
```

> [MODIFY] 切换数据类型：`matmul_tiling.h` 内按 `DATA_SIZE_FP16 = 2` 做 L1 预算。切 fp8 (=1)、fp32 (=4)、fp4×2 (=0.5) 时需新增分支或调整 byte 常量。

## 4. BlockScheduler：Serpentine 遍历

`MatmulBlockScheduler` 把 `[mCnt][nCnt]` 的 block 格分配到各 AIC。默认使用 **serpentine 行遍历**（偶数行正向、奇数行反向），相比 row-major 可以：

```
mCoreNum=4, nCnt=6 (一行 4 个核服务 6 列)
Row 0 (→) : core0(b0,b1), core1(b2), core2(b3), core3(b4,b5)
Row 1 (←) : core3(b5,b4), core2(b3), core1(b2), core0(b1,b0)
            ↑ nTileIdx 倒序，使两行相邻 core 的 N 坐标复用
```
**优点**：相邻行的同一 core 重用上一轮的 B 侧 L1 数据，减少 GM→L1 搬运。

**何时改成 row-major**：调度策略不影响正确性，只影响 cache 复用效率。把 `GetTileIdx` 内的 serpentine 反转段（`if (rowIdx & 1) { nTileIdx = nCnt_ - 1 - nTileIdx; }`）删掉即可退化为顺序平铺。

## 5. BlockMmad 流水：L1/L0/L0C 三级 ping-pong

```
K 循环 (kL1 窗口) ──┐
                    │
GM ─CopyGM2L1(A)─▶ L1.A[buf_i] ┐
GM ─CopyGM2L1(B)─▶ L1.B[buf_i] ┤  ping-pong on L1_BUFFER_NUM
                               │
K 子循环 (baseK 切片)          │
L1.A[buf_i] ─CopyL12L0─▶ L0A[pp]  ┐
L1.B[buf_i] ─CopyL12L0─▶ L0B[pp]  ┤  ping-pong on L0
                                  │
L0A[pp] × L0B[pp] ─Mad─▶ L0C[pp]  ┤  ping-pong on l0cDB
                                  │
(最后一次 baseK) ─CopyL0C2GM─▶ GM  ┘  fixpipe 写回 + 自动 quantPre
```

- **L1_BUFFER_NUM = 2**：每半个 L1 放 `(A|B)` 一组，2 组双缓冲。调成 4 需要在 dispatch policy 加 `STAGES_` 模板参，并重排 `l1BufferAOffset_ / l1BufferBOffset_`。
- **L0 ping-pong**：`EVENT_ID0/EVENT_ID1` 在 `MTE1↔M` 事件上切换，掩盖 load_data 延迟。
- **L0C ping-pong**（`l0cDB=2`）：两片 L0C 交替累加→fixpipe，掩盖 fixpipe 回写延迟。

## 6. LayoutA / LayoutB 运行时选择

A/B 两侧的 layout 必须是**编译期模板参数**（`CopyL12L0` 在 `NzLayout` / `ZnLayout` 之间静态分派），但 host 侧通常按 `transA/transB` 动态切换。套路：**launcher 按 4 种组合实例化 kernel 模板**。

```cpp
template <class LayoutA, class LayoutB> __global__ __aicore__ __cube__ void matmul_xxx(...);

if (transA && transB)        matmul_xxx<layout::ColumnMajor, layout::ColumnMajor><<<...>>>(...);
else if (!transA && transB)  matmul_xxx<layout::RowMajor,    layout::ColumnMajor><<<...>>>(...);
else if (transA && !transB)  matmul_xxx<layout::ColumnMajor, layout::RowMajor   ><<<...>>>(...);
else                         matmul_xxx<layout::RowMajor,    layout::RowMajor   ><<<...>>>(...);
```

`MatmulKernel` 内对 A/B **对称地**用 `std::conditional_t` 把 Layout tag 映射到 tensor_api format：

```cpp
using MakeLayoutA = std::conditional_t<
    std::is_same_v<LayoutA, layout::ColumnMajor>,
    AscendC::Te::DNLayoutFormat<AType>,   // host 文件按 (K, M) 落盘，device 逻辑 (M, K)
    AscendC::Te::NDLayoutFormat<AType>>;  // host 文件按 (M, K) 落盘
using MakeLayoutB = std::conditional_t<
    std::is_same_v<LayoutB, layout::ColumnMajor>,
    AscendC::Te::DNLayoutFormat<BType>,   // host (N, K) → device 逻辑 (K, N)
    AscendC::Te::NDLayoutFormat<BType>>;  // host (K, N)
```

> **常见错误**：只对 LayoutB 做 conditional_t 映射、却把 LayoutA 硬编码成 `NDLayoutFormat`。这种模板编译时不会报错，但 transA=true 时跑出来 ≈100% mismatch（K 维和 M 维错位）。如果新算子需要支持 transA，必须**对 LayoutA 同样做 conditional_t**。

## 7. Launcher 的 `<<<>>>` 启动要点

```cpp
matmul_custom<LayoutB><<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
```

- **第一参数**：`usedCoreNum`（混合核工程通常再乘以 `GetTaskRation()`，纯 Cube 场景直接用）。
- **第二参数**：`nullptr`（tpipe，Matmul 直调不需要）。
- **第三参数**：`stream`（ACL stream）。
- **tilingData** 按值传递。kernel 内用 `BlockSchedulerParams{td.baseM, td.baseN, td.mTailCnt, ...}` 组装。

`MatmulKernel::Params` 是聚合初始化，顺序必须与结构体内字段匹配：
```
Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;     // {aAddr, bAddr, cAddr}
    L1Params       l1Params;        // {kL1}
    BlockSchedulerParams schParams; // {baseM, baseN, mTailTile, nTailTile, ...}
    MatmulTiling   qbmmParams;      // {baseM, baseN, baseK, dbL0C}
};
```
改名 / 乱序会直接触发 `excess elements in scalar initializer`。

## 8. 常见改造

### 8.1 切换 dtype（bf16 → fp16 / fp32 / fp8 / int8）

| 位置 | 改动 |
|------|------|
| Launcher | `AType/BType/CType`、`sizeA/B/C` 字节数 |
| gen_data.py | 输入 dtype；golden 在高精度上算后 cast |
| verify_result.py | 读取 dtype 与 `ERROR_TOL` 容差（int8→int32 应 bit-exact 比对） |
| BlockMmad | `BLOCK_CUBE`（bf16/fp16=16, int8/fp8=32, fp4=64）；`L0CType` 选择（int8 in→int32 acc，否则 fp32） |
| tiling | `DATA_SIZE_FP16` 分支或新增常量（按"输入字节数"语义） |

> **L0C 累加 dtype**：硬件静态检查（`check_data_type_3510.h`）拒绝 `int8 → fp32 L0C` 与 `fp16 → int32 L0C` 等错误组合。bf16/fp16/fp8 使用 fp32 L0C；**int8 必须使用 int32 L0C**。模板中 `BlockMmad::L0CType = conditional_t<is_same_v<AType, int8_t>, int32_t, float>`。同时 `MakeL0CmemPtr<L0CType>` 与 `HALF_L0C_SIZE` 的字节单位需保持一致。

### 8.2 增加 Bias / Scale 输入

1. `MatmulTilingData` 里加 `biasAddr / scaleAddr`（或直接在 kernel 签名加 `GM_ADDR`）。
2. `BlockMmad::Params` 追加字段；`MatmulKernel::ResetGmAddr` 里映射。
3. `CopyL0C2GM` 的 `FixpipeParams` 打开 `biasFlag` 或 `quantPreScaleFlag`。
4. L1 预留 `BIAS_TABLE_NUM` 空间（默认 0，见 `matmul_constant.h`）。

### 8.3 新增 MX/量化 Trait

参考 `ascendc-direct-invoke-template/references/matmul_kernel/include/block/matmul_block_mmad.h` 里的 SFINAE 头：

```cpp
template <...>
class BlockMmad<
    ..., std::enable_if_t<std::is_base_of_v<
        QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE>, DispatchPolicy_>>>
{ ... };
```

增加新 dispatch tag（`policy/dispatch_policy.h`）→ 新增 BlockMmad SFINAE 特化 → 在 launcher 切换 DispatchPolicy 即可并存多条计算路径，模板层自动分派。

### 8.4 调大 L1 stage 数（2 → 4）

1. `policy/dispatch_policy.h` 加 `STAGES_` 模板参数：`MatmulMultiBlockPolicy<FULL_LOAD, STAGES>`。
2. BlockMmad 把 `L1_BUFFER_NUM` 改为 `DispatchPolicy::stages`。
3. 重算 `l1BufferAOffset_/l1BufferBOffset_`（4 个半区 interleave，而不是 2 组）。
4. Tiling 需要缩小 `kL1`（半区容量减半）。

## 9. 尺寸 / 对齐约束速查

| 维度 | 约束 |
|------|------|
| M / N | baseM / baseN 通常是 16 的倍数（BLOCK_CUBE=16）；tiling 自动兜底 |
| K | `kL1` 是 `baseK` 的整数倍；`baseK` ≤ 256（dav-3510 L0A 限制） |
| A 地址 | GM 地址 1B 对齐即可 |
| B 地址 | 同上 |
| C 地址 | GM 地址 1B 对齐即可 |
| Mad m/k/n | 非 mx 场景直接传实际值，无需 16 对齐（mx 场景另有 BLOCK_CUBE 颗粒度要求） |

## 10. 编译期 / 运行期排障

| 报错 | 原因 |
|------|------|
| `excess elements in scalar initializer` | `Params` 字段顺序/名称与 Kernel 声明不一致 |
| `a type specifier is required for all declarations`（构造函数处）| BlockScheduler 构造函数名没有和类名一致，或 clone 模板后忘记改名 |
| `no matching function for call to MatmulKernelImpl` | `Params` 构造失败后被推断成 `int`，连锁错误；先修上面两项 |
| `static_assert ... The data type is not supported for L0C position` | L0C dtype 与 A/B dtype 不匹配。int8 输入必须 L0C=int32；bf16/fp16/fp8 输入用 L0C=fp32。改 `L0CType` 与 `MakeL0CmemPtr<L0CType>` |
| `static_assert ... The data type is not supported`（fixpipe L0C2GM）| `CType` 与 L0CType 组合非法（如 `__gm__ int + __cc__ float`）。同步上一条 |
| fixpipe 写回全 0 | `CType` 设成 `float` 但 L0C 路径仍按 bf16；或 `FixpipeParams` 未设 `FINAL_ACCUMULATION` |
| 精度大面积 mismatch（≈100%）且仅 transA=true 触发 | `MakeLayoutA` 没有对 `LayoutA` 做 conditional_t 分派，仍硬编码 `NDLayoutFormat`。见 §6 |
| 精度偏大 | `kL1` 窗口不是 `baseK` 倍数 / Mmad 的 `initVal` 没在首次累加置 true |

## 参考

- tensor_api 硬件封装：`references/matmul_kernel/third_party/tensor_api/impl/arch/npu_arch_3510/`
- 一般通用优化（Double Buffer / PipeBarrier / TILE_LENGTH）见同目录 `kernel_launch_details.md`
