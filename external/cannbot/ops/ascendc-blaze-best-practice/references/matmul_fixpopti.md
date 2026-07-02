# Matmul FixpOpti 模板（AIC+AIV 混合 Fixpipe→UB→MTE3）

> **适用架构**：DAV_3510
>
> **基底模板**：`references/matmul_custom/`（所有文件从此出）
>
> 本模板描述如何从纯AIC 基底生成 FixpOpti 算子工程。前置阅读：[`matmul_pattern.md`](matmul_pattern.md) §10。

## 1. 适用判据

| 条件 | 说明 |
|------|------|
| 需要 Pipeline overlap | MTE3 写回延迟被 AIC 计算隐藏 |
| 多 tile 场景 | 单个 tile 无 overlap 余地 |
| epilogue 可定制 | AIV 侧可插入 cast/quant/格式转换 |

## 2. 架构与数据流

```
AIC 核                                AIV 核
GM → L1 → L0 → MMAD → L0C
                         │
                    Fixpipe L0C→UB (SPLIT_M)
                         │
              CrossCoreSetFlag ────────▶ WaitFlag(AIC→AIV)
                         │                │
                         │          Epilogue: Cast + DataCopyPad → GM
                         │                │
              ◀─────── CrossCoreSetFlag ──┘ (AIV→AIC 背压)
```

## 3. 生成步骤

### 3.1 复制基底

```bash
cp -r references/matmul_custom <your_project> && cd <your_project>
```

### 3.2 必改 [N]

| # | 文件 | 操作 |
|---|------|------|
| N1 | `.cpp` 启动器 | 用 `matmul_fixpopti.cpp` 替换 `matmul_custom.cpp`，全局替换工程名 |
| | `CMakeLists.txt` | `matmul_custom` → 目标名 |
| | `run.sh` | `OP_NAME` → 目标名；`TRANS_B` 默认改为 `false`（与 FixpOpti 启动器 NN 默认一致） |
| N2 | 启动器 `.cpp` | 修改 `AType/BType/CType`、`sizeA/B/C` 的 `sizeof` |
| | `include/utils/matmul_tiling_constant.h` | 修改 `DATA_SIZE_FP16` |
| N3 | `scripts/gen_data.py` | 按需改 dtype / golden / 容差 |

### 3.3 常改 [C]

| # | 文件 | 操作 |
|---|------|------|
| C1 | 启动器 `.cpp` | 修改 `transA/transB` 默认值、`LayoutA/LayoutB` 模板实例化 |
| C2 | `include/tiling/matmul_tiling_data.h` + 启动器 | 增删 TilingData / Params 字段 |

### 3.4 选改 [A]

| # | 文件 | 操作 |
|---|------|------|
| A1 | 启动器 `.cpp` | `NO_FULL_LOAD_MODE` → `A_FULL_LOAD_MODE`；`MatmulTilingSwat` → `MatmulTilingAFullLoad` |
| A2 | 启动器 `.cpp` | NZ/ZN 分形预重排输入（A-NZ / B-NZ），GM→L1 走块拷贝省掉格式转换带宽。改造步骤详见 [`matmul_pattern.md`](matmul_pattern.md) §5.4 |
| E1 | 启动器 `.cpp` | `IdentityEpilogue<CType>` → 自定义 Epilogue 类。**推荐 RegBase 路径**：<br>- **RegBase（推荐）**：`epilogue_fusion_regbase.h` 模板，用 `__VEC_SCOPE__` + `Reg::Cast/Mul/Exp` 等 RegTensor API，详见 [`matmul_fixpopti_regbase_epilogue.md`](matmul_fixpopti_regbase_epilogue.md)<br>- **MemBase（仅限简单场景）**：`epilogue_fusion_membase.h` 参考样例，仅当 vector 操作为单个 AscendC API 调用（如 `AscendC::Mul/Add/Div`）时使用 |

### 3.5 BlockMmad 输出 Tensor location（已内置）

纯AIC direct 路径需要 `CopyL0C2GM` 将 L0C 写回 GM；FixpOpti/Fusion 路径需要
`CopyL0C2UB` 将 L0C 写到 UB，AIV 的 Epilogue 才能读取。这个差异已经内置在共享
`BlockMmad` 中，通过传入输出 Tensor 的 location 编译期分流：

| 路径 | 输出 Tensor | BlockMmad 终端输出 |
|------|-----------------|--------------------|
| 纯AIC direct | GM Tensor | `CopyL0C2GM` |
| FixpOpti/Fusion | UB Tensor | `CopyL0C2UB` + SPLIT_M Trait |

因此生成 FixpOpti 算子时使用 `MatmulKernelFused`，由 fused kernel 构造 UB Tensor
传给 `BlockMmad`。不要复制或回改 `include/block/matmul_block_mmad*.h`，也不要给
`BlockMmad` 增加模板参数或给 `DispatchPolicy` 增加输出模式。

**NO_FULL_LOAD FixpOpti 默认写法**：

```cpp
using BlockScheduler = MatmulSwatScheduler<NO_FULL_LOAD_MODE>;
using DispatchPolicy = MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>;
```

**A_FULL_LOAD FixpOpti 写法**：

```cpp
using BlockScheduler = MatmulSwatScheduler<A_FULL_LOAD_MODE>;
using DispatchPolicy = MatmulMultiBlockPolicy<A_FULL_LOAD_MODE>;
// host main 同步使用 MatmulTilingAFullLoad
```

`DUAL_DST_SPLIT_M` 是非 fp32 输出走 SPLIT_M 的关键：fp32 输出走 DUAL_DST 单指令广播，非 fp32 必须拆为 2 条 fixpipe 指令分别路由到 AIV0/AIV1（陷阱 P5）。

### 3.6 新增文件（复制即用，无需修改）

以下文件已存在于 `references/matmul_custom/`，按 §3.1 复制基底后直接可用：

| 文件 | 用途 |
|------|------|
| `matmul_fixpopti.cpp` | FixpOpti 启动器（`__mix__(1, 2)`，AIC+AIV + CV 同步） |
| `include/kernel/matmul_kernel_fused.h` | AIC+AIV 统一循环驱动 |
| `include/epilogue/identity_epilogue.h` | float→bf16 Cast + DataCopyPad + SetFlag |
| `include/epilogue/epilogue_fusion_regbase.h` | RegBase 后融合框架（推荐） |
| `include/epilogue/epilogue_fusion_membase.h` | MemBase 后融合参考样例（简单 vector 场景） |
| `include/epilogue/cv_sync_constants.h` | CV Flag 常量 |

## 4. 常见陷阱

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| P1 | AIC hang | AIC 忘了 drain 末轮 WaitFlag | 循环结束后补 WaitFlag(AIV→AIC) |
| P2 | AIV hang | 空闲核未发 flag | 空闲核 return 前补 CrossCoreSetFlag |
| P3 | 写回数据错乱 | AIC 覆写 AIV 未读完的 UB | CV sync count 错开 |
| P4 | 精度偏差 | unitFlag 未设 FINAL_ACCUMULATION | 末次 Fixpipe 必须 FINAL_ACCUMULATION |
| P5 | 仅一半 rows 正确 | SPLIT_M 时 UB 保持 float，cast 由 Epilogue 完成 | 非 fp32 输出必须 SPLIT_M + Epilogue Cast |
| P6 | AIV1 数据全为零 | 误以为 AIV 共享 UB，给 AIV1 加了行偏移 | SPLIT_M 下每个 AIV 有独立 UB 空间，offset=0 |
| P7 | 非对齐 N 大面积错误 | bf16 DataCopyPad 需 16 元素行对齐，而非 float 的 8 | `nAlignBf16 = ceil(N, 16)*16`；逐行 Cast |
| P8 | `MakeCopy` trait 编译错误 | 直接传 `CopyL0C2UBTrait` 缺少 `TraitType` | 用自定义 struct 包装（参考 `CopyL0C2UBTraitDefault`） |
| P9 | `PIPE_FIX`/`PIPE_V` 编译错误 | 这些在全局作用域 | 去掉 `AscendC::` 前缀 |

## 5. 与纯AIC 差异速查

| 维度 | 纯AIC | FixpOpti |
|------|-------|----------|
| 启动器 | `matmul_custom.cpp` | `matmul_fixpopti.cpp` |
| Kernel 属性 | `__cube__` | `__mix__(1, 2)` |
| Kernel 模板 | `MatmulKernel` | `MatmulKernelFused` |
| AIV 行为 | `return` | Epilogue 循环 |
| BlockMmad | kernel 传 GM Tensor，写 GM | kernel 传 UB Tensor，写 UB |
| CV 同步 | 无 | CrossCoreSetFlag/WaitFlag |
| 新增文件 | — | fused kernel、epilogue × 4（identity + cv_sync + regbase + membase） |
| 共享文件 | common/ tiling/ scheduler/ utils/ block/ | 完全复用 |
