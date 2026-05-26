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
| E1 | 启动器 `.cpp` | `IdentityEpilogue<CType>` → 自定义 Epilogue 类 |

### 3.5 BlockMmad Fixpipe 改造（必做）

纯AIC 的 `CopyL0C2GM` 将 L0C 写到 GM，AIV 无法访问。FixpOpti 必须改为 `CopyL0C2UB` 将 L0C 写到 UB，AIV 的 Epilogue 才能读取。

**文件**：`include/block/matmul_block_mmad.h`（NO_FULL_LOAD_MODE）

**原代码**（纯AIC L0C→GM）：
```cpp
        // L0C -> GM，由 fixpipe 完成 L0C(fp32/int32) -> CType 的量化/cast。
        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(CopyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
```

**替换为**（FixpOpti L0C→UB + SPLIT_M Trait）：
```cpp
        // FixpOpti: L0C→UB via Te::Copy(CopyL0C2UB{}) + 自定义 SPLIT_M Trait。
        (void)gmC;
        auto curMPad = (curM + 1L) & ~1L;
        constexpr int64_t UB_N_ALIGN_ELEM = 32L / static_cast<int64_t>(sizeof(L0CType));
        auto curNUbAlign = ((curN + UB_N_ALIGN_ELEM - 1L) / UB_N_ALIGN_ELEM) * UB_N_ALIGN_ELEM;

        auto layoutUB = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<BLOCK_CUBE_L0C>>(curMPad, curNUbAlign);
        auto ubTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(0), layoutUB);

        auto copyOp = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{}, CopyL0C2UBSplitMTrait{});
        AscendC::Te::Copy(copyOp, ubTensor, tensorL0C,
            AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
```

**并在文件头部 `namespace Block {` 之后插入 Trait 声明**：
```cpp
struct CopyL0C2UBSplitMTrait {
    using TraitType = AscendC::Te::CopyL0C2UBTrait;
    static constexpr const TraitType value{
        AscendC::Te::RoundMode::DEFAULT,
        false,
        false,
        AscendC::Te::DualDstMode::DUAL_DST_SPLIT_M
    };
};
```

`DUAL_DST_SPLIT_M` 是非 fp32 输出走 SPLIT_M 的关键：fp32 输出走 DUAL_DST 单指令广播，非 fp32 必须拆为 2 条 fixpipe 指令分别路由到 AIV0/AIV1（陷阱 P5）。

**同样改造 `include/block/matmul_block_mmad_a_full_load.h`**（A_FULL_LOAD_MODE 特化），Trait 声明复用 `matmul_block_mmad.h` 中的定义（A_FULL_LOAD 文件被前者 include）。

### 3.6 新增文件（复制即用，无需修改）

以下文件已存在于 `references/matmul_custom/`，按 §3.1 复制基底后直接可用：

| 文件 | 用途 |
|------|------|
| `matmul_fixpopti.cpp` | FixpOpti 启动器（`__mix__(1, 2)`，AIC+AIV + CV 同步） |
| `include/kernel/matmul_kernel_fused.h` | AIC+AIV 统一循环驱动 |
| `include/epilogue/identity_epilogue.h` | float→bf16 Cast + DataCopyPad + SetFlag |
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
| BlockMmad | `CopyL0C2GM` | `CopyL0C2UB` + SPLIT_M Trait |
| CV 同步 | 无 | CrossCoreSetFlag/WaitFlag |
| 新增文件 | — | fused kernel、epilogue × 2 |
| 共享文件 | common/ tiling/ scheduler/ utils/ | 完全复用 |
