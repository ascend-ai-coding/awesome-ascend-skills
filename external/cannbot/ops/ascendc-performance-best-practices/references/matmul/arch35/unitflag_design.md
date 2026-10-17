# matmul UnitFlag（MMAD 与 Fixpipe 细粒度流水并行）优化设计

## 1. 优化目标

将 matmul 的 **MMAD 计算** 与 **FIXPIPE 搬出** 从「整段完成依赖」（FIXPIPE 须等 MMAD 全部执行完才开始搬出 L0C）改为 **512B 粒度的细粒度流水并行**：MMAD 每计算完一个 512B 数据块，FIXPIPE 立即搬出该块。在 **无法开启 L0C Double Buffer** 的条件下（为充分发挥计算访存比，L0C 通常被完整输出基本块占满）提高计算与搬出的并行度。

适用 Bound 类型：**FIXPIPE Bound**（L0C→GM 搬出成为瓶颈）或 MMAD 与 FIXPIPE 串行执行造成流水空洞的场景。

> UnitFlag 是硬件指令级开关：通过合理配置 MMAD 和 Fixpipe 指令参数（`unitFlag` 字段），系统实现计算与搬运的流水线并行，无需额外 buffer 空间。

## 2. 架构概览

### 2.1 未优化路径（整段完成依赖）

```text
MMAD:    [=========== 全部 K 轮累加 ===========]
FIXPIPE:                                        [==== L0C → GM ====]   ← 完全串行
```

L0C 单份被完整输出块占满（无 Double Buffer 空间），`M_FIX` 事件只能做整段同步：MMAD 全部完成后 FIXPIPE 才能启动。

### 2.2 优化路径（512B 粒度同步）

```text
MMAD:    [块0][块1][块2][块3]...[尾块=FINAL_ACCUMULATION]
FIXPIPE:      [搬出块0]  [搬出块1]  [搬出块2] ...        ← 每块 512B 就绪即搬
```

`unitFlag` 取值语义：

| 值 | 常量名 | 含义 |
|----|--------|------|
| 0 | `UNITFLAG_DISABLE` | 不开启 unit_flag（默认） |
| 2 | `NO_FINAL_ACCUMULATION` | 非尾轮块：FIXPIPE 可提前搬出该 512B 块，但 L0C 累加尚未结束（结果未最终化） |
| 3 | `FINAL_ACCUMULATION` | 尾轮块：该块为最终累加结果，FIXPIPE 搬出后即最终输出 |

### 2.3 事件同步模型（相对基线的增量）

| 事件 | 基线用途 | UnitFlag 开启后 |
|------|---------|----------------|
| `M_FIX` / `FIX_M` | MMAD→FIXPIPE 整段同步 | **删除** MMAD 与 Fixpipe 之间的显式 `SetFlag/WaitFlag<M_FIX>`，由硬件按 512B 粒度自动同步 |

## 3. 关键参数配置

```cpp
constexpr uint32_t UNITFLAG_DISABLE = 0;        // 不开启 unit_flag
constexpr uint32_t NO_FINAL_ACCUMULATION = 2;   // 非尾轮标志
constexpr uint32_t FINAL_ACCUMULATION = 3;      // 尾轮标志
```

无新增 TilingData 字段，仅修改 MMAD 入参与 Fixpipe 参数。

## 4. 核心计算循环

### 4.1 改造前（默认配置）

```cpp
// 执行 M-MAD 操作
MmadParams para;
para.cmatrixInitVal = (iter1 == 0 && iter0 == 0);
para.m = actualCurM;
para.n = actualCurN;
para.k = curKL0;
AscendC::Te::Mad(MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{},
                 tensorL0C, tensorAL0, tensorBL0, para);

// 流水同步 MMAD 计算和 Fixpipe 搬出指令（整段依赖）
AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
AscendC::Te::Copy(copyL0C2GM, gmBlockC_, tensorL0C);
```

### 4.2 改造后

```cpp
MmadParams para;
para.cmatrixInitVal = (iter1 == 0 && iter0 == 0);
para.m = actualCurM;
para.n = actualCurN;
para.k = curKL0;

// 关键改动 1：根据迭代位置动态配置 unitFlag
if constexpr (EN_UNIT_FLAG == true) {
    if (iter1 == (kL0IterNum - 1) && iter0 == (kL1TileNum - 1)) {
        para.unitFlag = FINAL_ACCUMULATION;
    } else {
        para.unitFlag = NO_FINAL_ACCUMULATION;
    }
}
AscendC::Te::Mad(MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{},
                 tensorL0C, tensorAL0, tensorBL0, para);

// 关键改动 2：删除 MMAD 和 Fixpipe 之间的流水同步

// 关键改动 3：使用内置 unitFlag 的 Fixpipe 参数
auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
AscendC::Te::Copy(copyL0C2GM, gmBlockC_, tensorL0C,
                  AscendC::Te::FixpipeParams{UNITFLAG_EN_OUTER_LAST});
```

## 5. 优化的关键修改点

| 修改项 | 优化前 | 优化后 |
|--------|--------|--------|
| MMAD `unitFlag` | 未设置（默认 0） | 尾轮 `FINAL_ACCUMULATION(3)`，非尾轮 `NO_FINAL_ACCUMULATION(2)` |
| MMAD↔Fixpipe 事件同步 | `SetFlag/WaitFlag<M_FIX>` 整段同步 | 删除，硬件 512B 粒度自动同步 |
| Fixpipe 参数 | 默认 `FixpipeParams` | `FixpipeParams{UNITFLAG_EN_OUTER_LAST}` |

## 6. 注意事项与约束

1. **尾块配置**：循环边界处最后一块的 UnitFlag 与中间块不同（FINAL vs NO_FINAL），漏配会导致输出非最终累加结果、精度错误。
2. **与 L0C Double Buffer 的关系**：L0C 有空间开 Double Buffer 时优先用 Double Buffer 掩盖；UnitFlag 主要用于 L0C 被完整输出块占满、无法开缓冲的场景。
3. **流水图解读陷阱**：开启后 profiling 中 fixpipe 段长度会**变长**——这是指令下发时机提前、实际搬运延迟到尾轮累加结束的正常表现，不代表劣化；需对比 kernel 总时间判断收益。
4. **后续 Tile 的 L0C 复用冲突**：若 FIXPIPE 搬出尾部延续到下一 Tile 且下一 Tile 立即复用相同 L0C 地址，仍可能产生读写冲突造成 MMAD 空洞，此时需叠加 [k_tail_stepwise_copyout_design.md](k_tail_stepwise_copyout_design.md)。

## 7. 实施常见问题与解决方案

| # | 问题 | 根因 | 解决方案 |
|---|------|------|---------|
| 1 | 开启后精度错误 | 尾轮块误用 `NO_FINAL_ACCUMULATION` | 严格按 `iter1 == kL0IterNum-1 && iter0 == kL1TileNum-1` 判定 FINAL |
| 2 | kernel 时间未降反观察 fixpipe 段变长 | 指令提前下发但数据搬出仍受尾轮依赖约束 | 以 kernel 总时间与 MMAD 断流情况为验收指标，不以 fixpipe 段长度判断 |
| 3 | MMAD 仍断流 | FIXPIPE 尾部与下一 Tile 的 L0C 写冲突 | 叠加 K-Tail Stepwise Copyout（见 k_tail_stepwise_copyout_design.md） |

## 8. 实测性能

基础 MatMul（M=1024, K=2048, N=4096, dav-3510，profile 分解单位 μs）：

| candidate | kernel | mac | mte1 | mte2 | fixpipe |
|-----------|--------|-----|------|------|---------|
| matmul（未开启） | 86.870 | 43.804 | 12.997 | 51.857 | 2.970 |
| unit_flag | 85.907 | 42.257 | 11.139 | 35.639 | 22.802 |

FixPipe 与 MMAD 实现深度并行执行，MTE2 段同步缩短，整体 kernel 时间下降。

量化 matmul（MXFP4）实测：优化后 MMAD 执行不再等待 FIXPIPE，解决 MMAD 断流。

## 9. 与其他优化的叠加关系

- **前置依赖**：建议先开 pingpong 双缓冲（[pingpong_design.md](pingpong_design.md)），UnitFlag 解决的是 L0C 侧输出并行度，两者层级不同。
- **可叠加**：SWAT / StreamK / FullLoad / Scale 合并载入（Tiling 层优化与指令级开关正交）。SWAT 模板与 FullLoad 模板的标配手段之一。
- **互补**：K-Tail Stepwise Copyout 处理 UnitFlag 开启后遗留的 L0C 复用冲突。
