# Scatter 族 — 通用 Tiling 算法概览

> 代际限定：本文 RegBase / Reg / MicroAPI / VF 相关实现仅适用于 arch35（A5 / DAV_3510）；通用 Tiling 建模不受此限制。

> Scatter 族算子的六路径 Tiling 算法。所有按轴 `dim` + 单值 `indices` 的 Scatter 算子共享同一套路径语言。
>
> **与其他族的根本差别**：Scatter 的 Tiling 不是"怎么把数据切进 UB"——写地址由运行期 `indices` 决定，UB tile 化不成立。
> 它切的是**并行划分方式**（一个线程认领一条纤维，还是认领一个更新元素）与**冲突消解方式**（免原子 / 硬件原子 / 键仲裁），
> 输出的"tiling data"主要是 **block 数 + 每 block 线程数 + workspace 尺寸**，而不是 ubFactor / tileSize。

## 文档索引

| 文档 | 内容 |
|------|------|
| [tiling-flow.md](tiling-flow.md) | 合轴公式 + **六路径决策树** + 线程数/block 数推导 + workspace 预算 + 实测阈值来源 |
| [tiling-fields.md](tiling-fields.md) | TilingData 字段语义（含各路径消费哪些字段） |

## 路径一览

`starved = numFibers < coreNum × 1024`（纤维路径铺不满核）

| 路径 | 触发条件 | 核心策略 | 散射趟数 | 原子 | workspace |
|------|---------|---------|---------|------|-----------|
| ① 纤维独占 | `!starved` | 线程认领一条纤维，输出集互不相交 | 1 | **无** | 无 |
| ② 打包键单趟 | `none` + `esize==2` + `sIdx+1 ≤ 65535` | `key=(j+1)<<16\|bits16(v)` 单次 `atomic_max` | 1 | 32 位 | `int32[numel]` |
| ③ 两趟胜者 | `none` + 键装不进 32 位 | 趟1 裁决 `atomic_max(win,i+1)`，趟2 比中才写 | 2(+1) | 32 位 | `int32[numel]` |
| ④ 直接原子 | `add`/`amin`/`amax`/`multiply` | 顺序无关，冲突交给硬件原子（乘用 CAS 循环） | 1 | 32 位 | 半精度累加类需 `float[numel]` |
| ⑤ warp 段末 | `none` + `esize==4` + 键 ≤31 位 + `numFibers ≥ 1024` | warp 内 15 级 bitonic，同 index 段末者写 | 1 | **无** | **无** |
| ⑥ 通用逐维 stride | 三张量**非 dim 维范围不等** | 按每维独立 stride 还原坐标，仲裁同 ③/④ | 同上 | 同上 | 同上 |

## 典型场景

| 场景 | 推荐路径 | 原因 |
|------|---------|------|
| `inner` 大、纤维数远超核容量 | ① 纤维独占 | 免原子、免 workspace、免额外 pass，一趟到底 |
| fp16/bf16 的 `none`，纤维数不足 | ② 打包键 | 2 字节值 + `j` 能挤进同一个 32 位键，一趟 `atomic_max` 同时完成仲裁与传值 |
| fp32/int32 的 `none`，纤维数 ≥1024 | ⑤ warp 段末 | 唯一既单趟又免原子免 workspace 的路径 |
| int64 的 `none` | ③ 两趟胜者 | 值塞不进任何 32 位键，而 64 位原子只有 32 位的 1/3 吞吐；拆两趟顺序流量还减半 |
| `add` / `amax`，纤维数不足 | ④ 直接原子 | 结果与顺序无关，不需要仲裁，直接铺满线程 |
| fp16/bf16 的 `add` / `multiply` | ④ + fp32 累加器 | 硬件半精度原子每步舍回，累加误差过不了精度 |
| `indices`/`updates` 非 dim 维比 `data` 小 | ⑥ 通用路径 | 沿 dim 步长不再是三者通用的 `inner` |

## 设计原理与范式代码

本目录只讲**怎么算 tiling**。路径为什么这么设计、核心不变量的证明、可拷贝的 kernel 骨架，见最佳实践货架：

- [reference/scatter/scatter_design.md](../../../../../ascendc-performance-best-practices/references/scatter/scatter_design.md) — 总览、核心不变量、实测规律、证伪清单
- 六条路径各自的分册与 `code/` 下三份范式代码（VF 骨架 / warp 排序 / Host 选路）
