# Conversion 优化工作流（Transpose / Permute / Split / Concat）

> 适用：只改变数据布局、不改变元素值的算子。这类算子内部设计复杂（地址重排是全部代价），**必须按模式特化路由处理，禁止一个通用 gather kernel 打天下**。

## Step 0 — 族判定确认

命中 [README.md](README.md) 判定表第 7 行：**输出元素集合 = 输入元素集合，值不变只变位置/布局**。注意与 index_scatter 区分：conversion 的地址映射是编译期静态的（轴排列/切分点已知），gather/scatter 的索引是运行时数据。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。Conversion 族常见命中：**C4**（P1 双缓冲/P7 对齐是搬运类算子基本盘）。

## Step 2 — 模式特化路由（P127，R1→R5 严格按序）

| 路由 | 判定条件 | 动作 | 说明 |
|---|---|---|---|
| R1 View 短路 | identity permutation，或仅移动 size-1 维度 | **不启动 kernel**，直接返回 view | 零分配零拷贝，最先判 |
| R2 连续维合并 | 置换后仍连续的维度可合并 | reshape 合并降有效 rank，再进 R4 | 例：`[A,B,M,N]→[A,B,N,M]` 合并为 `(A*B,M,N)` 走 batch transpose |
| R3 退化 shape | batch 维为 1 | batch transpose 降级为 2D transpose kernel | 退化场景走更简单的专用 kernel |
| R4 模式分发 | 按置换模式选专用 kernel（见下表） | 各模式独立 kernel，tile-based 连续搬运 | 通用 gather 不得覆盖高频模式 |
| R5 通用 fallback | 罕见 permutation 模式 | 1D gather/scatter kernel，**正确性优先于性能** | 仅作最后分支 |

### R4 模式分发明细

| 置换模式 | 判定 | 专用路径 |
|---|---|---|
| 单维转置（2D transpose） | 仅两轴交换 | **P26**（stride 零拷贝：srcStride=0 连读 + dstStride 跳写，DataCopyPad 一次多行） |
| batch transpose | 批内两轴交换（如 `[B,M,N]→[B,N,M]`） | 专用 batch kernel（2D 模式 + batch 维外循环） |
| swap-adjacent | 相邻两维交换 | 专用 kernel（tile 内转置即可，搬运局部性最好） |
| small-channel（C≤16） | NCHW→NHWC 等小通道重排 | **P114**（统一建模 `[C,N]→[N,C]`，TransDataTo5HD+vnchwconv+Gather） |
| reverse dims | 轴整体逆序 | 专用 kernel（倒序 tile 搬运） |

## 两条禁令（违反视为未真正特化，必须重写）

1. **禁止单一 generic gather 作为主路径**：forward/分发逻辑里不得对所有输入调同一个标量 gather/scatter kernel；2D / batch transpose 等高频模式必须有独立 kernel，fallback 不得覆盖它们。
2. **专用 kernel 内禁止逐元素索引**：不允许用逐元素 div/mod 或 GetValue/SetValue 在专用 kernel 里做地址计算；必须是 tile-based 规则连续搬运（DataCopyPad/DataCopy 的 stride/blockCount 参数化）。

## 组合约束

- P114 与 P26/P7/P1 协同（P114 frontmatter 已声明）；UB 预算与 32B 对齐规则见 P114 卡片
- 完整路由卡的落地 SOP 见 **P127** playbook
