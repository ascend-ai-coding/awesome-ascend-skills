# Conversion 类算子场景路由

> 本文档是 Conversion 数据转换类的**入口索引**（也是 `routes.yaml` 中 `Conversion` 范式加载的入口文件），
> 用于**场景判定**和**策略选择**。确定子场景后，按链接进入对应详细文档。
> Conversion 类：改变布局/形状、合并/拆分张量的搬运算子——输出元素来自输入的重排、子集或扩展，通常不改变元素值（融合后处理除外）。

---

## 子场景判定流程

按“输出与输入的元素集合关系”判定，而不是按算子名称。**各分支互斥，自上而下首个匹配即命中**：

```text
给定: op 语义（输出元素相对输入的关系）

Step 1 — 元素集合关系判定（自上而下，首个匹配即命中）:
  ├─ 元素总数不变，仅顺序/布局重排
  │    ├─ 通道维较小（C ≤ 16）的 transpose / permute → [transpose-small-channel.md]
  │    └─ 其它重排/扩展/裁剪（reshape / view / tile / expand / broadcast_to 等）→ 未覆盖（见下）
  ├─ 输出是输入的有序子集（可带正向 stride），不改变元素值
  │    └─ Slice / StridedSlice                    → [strided-slice.md]
  ├─ 含“沿某维切块后沿相邻维拼接 + 末尾补零”的语义
  │    └─ ChunkCat（dim=0 切块 + 级联 + 零填充）   → [chunkcat.md]
  ├─ 输出是输入的超集（新增位置由常量/边缘/镜像填充），且不含切块重组
  │    └─ Pad / PadV3（constant / replicate / reflect） → [pad.md]
  └─ 其它                                        → 未覆盖（见下）
```

> **ChunkCat 分支必须先于 Pad 判定**：ChunkCat 的输出同样“变大 + 补零”，但放大来自**切块重组 + 尾部零填充**，不是单纯在边缘补值。若先判 Pad 会把 ChunkCat 误导向 [pad.md](pad.md)，而 pad.md 只覆盖常量/边缘/镜像三种模式、不含切块语义。

**当前未覆盖**：reshape / view / tile / expand / broadcast_to 等其它 LayoutTransform 语义、通用 Concat（不含切块与补零）、Split、大通道通用 transpose、5HD/NC1HWC0 布局转换。遇到时按 [SKILL.md](../../SKILL.md) 的「通用设计要素」设计，并先做 API 探针与正确性验证，不要强行套用上述子场景的门禁。

---

## 子场景索引

| 子场景 | 触发特征 | 关键设计 | 详细文档 |
|--------|---------|---------|---------|
| **Small-Channel Transpose** | 元素总数不变、维度顺序重排、小通道 `C ≤ 16` | 统一建模 `[C, N] → [N, C]`；`TransDataTo5HD + Gather`；offset 表按对齐后 `tileNA` 构建；`repeats ≤ 255` | [transpose-small-channel.md](transpose-small-channel.md) |
| **Slice / StridedSlice** | 输出是输入的有序子集、正向 stride、rank ≤ 8、元素 2/4/8 字节 | Host 按几何特征分路由（CONTIGUOUS / AFFINE_ROWS / PREFIX_SPANS / COMPOSITE / PERIODIC / MACRO_WINDOW / PACKED）；Kernel 真实 `switch(routeIndex)`；ownership group 保证 32B 写所有权 | [strided-slice.md](strided-slice.md) |
| **Pad** | 输出更大、最后维有 pad、`W×b` 非 32B 对齐（短块过多）、无切块重组 | Host 按模式/对齐分路由（FLAT_COPY / CONST_TRANSPOSED / PAD_INTEGRATED / REPLICATE_* / REFLECT_2D）；“4 行打包 + 转置域补边”改布局；通用路内部行与 H 带也必须批量搬运 | [pad.md](pad.md) |
| **ChunkCat（切块 + cat + 补零）** | dim=0 切块 + 沿下一维级联 + 末尾零填充，1–4 个连续输入 | 二维 block 网格搜索 + 列优先 UB tile 切分；`FAST_ROW_E → SEGMENT_BATCH → SEGMENT_SINGLE` 三路由；padding 必须显式写零 | [chunkcat.md](chunkcat.md) |

> **平台前提**：表中各子场景的基准数值来自 Ascend 910B3（DAV_2201）；Ascend 950PR（DAV_3510 / arch35）
> 的追加证据、平台专属机制（NDDMA / Loop Mode / RegBase / SIMT / `SetLoopModePara` / `DataCopyPad<Compact>`）
> 与其门禁，分别标注在各自文档的平台锚点（pad §B0 与 §3.8、chunkcat §0 与 §9、strided-slice §0 与 §5.12、
> transpose §0）。未显式标注「950PR 专属」时，910B3 路径仍然有效；
> 平台专属条目必须写成带平台谓词的分支，且保证 910B3 原路径可达。

---

## 跨子场景通用规则

以下规则取自各子场景文档的实测结论，但**验证覆盖度并不相同**；括号内标注的是实际来源，未标注的条目不要外推到 small-channel transpose：

1. **Host 决策、Kernel 只执行**：路由所需信息全部来自 shape/stride/dtype/平台参数，Host 一次算清并写入 TilingData；Kernel 必须真实消费 `routeIndex` 并分发到对应实现——只把 ID 写入 TilingData、所有分支仍走同一个 generic kernel，不算实现路由。（来源：strided-slice、chunkcat；pad 只覆盖“Host 决策 / Kernel 只执行”一层）
2. **fallback 永远可达**：每条特化路由都要有明确准入谓词与保守 fallback；fallback 不能吞掉已有专用方案覆盖的常见几何（逐 run、逐行、`blockCount=1` 都不得作为非对齐的默认答案）。（来源：strided-slice、pad、chunkcat）
3. **32B 写所有权先于性能**：写回以物理 32B 块为最小 owner 单位；非 32B 对齐的短 run 先在 UB 内拼满再由单一 owner 写回；批量路径的核边界必须落在完整 ownership group 上。（来源：strided-slice；pad / chunkcat 只覆盖“32B 对齐”一层，不涉及写所有权分组）
4. **同步按路径选，不做全量替换**：简单对齐单缓冲路径用精确事件；多 stage 或需要跨队列保护的路径保留 `PipeBarrier` / `TQue`。禁止用 `PIPE_ALL` 让 MTE2/V/MTE3 全局串行，也不要把所有路径一刀切改成精确事件。（来源：strided-slice、pad；chunkcat 有同步设计但未讨论 `PipeBarrier` / `PIPE_ALL` 取舍）
5. **搬运主导，批量化优先**：逐行 / 逐 run / 逐像素的小 DMA 是主要退化源。优先批量 DMA、布局变换、Gather/窗口；能通过改布局（如转置域补边）解决时，不要死磕非对齐。（来源：strided-slice、pad、chunkcat；transpose 只支持“逐像素 DMA 的 blockLen 过小”这一点）
6. **平台差异必须重新验证**：现有实测集中在两个平台——Ascend 910B3（DAV_2201，UB 192 KiB）与 Ascend 950PR
   （DAV_3510 / arch35），CANN 同为 9.1.0。换 SoC 或 CANN 版本后都要重测，不能直接外推。以下三条是**跨子场景
   门禁**，与具体数值无关，只在目标平台重新标定后才可放开；逐子场景的数值、机制与证据等级见各文档平台锚点
   （pad §B0 与 §3.8、chunkcat §0 与 §9、strided-slice §0 与 §5.12、transpose §0）：

   - **UB 预算不得写死**：预算取运行时可用的设备属性
     `aclrtGetDeviceInfo(dev, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, ...)`；**不得**用编译期常量、平台 INI 的
     `ub_size`，或用 `GetCoreMemSize(CoreMemType::UB, ...)` 的静态值代替——950PR 上按物理容量分配会**静默越界**
     （pad §B0，A）。
   - **平台 API 的编译期门禁**：950PR（arch35）的 `DataCopyExtParams.srcStride/dstStride` 为 `int64_t`
     （可取负值，用于源块重复 / 一行广播到多行）；`DataCopyPad` 的 `PaddingMode` 模板参数**只对 arch35
     系存在**，写进 910B3 目标会直接编译失败。两者都必须用平台谓词 / 条件编译隔离
     （chunkcat §9、strided-slice §5.12.1）。
   - **平台专属搬运的同步契约**：NDDMA 多维 `DataCopy` **无 pipe 注解**，其后 MTE3 读同一 UB 前必须显式
     `SetFlag/WaitFlag<HardEvent::MTE2_MTE3>`，否则**静默读到旧数据**；且不得与双缓冲的 `MTE3_MTE2` 流水线
     共用同一组事件 ID（pad §3.6、§4.5，A）。

7. **性能证据口径**：只采信同 device、同 session 的 paired 测量；跨 session 的微秒级差异和小于 3% 的 geomean 变化不作为确定性收益。（来源：仅 strided-slice 声明该口径）
