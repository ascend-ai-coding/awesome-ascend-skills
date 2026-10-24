# DAV_3510 TopK 模板使用指南

## 1. 模板定位

本文用于后续重新实现或优化 Ascend 950 / DAV_3510 TopK。A2 大规模稀疏选择仍参考
[radix_sort.md](../../arch22/radix_sort.md)，二者不可直接混用。

当前契约：连续 ND 输入，有序输出，indices 为 int64；支持 int8/uint8/int32/int64/fp16/fp32/bf16，
`1 <= k <= min(D, 2048)`，largest/smallest 均支持。

`dav3510/` 是 TopK 设备侧最佳版本快照，适用于 Ascend 950 / DAV_3510。快照保留 TilingData 的
direct-launch ABI、全部 Kernel 模板和统一 launch，不能只挑一个 RadixSort 模板使用。Host Tiling
按性能策略 skill 的 `references/tiling/sort/topk_host_tiling.md` 与 `script/topk_tiling.py` 复原，不在
本目录重复保存 C++ 模板。

`dav3510/` 下所有源码均以 `.template` 为最终后缀，例如 `top_k_kernel.cpp.template` 和
`top_k_launch.h.template`。它们是不可直接编译的参考模板；原始 `.cpp/.h/.txt` 后缀保留在文件名中，
用于在目标优化工程内去掉最后一个 `.template` 后缀后精确恢复。skill 原目录内禁止生成 `.cpp/.h`。

当前性能不能概括为“使用 RadixSort”。实现按 shape、dtype、k、UB 和核数在以下模板间选择：

| 路径 | 核心适用场景 | 决定性能的实现 |
|------|--------------|----------------|
| 固定小轴 | non-last，D=7/17，k=D，fp16/bf16 | 固定 compare-exchange network，沿 inner 宽向量并行 |
| transpose-fallback | outer=1 的超大 int32/fp32 非尾轴，inner 足够大且 k/D 小 | 设备侧物化为连续尾轴，复用完整尾轴 TopK |
| fallback L | int64 或其它 fallback 的连续行 | G 行批处理、D 分块、running TopK、连续 DMA |
| fallback S | 大非尾轴或高复用 int32 跨步行 | 2D Compact DMA、Gather 装配、G 行 TopK、UB 输出转置 |
| common mode 8 | D<=2048 的一般非尾轴 | UB 内重排为连续行，按 dtype 选 merge/radix |
| common merge/radix | 一般尾轴 | small merge、single-block/single-core/multi-core radix、fp32 merge |
| SortAndTopK | 超大 D 或 radix tile 不可行 | 多核全轴 radix sort，再复制前 k |

选路的精确门槛和 Host Tiling 真值由性能策略 skill 的 `references/tiling/sort/` 维护。完整接入必须同时
读取其中的 `topk.md`、`topk_host_tiling.md` 和 `script/topk_tiling.py`。本文只维护 Kernel、launch 的
实施经验、模板接入方式和复现验收，不复制 Host Tiling 规则。

## 2. 目录职责

| 路径 | 职责 | 迁移要求 |
|---|---|---|
| 上游 `references/tiling/sort/` | Host 一级路由、common tiler、ABI、UB/workspace | 按复原规格和 Python 模型生成 Host C++ |
| `dav3510/op_kernel/arch35/core/` | merge/radix/SortAndTopK 与固定模板 | 全部 `.template` 成套转换 |
| `dav3510/op_kernel/arch35/fallback/` | batched L/S fallback tiler 与 kernel | 全部 `.template` 成套转换 |
| `dav3510/op_kernel/arch35/sort*/` | TopK 依赖的 Sort/SortWithIndex 基类 | 转换时保持相对路径 |
| `dav3510/op_kernel/arch35/top_k_*` | special/common 分发和 launch 契约 | 全部 `.template` 成套转换 |
| `dav3510/op_kernel/arch35/transpose/` | tiling key 900003 的 TopK 私有 Transpose | 不得引用独立 Transpose 算子源码 |

## 3. 模板转换与复制方式

在优化产物目录中执行，`TOPK_DST` 指向目标工程的 TopK 源码目录。转换只操作目标工程，不修改
skill 中的 `.template`：

```bash
TEMPLATE=<skill-dir>/references/sort/arch35/templates/dav3510
TOPK_DST=<target-ops-dir>/top_k

mkdir -p "$TOPK_DST"

# 保留目录结构，并去掉最后一个 .template 后缀。
find "$TEMPLATE" -type f -name '*.template' \
  -print0 |
while IFS= read -r -d '' src; do
  rel=${src#"$TEMPLATE"/}
  dst="$TOPK_DST/${rel%.template}"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
done
```

转换后 `*.cpp.template -> *.cpp`、`*.h.template -> *.h`。该命令只转换本目录保留的设备侧模板；
Host 文件必须另按上游复原规格生成。本目录不提供构建文件，目标工程必须自行完成源码注册和构建
接线，并按目标框架适配注册入口。不得改写上游 Host 路由、TilingData ABI、Kernel 相对 include、
fallback 编译宏或 launch 属性来迁就框架。

## 4. 必须保留的构建与运行契约

1. 设备侧同时编译 `op_kernel/arch35/top_k_kernel.cpp`、
   `op_kernel/arch35/fallback/top_k_fallback.cpp` 和
   `op_kernel/arch35/transpose/top_k_transpose_kernel.cpp`，kernel include 根目录设为
   `op_kernel/arch35`。TopK 必须使用自己的 Transpose 文件，不得直接 include 或编译独立 Transpose 算子源码。
2. 按上游 `topk_host_tiling.md` 第 11 节生成 `op_host/top_k_tiling_adapter.cpp`、
   `op_host/top_k_tiling_base.h/.cpp`、`op_host/top_k_tiling_arch35.h/.cpp` 和
   `op_host/sort_with_index_tiling.h`，并用 `script/topk_tiling.py` 对齐 route 和字段。
3. 对 Python JSON 中的所有 `sdk_queries` 接入目标 CANN API，再编译 Host；探索估计不得进入生产实现。
4. Host tiling 源使用 `_GLIBCXX_USE_CXX11_ABI=0`。
5. `arch35/fallback/top_k_fallback.cpp` 保留 `TopKLaunchParams`、`calc_topk_params`、`launch_topk` 三个编译期重命名。
6. special key 900001/900002/900003 先于 common key 分发。
7. mixed kernel 保留 `KERNEL_TYPE_MIX_AIV_1_0`、`SCHEM_MODE=1` 和动态 UB launch 属性。
8. 系统 workspace 与用户 workspace 通过 `SetSysWorkspaceForce` / `GetUserWorkspace` 分离。
9. 私有 Transpose 的 host tiling 从目标平台查询 AIV 核数和 UB 容量；每块 UB tile 预算按双缓冲及 24 KiB 预留量计算。资源查询失败时返回零 `numBlocks`，调用方须将其视为 tiling 失败，不得启动 kernel。

## 5. 实施、适配和验收顺序

### 5.1 推荐实施顺序

1. 先冻结行为和基线：固定设备、CANN、编译选项、输入生成、warmup 和计时方式，保存每个 case 的
   values/indices 精度、耗时、tilingKey、modeType、kernel symbol、核数、UB 和 workspace。没有逐 case
   路由记录时，后续很难区分“kernel 变慢”与“tiler 选错”。
2. 整体迁移 common 模板栈：先让常规尾轴和 D<=2048 非尾轴通过，再加 special route。common 内保持
   `non-last mode 8 -> last-axis small MergeSort -> single-block radix -> fp32 inter-core merge ->
   fp32 intra-core merge -> single-core radix -> optimized multi-core radix -> general multi-core radix ->
   SortAndTopK`。这些模板共享 TilingData，但 mode 专属字段不同；host 判断、modeType、tilingKey 和模板
   实例必须一一对应。
3. 实现 fallback 的 running TopK：首 chunk 产生 running candidates，后续只在旧 running 与新 chunk
   TopK 的并集中继续选择，最后排序并写出 value 与原 axis index：

   ```text
   run = TopK(chunk0)
   for chunk in chunks[1:]:
       run = TopK(run union TopK(chunk))
   sort(run)
   store values and original axis indices
   ```

   `TopK(A union B) = TopK(TopK(A) union TopK(B))` 是截断成立的依据。每轮最多处理约 2k 个候选，
   但必须保留原 axis index，且最终显式有序化。
4. 分别优化 L 和 S 的数据布局：L 直接批量 DMA `[G,dChunk]`；S 先把 `[dChunk,G]` 以 2D Compact DMA
   搬入 UB，再 Gather 成 `[G,dChunk]`。S 的输出将 `[G,k]` 分块转为 `[k,G]`，随后批量 DMA 回
   `[outer,k,inner]`。
5. 最后增加窄门控专用路径：增加 D=7/17 sorting network 和 transpose-fallback。两者只在已 A/B 的
   条件下抢占 common/fallback，不要为追求某个 case 泛化到相邻 dtype 或输出比例。
6. 恢复 launch 资源契约：统一 special/common 分发，恢复 schedule mode 1、动态 UB、mixed AIV task type
   和用户 workspace 偏移。缺少 launch 属性时，kernel 代码相同也不代表运行条件相同。

### 5.2 模板适配和验收顺序

1. 按第 3 节转换设备侧模板，保留目录结构。
2. 按上游复原规格生成 Host 文件并完成 SDK 查询接线，只适配注册、头文件搜索路径和构建接线。
3. 运行 Python 和 C++ 代表 case，逐项对齐 tiling key、modeType、核数、动态 UB、workspace 和 ABI 字段。
4. 跑第 14 节的完整正确性回归矩阵。
5. 使用目标工程的性能测试集，在同设备、CANN、输入和计时口径下与优化前基线逐 case A/B。
6. 全量精度通过且无未经评估的单 case 性能退化后，才允许调整路由阈值。

不要直接修改 skill 内的 `.template`。所有适配只在转换后的优化产物中进行，并在优化报告中记录其
相对模板和 Host 复原规格的实际 diff。

## 6. 固定 D=7/17 排序网络

### 6.1 UB 布局与并行方向

把每个 axis value 看成一行，沿 inner tileWidth 向量化：

```text
values  [D, tileWidth] uint16 storage
indices [D, tileWidth] uint16
scratch value/index rows
compare mask + NaN mask
```

GM 以 axis 行搬入，index 行填 axis 编号。每个 compare-exchange 对整段 inner 列并行，value/index
必须使用同一个 swap mask。D=17 使用 72 次 compare-exchange、11 层依赖；不能换成动态插入排序后
期待相同性能。

### 6.2 正确性细节

- `largest` 在运行时改变 compare 方向，不需要两套网络。
- NaN 比较与交换必须显式遵守接口语义，不能依赖普通 `>`/`<` 的偶然结果。
- 尾 inner tile 仅写有效 width；scratch 和 mask 仍按对齐 tileWidth 分配。
- 最终 SIMT 写回 `[outer,D,inner]`，index 从 uint16 转 int64。

实现上可将比较对整理为 `constexpr TopKComparePair[]`，由固定编译期长度的 `for` 循环配合
`#pragma unroll` 展开；不要使用运行时生成的 pair 表或未展开的动态索引循环，否则可能引入额外的
循环控制和 UB 地址计算。

## 7. Fallback kernel 关键实践

### 7.1 G 行批处理摊薄固定开销

高阶 TopK/Sort API 每次调用有固定循环与同步成本。一次处理 G 行比逐行调用更重要，G 上限 128，
由 host 查询目标平台的 UB 容量、扣除 8 KiB 预留后与 dChunk 联合搜索。kernel 不应自行缩小 G；host/kernel 参数不一致会
造成 buffer 越界或性能丢失。

### 7.2 fp16/bf16 统一以 fp32 计算

Fallback 先把 raw 16 位数据精确 cast 到 fp32，TopK/merge 全程用 fp32，输出再 cast 回原 dtype。
这是对目标 CANN 高阶 TopK 16 位 index 收集缺陷的规避，不是一般精度偏好。需要同时预算 raw、cast、
compute 和回转 buffer。

### 7.3 int64 动态窄化

目标版本的 b64 TopK 存在 crash 风险，因此每个 chunk 先做 int32 范围检查：

```text
if all values in current chunk fit int32 exactly:
    cast int64 -> int32
    RADIX_SELECT
else:
    Sort<int64>(RADIX_SORT)
    truncate to k
```

范围检查必须逐 chunk；一行中可能同时存在窄值和宽值。窄化后的 index 仍是原 axis offset，最终转为
int64。RADIX_SELECT 结果不保证有序，running candidates 写出前必须 Sort。

### 7.4 Padding 是算法输入的一部分

尾块和 Gather 洞区先填方向相关最差值，再由 DMA 覆盖有效区：

| 方向 | 浮点 padding | 整数 padding |
|------|--------------|--------------|
| largest | `-inf` 或 dtype 最小值 | dtype 最小值 |
| smallest | `+inf` 或 dtype 最大值 | dtype 最大值 |

不初始化 padding 会把上一个 tile 的残留值选入结果。只填 value 不填或不维护 index，也会产生表面
value 正确、index 错误的问题。

## 8. S 分支：把跨步访问改成批量搬运

### 8.1 输入 2D Compact DMA + Gather

原布局中同一 TopK 行沿 D 的步长为 inner。一次搬入 D 个 block，每个 block 含 G 个连续元素，
Compact 模式在 UB 中形成 `[dChunk,G]`。随后：

- `dChunk <= 128` 且容量/offset 满足时，用一张全量偏移表做一次大 Gather；
- 其它场景逐行 Gather 到 `[G,dChunk]`，但不退回 GM 单元素读；
- 尾 dChunk 先填 padding，再用 2D DMA 覆盖有效部分。

Gather source offset 硬门槛：8 位不超过 65535B，16 位不超过 131071B。表项、offset 单位和 dtype
字节数必须一起校验；旧实现注释中的 `dChunk<=256` 已过期，当前整块输入 Gather 门槛是 128。

### 8.2 输出 UB 转置 + 批量 DMA

TopK 产生 `[G,k]`，GM 需要 `[k,G]` 片段。复用已经结束生命周期的 merge value/index 区作为输出
tile，Gather 转置后：

- G 对 32B 块对齐时发一次 2D DMA；
- 不对齐时按 k tile 发少量连续 DMA；
- G<8、offset 或 tile 容量不满足才逐行 scatter。

这一步同时处理 value 与 int64 index。每个输出 tile 写完后，在 Gather 覆写复用 buffer 前完成
MTE3-V 同步。

## 9. UB 复用与同步不变量

UB 优化应以生命周期图为依据，不是简单删 buffer：

1. L/int64 可让 merge scratch 复用已死亡的 input/index 区，但 `dChunk >= mergePad` 且容量要按较大者算。
2. S 输出转置可复用 merge 区，但必须在 running TopK 的最后一次读取完成后开始。
3. 下一 chunk DMA 覆写 stage 前，上一 chunk 的 Gather/TopK 已不再读取 stage。
4. 下一输出 tile Gather 覆写前，上一 tile 的 MTE3 写回完成。
5. API tmp 取 TopK、RADIX_SELECT floor 和 b64 Sort scratch 的最大值，不能按常见 dtype 省略条件项。

Host UB 公式必须逐项映射 kernel `InitBuffer`。改动 buffer 生命周期后，同时更新 host 预算、tiling
字段、kernel offset 和边界测试。

## 10. Transpose-fallback 实施约束

执行顺序为 input transpose、内部尾轴 TopK、values transpose、indices transpose。四个 workspace 区
按 512B 对齐，内部 TopK 使用递归 tiler 返回的完整核数、UB、workspace 和 tiling data。

这里不要：

- 在 host 上物化 transpose；
- 对所有 non-last shape 都 transpose；
- 递归时允许再次进入 transpose-fallback；
- 忘记 indices 是 int64，按 value 的 4B 大小分 workspace；
- 把内部系统 workspace 与外层用户 workspace offset 混用。

## 11. Common 模板不可丢失的经验

- non-last mode 8：D<=2048 时在 UB 内把跨步行重排为连续行；浮点优先 MergeSort，其余 RadixSelect。
- small MergeSort：D<=1024 的 fp16/fp32/bf16 尾轴；bf16 以 fp32 compute。
- fp32 `0.25 <= k/D < 0.5`：完整排序归并比反复选择更合适，依次尝试 inter-core、intra-core 两个
  独立模板。
- optimized multi-core：只有 `k*ceil(D/tile)<=tile` 时，所有局部候选才能在一次最终选择中放下。
- general multi-core：保留每 tile 的 k 候选并做后续选择，不可误套 optimized 条件。
- SortAndTopK：前述 tile 失败或 D 超大时的性能或能力兜底，不等于通用 M-way fallback。

这些模板的次序属于性能契约。任何顺序调整都要对被抢占模板的所有边界 case 做 A/B。

## 12. Launch 与 workspace 检查表

- [ ] special key 900001/900002/900003 在 common key 前分发。
- [ ] mixed kernel 使用 `KERNEL_TYPE_MIX_AIV_1_0`。
- [ ] launch 属性包含 `SCHEM_MODE=1`。
- [ ] launch 属性包含 tiler 计算的 `DYN_UBUF_SIZE`。
- [ ] `SetSysWorkspaceForce` 后通过 `GetUserWorkspace` 获取用户区。
- [ ] block dim、dynamic UB、workspaceBytes 与 host params 完全一致。
- [ ] common modeType 与实际模板实例一致，未链接到旧 `.so`。

## 13. 反模式

| 反模式 | 后果 | 替代做法 |
|--------|------|----------|
| 用一个 RadixSort 覆盖全部 TopK | 小轴、非尾轴、int64 和大 k case 明显回归 | 保留多模板路由 |
| 只按 dtype 选模板 | 同 dtype 不同 D/inner/k 落入错误路径 | 用归一化 shape + UB + 核数联合判定 |
| non-last 一律 transpose | 小 inner / 大 k 被三次额外搬运拖慢 | 使用严格搬运收益门控 |
| fallback 取第一个可行 tile | API 固定开销未摊薄 | 搜索 dChunk/G 并最小化调用成本 |
| fp16/bf16 直接走 fallback 16 位 TopK | 触发目标版本 index 缺陷 | fp32 compute 后 cast 回 |
| int64 直接走 b64 TopK | 目标版本可能 crash | 逐 chunk 窄化，否则 b64 Sort |
| 假设 RADIX_SELECT 已有序 | values 集合正确但顺序不满足接口 | 最终 candidates 显式 Sort |
| value/index 分开优化搬运 | 重复值和 swap 后 index 错配 | 全流程成对移动 |
| 复用 UB 不补同步 | 偶发脏数据或跨 tile 污染 | 按生产者/消费者生命周期加事件 |
| 普通 launch 漏动态 UB | mode 8 / mixed 模板资源改变 | host args 同时传 schedule 与 UB |
| 只看总体分数 | 专用路径回退被其它收益掩盖 | 核验逐 case route 和时间 |

## 14. 正确性回归矩阵

至少覆盖：7 种 dtype；largest/smallest；尾轴/non-last；k=1、k=D、k=2048；32B 非对齐；
NaN/Inf；重复值；int64 全窄、全宽、窄宽混合；S 的整块/逐行 Gather；输出批量/逐行路径；
transpose workspace；所有 common mode。检查 values 之外，还要用输出 index 回取输入确认 pair 一致。

## 15. 来源实现锚点

Kernel、统一 launch、fallback 和 TopK 私有 Transpose 快照保存在当前目录的 [dav3510/](dav3510/)。
Host Tiling 不再保存为 C++ 模板，必须从性能策略 skill 的 `references/tiling/sort/topk_host_tiling.md`
与 `script/topk_tiling.py` 复原。复核经验是否过期时，以同一提交的上游 Tiling 规格、保留的
Kernel/launch 模板和全量回归结果为准。
