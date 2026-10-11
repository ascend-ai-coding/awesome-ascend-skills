# 实现③：UB 内 Broadcast 指令

> 主文档：[broadcast_design.md](broadcast_design.md)　|　参考代码：[`code/broadcast_common.h`](code/broadcast_common.h)（`UbBroadcast` / `FillUbShape`）

## 思路

先把紧凑源 `DataCopyPad` 搬进 UB，再用 `AscendC::Broadcast` 矢量指令在 UB 内展开到目标 shape。占 Vector 单元，但对**尾轴对齐 + 小字节类型**吞吐最高。还是唯一能广播**中间计算结果**的实现（① ② 只能广播 GM 输入）。

## 适用

**前提：tile 内要有 UB Broadcast 能展开的维（`dShape>sShape`）。** 两种会出 nan 的反例（cann-bench Maximum case 18 实测）：
1. 广播轴**严格在外层**（`brcInTile=false`）：UB 展不了外层轴 → 必须走 ②（offset 寻址）。
2. 广播轴是切分轴**但 `ubFormer==1`**：tile 内仅 1 行，`dShape==sShape` 空广播 → 必须走 ① NDDMA。

即需 `inTileExpandable`（切分轴以下某轴广播，或切分轴广播且 `ubFormer>1`）。满足后任一成立即用 ③：
- 尾轴对齐且 dtype ∈ {INT8, UINT8, FP16, BF16, INT16, UINT16}（B8/B16）；
- 或 BigNLast：非尾轴广播且尾轴字节 ≥ `nddma dcache/2`（源码注释 ~4096B）；
- 或广播的是上游计算结果而非 GM 输入。

## Kernel 写法

```cpp
// 节选，完整见 code/broadcast_add_kernel.cpp
LocalTensor<T> src = bufSrc_.Get<T>();
// 源长度按"源 shape"算，不能用展开后的输出 tile 长度（否则 [M,1] 尾轴广播会多搬 N 倍）
int64_t srcLen = UbSrcLen(inputStrides[i], inputDims[i], ubSplitAxis, shapeLen, rows);
DataCopyPadCompact(yGm[off], src, srcLen);                                     // ① 紧凑搬入
UbBroadcast(dst, src, outputDims, inputDims[i], ubSplitAxis, shapeLen, rows);  // ② UB 内展开
```

`UbBroadcast` 内部（`code/broadcast_common.h`）：

```cpp
uint32_t dShape[8], sShape[8];
FillUbShape(dShape, sShape, outputDims, inDims, ubSplitAxis, shapeLen, curRows); // 首轴随主/尾块变
int64_t rank = shapeLen - ubSplitAxis;
// ★退化守卫：所有 dShape==sShape（无可展开维，如尾块 rows==1）→ 直接 UB→UB 拷贝，否则空广播出 nan
bool anyExpand = false; int64_t cnt = 1;
for (int k = 0; k < rank; k++) { if (dShape[k] > sShape[k]) anyExpand = true; cnt *= dShape[k]; }
if (!anyExpand) {                                       // 无可展开维 → dtype 无关 UB→UB 拷贝
    int32_t alignEle = 32 / sizeof(T);
    int32_t cntAlign = ((int32_t)cnt + alignEle - 1) / alignEle * alignEle;  // 向上对齐，buffer=elemNum 不越界
    AscendC::DataCopy(dst, src, cntAlign); return;
}
AscendC::BroadcastTiling bt;
AscendC::GetBroadcastTilingInfo<T>(rank, dShape, sShape, false, bt);
AscendC::Broadcast<T>(dst, src, dShape, sShape, &bt);   // ← UB 矢量广播指令
```

## rank（R）

- `R = shapeLen - ubSplitAxis`（UB tile 内参与广播的维数）。
- 静态已知且 ≤4 → 可用编译期 `Broadcast<T, R>` / `GetBroadcastTilingInfo<T, R>`，性能更好；
- 动态或超阈值 → 运行期 `Broadcast<T>`（如样例，R 作为运行期实参）。
- 尾块需把 `dstShape[0]`/`srcShape[0]` 改成 `ubTail` / 源尾块长度（`FillUbShape` 已按 `curRows` 处理）。

## 中间结果广播

若广播的是上游计算节点的输出（不是 GM 输入），**跳过 `DataCopyPadCompact`**，直接对上游 UB tensor 调 `UbBroadcast`。这是 ③ 相对 ① ② 的独占能力。

## 注意

- **退化守卫**：`UbBroadcast` 内先检查是否存在 `dShape>sShape` 的维；若无（如尾块 `rows==1` 或选型漏判），直接 `DataCopy(dst,src,对齐cnt)` 做 dtype 无关的 UB→UB 拷贝（不用 `Adds`，避免 int64 等不支持），而非调 `Broadcast`，防空广播出 nan。这是兜底，正确选型应优先在 host 侧避免（见上"前提"）。
- UB 预算要**多留一块紧凑源 buffer**（样例的 `bufSrc_`）：③ 是 src + dst 两块，① 只需一块。tiling 的 `aliveBuf` 要含这块。
- **每个走 ③ 的输入要各自一块 `bufSrc`**：若多个输入都可能 ③（如 binary 的 x1/x2 都广播），共用一块 `TBuf` 会 RAW 冒险（第二个输入的 `DataCopyPadCompact` 覆盖第一个的紧凑源）。binary 算子应配 `bufSrc1_/bufSrc2_`，`aliveBuf` 相应增大（cann-bench Maximum 落地经验）。
- `GetBroadcastTilingInfo` 的 rank 必须与 `Broadcast` 调用一致。
- src/dst shape 的首轴是 UB 切分轴方向的行数，后续轴是尾轴及以下；广播轴在 src 上填 1。
