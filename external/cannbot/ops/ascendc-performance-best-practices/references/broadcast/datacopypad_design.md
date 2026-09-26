# 实现②：DataCopyPad 紧凑搬入（外层广播，offset 寻址）

> 主文档：[broadcast_design.md](broadcast_design.md)　|　参考代码：[`code/broadcast_common.h`](code/broadcast_common.h)（`DataCopyPadCompact`）

## 思路

当广播轴**全部位于切分轴之上（外层循环轴）**时，UB tile 内数据连续、无需复制。广播由 **GM offset 算术**实现：外层广播轴 `stride=0`，`GetGmOffset` 在这些轴推进时给出同一段 GM 偏移，于是每个 tile 用普通 `DataCopyPad` 搬一段连续数据即可，"广播"体现在不同输出 tile 指向相同输入区。

## 适用

广播轴全在切分轴之上（`brcInTile == false`）、切分轴本身连续。同 ① 一样不占 Vector，且比 ① 更省——连 DMA 的 in-tile 复制都不需要，纯连续搬。

> ② 与 ① **共用同一段 kernel**：`BroadcastNddma` 内 `outputStrides[ubSplitAxis] == inputStrides[ubSplitAxis]`（切分轴非广播、tile 内无需复制）时自动落入 `DataCopyPadCompact`。

## Kernel 写法

```cpp
// 节选，完整见 code/broadcast_add_kernel.cpp
if (td_.brcMode[i] == BRC_DATACOPYPAD) {
    int64_t off = GetGmOffset(axes, inputStrides[i], ubSplitAxis, ubFormer); // 外层广播轴 stride=0 → off 复用
    int64_t len = rows * inputStrides[i][ubSplitAxis];                       // 切分轴非广播，len=tile
    DataCopyPadCompact(yGm[off], dst, len);                                  // 每轮搬，正确且安全
}
```

紧凑搬入 `DataCopyPadCompact`（`code/broadcast_common.h`）：

```cpp
AscendC::DataCopyExtParams ext{1, (uint32_t)(lenEle * sizeof(T)), 0, 0, 0}; // blockCount=1，连续一段
AscendC::DataCopyPad(ub, gm, ext, pad);                                     // lenEle 可非 32B 对齐
```

## 关于"复用"（可选优化，本样例未启用）

早期版本想靠"不重搬 + 复用上轮 buffer"省搬运，但**判据不能用"切分轴 stride"**：② 路径下切分轴恒非广播（`inputStrides[ubSplitAxis] != 0`），该判据永远要求重搬，复用分支是**死路径**，并不省。

正确的复用判据是「**本输入 GM offset 与上一轮相同**」——即本轮只推进了外层广播轴（stride=0）、切分轴未变。此时上轮 UB 内容仍有效可复用。注意：

- **DoubleBuffer 复杂性**：depth=2 的 TQue 每轮交替返回 ping/pong 两块。复用需按 buffer 槽位分别记录"该槽当前持有哪个 offset 的数据"，仅当目标槽已持有当前 offset 才跳过搬运。简单做法是给该广播输入改用单块 `TBuf`（非 ping-pong），自行管 MTE2→V 同步。
- **收益有限**：多搬几次连续数据是安全的（不影响正确性），仅多占 MTE2。**入门样例选择每轮都搬**，把复用留作 profiling 确认 ② 的搬运成为瓶颈后再加的优化。

## 注意

- ② 的广播完全靠 `inputStrides` 的 0 与 `GetGmOffset`，kernel 里**不需要任何广播指令或 DMA 复制**。
- `len` 用真实元素数（含非对齐尾块），`DataCopyPad` 自动 pad，无需手动处理。
- 若要做复用优化，复用 buffer 不能与每轮重写的输入共用，且判据用 offset 比较而非切分轴 stride。
