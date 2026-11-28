# 技术卡片 mem-02：Ping-Pong 预取下一 tile

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：访存 / 流水
- **一句话**：当前 tile 在计算时，向另一块 L1/UB buffer 发起下一 tile 的搬入；把每轮“先等数据再算”改成稳态重叠。

## 何时用（诊断特征）

- 每轮都在计算前等待 `MTE2` 或 L1 buffer 就绪，计算和 copy 时间线首尾相接而很少重叠。
- 连续 tile 之间没有数据依赖，且当前 tile 计算期间不会再写下一 tile 的目标 buffer。
- 单 buffer 的 tile 已能让计算和搬运各自达到合理粒度，UB/L1 仍容得下第二份输入。

## 原理

第一块预热后，tile `t` 的计算与 tile `t+1` 的搬入可并行。两个物理 buffer 轮换所有权：copy 只写“空闲”的 buffer，compute 只读“已就绪”的 buffer。关键不在多开一个 buffer，而在把下一轮的 copy 发射提到当前轮 compute 之前，并严格维护 MTE 与消费方的事件配对。

## 怎么改（before / after）

```cpp
// 伪代码：Wait*/Mark* 表示与实际 TQue/TBuf 和 event ID 对应的所有权、同步操作。
// before: 每轮搬入完成后才开始计算
for (uint32_t t = 0; t < tileNum; ++t) {
    CopyIn(buf, t);
    WaitCopyDone(buf);
    Compute(buf);
}

// after: 预热 tile 0；计算 t 时搬入 t+1 到另一 buffer
CopyIn(buf[0], 0);
MarkCopyDone(0);
for (uint32_t t = 0; t < tileNum; ++t) {
    uint32_t cur = t & 1;
    uint32_t next = cur ^ 1;
    if (t + 1 < tileNum) {
        WaitBufferReusable(next);
        CopyIn(buf[next], t + 1);       // 尽早发射，和下方 Compute(cur) 重叠
        MarkCopyDone(next);
    }
    WaitCopyDone(cur);
    Compute(buf[cur]);
    MarkBufferReusable(cur);
}
```

Tiling 侧同时验证 `2 * inputTileBytes + workspaceBytes + outputTileBytes`。若以 queue 实现，`InitBuffer` 的深度和 host UB 预算必须使用同一份 `bufferNum`，不能只改 kernel。

## 性能数据

用 msprof 时间线检查稳态区间：`MTE2` 应覆盖 Vector/Cube 的一部分而非整体串行。比较 `aiv_time` 与 MTE 等待空洞；首尾 prologue/epilogue 不应被误当作无法消除的稳态开销。

## 红线（Stop rules）

- tile 太小、copy 比 compute 长很多或反之时，双缓冲未必抵消额外 UB/L1 占用；先用单 buffer 基线测量。
- 最后一轮不能预取不存在的 tile；复用 buffer 前必须等原消费者完成，事件类型和 event ID 必须成对。
- 不要把仍需被当前计算读取的数据覆盖到 next buffer；跨阶段共享临时结果时先做 mem-04 的生命周期分析。
- 内存预算不足时减 tile 或回退单缓冲，不可为了开双缓冲挤掉对齐/padding 所需空间。

## 关联

- 源（5 次）：[ops-nn `eafe7968`](https://gitcode.com/cann/ops-nn/commit/eafe7968)（arch35 Conv2D：预热首块并在循环内预取下一 `cin` block 到另一 L1 buffer）、[ops-nn `162dd73b9`](https://gitcode.com/cann/ops-nn/commit/162dd73b9)（arch35 BatchNorm：BlockSplitR 流水）、[ops-transformer `a2ee2d928`](https://gitcode.com/cann/ops-transformer/commit/a2ee2d928)（arch35 QKV：tile 流水重排）、[ops-transformer `de8e0d03b`](https://gitcode.com/cann/ops-transformer/commit/de8e0d03b)（arch35 MegaMoE：动态 wave pipeline）、[ops-transformer `cad87fee0`](https://gitcode.com/cann/ops-transformer/commit/cad87fee0)（arch35 MegaMoE：按 UB 预算调整 pipeline buffer/batch）。
- 相邻卡片：mem-03（多 buffer 可同时服务于按行批搬）、mem-04（跨阶段临时量的复用）。
