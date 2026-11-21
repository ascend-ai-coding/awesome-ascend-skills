# 技术卡片 mem-04：复用 UB 缓冲并减少 GM 中转

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：访存 / UB
- **一句话**：临时结果只要仍在同一核的后续阶段被消费，就复用已有 UB buffer 或在 UB 内完成初始化；不要为清零、暂存或格式过渡先写 GM 再读回。

## 何时用（诊断特征）

- 相邻阶段之间存在 GM 写回后立即读回，或仅为清零、暂存、scale/metadata 准备开辟独立 GM 中转。
- 多个临时张量生命周期不重叠，却长期各自占用 UB；为此缩小 tile 或增加额外 copy。
- trace 显示短生命周期的 GM 写读、无效 buffer 初始化或额外同步多于真正的结果写回。

## 原理

UB 是同核流水的低延迟工作集。将生产者输出直接交给后续消费者，或让后续阶段复用前一阶段已经释放的 buffer，可同时减少 GM 流量、MTE 指令和 buffer 占用。关键是生命周期分析：复用仅在前一数据已被最后一次消费且不同步访问时安全；否则必须保留独立 buffer 与事件边界。

## 怎么改（before / after）

```cpp
// 伪代码：CanAlias 由生命周期、事件和物理布局共同证明；不是简单的指针复用。
// before: 清零或临时结果经过 GM 中转，后续阶段再搬回 UB。
FillGm(tmpGm, 0, bytes);
CopyIn(tmpUb, tmpGm, bytes);
Consume(tmpUb);

// after: 在 UB 内初始化；不重叠阶段复用已释放的 buffer。
LocalTensor<T> tmpUb = AcquireOrAliasUb(CanAlias(prevUb, nextStage));
Duplicate(tmpUb, static_cast<T>(0), elems);
Consume(tmpUb);
```

host tiling 需要把复用后的峰值 UB 用量而不是各逻辑 buffer 的静态和写入预算。若多个阶段有重叠，改为 ping-pong/ring buffer，并显式保存每个 buffer 的生产、消费、释放事件。

## 性能数据

记录 GM 读写字节、MTE 发射次数、峰值 UB 占用和 `aiv_time`。分别验证 buffer 生命周期相邻、部分重叠与完全并行的分支，确保复用只覆盖无重叠路径。

## 红线（Stop rules）

- 复用前必须完成前一消费者的读取；不能只因两个 `LocalTensor` 类型相同就 alias。
- 复用 buffer 的 dtype、长度、对齐、offset 和初始化语义必须兼容，tail 不能借用未初始化区域。
- 只在 UB 内清零不等价于保持 GM 输出语义；凡是外部可见或后续核读取的数据仍要按约定写回。
- 缓冲复用导致 tile 变小、增加同步或破坏双缓冲重叠时，应以峰值吞吐实测决定是否保留。

## 关联

- 源（4 次）：[ops-transformer `ee71f1b06`](https://gitcode.com/cann/ops-transformer/commit/ee71f1b06)（arch35 LightningIndexer：UB 复用优化）、[ops-transformer `96de5e5b4`](https://gitcode.com/cann/ops-transformer/commit/96de5e5b4)（arch35 LightningIndexer：继续收缩临时 UB buffer）、[ops-transformer `1820fdaf3`](https://gitcode.com/cann/ops-transformer/commit/1820fdaf3)（arch35 FlashAttentionScoreGrad：UB 清零替代 GM 清零）、[ops-transformer `316c08742`](https://gitcode.com/cann/ops-transformer/commit/316c08742)（arch35 QBSA：清理无效 buffer 并复用 `qscale` buffer）。
- 相邻卡片：mem-02（生命周期重叠时转双缓冲）、vec-05（可完全消除 UB staging 时进一步删中转）。
