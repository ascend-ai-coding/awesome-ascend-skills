# 阶段 5：流水优化 — 详细参考

## 5.1 CopyIn / Compute / CopyOut 范式

将算子划分为三级流水任务，使用 `TQue` 进行级间同步。不同阶段映射到独立的
硬件指令队列（MTE2/MTE3 搬运、V 矢量、M 矩阵），可并行执行。

```
CopyIn  → AllocTensor + DataCopy(GM→Local) + EnQue     [MTE2 队列]
Compute → DeQue + Vector/Cube 运算 + EnQue              [V / M 队列]
CopyOut → DeQue + DataCopy(Local→GM) + FreeTensor       [MTE3 队列]
```

基础框架：

```cpp
TPipe pipe;
TQue<VecIn, 1> queIn;
TQue<VecOut, 1> queOut;

pipe.InitBuffer(queIn, 2, 1024);   // double buffer

for (int i = 0; i < tileCount; i++) {
    // CopyIn
    auto tensor = queIn.AllocTensor<half>();
    DataCopy(tensor, gm, len);
    queIn.EnQue(tensor);

    // Compute
    auto tensorIn = queIn.DeQue<half>();
    auto tensorOut = queOut.AllocTensor<half>();
    Abs(tensorOut, tensorIn, 1024);
    queIn.FreeTensor(tensorIn);
    queOut.EnQue(tensorOut);

    // CopyOut
    auto result = queOut.DeQue<half>();
    DataCopy(gmOut, result, 1024);
    queOut.FreeTensor(result);
}
```

同一数据切片内，CopyIn → Compute → CopyOut 必须串行。但不同切片可重叠：
Compute 处理切片 N 时，CopyIn 可搬入切片 N+1，CopyOut 可搬出切片 N−1。

## 5.2 Double Buffer

`InitBuffer` 的 buffer 个数设为 **2**，使搬运与计算重叠执行。

**反例** — 未使能 double buffer（Vector 利用率约 33%）：

```cpp
pipe.InitBuffer(inQueueSrc0, 1, sizeSrc0 * sizeof(half));   // 单 buffer
pipe.InitBuffer(inQueueSrc1, 1, sizeSrc1 * sizeof(half));
pipe.InitBuffer(outQueueDst, 1, sizeDst0 * sizeof(half));

for (uint32_t index = 0; index < round * 2; ++index) {
    CopyIn(index);    // MTE2 忙，Vector 闲
    Compute();        // Vector 忙，MTE 闲
    CopyOut(index);   // MTE3 忙，Vector 闲
}
```

**正例** — 使能 double buffer（下一 tile 的 CopyIn 与当前 tile 的 Compute 重叠）：

```cpp
pipe.InitBuffer(inQueueSrc0, 2, sizeSrc0 * sizeof(half));   // double buffer
pipe.InitBuffer(inQueueSrc1, 2, sizeSrc1 * sizeof(half));
pipe.InitBuffer(outQueueDst, 2, sizeDst0 * sizeof(half));

for (uint32_t index = 0; index < round; ++index) {
    CopyIn(index);    // 可与前一次 CopyOut 重叠
    Compute();        // 可与下一次 CopyIn 重叠
    CopyOut(index);   // 可与下一次 Compute 重叠
}
```

**注意事项**：
- 内存开销翻倍（每个队列分配 2 块 buffer）。
- 循环次数须 >= 2 才能获益。
- 当计算时间远大于搬运时间时，搬运已被隐藏，double buffer 收益有限。
- 当数据量很小，一次即可完成全部计算时，无需 double buffer。

## 5.3 异步 Iterate（MIX 模式，AIC+AIV）

Matmul MIX 场景下，`Iterate` / `IterateAll` 会在 AIV（Vector 核）和
AIC（Cube 核）之间发送同步消息。同步模式控制消息频率：

- `Iterate<true>()`（同步）：**每次**迭代发一条消息——开销大。
- `Iterate<false>()`（异步）：仅**第一次**发消息，后续迭代无需 AIC/AIV 同步。

**同步模式** — 每次迭代都有消息开销：

```
AIV: send_msg → wait → send_msg → wait → send_msg → wait
AIC:           exec →           exec →           exec
```

**异步模式** — 仅首次发消息：

```
AIV: send_msg → continue → continue → continue
AIC:           exec     → exec     → exec
```

代码示例：

```cpp
TQueBind<TPosition::CO2, TPosition::VECIN>    qVecIn;
TQueBind<TPosition::VECIN, TPosition::VECOUT> qVecOut;

mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
mm.SetWorkspace(workspace, singleCoreM * singleCoreN * sizeof(float));

while (mm.template Iterate<false>()) {    // 异步模式
    auto cInUB = qVecIn.AllocTensor<float>();
    mm.GetTensorC(cInUB);
    qVecIn.EnQue(cInUB);
    cInUB = qVecIn.Deque<float>();
    auto cOutUB = qVecOut.AllocTensor<float>();
    Muls(cOutUB, cInUB, scalar, baseM * baseN);
    qVecIn.FreeTensor(cInUB);
    // ... 后续处理
}
```

MIX 场景下默认使用异步模式。仅在需要严格的逐次迭代顺序保证（防止地址踩踏）
时才回退为同步模式。
