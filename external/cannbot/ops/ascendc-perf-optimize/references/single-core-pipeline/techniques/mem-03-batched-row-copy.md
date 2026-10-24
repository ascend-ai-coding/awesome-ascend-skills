# 技术卡片 mem-03：按 UB 预算批量搬行与合并队列

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：访存 / MTE
- **一句话**：相邻行同进同出、生命周期相同的逻辑张量，按 UB 预算合并成一次多行搬运和一个队列，减少 copy 发射、EnQue/DeQue 与同步次数。

## 何时用（诊断特征）

- 循环中反复按单行或单个逻辑张量搬入/写出，而相邻行在 GM 上规则排布，且下游会在同一阶段消费。
- trace 有密集的小 `DataCopyPad`/`DataCopyExt`、多个短生命周期 queue；MTE 利用率不高但指令与同步开销明显。
- 输入行完整落入 UB 后可连续处理，或多个输出可在 UB 累积后一次写回。

## 原理

批量化先减少传输描述符和同步次数，再让 MTE 处理更长的连续块。合并队列将成对数据放在一个已对齐的物理 buffer 的不同 offset，使用一次 Alloc/EnQue/DeQue 管理；它只改变物理布局，不改变两个逻辑张量各自的计算语义。批量大小、queue 深度和留给索引/临时量的空间必须共同由 tiling 求解。

## 怎么改（before / after）

```cpp
// 伪代码：ChooseRowsPerCopy 由 host tiling 的 UB 预算求解；参数结构按实际 CANN API 填写。
// before: 每行单独搬运；两组同生命周期数据分别建 queue
for (uint32_t r = 0; r < rows; ++r) {
    DataCopyPad(xUb, xGm[r * rowStride], oneRow);
    ComputeOneRow(xUb);
    DataCopyPad(yGm[r * outStride], yUb, oneRow);
}

// after: 一次搬多行，成对逻辑张量在同一 queue 中分区
uint32_t rowsPerCopy = ChooseRowsPerCopy(availableUb, rowBytes, indexBytes, bufferNum);
LocalTensor<float> pair = pairQueue.AllocTensor<float>();
LocalTensor<float> a = pair;
LocalTensor<float> b = pair[alignedRowElems];
DataCopyPad(xUb, xGm[rowBase], {rowsPerCopy, rowBytes, srcStride, dstStride}, pad);
ComputeRows(xUb, rowsPerCopy, a, b);
pairQueue.EnQue(pair);
// 多个连续输出准备好后用同一批量参数写回
DataCopyPad(yGm[rowBase], yUb, {rowsPerCopy, outRowBytes, outSrcStride, outDstStride});
```

可以按 `rowsPerCopy * rowBytes`、索引/scale 临时区、输出区和 `bufferNum` 反推最大 batch；当整行能装入 UB 时再提高 buffer 深度。输出批量也要设置字节上限，避免一次过大 copy 让 UB 常驻数据失去空间。

## 性能数据

比较 MTE copy 次数、queue 操作次数和 `aiv_time`。重点观察小行、多行重复和成对结果场景；若行宽过小或行间不连续，应保留单行/通用搬运路径。

## 红线（Stop rules）

- 合并的逻辑张量必须生命周期一致，且每段 offset、长度和对齐由同一份 host/kernel 布局计算；不能为了合并而让一方覆盖另一方。
- `DataCopyPad` 的 `srcStride`、`dstStride` 分别按 GM 与 UB 的物理 pitch 推导，勿把 element 数和 byte 数混用。
- queue 深度增大时，host UB 可用空间也要同步扣减；空间不足、tail 很碎或行间无连续性时退回单行路径。
- 不能把“读一次多行”误用到后续不会消费的行；无效搬运会抵消收益。

## 关联

- 源（4 次）：[ops-transformer `8e2d0ae1`](https://gitcode.com/cann/ops-transformer/commit/8e2d0ae1)（arch35 MoE：按 UB 余量选择 `xCopyInQueueBufferNum`，读一次搬多行）、[ops-nn `cfa80545`](https://gitcode.com/cann/ops-nn/commit/cfa80545)（arch35 BatchNormGrad：成对 `dbeta/dgamma` 队列合并和多行 `DataCopyPad`）、[ops-nn `e2cad297f`](https://gitcode.com/cann/ops-nn/commit/e2cad297f)（arch35 `unique_dim`：提高 copy 粒度）、[ops-nn `2d82cda51`](https://gitcode.com/cann/ops-nn/commit/2d82cda51)（arch35 `scatter_update`：切分列后一次搬入多个 update）。
- 相邻卡片：mem-02（批量 tile 的双缓冲）、mem-01（连续主区的对齐搬运）。
