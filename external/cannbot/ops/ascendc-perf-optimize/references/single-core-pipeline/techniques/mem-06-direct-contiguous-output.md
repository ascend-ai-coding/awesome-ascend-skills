# 技术卡片 mem-06：连续结果一次性直写

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：访存 / MTE
- **一句话**：当输出区间已连续且本轮结果完整时，直接按大块写回 GM；删除 gather-out 的逐段搬出、冗余拷贝和无意义的中间队列。

## 何时用（诊断特征）

- 计算结果在 UB 中已按输出地址连续排布，却仍按 token、row、expert 或片段逐个 copy out。
- gather/scatter 的非连续分支覆盖了连续输出 case，导致普通 case 也承担索引计算和多次 MTE 发射。
- 后续没有对中间输出作变换，结果只需要写入最终 GM 区间。

## 原理

对连续完整的结果，最佳搬出路径是一次或少量大块 copy。将连续性作为 tiling 路由条件，连续 case 直接从 UB 写最终 GM；非连续 case 保留 gather/scatter 模板。收益来自少掉的 copy 描述、索引访问、queue 操作与同步，而不是改变最终字节数。

## 怎么改（before / after）

```cpp
// 伪代码：IsSingleContiguousRange 由输出 layout、offset 和本轮有效元素共同证明。
// before: 对每个逻辑片段计算目的地址并分别搬出。
for (uint32_t part = 0; part < partNum; ++part) {
    CopyOut(outGm[OutputOffset(part)], resultUb[ResultOffset(part)], partBytes);
}

// after: 连续完整区间直接一次性写回；其余规格走原通用模板。
if (IsSingleContiguousRange(outputLayout, validElems)) {
    CopyOut(outGm[baseOffset], resultUb, totalBytes);
} else {
    GatherOrStridedCopyOut();
}
```

如果连续区间超过 UB 中当前结果范围，可分为少量连续 chunk；不要为了凑成一次 copy 额外重排或复制整块数据。

## 性能数据

比较连续与非连续输出两组的 MTE copy 数、索引/Scalar 指令数和 `aiv_time`。连续快路径应覆盖完整结果区间；有 hole、重复写或交错 layout 的 case 必须回退。

## 红线（Stop rules）

- 连续性需包含目的地址递增、元素无洞、无别名冲突和本轮结果完整四项，不能只检查逻辑 token 连续。
- 写回长度只覆盖有效元素；尾块、drop/pad 和空输出不得按对齐长度越界写 GM。
- 如果非连续路径本身已由 DMA stride 高效表达，不要强行先在 UB 重排再直写。
- 输出存在跨核重叠或原子累加语义时，不能改为普通直写。

## 关联

- 源（3 次）：[ops-transformer `4bb447da4`](https://gitcode.com/cann/ops-transformer/commit/4bb447da4)（arch35 MoE Init Routing：非量化场景连续一次性搬出）、[ops-transformer `6462740a6`](https://gitcode.com/cann/ops-transformer/commit/6462740a6)（arch35 MoE：清理 gather-copy 冗余计算）、[ops-math `e11cf551`](https://gitcode.com/cann/ops-math/commit/e11cf551)（arch35 Concat：调整 pure-copy 准入条件）。
- 相邻卡片：mem-03（输入阶段按行批搬）、mem-07（整体语义退化为 copy 时走专用模板）。
