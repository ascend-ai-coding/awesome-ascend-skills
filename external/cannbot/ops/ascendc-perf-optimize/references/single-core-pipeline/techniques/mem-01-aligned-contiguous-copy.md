# 技术卡片 mem-01：对齐并连续化访存布局

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：访存 / MTE
- **一句话**：将 tile、行跨度和 copy 边界设计为硬件对齐的连续区间；非连续或尾块先走受控模板，不能让主路径携带非对齐的地址与 stride 开销。

## 何时用（诊断特征）

- 主路径频繁走 `DataCopyPad`、非对齐 copy 或短尾分支，MTE 实际传输效率低于连续大块搬运。
- 逻辑连续但物理 row stride、内轴长度、L1 切分块或格式转换边界没有满足对齐要求。
- 同一算子同时覆盖对齐和非对齐规格，后者迫使全部规格使用保守 copy 模板。

## 原理

连续、对齐的读写能降低传输描述符、padding 和尾块处理成本，并使 UB/L1 与 cache-line 分段稳定。优化不是把任意 shape 强行补齐，而是将合法的对齐条件显式进入 tiling：主模板只处理连续对齐区间，尾块或非连续输入使用独立正确路径。这样快路径没有额外分支，慢路径也不会越界或误读。

## 怎么改（before / after）

```cpp
// 伪代码：AlignBytes/IsContiguous/CopyAligned 代表由 tiling 和当前 CANN API 实现的判定与搬运。
// before: 所有规格沿用同一保守 copy，主路径也承担 stride、padding 和 tail。
CopyWithGenericStride(inGm, outUb, logicalRows, logicalCols, logicalStride);

// after: tiling 按物理连续性与对齐选择主/尾模板。
bool fastPath = IsContiguous(layout) && IsAligned(innerBytes, alignBytes) &&
                IsAligned(l1SliceBytes, alignBytes);
if (fastPath) {
    CopyAligned(inGm, outUb, alignedBytes);
} else {
    CopyStridedOrTail(inGm, outUb, physicalStride, validBytes);
}
```

对齐条件必须按 dtype 和实际物理 format 推导；NZ/ND、scale、不同转置方向的连续内轴并不相同。若只能让大部分 tile 对齐，应将主区与 tail 分开切分，避免主循环每轮做尾块判断。

## 性能数据

比较对齐与非对齐模板的 MTE 有效字节率、copy 指令数和 `aiv_time`。分开覆盖连续主区、非连续输入、未对齐内轴和尾块；优化后只有满足谓词的 case 应切到快路径。

## 红线（Stop rules）

- 不能仅按逻辑 shape 判定连续；要同时检查 format、实际 pitch、offset、转置和 dtype 字节数。
- padding 的填充值、tail mask、写回范围与访问上界必须在慢路径独立验证，不能复用快路径长度。
- 对齐阈值来自当前 API/硬件契约；950 历史提交中出现 16B、32B、64B、128B 等不同约束，不能抽成一个全局常量。
- 若对齐切分显著缩小 tile、降低并行度或增加无效搬运，应保留通用路径并以实测决定路由。

## 关联

- 源（5 次）：[ops-nn `52e3d09b1`](https://gitcode.com/cann/ops-nn/commit/52e3d09b1)（arch35 Conv2D：补齐 C04 `DataCopy` 地址对齐）、[ops-nn `0be050c50`](https://gitcode.com/cann/ops-nn/commit/0be050c50)（arch35 BatchNorm：为 aligned UB 采用独立模板）、[ops-transformer `45032c965`](https://gitcode.com/cann/ops-transformer/commit/45032c965)（arch35 MoE：新增非 64B 对齐搬运路径）、[ops-transformer `ab3ec092b`](https://gitcode.com/cann/ops-transformer/commit/ab3ec092b)（arch35 PANZ：scale 搬入 UB 的 32B 对齐）、[ops-transformer `191b48ba6`](https://gitcode.com/cann/ops-transformer/commit/191b48ba6)（arch35 DropPad：消除重复 `AlignBytes` 计算并优化对齐路径）。
- 相邻卡片：mem-03（将连续行合并批搬）。
