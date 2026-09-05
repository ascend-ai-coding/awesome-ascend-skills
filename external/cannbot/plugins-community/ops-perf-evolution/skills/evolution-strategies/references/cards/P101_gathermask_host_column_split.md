---
id: P101
bottlenecks: [mte2_stall, scalar_loading]
op_families: [elementwise, special, index_scatter, attention]
complexity: L1
conflicts_with: []
synergizes_with: [P100, P95, P69]
requires: []
has_preconditions: false
has_playbook: true
---

# P101: GatherMask 替代 Host 侧列拆分 (GatherMask Replaces Host-Side Column Split)

## 核心思想

当输入为 interleaved 多分量格式（如 `(N,4)` 的 `[x1,y1,x2,y2, x1,y1,x2,y2, ...]`）时，传统做法是 Host 侧 `.select(1,i).contiguous()` 拆成独立 1D 张量传入 kernel。GatherMask 内置固定模式 (built-in patterns 3/4/5/6) 可以在 kernel 内一条 VEC 指令拆分一个分量，消除 Host 侧拷贝和多余 DMA：

```
传统: Host 8×.select().contiguous() → Kernel 8+ 路 DMA → Compute
P101:  Host 传原始 interleaved 张量 → Kernel 1 路 DMA → 4×GatherMask → Compute
```

GatherMask 的 4 条 VEC 指令开销远小于 Host 侧拷贝 + 多路 DMA 的累计 overhead。

## 代码骨架

```cpp
// === 改造前：Host 侧列拆分 + 多路 DMA ===
// Host:
at::Tensor x1 = bboxes.select(1, 0).contiguous();  // 4 路 .select().contiguous()
at::Tensor y1 = bboxes.select(1, 1).contiguous();
at::Tensor x2 = bboxes.select(1, 2).contiguous();
at::Tensor y2 = bboxes.select(1, 3).contiguous();
// Kernel: 4 路 DMA
CopyIn(x1Tile, x1Gm[tileStart], tileLen);  // 4 次 DataCopyPad
CopyIn(y1Tile, y1Gm[tileStart], tileLen);
CopyIn(x2Tile, x2Gm[tileStart], tileLen);
CopyIn(y2Tile, y2Gm[tileStart], tileLen);

// === 改造后：Kernel 内 GatherMask 解交织 ===
// Host: 直接传原始 (N,4) 张量，不做列拆分
// Kernel: 1 路 DMA + 4×GatherMask
CopyIn(interleaved, bboxesGm[tileStart * 4], tileLen * 4);  // 1 次 DataCopyPad
PipeBarrier<PIPE_ALL>();

// Normal 模式: 每次 repeat 处理 256B = 64 float32 → 16 dst per repeat
// pattern 3 (00010001): 每 4 个取第 1 个 → x1
// pattern 4 (00100010): 每 4 个取第 2 个 → y1
// pattern 5 (01000100): 每 4 个取第 3 个 → x2
// pattern 6 (10001000): 每 4 个取第 4 个 → y2
int32_t alignedLen = (tileLen / 16) * 16;
uint16_t repeatTimes = alignedLen / 16;
GatherMaskParams gmParams(1, repeatTimes, 8, 0);  // stride=8 (256B) 不可写成 1

uint64_t rsvdCnt = 0;
GatherMask(x1, interleaved, static_cast<uint8_t>(3), false, 0, gmParams, rsvdCnt);
GatherMask(y1, interleaved, static_cast<uint8_t>(4), false, 0, gmParams, rsvdCnt);
GatherMask(x2, interleaved, static_cast<uint8_t>(5), false, 0, gmParams, rsvdCnt);
GatherMask(y2, interleaved, static_cast<uint8_t>(6), false, 0, gmParams, rsvdCnt);
PipeBarrier<PIPE_ALL>();

// 尾块: tileLen 非 16 倍数时标量补齐 alignedLen..tileLen-1
for (int32_t j = alignedLen; j < tileLen; ++j) {
    int32_t off = j * 4;
    x1.SetValue(j, interleaved.GetValue(off));
    y1.SetValue(j, interleaved.GetValue(off + 1));
    x2.SetValue(j, interleaved.GetValue(off + 2));
    y2.SetValue(j, interleaved.GetValue(off + 3));
}
```

## 关键修改点

1. **解交织维度放外层循环**：解交织后 UB 常驻，内层维度全量复用；放内层则每 tile 都要重 DMA + 重 GatherMask，DMA 复杂度 O(外层×内层/tile²)
2. **Tile 填到 UB 上限**：tile 过小会导致 DMA 次数随 N 线性增长，压过 GatherMask 收益。tile 填到 UB 上限（910B2C 约 2048）。

3. **Normal 模式 + `src0RepeatStride=8`**：Normal 模式每次 repeat 处理 256B（64 float32），stride=8 DataBlocks 保证 repeat 间不重叠。stride=1 会导致连续 repeat 重叠 224B → 全 case 精度 fail
4. **16 对齐 + 标量尾块**：Normal 模式每次 repeat 输出 16 个 dst 元素，tileLen 向下对齐到 16 的倍数交 GatherMask，剩余 0-15 个元素走标量补齐
5. **只解交织需要复用的大维度**：小维度用标量 GM 访问即可，两侧都做 GatherMask 增加 UB 压力和指令数但无额外复用收益

## 适用性检测 (grep)

```bash
# 检测 Host 侧列拆分（本策略的替代目标）
grep -nE "\.select\(.*\)\.contiguous\(\)" op_host/*.cpp

# 检测 interleaved 多分量输入格式 (N,4) (N,3) (N,2)
grep -nE "size\(1\)\s*==\s*[234]" op_host/*.cpp

# 检测 GatherMask 是否已使用 built-in patterns 3/4/5/6
grep -nE "GatherMask.*static_cast<uint8_t>\([3-6]\)" op_kernel/*.cpp
```

## 必然推论: 被消除的 GM Tensor 所承载的计算必须迁移到 Kernel 侧

当 Host 侧列拆分被消除后，GM 输入数量从 N 路减少到 1 路（interleaved）。**任何原先对这 N 路 GM tensor 做的 Host 侧预处理或预计算，其结果不再有独立的 GM 通道传入 kernel**，必须满足以下二选一：

| 情形 | 判定条件 | 处理方式 |
|------|---------|---------|
| 可以直接丢弃 | 预计算结果仅用于减少 kernel 计算量，kernel 有原始分量后可在 UB 上重算 | **在 kernel 内用向量指令重算**（通常 2-3 条 VEC 指令，成本远低于单独 DMA） |
| 不能丢弃 | 预计算结果依赖运行时才能确定的数据（如索引表、动态阈值） | 预计算结果保留为独立 GM 输入，但需评估：剩余 DMA 开销是否超过了 P101 节省的 DMA 开销？若超过，P101 不可行 |

**实施检查清单**（采用 P101 时必须逐项通过，否则为无效实现）：

```bash
# 1. 列出基线 op_host 中所有 GM 输入 tensor（包括预计算产物）
grep -nE "EXEC_KERNEL_CMD|SetGlobalBuffer" op_host/*.cpp op_kernel/*.cpp

# 2. 对每个被消除的 GM 输入，回答：
#    - 这个 tensor 是原始数据还是预计算结果？
#    - 如果是预计算结果：它的输入数据量是什么？能迁移到 kernel UB 上算吗？
#    - 如果不能迁移：它单独占一条 DMA——P101 省下的 DMA 够抵消吗？

# 3. 改造后验证：kernel 的 SetGlobalBuffer 数量应严格 ≤ 改造前 - 被消除的原始分量数
```

**典型反模式**：
- 消除了输入张量的 C 路列拆分 DMA
- **但没有消除**对拆分后各列做的预计算结果 DMA（如各列差值乘积，完全可以在 UB 上用 2-3 条向量指令重算）
- 结果：节省的 DMA 收益被残留的预计算结果 DMA 抵消，P101 退化为无效

**正确做法**：
- 列拆分 → GatherMask：消除 C 路 DMA
- 与拆分列相关的预计算 → kernel UB 内若干条向量指令重算：消除对应 DMA
- 与拆分列无关的预计算（原本走标量路径）→ 保持不变
- 最终 GM 输入数量应严格等于：未拆分的原始 tensor 数 + 与列拆分无关的辅助 tensor 数 + output

## 常见陷阱

⚠️ **解交织维度必须外层循环**：内层循环会导致每 tile 重 DMA + 重 GatherMask，DMA 复杂度从 O(N) 变为 O(M×N)
⚠️ **tile 过小导致收益被 DMA 次数抵消**：tile 越小，DMA 次数随 N 线性增长，大 N 时退化严重
⚠️ **残留的 GM 预计算结果会抵消 P101 收益**：见上方"必然推论"——每一条未迁移到 kernel 侧计算的预计算结果都有独立 DMA 开销

> GatherMask API 参数约束（stride=8、16 对齐、PipeBarrier 等）见 `api-gathermask.md`。

## 代码搜索关键词

```bash
grep -nE "GatherMask|GatherMaskParams|src0RepeatStride|interleaved|deinterleave" op_kernel/*.cpp
```

## 来源

- 31_IOU GatherMask vs Host 预拆分对照实验
- P100 (GatherMask 替代标量解交织) 覆盖 RoPE 奇偶拆分等步长取元素场景，本卡片覆盖列分量分离场景，两者互补
