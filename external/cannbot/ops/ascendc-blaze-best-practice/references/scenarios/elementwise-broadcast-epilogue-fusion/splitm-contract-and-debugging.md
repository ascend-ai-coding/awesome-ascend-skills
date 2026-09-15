# SplitM Elementwise/Broadcast Epilogue 契约与定位

本文只在当前需求、MatMul 基础合同和选定 Kernel 明确启用 M 维分片时使用。它不规定所有 SplitM 的 ratio、trait、slot 或同步；每个公式和地址必须绑定当前 witness 与验证记录。

## 1. 入口门禁

DESIGN 进入本专题前必须已有：

```text
matmul_base_analysis: ready
splitM_requested: true
splitM_source_contract: resolved
splitM_kernel_lifecycle: resolved
```

若未激活，写 `not_applicable`；若只发现名称线索或字段语义 unknown，保持 blocking/unknown，不推导行分配。

## 2. 四层合同

分别冻结：

1. **Block层**：L0C/UB 逻辑和物理 shape、Copy extent、alignment、final/partial、sub 可见性；
2. **Kernel层**：实际 AIC/AIV ratio、sub index、flag/pipe、slot、首轮、empty task、reuse 和 final drain；
3. **Epilogue层**：localRows、UB C 起点、GM operand/output 全局 offset、stage pitch、stride/extent 单位和 mask/tail；
4. **验证层**：实际 blockShape、odd/even M、N alignment、multi-tile、slot reuse、五模式和重复回归。

每层都必须有 source evidence；device 规范还必须有同一 Blaze 组装方案的正向/负向精度证据。单 shape 通过、相似实现或最终版本单次通过不足以标记已验证。

## 3. 行分配和地址规则

先从当前 Kernel witness 恢复实际分配规则，再在 DESIGN 中写成符号合同。例如，若证据确实证明两个 sub 均分当前 tile，才可以使用：

```text
chunkRows = ceil_div(curM, ratio)
subStart = sub * chunkRows
localRows = min(chunkRows, max(curM - subStart, 0))
```

`ratio`、sub 范围和 odd-M 分配不能预先固定。对每个对象分别说明：

| 对象 | 必须确认 |
|---|---|
| UB C | 当前 slot 是否已写入本 sub；是否允许再加 sub offset |
| GM full-tensor operand | 是否按全局行增加 subStart；row pitch/offset 单位 |
| GM output | 是否与 operand 使用同一全局行基准；non-overlap/final |
| Empty sub | Epilogue 是否跳过访问；Kernel 如何释放/通知 |

常见错误是对已经按 sub 写入本地 slot 的 UB C 再叠加 subStart；是否存在该风险必须由当前 Copy 和 adapter 证据决定。

## 4. Slot、alignment 和容量

对每个声明的 slot 方案写出：

```text
slot_count
slot_base_unit
slot_capacity
C/staging/output ranges
per-dtype alignment
stage row pitch
reuse wait and release
```

C、每路额外 input、中间值和 output 独立计算 alignment/row bytes；GM 有效 `curN`、Copy extent 和 UB pitch 不混用。所有 staging 必须落在当前 slot，整 VF load 的 guard 只有在目标 API 需要且已证明时才计入。

## 5. 同步和根因定位

先执行隔离路径，再按最早失败归因：

1. C-direct-GM：确认 MatMul 主体；
2. C-through-fusion：确认 L0C2UB、slot 和 C-ready；
3. V-zero-C/V-known-C：确认 operand offset、mask、公式和 output；
4. Full：确认真实 sub、reuse、RAW 和 final drain。

若怀疑某个 trait、offset、ratio 或 bridge，必须有固定基线、只撤变量的负向、只恢复变量的正向、相关 boundary 和清理后 Full 回归。否则结论标记 `unverified`，不得写成根因。

## 6. PLAN 输出

Step 3 只有在 DESIGN 激活 SplitM 时，才在 PLAN 中实例化：

- 真实 ratio/sub/slot/offset/row-pitch 字段；
- odd-M、empty sub、alignment/tail 和复用动作；
- row/column-encoded、known-C 和最早失败域诊断；
- required 负/正对照、最终 Full 和清理回归。

不激活的 SplitM action 必须显式记录 N/A 依据。任何新的 SplitN、ratio、dtype、Kernel 控制流或 shape 组合先回 Step 2/3，不由 Step 4扩展。
