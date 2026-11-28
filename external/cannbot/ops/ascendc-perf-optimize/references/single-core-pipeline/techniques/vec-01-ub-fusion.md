# 技术卡片 vec-01：UB 内融合，消除 GM 往返

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：VEC / 访存
- **一句话**：相邻 Vector 操作在 UB 内接力完成，中间结果不写回 GM，省掉多次 MTE2/MTE3 往返。

## 何时用（诊断特征）

- trace 中两个 VECTOR 活跃段之间插入了 **MTE3（写回）+ MTE2（读入）**，说明中间结果经过了 GM，未融合。
- `aiv_mte2_ratio` / `aiv_mte3_ratio` 偏高但计算量本身不大（搬运快于计算，却因往返而空转）。
- 算子由多个 elementwise/activation 步骤串成，且各步骤之间落 GM 中转。

## 原理

多个连续 Vector 操作直接在 UB（或向量寄存器）中传递中间结果，消除不必要的 GM↔UB 搬移。GM 访问延迟远高于 UB，把 N 步链路从「每步一来一回 GM」压成「入口 load 一次、出口 store 一次」，直接砍掉中间的 MTE2/MTE3 指令。

## 怎么改（before / after）

```cpp
// before：中间结果落 GM 再读回 —— 6 次 GM 访问
DataCopy(ub, gmIn,  len);   Compute1(ub);   DataCopy(gmMid, ub, len);  // 写回 GM
DataCopy(ub, gmMid, len);   Compute2(ub);   DataCopy(gmOut, ub, len);

// after：中间结果留在 UB 内接力 —— 2 次 GM 访问
DataCopy(ub, gmIn, len);
Compute1(ub);               // 结果留 UB
Compute2(ub);               // 直接在 UB 上接着算
DataCopy(gmOut, ub, len);
```

链式算子（如 `Mul` 后接 `Add`、多个激活函数串联）优先合并到一个 UB 驻留窗口内完成；能进一步用融合指令的转 vec-03。

**链已 UB 驻留后的下一步**：本卡片的 after 形态（中间结果留 UB 接力）达成后若 VEC 耗时仍主导，本卡片无动作——继续到 vec-05 的寄存器驻留改写（消除每个向量 API 的 UB 读写），这是深度 VEC bound 的正常分析路径而非例外。

## 性能数据

msprof `aiv_time` 前后对比：融合后 MTE2/MTE3 指令数按被消除的中转次数下降；访存往返是主要开销时收益明显。**待实测**（预期方向：中间落 GM 的步数越多，省掉的往返越多）。

## 红线（Stop rules）

- 中间结果尺寸必须放得下 UB（含 DoubleBuffer 时可用容量减半）；放不下就得分块，融合窗口随之受限。
- 若中间结果在别处仍被复用，不能简单删除中转。
- 融合改变了 buffer 生命周期，注意 EnQue/DeQue 同步不要漏，避免读到未写完的数据。

## 关联

- 源：`../vec.md` 策略 1（UB 融合）。
- 相邻卡片：vec-03（融合指令进一步减指令数）、vec-05（消除中间 UB staging 的寄存器内直算）。
