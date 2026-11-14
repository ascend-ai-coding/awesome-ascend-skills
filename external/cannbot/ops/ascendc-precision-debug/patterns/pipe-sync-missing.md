# pipe 同步缺失（SetFlag/WaitFlag）

## 形态指纹

- **同一输入多次上板运行，结果偶发不一致**（稳定性分流结论 = 同步/内存类）的最典型根因。
- 偶发超差：跑多次，有时过有时不过，无固定规律。
- 多发于：手写多 stage 流水（MTE2 搬运 → V 计算 → MTE3 写回）、复用同一块 UB 的场景。

## 根因机理

NPU 各 pipe（MTE2/V/MTE3 等）异步并行。对同一块 buffer 的读写跨 pipe 时，若缺少 SetFlag/WaitFlag 同步，下游 pipe 可能读到上游尚未写完的旧数据。时序竞争导致结果随运行波动。

## 验证方法

1. 同一输入连跑 5 次以上：出现偶发不一致 → 高度疑似（与"每次都错且错得一致"的逻辑问题区分）。
2. 在可疑读写之间插入同步（或保守起见插入全同步）后重跑：多次运行全部通过 → 确认。
3. 检查复用 UB 的位置：同一 LocalTensor 被两个 pipe 交叉使用处是重灾区。

## 修复 pattern

```cpp
// 反例：MTE2 搬运未完成，V 直接算
DataCopy(xUb, xGm, len);
Add(yUb, xUb, zUb, len);   // 可能读到未搬完的数据

// 正例：跨 pipe 加同步
DataCopy(xUb, xGm, len);
SetFlag<HardEvent::MTE2_V>(eventId);
WaitFlag<HardEvent::MTE2_V>(eventId);
Add(yUb, xUb, zUb, len);

// V 算完，MTE3 写回前同理：SetFlag<HardEvent::V_MTE3> / WaitFlag
```

注意 eventId 的申请/释放配对；TQue 管理的队列自带同步，手动 TBuf 复用才需要显式 flag。

## 反例（像但不是）

- 结果每次都稳定地错且错得一致 → 是逻辑/维度问题，不是同步；同步问题通常偶发。
- 仅个别随机 bit 翻转且无规律 → 先排除硬件/环境问题再归到同步。
