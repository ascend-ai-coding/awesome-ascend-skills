# 技术卡片 vec-07：硬件树形归约

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：VEC
- **一句话**：用硬件树形归约指令替代手写 for 逐元素归约，把归约链长从 N 降到 log N。

## 何时用（诊断特征）

- 算子含 reduction（sum/max/min），且实现是**手写 for 循环逐元素累加**。
- VECTOR 泳道出现因累加器写回等待的周期性空泡（归约串行链长）。
- `aiv_vec_ratio` 高，归约类指令（`vec_reduce_sum`/`vec_reduce_max`）占比可观。

## 原理

手写 `for` 逐元素归约是单累加器 RAW 串行链，链长 ≈ 元素数，硬件无法并行。硬件树形归约指令（`ReduceSum`/`ReduceMax`/`ReduceMin`）内部按树形两两合并，链长降到 log 级，并吃满 lane。

## 怎么改（before / after）

```cpp
// before：手写 for 逐元素归约，串行链长 = count
float acc = 0.0f;
for (uint32_t i = 0; i < count; ++i) {
    acc += src.GetValue(i);          // 逐元素标量累加，V_S/S_V 同步伤流水
}

// after：硬件树形归约指令
ReduceSum(dstLocal, srcLocal, workLocal, count);   // 树形归约，lane 并行
// 归约维 R ≪ VL、需按行归约时，进一步用蝶式归约见 vec-06
```

## 性能数据

msprof `aiv_time`：手写标量归约 → 硬件 `ReduceSum`，消除逐元素 V_S/S_V 同步与串行链。**待实测**（预期：归约元素数越大，树形化收益越明显）。

## 红线（Stop rules）

- 树形归约**改变浮点累加次序**，与串行结果不 bit 一致——fp32 中间累加缓解，须过 `func_cases` 精度回归。
- `ReduceSum`/`Brcb` 等需 64 对齐，相关 buffer 按 64 对齐分配。
- 结果落在 lane0，若后续要逐元素用需先广播（`Duplicate`/`Brcb`），见 vec-04/vec-05。

## 关联

- 源：`../vec.md` 策略 4（低延迟归约）。
- 相邻卡片：vec-08（单累加器串行链→多路累加）、vec-06（R≪VL 蝶式归约）、vec-04（归约后标量广播向量化）。
