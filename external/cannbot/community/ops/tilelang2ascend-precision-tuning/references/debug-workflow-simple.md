# 简单 Vector 算子调试流程

适用于单核、线性数据流的算子（Add、Mul、Softmax、LayerNorm 等）。

## 核心原则

- 在 GM 数据点上追踪：CopyIn 后、Compute 步骤、CopyOut 前。
- 系统化编号便于定位。
- 先验证输入，再验证计算。
- 某一步结果错误时，必须先回溯该步输入是否已经错误，再检查该步公式/API 参数。
- 默认先 dump 首个 tile 的前 8/16/32 个元素；输出过多时先缩 case，再扩大 dump。

## 调试步骤

### Step 1：验证输入数据

在 CopyIn（DataCopy 从 GM 到 LocalTensor）后：

```cpp
LocalTensor<T> inputLocal = inQueue.DeQue<T>();
// Dump 输入
DumpTensor(inputLocal, 400, 32);
```

### Step 2：验证计算中间结果

每个 Compute 步骤后：

```cpp
// 计算 1
Adds(tmpLocal, inputLocal, 1.0f, tileLength);
DumpTensor(tmpLocal, 500, 32);

// 计算 2
Mul(outputLocal, tmpLocal, scaleLocal, tileLength);
DumpTensor(outputLocal, 510, 32);
```

### Step 3：验证输出数据

在 CopyOut（DataCopy 写回 GM）前：

```cpp
// 写回前 dump
DumpTensor(outputLocal, 600, 32);
outQueue.EnQue(outputLocal);
```

## 编号约定

| 范围 | 阶段 | 说明 |
|------|------|------|
| 400-499 | V输入 | CopyIn 后，队列 DeQue 后 |
| 500-599 | V中间 | 每个 Compute 步骤后，步内递增 10 |
| 600-699 | V输出 | CopyOut 前，队列 EnQue 前 |

## CPU Golden 对照

在 PyTorch 参考实现中，打印相同位置：

```python
# 输入
print(f"[CPU-400] input[:8]: {input_tensor[:8]}")

# 计算中间
tmp = input + 1.0
print(f"[CPU-500] tmp[:8]: {tmp[:8]}")

# 输出
output = tmp * scale
print(f"[CPU-600] output[:8]: {output[:8]}")
```

## 快速定位策略

1. **输入错误** → 检查 DataCopy 参数、GM offset 计算、stride/shape
2. **计算中间错误** → 先回查该步输入 dump 是否正确，再检查 API 参数、顺序、广播、类型转换、同步
3. **输出正确但 GM 写入错误** → 检查 CopyOut offset、stride、EnQue/DeQue 顺序

## 输出采集

- DumpTensor 直接输出到屏幕，不会额外生成日志。
- 屏幕信息过多时，把任务输出重定向到 debug 文件，再按 `desc` 过滤读取。

## 定位完成后

移除所有 DumpTensor，重新验证。
