# mssanitizer 错误类型详解

本文档详细说明 mssanitizer 检测到的各类内存错误，包括错误原因、诊断方法和解决方案。

## 错误类型索引

| 错误类型 | 严重程度 | 常见场景 |
|---------|---------|---------|
| [非法内存访问](#非法内存访问-illegal-readwrite) | 🔴 严重 | DataCopy 参数错误、offset 计算错误 |
| [非法释放](#非法释放-illegal-free) | 🔴 严重 | 重复释放、释放未分配内存 |
| [内存泄漏](#内存泄漏-memory-leak) | 🟡 中等 | 缺少 FreeTensor、资源清理不完整 |
| [UB 地址越界](#ub-地址越界) | 🔴 严重 | DataCopyPad 未对齐、buffer 大小错误 |

---

## 非法内存访问

### 错误日志

```
====== ERROR: illegal read of size <N>
======    at 0x<address> on GM

====== ERROR: illegal write of size <N>
======    at 0x<address> on GM
```

### 常见原因与解决方案

#### 1. DataCopy 参数错误

```cpp
// ❌ 错误: DataCopy 长度计算错误
uint32_t elements = 1000;
DataCopy(xLocal, xGlobal[offset], elements);  // 可能访问越界

// ✅ 正确: 使用 DataCopyPad 并正确计算长度
uint32_t elements = 1000;
DataCopyExtParams copyParams{1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
DataCopyPad(xLocal, xGlobal[offset], copyParams, padParams);
```

#### 2. offset 计算错误

```cpp
// ❌ 错误: offset 计算未考虑边界
uint32_t offset = blockIdx * elementsPerCore;
DataCopyPad(xLocal, xGlobal[offset], copyParams, padParams);
// 可能超出 totalElements 范围

// ✅ 正确: 检查边界
uint32_t offset = blockIdx * elementsPerCore;
uint32_t actualElements = elementsPerCore;

if (offset + elementsPerCore > totalElements) {
    actualElements = totalElements - offset;
}

if (actualElements > 0) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(actualElements * sizeof(T)), 0, 0, 0};
    DataCopyPad(xLocal, xGlobal[offset], copyParams, padParams);
}
```

#### 3. 未正确处理尾部元素

```cpp
// ❌ 错误: 未处理尾部元素
for (uint32_t i = 0; i < loopNum; i++) {
    uint32_t elements = maxElementsPerLoop;  // 总是使用最大值
    Compute(offset, elements);  // 最后一次循环可能越界
}

// ✅ 正确: 处理尾部元素
for (uint32_t i = 0; i < loopNum; i++) {
    uint32_t offset = startOffset + i * maxElementsPerLoop;
    uint32_t elements = maxElementsPerLoop;
    
    if (offset + elements > startOffset + actualElements) {
        elements = startOffset + actualElements - offset;
    }
    
    if (elements > 0) {
        Compute(offset, elements);
    }
}
```

### 诊断方法

```cpp
// 使用 PRINTF 调试
AscendC::PRINTF("offset=%u, elements=%u, totalElements=%u\n", 
                offset, elements, totalElements);

// 检查 DataCopy 参数
AscendC::PRINTF("DataCopy: len=%u, sizeof(T)=%lu\n", 
                copyParams.len, sizeof(T));
```

```bash
# 开启日志打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=1
./your_test
```

---

## 非法释放

### 错误日志

```
[E] illegal free. memInfoSrc: 1, addr: 0x<address>
```

### 常见原因与解决方案

#### 1. 重复释放

```cpp
// ❌ 错误: 重复释放
LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
// ... 使用 xLocal
inQueueX.FreeTensor(xLocal);
// ...
inQueueX.FreeTensor(xLocal);  // 重复释放！

// ✅ 正确: 只释放一次
LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
// ... 使用 xLocal
inQueueX.FreeTensor(xLocal);
// 不再重复释放
```

#### 2. 释放未分配的内存

```cpp
// ❌ 错误: 释放未分配的内存
LocalTensor<T> xLocal;  // 未调用 AllocTensor
inQueueX.FreeTensor(xLocal);  // 非法释放！

// ✅ 正确: 先分配再释放
LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
// ... 使用 xLocal
inQueueX.FreeTensor(xLocal);
```

#### 3. CANN Runtime 内部问题

某些情况下，CANN Runtime 内部的内存管理可能导致误报。

**诊断方法**:
```bash
# 检查 CANN 版本
cat /usr/local/Ascend/cann/version.info

# 查看详细日志
export ASCEND_SLOG_PRINT_TO_STDOUT=1
./your_test
```

**解决方案**:
- 升级 CANN 版本
- 检查 CANN Runtime 版本兼容性
- 联系华为技术支持

---

## 内存泄漏

### 错误日志

```
====== ERROR: LeakCheck: detected memory leaks
====== Leaked block at 0x<address> of size <N>
```

### 常见原因与解决方案

#### 1. 缺少 FreeTensor 调用

```cpp
// ❌ 错误: 缺少 FreeTensor
LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
// ... 使用 xLocal
// 缺少: inQueueX.FreeTensor(xLocal);

// ✅ 正确: 配对使用
LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
// ... 使用 xLocal
inQueueX.FreeTensor(xLocal);
```

#### 2. 异常路径未清理资源

```cpp
// ❌ 错误: 异常路径未清理
void* devInput = nullptr;
aclrtMalloc(&devInput, size, ACL_MEM_MALLOC_HUGE_FIRST);

if (some_error) {
    return;  // 未释放 devInput！
}

aclrtFree(devInput);

// ✅ 正确: 所有路径都清理资源
void* devInput = nullptr;
aclrtMalloc(&devInput, size, ACL_MEM_MALLOC_HUGE_FIRST);

if (some_error) {
    aclrtFree(devInput);  // 清理资源
    return;
}

aclrtFree(devInput);
```

#### 3. 资源清理顺序错误

```cpp
// ❌ 错误: 清理顺序错误
aclrtFree(devInput);   // 先释放输入
aclrtFree(devOutput);  // 再释放输出
aclrtDestroyStream(stream);  // 最后销毁 stream

// ✅ 正确: 按创建的相反顺序清理
aclrtDestroyStream(stream);  // 先销毁 stream
aclrtFree(devOutput);        // 再释放输出
aclrtFree(devInput);         // 最后释放输入
```

---

## UB 地址越界

### 错误日志

```
VEC instruction error: the ub address out of bounds
CCU instruction address check error
```

### 常见原因与解决方案

#### 1. DataCopyPad 大小未 32 字节对齐

```cpp
// ❌ 错误: 未对齐
uint32_t elements = 100;
DataCopyExtParams copyParams{1, elements * sizeof(T), 0, 0, 0};
// 100 * 4 = 400 字节，不是 32 的倍数

// ✅ 正确: 32 字节对齐
constexpr uint32_t BLOCK_SIZE = 32;
uint32_t elements = 100;
uint32_t alignedBytes = ((elements * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
DataCopyExtParams copyParams{1, alignedBytes, 0, 0, 0};
// 400 -> 416 字节（32 的倍数）
```

#### 2. 超出 UB 缓冲区容量

```cpp
// ❌ 错误: buffer 大小不足
pipe.InitBuffer(inQueueX, BUFFER_NUM, 1024);  // 太小
// ...
DataCopyExtParams copyParams{1, 2048, 0, 0, 0};  // 超出 buffer 大小

// ✅ 正确: 正确计算 buffer 大小
uint32_t maxElementsPerLoop = ubAvailable / dataTypeSize;
pipe.InitBuffer(inQueueX, BUFFER_NUM, maxElementsPerLoop * sizeof(T));
```

#### 3. buffer 大小计算错误

```cpp
// ❌ 错误: 未考虑所有 buffer
uint32_t ubAvailable = ubSize / 2;  // 只考虑了一个 buffer

// ✅ 正确: 考虑所有 buffer
constexpr uint32_t BUFFER_NUM = 2;  // 输入输出两个 buffer
uint32_t ubAvailable = ubSize / (2 * BUFFER_NUM);

// 还需要考虑 tmpBuffer
if (tmpBufferSize > 0) {
    ubAvailable = (ubAvailable > tmpBufferSize) ? (ubAvailable - tmpBufferSize) : 0;
}
```

---

## 最佳实践

### 1. 内存对齐

```cpp
constexpr uint32_t BLOCK_SIZE = 32;

// 对齐计算
uint32_t alignedBytes = ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

// 元素对齐
uint32_t blockSizeElements = BLOCK_SIZE / sizeof(T);
uint32_t alignedElements = ((elements + blockSizeElements - 1) / blockSizeElements) * blockSizeElements;
```

### 2. 缓冲区管理

```cpp
// ✅ 正确: 配对使用
LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
// ... 使用 xLocal
inQueueX.FreeTensor(xLocal);

// ✅ 正确: 队列管理
inQueueX.EnQue(xLocal);
xLocal = inQueueX.DeQue<T>();
```

### 3. 边界检查

```cpp
// ✅ 正确: 检查边界
uint32_t actualElements = std::min(remainingElements, maxElementsPerLoop);
if (actualElements > 0) {
    // 处理数据
}
```

### 4. 资源清理

```cpp
// ✅ 正确: 按创建的相反顺序清理
// 创建顺序: stream -> devInput -> devOutput
// 清理顺序: devOutput -> devInput -> stream
aclrtFree(devOutput);
aclrtFree(devInput);
aclrtDestroyStream(stream);
```

---

## 调试技巧

### 1. 使用 PRINTF

```cpp
AscendC::PRINTF("offset=%u, elements=%u, totalElements=%u\n", 
                offset, elements, totalElements);
```

### 2. 开启日志打屏

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
./your_test
```

### 3. 使用 parse_mssanitizer_log.py

```bash
python3 scripts/parse_mssanitizer_log.py mssanitizer.log --output report.md
```

### 4. 对比参考实现

```bash
# 搜索类似算子的实现
grep -r "DataCopyPad" examples/
```

---

## 相关资源

- **[skill.md](../skill.md)**: mssanitizer 使用指南
- **[parse_mssanitizer_log.py](../scripts/parse_mssanitizer_log.py)**: 日志解析工具
