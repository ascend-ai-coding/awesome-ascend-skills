# mssanitizer 使用指南

mssanitizer 是昇腾 CANN 提供的算子异常检测工具，用于检测 Ascend C 算子的内存问题、竞争条件和未初始化内存访问。

## 检测模式

| 模式 | 命令 | 检测内容 |
|------|------|----------|
| 内存检测 | `-t memcheck` | 非法内存访问、非法释放、内存泄漏、UB地址越界 |
| 竞争检测 | `-t racecheck` | 多核并行竞争条件、数据竞争 |
| 未初始化检测 | `-t initcheck` | 未初始化内存读取 |
| 同步检测 | `-t synccheck` | 同步错误、同步点问题 |

---

## 一、内存检测 (memcheck)

### 检测流程

```
环境准备 → 编译算子 → 构建测试 → 运行检测 → 分析结果 → 修复验证
```

### 1. 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

### 2. 编译算子

```bash
# 方式一：使用 build.sh
bash build.sh --opkernel --soc=ascend910b --ops=<operator_name> --mssanitizer

# 方式二：CMakeLists.txt 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cce-enable-sanitizer -g")
```

> **注意**：必须添加 `-g` 生成调试信息；避免使用 `-O2/-O3`，建议用 `-O0`

### 3. 运行检测

```bash
# 基础内存检测
mssanitizer -t memcheck ./<test_program>

# 完整检测（含内存泄漏）
mssanitizer -t memcheck --leak-check=yes --check-device-heap=yes ./<test_program>

# 指定 kernel 检测
mssanitizer -t memcheck --leak-check=yes --kernel-name=<name> ./<test_program>
```

### 4. CANN 软件栈内存泄漏定位

当 `npu-smi info` 显示设备内存持续增长时：

```bash
# 步骤1：检测 host 侧泄漏
mssanitizer --check-device-heap=yes --leak-check=yes ./<test_program>

# 步骤2：检测 AscendCL 接口泄漏
mssanitizer --check-cann-heap=yes --leak-check=yes ./<test_program>

# 步骤3：定位代码位置（替换头文件并链接动态库后）
# 头文件: ${INSTALL_DIR}/tools/mssanitizer/include/acl/acl.h
# 动态库: -L ${INSTALL_DIR}/tools/mssanitizer/lib64 -lascend_acl_hook
mssanitizer --check-cann-heap=yes --leak-check=yes ./<test_program>
```

### 5. 检测结果示例

| 错误类型 | 日志示例 |
|---------|---------|
| 内存泄漏 | `ERROR: LeakCheck: detected memory leaks` |
| 非法访问 | `ERROR: illegal read/write of size N` |
| 非法释放 | `[E] illegal free. addr: 0x...` |
| UB越界 | `VEC instruction error: the ub address out of bounds` |

### 6. 常见问题修复

**非法内存访问**：检查 DataCopy 参数、offset 计算、尾部元素处理

```cpp
// 正确示例：边界检查
uint32_t actualElements = std::min(remainingElements, maxElementsPerLoop);
if (actualElements > 0) {
    DataCopyPad(xLocal, xGlobal[offset], copyParams, padParams);
}
```

**内存泄漏**：确保 AllocTensor/FreeTensor 配对使用

```cpp
LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
// ... 使用 xLocal
inQueueX.FreeTensor(xLocal);  // 必须释放
```

**UB越界**：确保 32 字节对齐

```cpp
constexpr uint32_t BLOCK_SIZE = 32;
uint32_t alignedBytes = ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
```

---

## 二、竞争检测 (racecheck)

### 检测场景

- 多核并行竞争
- 数据竞争
- 同步点问题

### 运行检测

```bash
mssanitizer -t racecheck ./<test_program>
```

### 错误示例

```
====== ERROR: Race condition detected
======    Thread 1: read at 0x...
======    Thread 2: write at 0x...
```

### 解决方案

```cpp
AscendC::SyncAll();  // 或 pipe.Barrier();
```

---

## 三、未初始化检测 (initcheck)

### 运行检测

```bash
mssanitizer -t initcheck ./<test_program>
```

### 错误示例

```
====== ERROR: Use of uninitialized value
======    at 0x... in kernel.cpp:128
```

### 解决方案

```cpp
// 错误：未初始化直接使用
LocalTensor<half> xLocal;

// 正确：先分配再使用
LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
```

---

## 四、同步检测 (synccheck)

```bash
mssanitizer -t synccheck ./<test_program>
```

---

## 五、完整示例

```bash
# 1. 编译算子
bash build.sh --opkernel --soc=ascend910b --ops=add_custom --mssanitizer

# 2. 构建测试
cd examples/build && cmake .. && make

# 3. 运行所有检测
mssanitizer -t memcheck --leak-check=yes --check-device-heap=yes ./bin/test
mssanitizer -t racecheck ./bin/test
mssanitizer -t initcheck ./bin/test
```

---

## 六、FAQ

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 文件名显示 `<unknown>:0` | 未启用 `-g` 或使用了优化 | 添加 `-g -O0` |
| InputSection too large | 调试信息过大 | 分模块编译 |
| records undetected | 记录数超限 | 添加 `--cache-size=100000` |

---

## 七、最佳实践

| 阶段 | 建议 |
|------|------|
| 开发 | 每次修改后运行检测，使用完整选项 |
| 测试 | 运行所有检测模式，测试边界条件 |
| 发布 | 完整检测流程，验证修复，回归测试 |

---

## 相关资源

- [昇腾社区 mssanitizer 文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/devaids/auxiliarydevtool/atlasprofiling_16_0041.html)
- [错误类型详解](./error_types.md)
- [日志解析工具](../scripts/parse_mssanitizer_log.py)
