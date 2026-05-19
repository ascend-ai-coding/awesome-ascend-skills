# 卡死/崩溃调试工作流程

## 快速决策树

```
程序无法运行完
    │
    ├─ 程序崩溃？
    │   └─ Coredump 调试
    │       └─ GDB 分析 coredump 文件
    │
    └─ 程序卡死？
        └─ Kernel挂起调试 → 查看plog → 定位卡死位置
```

## 流程1：Kernel挂起调试

### Step 1: 查看plog日志

```bash
# plog 默认路径
ls $HOME/ascend/log/debug/plog/plog-pid_*.log

# 或开启日志打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 使用 parse_plog.py 解析日志（脚本位于 crash-debug skill 的 scripts/ 目录）
python3 scripts/parse_plog.py <plog_file>
```

### Step 2: 分析plog内容

**核心超时**

```
症状：日志中出现 "timeout" 或程序长时间无响应

可能原因：
    ├─ Buffer 未释放 → 检查 AllocTensor/FreeTensor 配对
    ├─ 死锁 → 检查 EnQue/DeQue 配对
    ├─ 无限循环 → 检查循环终止条件
    └─ 阻塞操作 → 检查同步点
```

**内存访问越界**

```
症状：aic error 或 程序长时间无响应

可能原因：
    ├─ DataCopy 长度错误 → 检查 size 参数
    ├─ GM 地址错误 → 检查 offset 计算
    ├─ UB 访问越界 → 检查 buffer 大小
    └─ 非对齐访问 → 检查 32 字节对齐
```

### Step 3: Kernel调试方法

**方法1：Printf调试**

```cpp
// 在 Kernel 中打印关键变量
AscendC::PRINTF("blockLength=%llu, tileNum=%llu\n", blockLength_, tileNum_);
```

**方法2：DumpTensor调试**

```cpp
// 打印 tensor 内容
AscendC::LocalTensor<T> xLocal = inQueue.DeQue<T>();
DumpTensor(xLocal, 0, 128);  // 打印前128个元素
```

**方法3：单步调试（msDebug）**

```bash
# 使用 msDebug 工具进行单步调试
# 参考：https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug
```

## 流程2：Coredump 调试（程序崩溃）

**适用场景**：程序崩溃、Segmentation Fault、Abort

### Step 1: 启用 coredump

```bash
ulimit -c unlimited  # 启用 coredump
```

### Step 2: 生成并分析 coredump

```bash
# 运行程序（崩溃时生成 core 文件）
./your_executable

# 使用 GDB 分析 coredump
gdb <executable> <core_file>

# GDB 常用命令
bt              # 查看调用栈
bt full         # 查看完整调用栈（包含局部变量）
frame N         # 切换到第 N 层栈帧
info locals     # 查看局部变量
p variable      # 打印变量值
```

### Step 3: 定位问题

常见崩溃原因：
- **空指针解引用**：检查 tensor 是否为 nullptr
- **内存越界**：检查 DataCopy 长度、GM/UB 访问范围
- **栈溢出**：检查递归深度或大数组

## Buffer 相关卡死/崩溃

以下问题可能导致程序卡死或崩溃：

| 问题 | 表现 | 解决方案 |
|------|------|----------|
| Buffer 未释放 | 核心挂起/超时 | 循环内 Alloc 后必须 Free |
| 核心超时/挂起 | 程序无响应 | 检查 Buffer 冲突/死锁 → Alloc / Free 配对 |
| VECIN 用于输出 | 输出等于输入 | 输出必须用 VECOUT 队列 |
| Double Buffer 漏算 | 阈值错误 | 计算阈值时 ×2 |

### 死锁与 Buffer 冲突

```
常见死锁模式：
    ├─ Buffer 分配与释放不配对 → 循环中对同一 buffer 多次 Alloc 不 Free
    ├─ EnQue/DeQue 不配对 → 队列空等或满等
    ├─ 多核同步缺失 → CrossCoreWaitFlag 无对应 SetFlag
    └─ PipeBarrier 滥用 → 全流水线停顿，后续操作依赖未完成数据
```

## 调试工具速查

| 工具/方法 | 用途 | 使用场景 |
|----------|------|---------|
| `plog日志` | 查看运行时日志 | 卡死/崩溃分析 |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 日志打屏 | 需要实时查看日志 |
| `parse_plog.py` | 日志解析 | 自动提取错误/超时/崩溃信息 |
| `AscendC::PRINTF` | Kernel内打印 | Kernel逻辑调试 |
| `DumpTensor` | 打印tensor内容 | 数据验证 |
| `msDebug` | 单步调试 | 复杂问题（卡死、越界） |
| `ulimit -c unlimited` | 启用 coredump | 程序崩溃前设置 |
| `gdb <exe> <core>` | 分析 coredump | 程序崩溃时优先使用 |
