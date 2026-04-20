# AscendC 资源管理接口总结

## 一、TPipe

**用途**: 统一管理Device端内存和同步事件资源，一个Kernel必须且只能有一个TPipe对象。

### 核心功能

1. **内存资源管理**: 通过InitBuffer为TQue和TBuf分配内存
2. **同步事件管理**: 通过AllocEventID/ReleaseEventID管理事件ID

### 关键接口

| 接口 | 功能 |
|------|------|
| `InitBuffer(que, num, len)` | 为TQue分配内存（num块，每块len字节） |
| `InitBuffer(buf, len)` | 为TBuf分配内存（len字节） |
| `AllocEventID<HardEvent>()` | 申请EventID（占用型） |
| `FetchEventID(HardEvent)` | 获取EventID（非占用型） |
| `ReleaseEventID<HardEvent>(id)` | 释放EventID |
| `Reset()` | 重置资源 |
| `GetBaseAddr()` | 获取基地址 |

### 示例

```cpp
AscendC::TPipe pipe;

// 为TQue分配内存
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
pipe.InitBuffer(inQueue, 2, 1024);  // 2块，每块1024字节

// 为TBuf分配内存
AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
pipe.InitBuffer(tmpBuf, 512);  // 512字节

// EventID管理
AscendC::TEventID eventId = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>();
// ... 使用EventID ...
GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventId);
```

---

## 二、TQue

**用途**: 管理队列操作，实现流水线并行。

### 模板参数

```cpp
template <TPosition pos, int32_t depth, auto mask = 0>
class TQue {...};
```

| 参数 | 说明 |
|------|------|
| pos | 逻辑位置（VECIN/VECOUT/A1/A2/B1/B2/CO1/CO2） |
| depth | 队列深度（推荐设为1，Tensor原地操作设为0） |
| mask | 可选配置（ND↔NZ转换等） |

### 核心接口

| 接口 | 功能 |
|------|------|
| `AllocTensor<T>()` | 分配Tensor |
| `AllocTensor<T>(tensor)` | inplace方式分配 |
| `EnQue(tensor)` | Tensor入队 |
| `DeQue<T>()` | Tensor出队 |
| `DeQue<T>(tensor)` | inplace方式出队 |
| `FreeTensor(tensor)` | 释放Tensor |
| `HasTensorInQue()` | 队列是否有数据 |
| `HasIdleBuffer()` | 是否有空闲Buffer |
| `GetTensorCountInQue()` | 获取队列中Tensor数量 |

### Buffer数量限制

| 产品 | EventID数量 | 最大Buffer数 |
|------|-------------|--------------|
| Atlas 训练 | 4 | 4 |
| Atlas 推理 AI Core | 8 | 8 |
| Atlas A2/A3 | 8 | 8 |
| Atlas 200I/500 A2 | 8 | 8 |

### 标准流水线模式

```cpp
// CopyIn -> Compute -> CopyOut
AscendC::LocalTensor<half> srcLocal = inQueue.AllocTensor<half>();
AscendC::DataCopy(srcLocal, srcGlobal, dataSize);
inQueue.EnQue(srcLocal);

srcLocal = inQueue.DeQue<half>();
// ... 计算操作 ...
inQueue.FreeTensor(srcLocal);
```

### Double Buffer 模式

Double Buffer 的目的是**让 DMA 搬运（MTE2/MTE3）与 Vector 计算并行执行**，而非简单的"两块内存做计算"。

```cpp
// MTE 通路队列: depth=1, num=2 (double buffer)
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
pipe.InitBuffer(inQueue, 2, len);   // 2块: 一块在搬运，一块在计算
pipe.InitBuffer(outQueue, 2, len);

// 纯计算临时缓冲区: 无需 double buffer
AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
pipe.InitBuffer(tmpBuf, len);       // 1块即可
```

**何时用 TQue vs TBuf**:

| 场景 | 用 TQue (VECIN/VECOUT) | 用 TBuf (VECCALC) |
|------|------------------------|-------------------|
| GM↔UB 搬运 | ✅ depth=1, num=2 | - |
| 纯计算中间变量 | - | ✅ depth=1 |
| 归约 tmpBuffer | - | ✅ |
| 升精度 FP32 workspace | - | ✅ |

---

## 三、TBuf

**用途**: 管理临时变量内存，不支持入队出队操作。

### 特点

- 只能参与计算，无法执行队列操作
- TPipe只为TBuf分配一块内存
- 获取的Tensor无需释放
- 多个临时变量需要定义多个TBuf

### 接口

| 接口 | 功能 |
|------|------|
| `Get<T>()` | 获取指定类型的Tensor |
| `GetWithOffset<T>(offset)` | 获取带偏移的Tensor |

### 示例

```cpp
AscendC::TPipe pipe;
AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
pipe.InitBuffer(tmpBuf, 512);

// 获取Tensor使用
AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
// 使用tmp进行计算，无需释放
```

---

## 四、TQueBind

**用途**: 绑定VECIN和VECOUT实现内存复用。

### 模板参数

```cpp
template <TPosition srcPos, TPosition dstPos, int32_t depth>
class TQueBind {...};
```

### 使用场景

用于存在Vector计算时实现VECIN和VECOUT内存复用。

### 示例

```cpp
AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> que;
pipe.InitBuffer(que, 2, 1024);

AscendC::LocalTensor<half> tensor = que.AllocTensor<half>();
que.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, half>(tensor);
tensor = que.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, half>();
```

---

## 五、Workspace管理

### GetUserWorkspace

获取用户使用的workspace指针：

```cpp
__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace);
```

```cpp
GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
```

### GetSysWorkSpacePtr

获取系统workspace指针（用于Matmul等高阶API）：

```cpp
__aicore__ inline __gm__ uint8_t* GetSysWorkSpacePtr();
```

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
```

### SetSysWorkSpace

设置系统workspace（Kernel直调场景）：

```cpp
__aicore__ inline void SetSysWorkspace(GM_ADDR workspace);
```

```cpp
AscendC::SetSysWorkspace(workspace);
if (GetSysWorkSpacePtr() == nullptr) {
    return;
}
```

### Host侧配置

```cpp
// Tiling函数中
size_t usrSize = 256;
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
size_t *currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = usrSize + sysWorkspaceSize;
```

---

## 六、内存管理约束

### InitBuffer 约束

- 申请的内存会在 TPipe 析构时自动释放
- 一个 kernel 中所有 Buffer 数量之和不能超过 64
- 自定义地址方式与不指定地址方式不建议混用
- len 不满足 32 字节对齐时会自动补齐

### UB 容量与行数计算

Host 端通过平台 API 获取 UB 大小后，计算每次处理的行数:

```cpp
// Host 侧
uint64_t ubSize;
ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

// 每行占用 UB 字节 = alignedCols * sizeof(T)
// 考虑多个 buffer（如 inQueue×2 + outQueue×2 + tmpBuf×1 + calcBuf×1 = 6份）
uint32_t tileRows = ubSize / (alignedCols * sizeof(T) * bufferCount);
```

### DataCopyPad blockCount 限制

`DataCopyExtParams.blockCount` 最大值为 **4095**。当 `tileRows > 4095` 时需要分批:
```cpp
tileRows = std::min(tileRows, (uint32_t)4095);
```

### AllocTensor约束

同一TPosition上连续Alloc的Tensor数量限制：

| 产品 | 最大数量 |
|------|----------|
| Atlas 训练 | 4 |
| Atlas 推理 AI Core | 8 |
| Atlas A2/A3 | 8 |

### 解决Buffer不足

```cpp
// 方法1: 合并多个buffer到一块，通过偏移使用
pipe.InitBuffer(que0, 1, len * 3);
AscendC::LocalTensor<T> local1 = que0.AllocTensor<T>();
AscendC::LocalTensor<T> local2 = local1[len];
AscendC::LocalTensor<T> local3 = local1[len * 2];

// 方法2: 释放不用的TQue
que0.FreeAllEvent();
```

---

## 七、典型使用模式

### Vector算子标准模式

```cpp
class KernelOp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t size) {
        xGlobal.SetGlobalBuffer((__gm__ half*)x, size);
        yGlobal.SetGlobalBuffer((__gm__ half*)y, size);
        pipe.InitBuffer(inQueue, 1, size * sizeof(half));
        pipe.InitBuffer(outQueue, 1, size * sizeof(half));
    }

    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        AscendC::LocalTensor<half> xLocal = inQueue.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGlobal, dataSize);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<half> xLocal = inQueue.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueue.AllocTensor<half>();
        AscendC::Add(yLocal, xLocal, xLocal, dataSize);
        outQueue.EnQue<half>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<half> yLocal = outQueue.DeQue<half>();
        AscendC::DataCopy(yGlobal, yLocal, dataSize);
        outQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::GlobalTensor<half> xGlobal, yGlobal;
    uint32_t dataSize;
};
```
