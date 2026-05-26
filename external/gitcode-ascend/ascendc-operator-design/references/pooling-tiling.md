# Pooling Operator Tiling Strategy

适用于池化操作（如 avg_pool, max_pool 等）的 tiling 策略与优化方案。

## 算子特性

- **计算模式**: 滑动窗口操作，每个输出位置依赖局部输入区域
- **访存模式**: 不规则访问，存在数据复用机会
- **计算强度**: 中等（通常是 memory-bound，但比 elementwise 高）
- **优化重点**: 最大化数据复用，减少重复加载，优化窗口访问

## 两级 Tiling 策略概述

```
┌─────────────────────────────────────────────────────────────┐
│                    全局内存 (GM)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         NCDHW 或 NDHWC 格式数据                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Core 0  │     │  Core 1  │ ... │ Core N   │   ← Block级Tiling (按输出点数量划分)
    │  s0~e0   │     │  s1~e1   │     │  s2~e2   │
    │ region 0 │     │ region 1 │     │ region N │
    └──────────┘     └──────────┘     └──────────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   UB 0   │     │   UB 1   │     │  UB N    │   ← UB级Tiling (按UB空间单次处理数据量循环遍历处理)
    │  o1~o2   │     │  o1~o2   │     │  o1~o2   │
    │ in tiles │     │ in tiles │     │ in tiles │
    └──────────┘     └──────────┘     └──────────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   UB 0   │     │   UB 1   │     │  UB N    │   
    │  o2~o3   │     │  o2~o3   │     │  o2~o3   │
    │ in tiles │     │ in tiles │     │ in tiles │
    └──────────┘     └──────────┘     └──────────┘
        ...               ...              ...
```

## Tiling 参数定义

### 参数结构体

```cpp
struct PoolingTilingData {
    // 输入输出形状
    int64_t batchSize;
    int64_t channels;
    int64_t inputD, inputH, inputW;
    int64_t outputD, outputH, outputW;
    
    // 池化参数
    int64_t kernelD, kernelH, kernelW;
    int64_t strideD, strideH, strideW;
    int64_t padD, padH, padW;

    // Block级 Tiling
    int64_t totalSpatial;       // 总空间位置数 (N * OD * OH * OW)
    int64_t usedCoreNum;        // 实际使用的核数
    int64_t formerNum;          // 整核数量
    int64_t formerLength;       // 整核数据长度
    int64_t tailNum;            // 尾核数量
    int64_t tailLength;         // 尾核数据长度

    // UB级 Tiling
    int64_t windowWNum;         // UB空间单次能够处理窗口的数量
};
```

### 关键常量定义

```cpp
constexpr int64_t CORE_NUM = GetBlockNum();                  // 核心数量（实际编码时通过接口获取，按实际使用核数调整）
constexpr int64_t UB_SIZE_LIMIT = 192 * 1024;     // UB 大小限制（实际编码时通过接口获取）,UB空间使用最好预留1k
constexpr int64_t ALIGN_NUM = 32 / sizeof(T);                 // 32字节对齐要求-元素对齐数量，T为输入数据类型
```

## 数据布局优化

### NCDHW vs NDHWC

池化算子的数据布局选择对性能影响巨大：

| 布局 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| NCDHW | PyTorch默认格式 | 通道维度不连续，难以向量化 | CPU计算 |
| NDHWC | 通道维度连续，易于向量化 | 需要转换 | NPU/GPU计算 |

### 推荐策略

```
输入 (NCDHW) → 转换 (NDHWC) → 池化计算 → 转换 (NCDHW) → 输出
```

## Block级Tiling（空间维度切分）

### 策略要点

1. **按空间位置切分**: 每个核处理不同的空间位置 (N, OD, OH, OW)
2. **通道完整处理**: 每个空间位置处理完整的通道维度
3. **负载均衡**: 整核/尾核策略
4. **复用数据**: 每个核最小循环处理windowWNum个窗口

### 参数计算公式

```cpp
// 实际使用的核数
int64_t usedCoreNum = std::min(totalSpatial, CORE_NUM);

// 整核和尾核
int64_t formerNum, tailNum, formerLength, tailLength;
if (totalSpatial < usedCoreNum) {
    formerNum = totalSpatial; tailNum = 0;
    formerLength = 1; tailLength = 0;
} else if (totalSpatial % usedCoreNum == 0) {
    formerNum = usedCoreNum; tailNum = 0;
    formerLength = totalSpatial / usedCoreNum; tailLength = 0;
} else {
    formerNum = totalSpatial % usedCoreNum; tailNum = usedCoreNum - formerNum;
    formerLength = totalSpatial / usedCoreNum + 1; tailLength = totalSpatial / usedCoreNum;
}
```

### 核间数据分布示例

假设 `batchSize=2, outputD=8, outputH=8, outputW=8`, `CORE_NUM=40`：

| 核类型 | 数量 | 每核空间位置数 | 说明 |
|--------|------|--------------|------|
| 整核 | 24 | 11 | 处理略多的空间位置 |
| 尾核 | 16 | 10 | 处理略少的空间位置 |

### 核内数据分布示例

每个核根据自己分配到的输出点处理个数，那么分两种情况:
情况1：单次UB空间能够同时处理DHW方向一个/多个窗口位置的输出点（一个窗口位置有C个输出点），假如能处理windowWNum个窗口。
①单次搬入数据为(windowWNum * sw + kw - 1) * alignC数量，其中alignC是channel进行32字节对齐，多余部分搬入时填充0值。
相当于一次加载windowWNum*C个输出点所需要的输入数据，复用w方向的数据进行累加，减少GM数据搬运次数。
当windowWNum=1时，表示一次只能够处理一个窗口位置（共C个值）。

情况2：单次UB空间无法处理一个窗口位置的输出点，
①如果K极大，单次UB空间无法处理一个窗口数据点，则该窗口以C作为最小单元进行搬运累加/最大值计算，完成一个窗口位置的计算。
②如果C通道数量极大，单次UB空间无法处理C个数据点，只能处理len个数据点(len < C)，则按一个窗口拆分多次循环以len作为最小单元循环多次进行累加/最大值计算，每次处理len个数据，完成一个窗口位置的计算。910b芯片192kb能处理C值上万，所以场景很少，是兜底场景。

## UB级Tiling（通道维度切分）

### 策略要点

1. **通道对齐**: AscendC Add高维切分用法 要求 32字节对齐
2. **向量化处理**: 使用向量指令并行处理多个通道
3. **非对齐处理**: DataCopyPad自动 padding 到对齐边界

### 精度处理说明

**重要**: NPU 计算单元不支持 bfloat16 数据类型的直接浮点计算，且为保证更高精度结果，平均池化和最大池化中 FP16、BF16 均统一升精度到 float32 后再进行计算，FP32 直接计算无需转换。

#### 数据类型处理策略

| 输入数据类型 | 搬运对齐 | 计算精度 | 转换方式 | UB 影响 |
|------------|---------|---------|---------|--------|
| float32 | 8元素(32字节) | float32 | 无需转换 | 无额外开销 |
| float16 | **16元素(32字节)** | float32 | `Cast<fp16, fp32>` | 需要 fp32 计算缓冲 |
| bfloat16 | **16元素(32字节)** | float32 | `Cast<bf16, fp32>` | 需要 fp32 计算缓冲 |


#### Pool 数据流

```
┌─────────────────────────────────────────────────────────────┐
│  GM (float16/bfloat16/float32)                              │
│  对齐要求: fp16/bf16 16元素, fp32 8元素                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ DataCopyPad (按原始类型搬运)
┌─────────────────────────────────────────────────────────────┐
│  UB - 输入缓冲 (原始类型)                                     │
│  fp16/bf16: 需要Cast转fp32                                   │
│  fp32: 直接使用                                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ Cast (fp16/bf16 → fp32)
┌─────────────────────────────────────────────────────────────┐
│  UB - 计算缓冲 (float32)                                     │
│  累加操作: AscendC::Add                                      │
│  平均操作: AscendC::Muls                                     │
│  最大值操作: AscendC::Max                                    │
│  比较操作: AscendC::Compare（最大池化索引专用）                │
│  选择操作: AscendC::Select（最大池化索引专用）                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ Cast (fp32 → fp16/bf16)
┌─────────────────────────────────────────────────────────────┐
│  UB - 输出缓冲 (原始类型)                                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ DataCopyPad (按原始类型写出)
┌─────────────────────────────────────────────────────────────┐
│  GM (float16/bfloat16/float32)                              │
└─────────────────────────────────────────────────────────────┘
```

### UB 分配表

**输入（NDHWC格式）:**

| UB空间参数 名称 | 大小（字节） | 用途 |
|------------|------------|------|
| dataLocal | (windowWNum * sw + kw - 1) * alignC * sizeof(T) | 输入数据空间 |
| castLocal | (windowWNum * sw + kw - 1) * alignC * sizeof(float) | cast数据缓冲 |
| sumBufLocal/maxBufLocal | windowWNum * alignC * sizeof(float) | 累加求和/最大值空间 |
| indicesLocal | windowWNum * alignC * sizeof(int32_t) | 最大值索引当前值 | (最大池化索引专用) |
| indicesUpdLocal | windowWNum * alignC * sizeof(int32_t) | 候选索引值 | (最大池化索引专用) |
| maskBufLocal | windowWNum * ceil(C/64) * 32 | Compare/Select掩码 | (最大池化索引专用, 32B对齐) |

### 累加器/最大值缓冲区初始化

| 算子类型 | 缓冲区 | 初始值 | 说明 |
|---------|--------|-------|------| 
| AvgPool3d | sumBufLocal | 0.0f | 累加求和，从0开始 |
| MaxPool3d | maxBufLocal | -3.4028235e38f | 最大值比较，初始化为float最小负值 |

### 通道对齐处理

```cpp
// 计算对齐后的通道数
int64_t alignC = ((channels + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;  //FP32时ALIGN_NUM=8,FP16、BF16时ALIGN_NUM=16
```

## Kernel侧实现模板
参考skills/ascendc-operator-code-gen/templates/pool_ndhwc_op_kernel.cpp

## Host端实现模板
参考skills/ascendc-operator-code-gen/templates/pool_ndhwc_op_host.cpp

## 关键函数使用文档
DataCopyPad：参考skills/ascendc-operator-code-gen/references/data-copy-api.md
Add高维切分：参考skills/ascendc-operator-code-gen/references/vector-compute-api.md
Max高维切分：使用方式与Add高维切分一致
Compare接口:参考skills/ascendc-operator-code-gen/references/vector-compute-api.md(最大池化索引专用)
Select接口:参考skills/ascendc-operator-code-gen/references/vector-compute-api.md(最大池化索引专用)

### 边界场景注意
  - channel和alignC，一个原始通道数，一个是32字节对齐后的通道数，不同的地方要使用正确。
  - pad!=0、ceil_mode=True主要注意：
    1. AvgPool: 边界处求和时不要累加到脏数据（segment scanning自动根据isValid跳过
       无效位置，padding区域不会被累加，无需Duplicate清零）
    2. AvgPool: 求平均值时不要除错窗口大小（通过isSamePoolSize判断poolSize是否一致）
       - isSamePoolSize=True: 所有窗口poolSize=kD*kH*kW，批量Muls
       - isSamePoolSize=False: 逐窗口单独计算除数，规则如下:
         a) divisorOverride>0: 使用用户指定值
         b) countIncludePad=True 且 pad>0: 除数=kD*kH*kW，padding零计入
            有效范围扩大到[-pad, input_size+pad)
         c) countIncludePad=False 或 pad=0: 除数=实际落在有效范围内的kernel元素个数
            有效范围[0, input_size)
       - 注意: countIncludePad只影响显式padding(pad>0)，对pad=0场景无效
    3. MaxPool: 初始化maxBufLocal为float最小负值，防止padding零值被错误选中
    4. 通用: 边界分段扫描中src1Start = (segIw - clipWStart) * alignC
       其中segIw复用上方iw计算公式: (startOw + segStart)*strideW + kw - padW
       clipWStart为CopyIn搬入的起始W坐标(dataLocal[0]对应的globalW位置)
    5. ceil_mode边界 vs 显式padding的重要区分:
       a) 显式padding(padding参数>0): 在输入四周添加零值, 增大有效输入范围
          countIncludePad决定这些零值是否计入AvgPool的除数
       b) ceil_mode边界: 仅影响输出尺寸计算(ceil vs floor)
          当kernel在右/下边界超出"原始输入+显式padding"总范围时, 超出部分
          既不属于显式padding也不受countIncludePad控制, 除数只取决于实际落在
          有效范围内的kernel元素个数
       c) 示例: ceil_mode=True且padding=0时, countIncludePad=True对除数无影响
          (因为没有显式padding零可供"include"), 边界窗口除数=实际有效元素个数