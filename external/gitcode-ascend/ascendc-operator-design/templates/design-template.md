# [算子名称] 设计文档

## 1. 算子接口

### 1.1 函数签名
```cpp
at::Tensor [operator_name](
    const at::Tensor &input1,
    const at::Tensor &input2,
    /* 其他参数 */
);
```

### 1.2 参数说明
| 参数名 | 类型 | 输入/输出 | 支持的数据类型 | 描述 | 约束条件 |
|--------|------|-------|-------|--------|------|
| input1 | at::Tensor | 输入 | bfloat16/float16/float32 | 输入tensor1 | 支持ND |
| input2 | at::Tensor | 输入 | bfloat16/float16/float32 | 输入tensor2 | 支持ND |
| output | at::Tensor | 输出 | bfloat16/float16/float32 | 输出tensor | 支持ND |

### 1.3 支持的数据类型
- [ ] bfloat16
- [ ] float16
- [ ] float32

---

## 2. 计算逻辑

### 2.1 算法描述
[详细描述算子的计算步骤和数据流动]

### 2.2 伪代码
```
for each tile in input:
    load tile to local memory
    compute on tile
    store result to global memory
```

### 2.3 实现路径选择
- [ ] AscendC Kernel（纯vector实现）
- [ ] CATLASS模板库（矩阵乘法类）
- [ ] ACLNN封装（CANN内置算子）

**选择理由**: [说明为什么选择这种实现方式]

---

## 3. Tiling策略

AscendC 算子采用**两级 Tiling 策略**来充分利用硬件并行能力。

### 两级 Tiling 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    全局内存 (GM)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              totalLength 元素数据                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Core 0  │     │  Core 1  │ ... │ Core 39  │   ← Block级Tiling (核间切分)
    │ formerLen│     │ formerLen│     │ tailLen  │
    └──────────┘     └──────────┘     └──────────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   UB 0   │     │   UB 1   │     │  UB 39   │   ← UB级Tiling (核内切分)
    │ tileLen  │     │ tileLen  │     │ tileLen  │
    │ tileLen  │     │ tileLen  │     │ tileLen  │
    │   ...    │     │   ...    │     │   ...    │
    └──────────┘     └──────────┘     └──────────┘
```

### 核心原则

- **Block级Tiling (核间切分)**: 确保每个 Core 处理的计算量相对均衡，使用整核/尾核策略
- **UB级Tiling (核内切分)**: 根据 UB 分配表确定 buffer 需求，计算单次循环处理量

**算子类型**: elementwise / reduction / 其他

**参考文档**:
- 硬件说明: `references/hardware-architecture.md`
- 逐元素操作: `references/elementwise-tiling.md`
- 归约操作: `references/reduction-tiling.md`
- 通用原则: `references/general-tiling-principles.md`

### 3.1 Tiling参数结构体定义

```cpp
struct [OperatorName]TilingData {
    int64_t totalLength;        // 总数据长度

    int64_t formerNum;          // 整核数量
    int64_t formerLength;       // 整核数据长度
    int64_t tailNum;            // 尾核数量
    int64_t tailLength;         // 尾核数据长度

    int64_t tileLength;         // UB单次处理长度
};
```

### 3.2 Block级Tiling（核间切分）

**策略要点**:
1. **Cache Line对齐**: 每个核处理的数据块 512 字节对齐
2. **负载均衡**: 整核/尾核策略

| 参数 | 计算公式 | 值 |
|------|----------|-----|
| totalLengthCore | (totalLength + CORE_NUM - 1) / CORE_NUM | [值] |
| totalLengthCoreAlign | (totalLengthCore + 512 - 1) / 512 * 512 | [值] |
| usedCoreNum | (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign | [值] |
| formerNum | usedCoreNum - 1 | [值] |
| tailNum | 1 | [值] |
| formerLength | totalLengthCoreAlign | [值] |
| tailLength | totalLength - (usedCoreNum - 1) * formerLength | [值] |

**负载均衡验证**:
- 整核/尾核数据差异: formerLength >= tailLength
**核间切分验证**:
- 计算数据量是否正确: formerNum * formerLength + tailNum * tailLength == totalLength

### 3.3 UB级Tiling（核内切分）

**策略要点**:
1. 根据 UB 分配表确定 buffer 系数
2. tileLength 由 Host 端根据 UB 大小计算
3. 32 字节对齐（UB 内部对齐要求）

#### 精度处理说明

**重要**: NPU 计算单元不支持 float16/bfloat16 数据类型的直接计算，必须升精度到 float32 后再进行计算。

| 输入数据类型 | 处理方式 | 计算精度 | UB 影响 |
|------------|---------|---------|--------|
| float16 | **升精度到 float32** | float32 | 需要额外 float32 buffer |
| bfloat16 | **升精度到 float32** | float32 | 需要额外 float32 buffer |
| float32 | 直接计算 | float32 | 无额外开销 |

#### UB 分配表

| Buffer名称 | 大小(字节) | 用途 | 数量 | 总大小 |
|-----------|-----------|------|------|--------|
| inQueueX | tileLength * dtypeSize | 输入数据缓冲 | BUFFER_NUM | [计算值] |
| inQueueY | tileLength * dtypeSize | 输入数据缓冲 | BUFFER_NUM | [计算值] |
| outQueueZ | tileLength * dtypeSize | 输出数据缓冲 | BUFFER_NUM | [计算值] |
| tempBuffer (fp16/bf16时) | tileLength * 4 | float32计算缓冲 | 1 | [计算值] |
| **总计** | - | - | - | **[总UB使用]** |

**float16/bfloat16 输入时的额外 UB 分配**:
- [ ] 需要分配 float32 计算缓冲区
- [ ] 每个 float16/bfloat16 输入/输出需要对应的 float32 buffer
- [ ] tileLength 计算需考虑额外的 buffer 开销

#### tileLength 计算

| 参数 | 计算公式 | 值 |
|------|----------|-----|
| bufferCoefficient | 根据UB分配表确定 | [值] |
| maxTileElements | UB_SIZE_LIMIT / bufferCoefficient（UB_SIZE_LIMIT 实际编码时通过接口获取） | [值] |
| alignElements | 32 / dtypeSize | [值] |
| tileLength | (maxTileElements / alignElements) * alignElements | [值] |

#### UB 约束验证

- **UB使用**: [X] bytes ([Y]% of UB_SIZE_LIMIT)
- **UB限制**: [Z] bytes（UB_SIZE_LIMIT 实际编码时通过接口获取，示例值 192KB）
- **是否满足约束**: [是/否]
- **对齐要求**: 32字节对齐

---

## 4. Workspace需求

### 4.1 Workspace 大小计算
根据算子类别确定：

| 算子类别 | workspace size | 说明 |
|----------|---------------|------|
| elementwise 类 | SYSTEM_WORKSPACE_SIZE | 通常为 16MB |
| 其他类算子 | sizeof([OperatorName]TilingData) | tiling data 大小 |

- **tiling data size**: sizeof([OperatorName]TilingData)

### 4.2 Workspace 分配示例
```cpp
// Host 端分配 workspace
constexpr int64_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;  // 16MB
size_t workspaceSize = SYSTEM_WORKSPACE_SIZE;  // elementwise 类算子
auto workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                           at::TensorOptions().dtype(at::kByte).device(x.device()));
```

---

## 5. 性能优化

### 5.1 关键优化点
1. [优化点1 - 如：使用double buffer隐藏内存延迟]
2. [优化点2 - 如：Cache Line对齐优化核间负载均衡]
3. [优化点3 - 如：减少GM访问次数]
4. [优化点4 - 如：向量化计算]

### 5.2 算子特性
- **计算模式**: [memory-bound / compute-bound / balance]
- **访存模式**: [顺序访问 / 随机访问 / 跨轴访问]
- **并行性**: [高 / 中 / 低]

---

## 6. Kernel端实现要点

### 6.1 执行流程（核内循环）

```cpp
__aicore__ inline void Process() {
    int64_t coreLength = AscendC::GetBlockIdx() == tiling->usedCoreNum - 1 ? this->tailLength : this->formerLength;
    int64_t tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
    int64_t tailTileLength = this->blockLength - (tileNum - 1) * this->tileLength;

    for (int64_t i = 0; i < tileNum - 1; ++i) {
        // 处理整块 tileLength
        CopyIn(i, this->tileLength);
        Compute(i, this->tileLength);
        CopyOut(i, this->tileLength);
    }
    // 处理尾块 tailTileLength
    CopyIn(tileNum - 1, tailTileLength);
    Compute(tileNum - 1, tailTileLength);
    CopyOut(tileNum - 1, tailTileLength);
}
```

---

## 7. 实现检查清单

### 7.1 文件结构
- [ ] `csrc/ops/<operator_name>/CMakeLists.txt`
- [ ] `csrc/ops/<operator_name>/op_host/<operator_name>.cpp`
- [ ] `csrc/ops/<operator_name>/op_kernel/<operator_name>.cpp`
- [ ] `csrc/ops.h` (添加声明)
- [ ] `csrc/register.cpp` (添加注册)

### 7.2 Host端实现
- [ ] 定义TilingData结构体（包含两级Tiling参数）
- [ ] 实现Block级Tiling参数计算（Cache Line对齐）
- [ ] 根据UB分配表确定buffer系数
- [ ] 实现UB级Tiling参数计算（32字节对齐）
- [ ] 分配workspace（如需要）
- [ ] 调用kernel入口函数

### 7.3 Kernel端实现
- [ ] 实现Init函数（整核/尾核偏移计算）
- [ ] 实现CopyIn函数（GM -> UB）
- [ ] 实现Compute函数（核心计算，fp16/bf16时需升精度）
- [ ] 实现CopyOut函数（UB -> GM）
- [ ] 实现Process主循环（处理尾块）
- [ ] 处理最后一个tile的边界情况

### 7.4 测试验证
- [ ] 准备测试数据（小规模）
- [ ] 准备测试数据（大规模）
- [ ] 正确性验证（与PyTorch对比）
- [ ] 性能测试
- [ ] 边界case测试

---

## 8. 参考实现

- **相似算子**: [项目中已有的类似算子路径]
- **PyTorch参考**: [对应的PyTorch函数]
- **文档参考**: [相关技术文档链接]

---

## 使用说明

填充此模板时：
1. **方括号内容**: 替换所有 `[placeholder]` 为实际内容
2. **复选框**: 勾选适用的选项
3. **表格**: 填写具体的数值和描述
4. **代码块**: 根据实际算子调整参数名和类型
5. **删除不适用部分**: 如果某节不适用于当前算子，可以删除

**设计完成后**，使用 `ascendc-operator-code-gen` skill 生成具体代码实现。
