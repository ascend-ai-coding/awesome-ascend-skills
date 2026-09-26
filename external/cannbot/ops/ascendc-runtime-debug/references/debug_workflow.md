# 运行时错误调试工作流程

## 快速决策树

```
运行时错误
    │
    ├─ 返回码非0？
    │   └─ 是 → 获取错误信息 (aclGetRecentErrMsg())
    │       │
    │       ├─ 错误信息清楚 → 针对性处理
    │       │
    │       └─ 错误信息不清楚 → 按错误码类型排查
    │           ├─ 161xxx（参数错误）→ 检查参数
    │           ├─ 361xxx（Runtime）→ 检查环境/设备
    │           └─ 561xxx（内部错误）→ 按具体错误码处理
    │
    └─ 复杂场景 → 开启日志调试
```

## 详细流程

### 流程1：错误码处理

#### Step 1: 获取错误信息

```cpp
// 在 aclnn 调用后立即检查
aclnnStatus status = aclnnXxxGetWorkspaceSize(...);
if (status != ACLNN_SUCCESS) {
    const char* error_msg = aclGetRecentErrMsg();
    printf("Error: %s\n", error_msg);
    // 根据错误码进入对应处理流程
}
```

#### Step 2: 按错误码分类处理

##### 161xxx - 参数错误

```
ACLNN_ERR_PARAM_NULLPTR (161001)
    └─ 参数校验错误，参数中存在非法的nullptr

ACLNN_ERR_PARAM_INVALID (161002)
    └─ 参数校验错误
        │
        ├─ 检查输入/输出tensor的dtype是否匹配
        │   └─ 如：输入的两个数据类型不满足输入类型推导关系
        ├─ 检查输入/输出tensor的shape是否满足要求
        ├─ 检查属性值是否在合法范围内
        └─ 检查tensor format是否支持
```

#### UT调试方案 - 参数校验错误（161xxx）

**优势**：
- 无需 NPU 设备，在 Host 层快速迭代
- 可自动化批量测试各种参数组合
- 隔离环境因素，聚焦参数逻辑问题

**调试流程**：
1. 根据错误码判断问题参数
2. 使用 `ascendc-ut-generator` 技能生成参数校验测试
3. 运行 UT 定位具体参数问题

**详细 UT 开发与运行**：参考 `ascendc-ut-generator` 技能

##### 361xxx - Runtime错误

```
ACLNN_ERR_RUNTIME_ERROR (361001)
    │
    └─ API内部调用npu runtime接口异常
        ├─ 使用 aclGetRecentErrMsg() 获取详细错误信息
        └─ 根据报错提示排查
```

##### 561002 - Tiling错误

```
ACLNN_ERR_INNER_TILING_ERROR
    └─ Tiling 处理异常
        │
        ├─ 检查 TilingFunc 函数实现
        │   ├─ 确保所有分支都设置了 TilingKey
        │   └─ 确保没有除零、数组越界等异常
        │
        └─ 检查输入参数
            ├─ shape 元素个数是否超过限制
            └─ 参数组合是否在支持范围内
```

#### UT调试方案 - Tiling错误（561002）

**优势**：
- 快速验证 Tiling 逻辑，无需 NPU 设备
- 可测试所有 TilingKey 分支覆盖情况
- 快速定位是哪个输入组合导致 Tiling 失败
- 隔离 Tiling 逻辑问题与 Kernel 执行问题

**调试流程**：
1. 记录 ST 失败的输入参数（shape/dtype/属性）
2. 使用 `ascendc-ut-generator` 技能生成 Tiling 逻辑测试
3. 运行 UT 验证 Tiling 逻辑是否正确

**快速诊断**：
- UT 通过 → Tiling 逻辑正确，检查 ST 环境/Kernel
- UT 失败 → Tiling 逻辑有问题，修复 Host 代码

**详细 UT 开发与运行**：参考 `ascendc-ut-generator` 技能

##### 561003 - Kernel查找失败

```
ACLNN_ERR_INNER_FIND_KERNEL_ERROR
    │
    └─ Kernel 查找失败
        ├─ 检查 TilingKey/SEL 匹配 → [kernel_binary_debug.md](kernel_binary_debug.md#流程2tilingkey--sel-不匹配)
        ├─ 检查 SEL dtype 条目 → [kernel_binary_debug.md](kernel_binary_debug.md#流程3sel-dtype-条目缺失)
        │
        ├─ 检查算子是否已安装
        │   ├─ ls $ASCEND_OPP_PATH/vendors/<vendor_name>/op_impl/ai_core/tbe/op_api/lib/
        │   └─ 确认有对应的 .so 文件
        │
        ├─ 检查 vendor_name
        │   ├─ 编译时 --vendor_name 参数
        │   └─ 安装后的目录名是否一致
        │
        ├─ 检查 SOC 版本
        │   ├─ 编译时 --soc 参数
        │   └─ 运行时设备 SOC 是否匹配
        │
        ├─ 检查算子二进制编译
        │   └─ 算子二进制编译失败 → 检查编译日志
        │
        ├─ 检查输入类型匹配
        │   └─ 输入类型和信息库不匹配 → 检查 dtype/shape 是否在{op_name}_def.cpp原型库支持范围内
        │
        ├─ 检查多版本 vendor 包冲突 → [kernel_binary_debug.md](kernel_binary_debug.md#流程4多版本-vendor-包冲突)
        │
        └─ 检查环境变量
            ├─ export ASCEND_OPP_PATH=/path/to/opp
            └─ export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/<vendor_name>/op_api/lib/:$LD_LIBRARY_PATH
```

##### 561107 - 环境变量缺失

详细配置步骤及常见错误见 [ascendc-env-check skill](skill:ascendc-env-check)

##### 561112 - 算子二进制包未加载

```
ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND
    │
    └─ 算子二进制包未加载
        │
        └─ 安装算子包
```

### 507035 向量核异常

```
错误码: 507035
官方宏名: ACL_ERROR_RT_VECTOR_CORE_EXCEPTION
错误信息: 向量核异常 (vector core exception)
来源: NPU 硬件/驱动层
机制: 向量核执行期间发生硬件异常，通过 aclrtStreamSynchronize() 报告给 Host
注意: 不要与 507046 (ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) 混淆
```

#### 根因分类

507035 是向量核硬件异常信号，按历史经验置信度从高到低排列：

```
507035 向量核异常
    │
    ├─ 高置信度
    │   ├─ UB 端起始地址非 32B 对齐
    │   │   ├─ DataCopy/DataCopyPad 的 UB 侧地址（含子张量偏移）→ 确保起始地址 32B 对齐
    │   │   ├─ VEC 计算指令 (Add/Muls/Duplicate 等) 操作数起始地址非 32B 对齐 → 检查子张量偏移
    │   │   └─ 动态 chunk 大小非 32B 对齐 → Host 侧对齐计算
    │   └─ DataCopyPad 填充参数违规（padParams 仅 GM→UB 方向存在）
    │       └─ leftPadding/rightPadding 所占字节数超过 32 → 检查 padParams
    │
    │   注意: blockLen 无 32B 对齐要求（DataCopyPad 支持非对齐搬运，
    │   blockLen 只需为 sizeof(T) 的整数倍、单位为字节；DataCopy 的
    │   count 形式非对齐时是静默向下取整，不触发 507035）
    │
    ├─ 中置信度
    │   └─ UB 溢出 / Buffer 冲突
    │       ├─ tmpBuf 大小不足 → 使用 GetReduceMaxMaxMinTmpSize API
    │       ├─ 动态 UB 预算计算有误 → 验证 allBufSize < UB_LIMIT
    │       └─ Buffer 越界覆盖 → 检查 AllocTensor/FreeTensor 配对
    │
    ├─ 低置信度
    │   └─ DataCopyPad 读取越界
    │       ├─ srcStride 或 xRow 超 GM buffer 范围 → 检查地址计算
    │       └─ 高维 shape 下 offset 计算超出实际 GM 分配 → 验证 GM 大小
    │
    └─ 极低置信度
        └─ 其他 DMA 异常
            ├─ 多核并发写冲突 → 检查 workspace 分配
            ├─ 硬件资源耗尽 → 减少 block_dim
            └─ 编译优化问题 → 清理编译缓存后重试
```

#### 诊断步骤

**Step 1: 缩小范围**

```bash
# 仅测试最小可复现用例
# 仅测试 32B 对齐的简单 shape (如 2D with innerDim=16/32/64)
# 仅测试 FP32（排除 dtype 因素）
```

**Step 2: 检查 DataCopyPad 参数**

```cpp
// 约束（依据官方 API 文档）：
// - blockLen ∈ [0, 2097151]，单位为字节，只需是 sizeof(T) 的整数倍，
//   直接传有效数据长度，禁止向上对齐（多搬数据会导致越界/数据错误）
// - 真正的 32B 对齐约束在地址上：
//   GM→UB: dst(UB) 起始地址必须 32B 对齐，src(GM) 1B 对齐即可
//   UB→GM: src(UB) 起始地址必须 32B 对齐，dst(GM) 无对齐约束
// - padParams（DataCopyPadExtParams）仅 GM→UB 方向有效，
//   leftPadding/rightPadding 所占字节数均不能超过 32B
// - stride 单位取决于所在侧：GM 侧为字节，UB 侧为 dataBlock(32B)
//   GM→UB 时 srcStride=字节、dstStride=dataBlock，UB→GM 时相反
// - 与 DataCopy（连续搬运）区分：DataCopy 的 count*sizeof(T) 须 32B 对齐
//   （非对齐静默向下取整），且 GM 侧地址须按 sizeof(T) 对齐

// GM→UB（4 参数，含 padParams）：
// ❌ 错误：blockLen 传了元素个数（单位应为字节）
DataCopyPad(xLocal, xGm, {1, 1, static_cast<uint32_t>(innerDim), 0, 0}, padParams);

// ✅ 正确：blockLen 传有效字节数（可非 32B 对齐），UB 行偏移另行对齐
uint32_t validBytes = innerDim * sizeof(T);  // 有效长度，不向上取整
DataCopyPad(xLocal, xGm, {1, 1, validBytes, 0, 0}, padParams);
// 若 xLocal 为 UB 子张量，其偏移须 32B 对齐（见 Step 3）

// UB→GM（3 参数，无 padParams；读 UB 时硬件补 dummy，写 GM 时自动丢弃）：
DataCopyPad(yGm, yLocal, {1, 1, validBytes, 0, 0});
```

**Step 3: 检查 UB 子张量偏移**

```cpp
// 所有 Get<T>(offset) 或 tensor[N] 创建的 UB 子张量，
// 其字节偏移必须是 32B 对齐

// ❌ 错误：chunkA0=8 (FP16), r=1 → 偏移 8*2=16B，非 32B 对齐
auto row = hBuf[r * chunkA0_];

// ✅ 正确：Host 侧将 chunkA0 对齐到 32B 边界
// FP16/BF16: alignElements = 32/2 = 16
// FP32: alignElements = 32/4 = 8
chunkA0_ = ((chunkA0_ + alignElements - 1) / alignElements) * alignElements;
```

**Step 4: 临时验证 - 强制 Scale up tileRows**

```cpp
// 如果怀疑 chunk size 太小导致分配碎片，可以临时减少 tileRows
// 若减少 tileRows 后 507035 消失 → 确认 chunk 对齐问题
uint32_t testTileRows = 1;  // 最小测试
```

#### 内置检查清单

遇到 507035 时按顺序检查：
1. [ ] DataCopy/DataCopyPad 的 UB 端起始地址（含 `Get(offset)` 子张量偏移，offset * sizeof(T)）是 32B 的倍数吗？
2. [ ] VEC 计算指令 (Add/Muls/Duplicate/Cast 等) 的操作数起始地址是 32B 对齐的吗？
3. [ ] DataCopyPad 的 blockLen 是 sizeof(T) 的整数倍、单位为字节、传有效长度吗？（GM→UB）leftPadding/rightPadding ≤ 32B 吗？
4. [ ] 动态 chunk 大小 (chunkA0, tileRows 等) 的对齐计算在 Host 侧做了吗？
5. [ ] tmpBuf 大小是否使用了官方 API (`GetReduceMaxMaxMinTmpSize`) 而非手动估算？
6. [ ] 总 UB 使用量 < UB 容量限制吗？

### 流程4：环境检查

使用 [ascendc-env-check skill](skill:ascendc-env-check) 进行环境检查

## 调试工具速查

| 工具/方法 | 用途 | 使用场景 | 适用错误 |
|----------|------|---------|---------|
| `aclGetRecentErrMsg()` | 获取错误详情 | 返回码非0时 | 所有错误 |
| `plog日志` | 查看运行时日志 | 所有错误 | 所有错误 |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 日志打屏 | 需要实时查看日志 | 所有错误 |
| **UT 测试** | **Host 层调试** | **参数校验/Tiling逻辑** | **161xxx/561002** |

## 未知错误码处理（兜底方案）

遇到速查表中未列出的错误码时：

### Step 1: 获取详细错误信息

```cpp
if (status != ACLNN_SUCCESS) {
    const char* error_msg = aclGetRecentErrMsg();
    printf("Error code: %d, Message: %s\n", status, error_msg);
}
```

查看 plog 日志：
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
# 或
ls $HOME/ascend/log/debug/plog/
```

### Step 2: 搜索官方文档

使用 `ascendc-docs-search` 技能搜索：
- 关键词：错误码数值 + 错误信息关键字
- 示例：搜索 "561118" 或 "workspace invalid"

### Step 3: 搜索社区资源

- [CANN 社区 Issue](https://gitee.com/ascend/cann/issues)
- [昇腾论坛](https://www.hiascend.com/forum)
- 搜索关键词：`错误码 + 错误信息`
