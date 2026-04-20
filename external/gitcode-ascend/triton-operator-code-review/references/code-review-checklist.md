# Triton 算子静态代码检视检查清单

本清单仅包含通过阅读代码可直接判断的检查项。

## Host 侧

### 接口设计
- [ ] 有输入形状校验（assert）
- [ ] 有数据类型校验
- [ ] 有设备一致性校验（`x.device == y.device`）
- [ ] 有边界/空输入处理

### Grid 配置
- [ ] 核数**非硬编码**（无 `grid = (20,)` 等字面量）
- [ ] 含 `tl.dot` 用 AI Core（`num_aicore`），其他用 Vector Core
- [ ] Grid 大小合理（推荐 1D）

### Block Size
- [ ] BLOCK_SIZE 声明为 `tl.constexpr`
- [ ] 矩阵运算 BLOCK_M/N/K 为 16 的倍数
- [ ] BLOCK_K 满足对齐（`kalign = 32 // dtype_bytes`）

## Device 侧

### Mask 完整性
- [ ] 所有 `tl.load` 有 `mask=` + `other=`（或使用 `make_block_ptr`）
- [ ] 所有 `tl.store` 有 `mask=`（或使用 `make_block_ptr`）

### 数据类型合规
- [ ] `tl.dot` 输入仅用 int8/fp16/fp32/bf16
- [ ] 未使用 `dot_scaled`
- [ ] `permute`/`trans` 未使用 int64
- [ ] `permute`/`trans` 3D (2,1,0) 注意兼容性

### 精度处理
- [ ] 归约操作前升精度到 FP32（`.to(tl.float32)`）
- [ ] `tl.dot` 的 `out_dtype`（浮点默认 fp32、int8 仅 int32 可选，显式指定非必要）
- [ ] Softmax 减最大值（`tl.exp(x - max_x)`）
- [ ] 输出前转回目标精度

### Atomic 操作
- [ ] `atomic_or/xor/and/xchg/cas` 未在 `for` 循环体内
- [ ] 多核 kernel 中 `tl.atomic_add` 返回值未被使用

### 代码模式
- [ ] 小且固定次数的循环可考虑 `tl.static_range`（loop 数较大时可能劣化，非强制）
- [ ] kernel 内未调用第三方库
- [ ] 向量化计算（非逐元素）

## 性能隐患

### 内存
- [ ] 无冗余 GM 访问（同一 ptr 多次 `tl.load`）
- [ ] 连续访存（无 `tl.arange * stride` 跳跃）
- [ ] 数据复用充分

### 计算
- [ ] Cube BLOCK 为 16 倍数（字面量可检查）
- [ ] 小循环可考虑 `tl.static_range`（大 loop 慎用）
- [ ] 向量化计算

### 同步
- [ ] Host 热路径无 `.item()`
- [ ] CPU-NPU 同步最小化
