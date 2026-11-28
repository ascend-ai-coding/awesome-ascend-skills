# ACLNN Runner 开发

本文件是 migration 工程测试 runner 的唯一详细开发指导。正式功能验证只使用基于目标算子 ACLNN example 和 ACLNN 参考文档生成的直接 ACLNN runner；不得使用 TTK、ST、ATK、kernel 专用 runner 或其他替代验证后端。

## 目录

1. 资料调查与依赖预检
2. Runner 合同
3. Runtime 生命周期
4. Tensor 与二进制 I/O
5. OPP 环境
6. 开发顺序
7. G4/G5 结果
8. 反模式检查

## 1. 资料调查与依赖预检

编码前依次调查目标算子的对外接口文档、ACLNN example、ACLNN 头文件和 ACLNN 参考文档；必要时补充 checker、shape inference、Host Tiling 和活动实现证据。恢复并记录：

- 参数顺序、输入输出数量和可选参数；
- dtype、format、逻辑 shape、storage shape、stride、offset 和布局语义；
- workspace 查询、执行、同步和资源释放顺序；
- 编译器、头文件、库、链接选项和运行时依赖。

正式编码前检查所有依赖：头文件和库存在，符号链接可解析，ACL 类型和枚举来自当前环境，基础 OPP 与自定义 OPP 路径有效。不能假定第三方 JSON 库或其他未探测依赖存在；依赖选择必须记录并可复现。

## 2. Runner 合同

Runner 在 G3 开发，必须读取 G3 依据 G1 验证义务冻结的 design 用例和输入资产，每次执行一个 `case_id`，并支持 original 与 Blaze 两个 role。具体 tensor 名称、数量、文件名和属性完全由目标算子接口及 design 决定，本文件不预设算子字段。

建议参数包含：

```text
--case <case-id>
--role <original|blaze>
--input-dir <dir>
--output-dir <dir>
--device <id>
--warmup <n>
--repeat <n>
--environment-state <path>
```

参数必须实际影响执行；调用者不得用参数把正式用例集合缩小为子集。runner 必须保留 ACLNN workspace 查询和执行两阶段，并保存输出、元数据、日志、执行次数、实际路由和身份信息。

## 3. Runtime 生命周期

先选择并记录生命周期模型：

- 单设备、单进程、单 stream 场景只创建所需的当前 device 和 stream；
- 多设备、多 context 或显式 context 切换时才创建和管理 context；
- stream 必须属于当前有效 context；
- 初始化、创建、同步和释放顺序必须与 ACL runtime 约束一致；
- 不得混用隐式当前 context 和不匹配的显式 context。

通用顺序为：初始化 runtime、选择 device、确定 context、创建 stream、创建 Tensor、查询 workspace、执行 ACLNN、同步、拷贝输出、释放资源。生命周期选择不能由错误的单一示例推导，必须与实际调用合同一致。

## 4. Tensor 与二进制 I/O

### 4.1 Tensor 语义

每个 Tensor 分别记录：逻辑 view shape、物理 storage shape、元素 stride、offset、dtype、format、packing 和 transpose/layout 状态。创建前检查 rank、shape、stride、storage 容量、offset、format 和关联 Tensor 的约束一致。

转置、非连续布局或物理重排不能只交换 shape 表示。必须根据 ACLNN 文档或实现证据确认：API 看到的逻辑 view、物理数据排列、stride、storage shape 以及相关 Tensor 是否同步变化。无法证明时保持 `unknown`，不得通过试错修改冻结 design。

### 4.2 二进制输入

输入目录按用例组织：

```text
validation/inputs/<case-id>/
├── input-manifest.json
└── tensors/<tensor-id>.bin
```

manifest 记录实际 ACL 参数绑定、dtype、format、逻辑/物理 shape、stride、offset、packing、字节数、生成 seed 和可选状态。runner 统一计算元素大小和预期文件长度，执行前检查文件存在、大小、SHA256、布局和可选输入状态。

### 4.3 二进制输出

输出按 role 和用例保存：

```text
validation/results/<role>/<case-id>/
├── outputs/<output-tensor-id>.bin
├── output-manifest.json
├── result.json
└── runner.log
```

输出保留原始 dtype 和布局，不统一转换为 FP32。必须保存全部输出 Tensor、动态元数据和必要的 inplace after-state；输出缺失、长度错误或保存失败直接生成 `FAIL` 或 `NOT_RUN`。

ACLNN 内部 workspace、padding 和内部 Tensor storage 对 runner 不透明，不能声称已对这些区域设置物理 guard。runner 只检查其可控制的 host buffer canary、输出初始化、文件长度、元数据、inplace after-state 和重复稳定性；全零不能作为通用越界判断。

## 5. OPP 环境

系统基础 OPP 与本次自定义 OPP 必须分层：基础 OPP 提供运行时依赖，自定义 OPP 通过 `ASCEND_CUSTOM_OPP_PATH` 叠加。不得用独立自定义路径覆盖基础 `ASCEND_OPP_PATH`。original 与 Blaze 在独立进程和独立自定义 OPP 根中运行，结果必须记录最终生效路径、package、vendor、Kernel 和环境 revision。

## 6. 开发顺序

1. **单用例闭环**：使用 G3 设计的最小合法用例，完成 runtime、Tensor、workspace、ACLNN 执行、同步、输出 bin 和结果记录。
2. **静态与运行检查**：验证 dtype/字节数、shape/storage/stride、布局、输出元数据和重复稳定性。
3. **参数化扩展**：严格按 G3 用例表扩展 dtype、format、layout、transpose、shape、属性、边界、fallback 和非法场景。
4. **集合冻结**：核对 G1 义务、G3 用例表、输入、runner 注册及 G4/G5 执行清单，构建 runner 但不执行正式验证。
5. **Original 基线**：G4 用同一 runner 执行全部冻结用例，保存稳定 original 输出。
6. **Blaze 验收**：G5 复用同一 runner、输入、参数和输出协议，只切换被测 OPP。

runner 开发不得根据 Blaze 当前实现或试运行结果删除、跳过、合并或重分类 G1 义务或 G3 用例。具体用例/输入设计错误返回 G3；支持域或源行为错误返回 G1。

## 7. G4/G5 结果

### G4 original 基线

G4 只证明原始实现能够真实、完整、稳定地产生二进制基线。每个冻结用例必须有真实执行、非零执行次数、完整输出 bin、必要元数据、正确路由和重复执行逐字节稳定性。G4 不要求 CPU golden、NumPy reference 或其他独立参考实现比较。

### G5 Blaze 比较

G5 对相同输入、相同 runner、相同执行参数和相同用例集合执行 Blaze。只有 original 与 Blaze 的输出 bin、必要动态元数据和 inplace after-state 全部逐字节一致，且身份、路由、执行次数和重复稳定性通过时，case 才能 `PASS`。

`close`、`rtol`、`atol`、平均误差和外部 golden 只能作为诊断信息，不能生成 `PASS` 或豁免逐字节差异。

结果状态只允许 `PASS`、`FAIL`、`NOT_RUN`，并由 runner 逐 case 生成；人工汇总不能覆盖失败。

## 8. 反模式检查

- 未调查 ACLNN example、头文件和参考文档即编码；
- 引入未探测的第三方依赖；
- 猜测 ACL 类型、元素大小或参数顺序；
- 用单一 shape 表示逻辑 view 和物理 storage；
- 仅交换 shape 表示 transpose；
- 无依据创建 context 或使用不属于当前 context 的 stream；
- 用自定义 OPP 覆盖系统基础 OPP；
- 直接使用 TTK、ST、ATK 或其他替代验证后端；
- 用文本摘要代替真实输入/输出 bin；
- 把 G4 original 稳定性误写成 CPU golden 正确性；
- 用 close 误差替代 G5 逐字节比较；
- 在 G2 迁移开发完成前开发正式 runner，或用 Blaze 试运行结果决定 G3 用例；
- 通过删除、跳过或伪造结果隐藏失败。
