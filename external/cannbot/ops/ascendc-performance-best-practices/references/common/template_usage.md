# 模板源码查找、复制与构建契约

先完成目标代际与能力筛选。不匹配的模板只能作为算法参考，不能因为目录中存在源码就直接落地。

## 查找链路

从 `SKILL.md` 的算子分类表进入目标族 `guide.md`，在模板表选择方案，读取说明文档及其链接的 `.h`、`.cpp` 或 `.template` 源码。`.template` 是参考模板源码，先按所在目录的 `usage_guide.md` 转成同名 `.h`/`.cpp`，再编译。实现指导型算子族的 `code/` 目录也由族级指南索引，不能跳过适用条件。

查到模板后，先复制到候选副本以保留算法和数据流，再按目标工程构建契约适配：

1. 将源码放进项目的 `op_kernel/` 目录，保留 `dav310/` 等必要的相对子目录结构；在 `kernel.cpp` 中按分支条件 include 并 dispatch。
2. 对比源/目标构建契约，受控调整 Reg 声明作用域、VF 入口、include、API 签名和 TilingData；逐项记录原因并以最小编译探针验证。不得修改 SDK 或无条件添加命名空间别名。
3. 保留算法语义、寄存器热链、mask/tail、cast/pack 与数值顺序；不同构建契约下不要求逐字保留声明位置或入口包装。

禁止查到模板后仍从零编写 kernel。

## Reg VF 实施门禁

仅 arch35 编写或改造 Reg VF 时，同时查询本库 [VF 范例库](arch35/regbase/README.md) 的 `catalog.json` 和 `ascendc-regbase-best-practice` 的 `references/reference-ops/open_source_operator_table.md`。命中任意一边后实际读取代码并记录可/不可复用结论；两边均无时记录 `no_matching_reference_implementation`。本地条目按计算模式、shape/layout 前置条件和 readiness 选择：

- `reference_template`：复制模板及依赖，按使用指南接入。
- `integration_skeleton`：只把已收录函数作为骨架，补齐目标工程上下文。
- `annotated_excerpt`：读取摘录代码和注解，补齐目标工程类上下文；不得把省略分支当作完整实现。

具体 API 必须用目标 SDK 头文件核验。`readiness=integration_skeleton` 提供已收录函数的集成骨架；`annotated_excerpt` 必须补上下文，`reference_only` 必须先建立目标构建契约。所有候选均须在目标工程完成编译、精度和性能验证。

范例算法结构可复用，但声明作用域、VF 入口和构建 flags 只在 `catalog.json` 的 `example_build_contract` 下成立。复制前运行 `scripts/vf_preflight.py --project <candidate> --output <candidate>/vf-preflight.json --strict`，逐项读取 `vf_translation_units`，并与范例契约比较。实施契约 `expectations.vf/casttrait` 为 true 时分别追加 `--expect-vf`/`--expect-cast`；删除 VF 后的 `not_applicable` 不是通过。每个 VF 翻译单元及 include 闭包都须有自己的最终后端契约，顶层 `contracts` 只是证据清单。契约不同或未知时，最多标为算法可参考、集成边界待验证。

## 编译失败时

首次编译或集成失败后先读 [CANN / Reg VF 编译问题 FAQ](arch35/regbase/cann_version_compat_faq.md)，保存真实编译命令与首错，再按目标 SDK 和同环境反例诊断：

- CANN 9.0+ 新代码使用 `AscendC::Reg::`。若 `MicroAPI` 未声明，先核对 `--npu-arch`、include 链和真实编译模式；不手工补命名空间别名。
- API 签名以当前 SDK 头文件为准，不凭固定安装路径猜测。`no member named ... in AscendC::Reg` 只说明当前翻译 pass；SDK 有声明或同环境有成功反例时，先按 `build_contract_mismatch` / `host_pass_leakage` 处理。
- `CastTrait` 在 device-only 源码可位于命名空间/类级；直接 `-xasc` 联合编译时复用同工程成功的架构保护或 VF 函数边界。函数内 `Reg::Cast` trait 使用 `static constexpr CastTrait`；涉及 Cast 时运行 `scripts/run_reg_cast_probe.py` 复用目标真实命令。
- 临时手写、无架构保护或将完整 device-only 模板塞入 `xasc_combined` 的 probe 只能诊断，失败不能成为 blocker。`CreateMask/MaskReg/RegTensor` 与 `__local_mem__` 同时在 combined host pass 报错时，先按 device-only 代码泄漏处理；最终以修正边界后的真实候选构建为门禁。

不得因一次辅助 probe 失败直接降级为高层 API，也不得修改 SDK。
