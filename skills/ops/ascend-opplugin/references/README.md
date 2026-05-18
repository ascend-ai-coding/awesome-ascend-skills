# 参考资料（ascend-opplugin）

这个目录存放 `ascend-opplugin` skill 的可复用参考资料，需要配合主文档 `../SKILL.md` 一起阅读。

阅读建议：

- **先看 `../SKILL.md`**：它包含 Ascend C 算子接入 torch_npu 的主流程（环境、工程布置、Host 注册、构建与测试等）
- **遇到版本矩阵或环境问题时**：优先查看 `reference.md`
- **需要新增一个自定义算子时**：查看 `examples.md` 里的通用清单

| 文件 | 主题 |
| --- | --- |
| [reference.md](reference.md) | `op-plugin` 分支与 `torch_npu` 的版本矩阵、`SOC_VERSION` 设置、常用链接 |
| [examples.md](examples.md) | 新增自定义算子的通用步骤（无 workspace / 有 workspace / 复用图算子等） |
| [pybind.md](pybind.md) | `Pybind` 路线的适用场景、限制与最小闭环 |
| [prompt-examples-beginner.md](prompt-examples-beginner.md) | 第一次使用本 skill 时可直接参考的 5 条精选 prompt |
| [prompt-examples.md](prompt-examples.md) | 使用本 skill 时可参考的 prompt 示例 |
| [container-cli-driver.md](container-cli-driver.md) | 容器内环境、容器外命令行驱动约定（含 OpCommand 等模板与常见报错修复） |

> 兼容性说明：仓库根目录保留了 `examples.md` / `reference.md` 的跳转页，用于避免旧链接失效。

