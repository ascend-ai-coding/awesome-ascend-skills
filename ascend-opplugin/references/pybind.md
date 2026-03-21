# Pybind 路线说明

## 适用场景

`Pybind` 适合下面这类需求：

- 只想快速把 C++ / AscendC 侧的函数暴露给 Python 做联调
- 当前目标是先验证 kernel、参数、返回值或基本功能是否跑通
- 暂时不要求通过 `torch.ops.*` 调用
- 暂时不要求 schema、DispatchKey、图追踪或长期维护

## 不适合作为默认正式接入的原因

与 `torch.library` / `TORCH_LIBRARY[_IMPL]` 路线相比，`Pybind` 更像“Python 与 C++ 的桥接层”，而不是正式的 PyTorch 算子注册方式。因此通常不作为本 skill 的默认主线，原因包括：

- 不属于完整的 PyTorch 算子系统注册
- 通常没有标准 schema 定义
- 不适合作为 `torch.ops.<namespace>.<op>` 的长期正式入口
- 对后续图追踪、编译优化、算子系统集成支持较弱

## 最小闭环

如果用户明确要走 `Pybind` 做快速验证，建议至少完成下面 5 步：

1. 准备环境：确保 `torch`、`torch_npu`、CANN、NPU 环境可用
2. 实现 C++ 函数：在函数里完成输入检查、NPU 调用和输出分配
3. 定义 `PYBIND11_MODULE(...)`：把 C++ 函数暴露成 Python 模块函数
4. 编译动态库：生成可被 Python 导入的 `.so`
5. 写最小测试：`import <module>` 后调用函数，检查 dtype / shape / 数值或至少冒烟通过

## 最小结构示意

```text
<project_root>/
├── csrc/
│   ├── kernel/
│   │   └── {kernel_name}_custom.cpp
│   └── host/
│       └── {module_name}.cpp        # 含 run_xxx + PYBIND11_MODULE
├── setup.py or CMakeLists.txt
└── test/
    └── test.py
```

其中 `csrc/host/{module_name}.cpp` 一般至少包含两部分：

- `run_xxx(...)`：真正调用 NPU kernel 或底层 C++ 实现
- `PYBIND11_MODULE({module_name}, m)`：把 `run_xxx` 暴露给 Python

## 与本 skill 主线的关系

- 如果用户说的是“先快速验证一下”“先把 C++ 函数调起来”，可以考虑 `Pybind`
- 如果用户说的是“正式接入算子”“要在 PyTorch 里注册”“希望通过 `torch.ops.*` 调用”，仍然优先走本 skill 主线，也就是 `torch.library` / Pattern A/B/C

## 迁移建议

很多场景下，`Pybind` 只是一个临时验证手段。验证通过后，通常应迁移到正式接入路线：

- 自研 kernel：迁移到 Pattern A / Pattern B
- 已有图算子：迁移到 Pattern C

也就是说，`Pybind` 更适合“先证明可行”，而不是“最终交付形态”。

