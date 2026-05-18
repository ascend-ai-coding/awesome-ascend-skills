# op-plugin 参考说明

## 版本表（整理自 op-plugin README）

**`op-plugin` 分支与 Ascend Extension for PyTorch（`torch_npu`）的对应关系：**

| op-plugin branch | torch_npu version (branch)   |
|------------------|-----------------------------|
| master           | mainline, e.g. v2.7.1       |
| 7.3.0            | e.g. v2.7.1-7.3.0          |
| 7.2.0            | e.g. v2.7.1-7.2.0          |
| 7.1.0 / 7.0.0 / 6.x / 5.x | see op-plugin README |

**构建脚本：** `bash ci/build.sh --python=<ver> --pytorch=<branch>`。例如：`--python=3.9 --pytorch=v2.7.1-7.3.0`。

**PyTorch / Python / GCC 对应关系（代表性示例）：**

| PyTorch   | Python        | GCC (ARM / x86) |
|-----------|---------------|------------------|
| v2.6.0    | 3.9, 3.10, 3.11 | 11.2 / 9.3       |
| v2.7.1    | 3.9, 3.10, 3.11 | 11.2             |
| v2.8.0+   | 3.9, 3.10, 3.11 (3.10+ for 2.9+) | 13.3   |

PyTorch 2.6 及以上版本不再支持 Python 3.8。更准确的完整版本矩阵请以 `op-plugin` 官方 README 为准。

## SOC_VERSION

- 在 `CMakeLists.txt` 中通常写成：`set(SOC_VERSION "Ascendxxxyy" CACHE STRING "system on chip type")`
- 在机器上执行 `npu-smi info`，读取其中的 **Chip Name**，然后把 `SOC_VERSION` 设置为 `Ascend` + 对应芯片名（例如 `Ascend910B`）

## 常用链接

- [op-plugin (gitcode)](https://gitcode.com/ascend/op-plugin)
- [Ascend Extension for PyTorch (torch_npu)](https://gitcode.com/ascend/pytorch)
- `op-plugin` 仓库中的 `examples/cpp_extension/README.md`：包含目录布局、kernel/host/tiling 说明以及运行步骤（仓库内路径：`examples/cpp_extension/README.md`）
- [Ascend C](https://www.hiascend.com/ascend-c) — Ascend C kernel 开发文档

