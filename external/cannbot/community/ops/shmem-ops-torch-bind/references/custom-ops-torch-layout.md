# custom-ops Torch 共享外层布局

> **仓内路径**：下文 `examples/torch_binding` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

`custom-ops/` 下的算子 **MUST** 采用与 `${SHMEM_REPO}/examples/torch_binding` 相同的**共享外层**模式，**禁止**在每个算子目录内再建完整 `torch_binding/` 子工程。

## 目录结构（MUST）

```text
custom-ops/
├── torch_binding/                    # 共享 Torch 接入层（仅一份）
│   ├── CMakeLists.txt                # 统一构建 shmem_custom_ops_torch.so
│   ├── include/
│   │   ├── torch_register.h          # 可从 examples/torch_binding 复用或 symlink
│   │   └── shmem_torch_kernel.h      # 声明各算子 kernel 入口
│   └── src/
│       ├── torch_bindings.cpp        # Manager 类（一次生成，多算子共用）
│       └── torch_bind_<op_name>.cpp  # 每新增一个算子只加此文件 + CMake 一行
├── <op_a>/                           # 独立算子工程（仅 C++ kernel + scripts）
│   ├── src/
│   ├── scripts/
│   └── docs/
└── <op_b>/
    └── ...
```

## 与 shmem examples 的差异

| 项 | `examples/torch_binding` | `custom-ops/torch_binding` |
| --- | --- | --- |
| 产物 `.so` 名 | `aclshmem_torch.so` | **`shmem_custom_ops_torch.so`**（NEVER 同名） |
| CMake 变量 | `ACLSHMEM_ALL_KERNEL_TARGETS`（父 CMake 注入） | `CUSTOM_OPS_KERNEL_TARGETS`（各算子 build 产物路径/target） |
| 算子位置 | `examples/<op>/` | `custom-ops/<op>/` |
| 新增算子改动 | 追加 `torch_bind_*.cpp` + kernel target | **同上**，不改 Manager / 不重写 CMake 框架 |

## 新增算子接入清单（Checklist）

1. 在 `custom-ops/<op>/` 完成 kernel 编译，产出 `build/lib/lib<op>_kernel.so` 或 CMake target
2. 在 `custom-ops/torch_binding/include/shmem_torch_kernel.h` 追加 kernel 声明
3. 新增 `custom-ops/torch_binding/src/torch_bind_<op>.cpp`
4. 在 `custom-ops/torch_binding/CMakeLists.txt`：
   - `list(APPEND CUSTOM_OPS_KERNEL_TARGETS ...)` 或 `target_link_libraries` 追加 kernel
   - `target_sources(shmem_custom_ops_torch PRIVATE src/torch_bind_<op>.cpp)`
   - `REGISTER_SHMEM_OPS_CLASS(<OpName>, ...)`
5. 在 `custom-ops/<op>/scripts/torch_test_<op>.py` 中 `load_torch_library('shmem_custom_ops_torch.so')`
6. **禁止** 在 `custom-ops/<op>/torch_binding/` 再建第二套 CMake

## 编译脚本（MUST 使用）

| 步骤 | 命令 | 产物 / 用途 |
| --- | --- | --- |
| 编译算子 kernel | [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §1 | `build/bin/<op>`, `build/lib/lib<op>_kernel.so` |
| 运行正确性 | [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §2 | 转发至算子 `scripts/run.sh` |
| 批量 case | [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §3 | case matrix 回归 |
| 性能采集 | [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md) §1 阶段 B | `[PERF]` 输出 |
| 编译共享 Torch | [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §4 | `shmem_custom_ops_torch.so` |

**NEVER** 在 skill/README 中把裸 `cmake -S ... -B ...` 作为首选入口；脚本内部可调用 cmake。

## 反模式

- ❌ 每个算子目录内独立 `torch_binding/` + `aclshmem_torch.so`（与 shmem 仓混淆、重复 Manager）
- ❌ custom-ops 产物仍命名为 `aclshmem_torch.so`
- ❌ 新增算子时复制整份 CMakeLists / Manager 代码
