# Linalg IR → 设备二进制

## 概述

Linalg IR → 设备二进制是 Triton-Ascend 编译流水线的最后阶段，将 Linalg IR 通过 BiSheng Compiler 编译为昇腾 NPU 可执行的设备二进制文件。该阶段涉及 Linalg IR → LLVM IR 的标准降低、BiSheng Compiler 的 HFusion/HIVM 优化、以及最终的目标代码生成。

根据硬件平台不同，使用不同的编译函数：
- 910_95 平台：[linalg_to_bin_enable_npu_compile_910_95()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L403)
- A2/A3 平台：[linalg_to_bin_enable_npu_compile_A2_A3()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L607)
- CPU 后端：[linalg_to_llir()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L171) + [llir_to_cpuasm()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L229)

## 关键概念

| 概念 | 说明 |
|------|------|
| BiSheng Compiler | 昇腾编译器（bishengir-compile），将 Linalg IR 编译为 NPU 二进制 |
| HFusion | 昇腾算子融合 IR，用于 CV 流水线优化 |
| HIVM | 昇腾设备指令层 IR，描述 Cube/Vector 核心执行指令 |
| kernel.o | NPU 目标文件，包含可执行的设备代码 |
| libkernel.so | 编译生成的共享库，包含元数据回调函数 |
| UB (Unified Buffer) | 统一缓冲区，Vector 核心的片上存储 |
| callback | 编译生成的回调函数，用于提取 task_type、workspace_size 等元数据 |

## linalg_to_llir()：mlir-opt 标准降低流水线

[linalg_to_llir()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L171) 用于 CPU 后端，通过 mlir-opt 的标准 Pass 将 Linalg IR 降低为 LLVM IR：

```python
def linalg_to_llir(linalg: str, metadata, opt):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        llmlir_path = os.path.join(tmpdir, "kernel.llir.mlir")
        llir_path = os.path.join(tmpdir, "kernel.ll")
        Path(ttadapter_path).write_text(linalg)

        mlir_opt_path = _get_mlir_path("bin", "mlir-opt")

        # Linalg-MLIR → LLVM-MLIR
        subprocess.check_call([
            mlir_opt_path, ttadapter_path,
            "--convert-linalg-to-affine-loops",
            "--eliminate-empty-tensors",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "--lower-affine",
            "--convert-linalg-to-loops",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-math-to-llvm",
            "--convert-complex-to-llvm",
            "--convert-vector-to-llvm",
            "--convert-index-to-llvm",
            "--memref-expand",
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--lower-affine",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            "-o", llmlir_path,
        ])

        # LLVM-MLIR → LLVM-IR
        mlir_translate_path = _get_mlir_path("bin", "mlir-translate")
        subprocess.check_call([
            mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path
        ])

        return Path(llir_path).read_text()
```

### mlir-opt 降低 Pass 序列

| 序号 | Pass | 说明 |
|------|------|------|
| 1 | `--convert-linalg-to-affine-loops` | Linalg → Affine 循环 |
| 2 | `--eliminate-empty-tensors` | 消除空张量 |
| 3 | `--empty-tensor-to-alloc-tensor` | 空张量 → 分配张量 |
| 4 | `--one-shot-bufferize` | 一次性缓冲区化 |
| 5 | `--lower-affine` | Affine → 标准操作 |
| 6 | `--convert-linalg-to-loops` | Linalg → 标准循环 |
| 7 | `--convert-scf-to-cf` | SCF → CF 控制流 |
| 8 | `--convert-cf-to-llvm` | CF → LLVM |
| 9 | `--convert-arith-to-llvm` | Arith → LLVM |
| 10 | `--convert-math-to-llvm` | Math → LLVM |
| 11 | `--convert-complex-to-llvm` | Complex → LLVM |
| 12 | `--convert-vector-to-llvm` | Vector → LLVM |
| 13 | `--convert-index-to-llvm` | Index → LLVM |
| 14 | `--memref-expand` | MemRef 展开 |
| 15 | `--expand-strided-metadata` | 展开步幅元数据 |
| 16 | `--finalize-memref-to-llvm` | MemRef → LLVM 最终化 |
| 17 | `--convert-func-to-llvm` | Func → LLVM |
| 18 | `--lower-affine` | 二次 Affine 降低 |
| 19 | `--convert-arith-to-llvm` | 二次 Arith 降低 |
| 20 | `--reconcile-unrealized-casts` | 消除未实现的类型转换 |

## llir_to_cpuasm()：LLVM IR → CPU 汇编

[llir_to_cpuasm()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L229) 用于 CPU 后端，将 LLVM IR 编译为 CPU 汇编代码：

```python
def llir_to_cpuasm(llir: str, metadata, opt):
    metadata["shared"] = 1
    fn_name = llir.split("define void @")[1].split("(")[0].strip()
    metadata["name"] = fn_name + " cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        linked_path = os.path.join(tmpdir, "kernel_linked.ll")
        dst_path = os.path.join(tmpdir, "kernel.s")

        # LLVM IR 降级
        llir = downgrade_llir(llir)
        Path(src_path).write_text(llir)

        # 链接 libdevice
        linker_path = _get_llvm_path("bin", "llvm-link")
        libclc_path = _get_llvm_path("lib", "clc", "libspirv-aarch64--.bc")
        subprocess.check_call([
            linker_path, src_path, libclc_path, "--only-needed", "-S", "-o", linked_path
        ])

        # LLVM IR → CPU 汇编
        llc_path = _get_llvm_path("bin", "llc")
        subprocess.check_call([llc_path, linked_path, "-o", dst_path])

        return Path(dst_path).read_text()
```

### CPU 后端编译流程

```
Linalg IR → mlir-opt → LLVM-MLIR → mlir-translate → LLVM IR
    → downgrade_llir() → llvm-link + libdevice → llc → CPU 汇编
```

## BiSheng Compiler 的作用

### 概述

BiSheng Compiler（bishengir-compile）是昇腾 NPU 的核心编译器，接收 Linalg IR 并执行以下优化和转换：

1. **HFusion 编译**：算子融合，将多个算子融合为 CV 流水线
2. **HIVM 编译**：设备指令生成，将融合后的算子转换为 Cube/Vector 核心指令
3. **LLVM 降低**：将 HIVM IR 降低为 LLVM IR
4. **目标代码生成**：生成 NPU 可执行的二进制文件

### 编译流程

```
Linalg IR
    │
    │  bishengir-compile
    │  --enable-hfusion-compile=true
    │  --enable-triton-kernel-compile=true
    │
    ▼
HFusion IR (算子融合)
    │
    ▼
HIVM IR (设备指令)
    │
    ▼
LLVM IR
    │
    ▼
kernel.o (NPU 二进制)
```

### 910_95 平台编译

[linalg_to_bin_enable_npu_compile_910_95()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L403) 的核心流程：

1. **解析元数据**：调用 `_parse_linalg_metadata()` 提取 mix_mode、kernel_name 等
2. **写入临时文件**：将 Linalg IR 写入 `kernel.ttadapter.mlir`
3. **构建编译选项**：根据 metadata 和 opt 构建编译选项列表
4. **调用 BiSheng Compiler**：通过 `subprocess.run()` 执行 bishengir-compile
5. **提取元数据**：从输出中解析 UB 占用信息
6. **加载回调函数**：从 `libkernel.so` 中加载元数据回调函数

### A2/A3 平台编译

[linalg_to_bin_enable_npu_compile_A2_A3()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L607) 与 910_95 类似，但有以下差异：

- 支持 `--enable-ubuf-saving` 选项
- 支持 `--tile-mix-vector-loop` 和 `--tile-mix-cube-loop` 选项
- 使用 `--reg-based=true` 或 `--enable-hivm-compile=true` 选项
- 支持 `--enable-memory-display` 选项

## 内核二进制格式

### 输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `kernel.o` 或 `kernel_reloc.o` | ELF 目标文件 | NPU 可执行的设备代码 |
| `libkernel.so` | 共享库 | 包含元数据回调函数（可选） |

### 二进制文件名确定

```python
if _check_bishengir_api_change():
    bin_file_with_ext = "kernel.o"
else:
    bin_file_with_ext = "kernel_reloc.o"
```

### 元数据回调函数

编译成功后，如果 `libkernel.so` 存在，从中加载以下回调函数：

| 回调函数 | 元数据字段 | 说明 |
|----------|-----------|------|
| `_infer_task_type_function` | `bs_task_type` | 推断任务类型 |
| `_infer_workspace_shape_function` | `workspace_size` | 推断工作空间大小 |
| `_infer_sync_block_lock_num_function` | `lock_num` | 推断同步锁数量 |
| `_infer_sync_block_lock_init_function` | `lock_init_val` | 推断同步锁初始值 |

```python
if Path(callback_path).is_file():
    lib = ctypes.CDLL(callback_path)
    __get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
    __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
    __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata, "lock_num")
    __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")
```

### UB 占用信息

BiSheng Compiler 在编译输出中包含 UB 占用信息：

```
UB size = 123456 bits
```

该信息通过正则表达式提取，用于 Inductor autotune：

```python
stdout_str = ret.stdout.decode('utf-8') if ret.stdout else ''
match = re.search(r'UB\s+size\s*=\s*(\d+)\s*bits', stdout_str)
if match:
    metadata["required_ub_bits"] = int(match.group(1))
```

## 链接和加载过程

### 编译时链接

BiSheng Compiler 在编译时链接以下内容：

1. **libdevice bitcode**：提供数学函数（sin、cos、exp 等）的底层实现
2. **用户指定的 bitcode**：通过 `--link-aicore-bitcode` 选项链接

```python
bisheng_options = metadata["bisheng_options"]
if bisheng_options is not None:
    _compile_option_list += [f"--append-bisheng-options={bisheng_options}"]

bitcodes = metadata["bitcodes"]
if bitcodes is not None:
    for bitcode in bitcodes:
        _compile_option_list += [f"--link-aicore-bitcode={bitcode}"]
```

### 运行时加载

编译后的二进制在运行时通过 NPU Driver 加载：

1. **加载二进制**：`driver.active.utils.load_binary()` 加载 `.o` 文件
2. **分配工作空间**：根据 `workspace_size` 分配 GM 工作空间
3. **配置同步**：根据 `lock_num` 和 `lock_init_val` 配置同步机制
4. **启动 Kernel**：`driver.active.launcher_cls` 创建启动器并执行

### CPU 后端链接

CPU 后端使用 `llvm-link` 链接 libdevice：

```python
linker_path = _get_llvm_path("bin", "llvm-link")
libclc_path = _get_llvm_path("lib", "clc", "libspirv-aarch64--.bc")
subprocess.check_call([
    linker_path, src_path, libclc_path, "--only-needed", "-S", "-o", linked_path
])
```

## BiSheng Compiler 编译选项详解

### 通用选项

| 选项 | 说明 |
|------|------|
| `--target=<arch>` | 目标架构（如 `ascend910b`、`ascend910_95`） |
| `--enable-hfusion-compile=true` | 启用 HFusion 编译 |
| `--enable-triton-kernel-compile=true` | 启用 Triton Kernel 编译 |
| `-o <output>` | 输出文件路径 |

### 内存优化选项

| 选项 | 说明 | 适用平台 |
|------|------|----------|
| `--enable-auto-multi-buffer=<n>` | 启用多缓冲（ping-pong pipeline） | 910_95/A2/A3 |
| `--enable-ubuf-saving=<n>` | 启用 UB 节省 | A2/A3 |
| `--limit-auto-multi-buffer-only-for-local-buffer=<n>` | 限制多缓冲仅用于本地缓冲 | 910_95/A2/A3 |
| `--limit-auto-multi-buffer-of-local-buffer=<n>` | 限制本地缓冲的多缓冲 | 910_95/A2/A3 |
| `--set-workspace-multibuffer=<n>` | 设置工作空间多缓冲 | 910_95/A2/A3 |
| `--disable-tightly-coupled-buffer-reuse` | 禁用紧耦合缓冲区复用 | 910_95 |

### CV 流水线选项

| 选项 | 说明 | 适用平台 |
|------|------|----------|
| `--enable-auto-bind-sub-block=<n>` | 启用自动绑定子块 | 910_95/A2/A3 |
| `--enable-hivm-auto-cv-balance=<n>` | 启用 CV 自动平衡 | 910_95/A2/A3 |
| `--enable-hivm-graph-sync-solver=<n>` | 启用同步求解器 | 910_95/A2/A3 |
| `--enable-hivm-unit-flag-sync=<n>` | 启用单元标志同步 | 910_95/A2/A3 |
| `--enable-hivm-inject-barrier-all-sync=<n>` | 启用全屏障同步 | 910_95/A2/A3 |
| `--enable-hivm-inject-block-all-sync=<n>` | 启用全块同步 | 910_95/A2/A3 |
| `--tile-mix-vector-loop=<n>` | 混合 Vector 循环 tiling | A2/A3 |
| `--tile-mix-cube-loop=<n>` | 混合 Cube 循环 tiling | A2/A3 |
| `--disable-auto-inject-block-sync=<n>` | 禁用自动注入块同步 | 910_95/A2/A3 |

### 向量化选项

| 选项 | 说明 | 适用平台 |
|------|------|----------|
| `--enable-vf-fusion` | 启用 VF 融合 | 910_95 |
| `--enable-drop-unit-dims=<n>` | 启用单位维度消除 | 910_95/A2/A3 |
| `--enable-flatten=<n>` | 启用展平 | 910_95/A2/A3 |
| `--enable-auto-vectorize-v2=<n>` | 启用自动向量化 v2 | 910_95/A2/A3 |
| `--hfusion-max-fused-ops-in-auto-vectorize-v2=<n>` | 自动向量化 v2 最大融合操作数 | 910_95 |
| `--hfusion-max-fused-elementwise-ops=<n>` | 最大融合逐元素操作数 | 910_95 |
| `--enable-vf-merge-level=<n>` | VF 合并级别 | 910_95 |
| `--hfusion-enable-multiple-consumer-fusion=<n>` | 启用多消费者融合 | 910_95 |

### SIMT 选项

| 选项 | 说明 |
|------|------|
| `--enable-hivm-compile=false` | 禁用 HIVM 编译（SIMT 模式） |
| `--enable-triton-ir-compile` | 启用 Triton IR 编译 |
| `--pure-simt` | 纯 SIMT 模式 |
| `--num-warps=<n>` | Warp 数量 |
| `--threads-per-warp=<n>` | 每 Warp 线程数 |
| `--enable-bishengir-simt-optimization=<n>` | SIMT 优化控制位 |
| `--simt-stack-limit=<n>` | SIMT 栈限制 |
| `--shared-mem-dynamic-size=<n>` | 共享内存动态大小 |
| `--enable-simt-reorder-instruction=true` | 启用 SIMT 指令重排 |
| `--disable-fma` | 禁用 FMA |

### 调试选项

| 选项 | 说明 |
|------|------|
| `--bishengir-print-ir-after=hivm-inject-sync` | 打印 HIVM 同步注入后的 IR |
| `--enable-sanitizer=true` | 启用 Sanitizer |
| `--enable-debug-info=true` | 启用调试信息 |
| `--enable-print-memory-allocated-size` | 打印内存分配大小 |
| `--enable-memory-display=true` | 显示内存信息 |
| `--disable-ffts` | 禁用 FFTS |

## NPU 适配要点

1. **平台区分**：910_95 和 A2/A3 使用不同的编译函数和选项
2. **BiSheng Compiler 版本**：`_check_bishengir_api_change()` 和 `_check_bishengir_is_regbased()` 检测编译器版本，影响输出文件名和编译选项
3. **UB 内存管理**：编译输出包含 UB 占用信息，用于 Inductor autotune 优化
4. **错误处理**：编译失败时保存 NPU IR 到 `kernel.npuir.mlir` 便于调试
5. **环境变量**：`TRITON_ENABLE_LIBDEVICE` 控制是否链接 libdevice bitcode

## 常见问题

### Q: 编译失败如何获取更多错误信息？

1. 设置 `opt.debug=True`，编译器会打印完整的命令行
2. 检查缓存目录中的 `kernel.npuir.mlir` 文件
3. 设置 `TRITON_KERNEL_DUMP=1` 转储各阶段 IR
4. 检查 BiSheng Compiler 的 stderr 输出

### Q: kernel.o 和 kernel_reloc.o 有何区别？

文件名取决于 BiSheng Compiler 的 API 版本。新版本 API 生成 `kernel.o`，旧版本生成 `kernel_reloc.o`。通过 `_check_bishengir_api_change()` 检测。

### Q: 如何控制 UB 内存使用？

1. 通过 `--enable-auto-multi-buffer` 控制多缓冲策略
2. 通过 `--limit-auto-multi-buffer-of-local-buffer` 限制本地缓冲多缓冲
3. 设置 `ENABLE_PRINT_UB_BITS=1` 获取 UB 占用信息
4. 在 autotune 中使用 `required_ub_bits` 进行配置选择

### Q: CPU 后端和 NPU 后端的编译路径有何区别？

CPU 后端使用 mlir-opt + llvm-link + llc 的标准 LLVM 工具链，生成 CPU 汇编。NPU 后端使用 BiSheng Compiler，经过 HFusion/HIVM 优化生成 NPU 二进制。

## 相关文档

- [01-pipeline-overview.md](01-pipeline-overview.md) - 编译流水线总览
- [05-triton-to-linalg.md](05-triton-to-linalg.md) - Triton IR → Linalg IR 转换
- [07-compile-options.md](07-compile-options.md) - 编译选项与环境变量参考
