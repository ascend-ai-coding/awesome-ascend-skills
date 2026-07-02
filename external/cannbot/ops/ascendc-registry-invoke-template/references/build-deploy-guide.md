# 编译部署指南

> 完整示例参考：`references/add_example/`

## 构建流程

```
cmake配置 → 编译(host+kernel+graph) → 打包(.run) → 安装部署
```

## build.sh 脚本

### 常用命令

```bash
bash build.sh --soc=ascend910b -j8           # 编译指定芯片
bash build.sh --soc=ascend910b -j8 -u        # 编译 + UT测试
bash build.sh --soc=ascend910b -j8 -s        # 编译 + ST测试
bash build.sh --soc=ascend910b -j8 -e        # 编译 + 运行aclnn调用示例
bash build.sh --soc=ascend910b -j8 -e --graph # 编译 + 运行图模式调用示例
bash build.sh --soc=ascend910b -j8 -a        # 编译 + 全部测试
bash build.sh --make_clean                   # 清理构建目录
bash build.sh --list-socs                    # 查看支持的芯片
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--soc=<version>` | 目标芯片（必填）：ascend910b/ascend910_93/ascend950 |
| `-j[n]` | 编译线程数，默认自动检测（最大8） |
| `-u, --ut` | 运行 UT 测试 |
| `-s, --st` | 运行 ST 测试 |
| `-e, --example` | 运行调用示例（默认 aclnn，需 NPU + 算子包已安装） |
| `--eager` | 运行 aclnn 调用示例（-e 的默认模式） |
| `--graph` | 运行图模式 (GE IR) 调用示例 |
| `-a, --all` | 运行全部测试 |
| `--make_clean` | 清理 build/ 和 build_out/ |

### 构建产物

| 产物 | 路径 | 说明 |
|------|------|------|
| Kernel 二进制 | `build/op_kernel/ascendc_kernels/binary/<soc>/*.o` | 算子核函数 |
| 安装包 | `build/custom_opp_ubuntu_aarch64.run` | 部署包 |

---

## CMakeLists.txt 配置

### 根目录配置

```cmake
set(ARCH32_COMPUTE_UNITS ascend910b ascend910_93)
set(ARCH35_COMPUTE_UNITS ascend950)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR}
)
```

### 代际架构映射

| 架构 | 芯片 | Kernel入口 |
|------|------|------------|
| arch22 | ascend910b, ascend910_93 | `{op}_arch22.cpp` |
| arch35 | ascend950 | `{op}_arch35.cpp` |

> **注意**：cmake 默认 `make` 不编译 kernel binary，需使用 `bash build.sh` 或 `cmake --build . --target all binary package`

---

## 安装部署

### 安装

```bash
./build/custom_opp_ubuntu_aarch64.run
```

### 卸载

```bash
./build/scripts/uninstall.sh
```

### 升级

```bash
./build/scripts/upgrade.sh
```

---

## UT 测试

详见 `references/add_example/tests/ut/README.md`

```bash
bash build.sh -u --soc=ascend910b   # 推荐：从根目录运行
cd tests/ut && ./run.sh              # 或进入 UT 目录运行
```

---

## ST 测试

详见 `references/st-test-guide.md` 和 `references/add_example/tests/st/README.md`

```bash
bash build.sh -s --soc=ascend910b   # Real 模式（需 NPU，需先安装算子）
cd tests/st && ./run.sh --mock      # Mock 模式（无需 NPU）
```

---

## 调用示例 (Examples)

详见 `references/example-guide.md` 和 `references/add_example/examples/`

```bash
bash build.sh -e --soc=ascend910b          # 通过 build.sh 集成运行（需 NPU + 算子包）
bash build.sh -e --graph --soc=ascend910b  # 图模式调用示例（需 NPU + 算子包）
cd examples && ./run.sh                    # 或进入 examples 目录独立运行
cd examples && ./run.sh --graph            # 图模式调用示例
```

**说明**：`-e` 会自动安装算子包（与 `-s` 相同逻辑），然后调用 `examples/run.sh`。也可在算子包已安装后直接运行 `examples/run.sh`。

---

## 常见问题

### Q1: 编译报错找不到 ASCEND_COMPUTE_UNIT

**A:** 必须指定 `--soc` 参数

### Q2: cmake 编译后 kernel binary 不存在

**A:** 使用 `bash build.sh --soc=<soc>` 或添加 `--target binary`

### Q3: UT 测试 ABI 兼容性问题

**A:** 运行 `./run.sh --clean` 重新编译
