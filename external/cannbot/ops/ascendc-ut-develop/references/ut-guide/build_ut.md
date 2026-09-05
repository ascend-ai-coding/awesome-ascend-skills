# UT 编译运行指南

## 编译运行所有模块

```bash
bash build.sh -u --ops=$op_name --cov
```

## 编译部分模块

```bash
bash build.sh -u --opapi --ops=$op_name --cov
bash build.sh -u --ophost --ops=$op_name --cov
bash build.sh -u --opkernel --ops=$op_name --cov
```

## 查看编译帮助

```bash
bash build.sh -h
```

## 常用参数说明

| 参数 | 说明 |
|------|------|
| `-u` | 编译UT |
| `--ops=<op_name>` | 指定算子名称 |
| `--soc=<soc_version>` | 指定SOC版本 |
| `--cov` | 生成覆盖率报告 |
| `--opapi` | 仅编译op_api层 |
| `--ophost` | 仅编译op_host层 |
| `--opkernel` | 仅编译op_kernel层 |

## 环境变量

在 UT 运行前，设置环境变量`export ASCEND_SLOG_PRINT_TO_STDOUT=1`，将 UT 运行的 LOG 重定向到标准输出，方便定位问题

较难定位时，可以开启更低的 LOG 等级，输出更详细的信息：

| 等级 | 数值 |
|------|------|
| DEBUG | 0 |
| INFO | 1 |
| WARNING | 2 |
| ERROR（默认） | 3 |

`export ASCEND_GLOBAL_LOG_LEVEL=${数值}`

## 覆盖率报告位置

编译完成后，覆盖率报告位于：
```
<repo>/build/tests/ut/cov_report/cpp_utest/ops.info_filtered
```

## 查看覆盖率

```bash
# 查看摘要
lcov --summary <repo>/build/tests/ut/cov_report/cpp_utest/ops.info_filtered

# 查看未覆盖代码
lcov --list <repo>/build/tests/ut/cov_report/cpp_utest/ops.info_filtered | grep ":0"
```