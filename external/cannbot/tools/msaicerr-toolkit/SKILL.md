---
name: external-cannbot-tools-msaicerr-toolkit
description: msaicerr 工具使用技能。用于：(1) 分析 AI Core Error 问题的故障信息，输出 info.txt 辅助定位，(2)
  将 Dump 文件解析为 .bin/.npy 文件（算子输入、输出、workspace），(3) 转换 *.bin 格式 Dump 文件的数据类型为 .npy，(4)
  运行内置算子样例检查软硬件环境，(5) 排查工具本身的执行约束与不支持算子清单。触发关键词：msaicerr、msaicerr.py、aicore error、aivec
  error、there is an aicore error、AI Core Error 分析、解析Dump文件、Dump文件转换、bin转npy、dest_dtype、bfloat16ext、exception_info、data-dump、report_path、内置算子样例、检查昇腾环境、debug_info.txt、info.txt。
original-name: msaicerr-toolkit
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# msaicerr 工具

分析 AI Core Error 故障信息、解析 Dump 文件、检查软硬件环境。工具随 CANN Toolkit 包部署，脚本路径为 `{install_path}/tools/msaicerr/msaicerr.py`。

## 适用范围与前置条件

| 项 | 要求 |
|----|------|
| 分析方式 | **仅支持本地分析**，部署工具的环境必须与日志产生环境为同一运行环境 |
| Python | 依赖 **python3.7.5 或以上版本**，需提前安装 |
| 产品形态 | **不支持** Ascend RC 形态 |
| 软件包 | 已安装 CANN Toolkit，并以 CANN 运行用户执行 `source {INSTALL_DIR}/set_env.sh` |
| 执行目录 | 使用前需 `cd` 到 `{install_path}/tools/msaicerr`；且**不能在 `-p` 指定目录或其子目录下执行** |

部分算子的 AI Core Error 问题暂不支持分析（MatmulAllReduce 类、AllGatherMatmul、MemSet 等），完整清单与约束见 [functions-and-restrictions.md](references/functions-and-restrictions.md)；环境准备步骤见 [environment-preparation.md](references/environment-preparation.md)。

环境自检：`bash scripts/preflight.sh`

## 场景路由

| 场景 | 命令 | 详细参考 |
|------|------|---------|
| 日志/屏显报 `there is an xx aicore error`、`aivec error` | `python3 msaicerr.py -p <故障信息目录> -out <结果目录>` | [aicore-error-analysis.md](references/aicore-error-analysis.md) |
| 把 Dump 文件解析成 .bin/.npy 看算子输入输出 | `python3 msaicerr.py -d <dump文件>` | [dump-parsing.md](references/dump-parsing.md) |
| .bin 文件按指定数据类型转成 .npy | `python3 msaicerr.py -d <bin文件> -dtype <dtype>` | [dump-dtype-conversion.md](references/dump-dtype-conversion.md) |
| 怀疑软硬件环境异常，先验证环境 | `python3 msaicerr.py -e -dev 0` | [environment-check.md](references/environment-check.md) |
| 确认工具能不能分析当前算子 | — | [functions-and-restrictions.md](references/functions-and-restrictions.md) |

## 核心命令速查

```bash
# 进入工具目录（{install_path} 替换为 CANN 实际安装路径）
cd /usr/local/Ascend/cann/tools/msaicerr

# 分析 AI Core Error（-p 必选，-out/-dev 可选）
python3 msaicerr.py -p $HOME/aic_err_info -out $HOME/result
python3 msaicerr.py -p $HOME/aic_err_info -out $HOME/result -dev 0

# 解析 Dump 文件为 .bin/.npy
python3 msaicerr.py -d /demo/extra-info/data-dump/0/exception_info.2.1.20250611171538370
python3 msaicerr.py -d <dump文件> -out $HOME/parsed

# 转换 *.bin 数据类型为 .npy（-dtype 必选）
python3 msaicerr.py -d <文件>.input.0.bin -dtype int8

# 检查环境（-e 必选，-dev 可选，默认 0）
python3 msaicerr.py -e
python3 msaicerr.py -e -dev 1
```

## 参数总览

| 参数 | 全称 | 适用场景 | 必选性 |
|------|------|---------|--------|
| `-p` | `--report_path` | AI Core Error 分析 | 该场景必选 |
| `-d` | `--data` | Dump 解析 / 数据类型转换 | 该场景必选 |
| `-dtype` | `--dest_dtype` | 数据类型转换 | 该场景必选 |
| `-e` | `--env` | 环境检查 | 该场景必选 |
| `-out` | `--output_path` | 全部场景 | 可选 |
| `-dev` | `--device_id` | AI Core Error 分析 / 环境检查 | 可选，默认 0 |

## 使用要点

**分析前先检查故障信息是否齐全**。在收集到的故障信息目录中确认：`dfx/data-dump` 下有 dump 文件、有异常算子编译信息（`*.o` 和 `*.json`），`dfx/log/host/cann` 下有日志文件。缺失则无法用 msaicerr 提取 AI Core Error 信息，应先补齐采集。

**路径关系是最常见的卡住原因**。工具的执行目录不能是 `-p` 指定目录或其子目录，`-out` 指定目录也不能是 `-p` 目录或其子目录，否则会出现解析卡住或失败。习惯做法：在 `{install_path}/tools/msaicerr` 下执行，`-p` 与 `-out` 各指向独立路径。

**多个 AI Core Error 只解析第一个**。故障信息中存在多个 AI Core Error 时，工具按日志时间解析**第一次出现**的问题。定位后续问题需要处理完首个问题后重新采集。

**结果读取路径**：分析结果看终端提示的 `info.txt`；工具自身执行日志在执行 msaicerr.py 的同级目录下的 `debug_info.txt` 或 `info_{时间戳}/debug_info.txt`，解析异常时先看这个文件。

**数据类型不识别时装第三方库**。`debug_info.txt` 提示 `Can not read with dtype xxx` 表示存在工具不能识别的数据类型，需自行安装对应第三方库，例如提示 `Can not read with dtype bfloat16` 则安装 `bfloat16ext` 库。

**AI Core Error 分析会顺带跑一次环境检查**。`-p` 分析过程中工具会运行一个内置算子样例检查软硬件环境是否正常，`-dev` 即指定该样例运行的 Device ID。

## 信息来源

本技能内容来自 CANN 社区仓 [cann/oam-tools](https://gitcode.com/cann/oam-tools) 的 `docs/zh/msaicerr` 官方文档。命令输出示例因版本而异，请以环境实际输出为准。
