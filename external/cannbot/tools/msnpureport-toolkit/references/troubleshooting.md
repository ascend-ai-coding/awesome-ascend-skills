# 常见问题排查

## 命令找不到

`msnpureport: command not found`：驱动软件包部署后工具位于 `{Driver安装目录}/driver/tools/msnpureport`，正常情况下可在任意目录执行。若找不到，配置 PATH：

```sh
export PATH=/usr/local/Ascend/driver/tools:$PATH
```

容器场景下必须配置该环境变量，且命令需带 `--docker` 参数。

## 权限相关

| 现象 | 原因与处理 |
|------|-----------|
| 导出或设置类命令执行失败 | 导出类与所有 `config --set` 类命令仅支持 root 用户执行 |
| 普通用户无 Device 侧权限 | 驱动需以 `--install-for-all` 参数安装，非 root 用户才具备 Device 侧权限 |
| 在某目录下无法执行 | 该目录可能被加锁。`lsattr` 查看目录属性，有 `i` 选项即加锁目录；可用 `chattr -i <目录>` 撤销，执行完建议 `chattr +i <目录>` 恢复。为安全起见不建议在加锁目录中执行 |

## 日志导出不全

**典型原因**：

- 任务执行时间过长，Device 侧日志已被老化清理。
- Device 出现异常，导致无法导出 Device 侧日志。

**处理**：在任务启动**之前**改用连续导出布防，见 [export-permanent.md](export-permanent.md)。这样能拿到 Device 异常前的完整日志。

## 黑匣子导出卡住或很慢

执行 `-a`、`-f`、`-t 2` 导出黑匣子相关信息时可能出现卡住或导出慢。建议执行芯片复位后重新导出：

```sh
npu-smi set -t reset -i <id> -c <chip_id>
```

详见《npu-smi 命令参考》。

## 并发导致的失败

| 场景 | 约束 |
|------|------|
| 同路径同时刻并发导出 | 会造成落盘时间戳目录名冲突，不要这样做 |
| 单次导出多进程 | 建议不超过 4 个，否则可能因资源不足执行失败 |
| `-t 2`、`-t 4` | 不支持多进程并发执行 |
| 连续导出 | 一个 Device 不支持并发；SMP 模式下一个 OS 内不支持多个导出进程并发 |
| `msnpureport_auto_export.sh` | 不建议并发执行，会重复采集浪费资源 |

## 容器场景限制

| 功能 | 普通容器 | 特权容器 |
|------|---------|---------|
| 单次导出 | 仅支持导出 Host 侧驱动内核日志 | 无限制 |
| 连续导出（`--permanent`） | 不支持 | 无限制 |
| `msnpureport_auto_export.sh` | 不支持（禁止映射进容器） | 不支持 |

容器内无法查看 syslog，建议设置 `msnpureport report --print 1` 打屏展示工具日志。

## 屏蔽核后算子执行失败

**原因**：涉及核间同步的算子必须一次拿到所需核数才执行，屏蔽后实际核数小于算子所需核数会导致执行失败。

**排查**：

1. 查询屏蔽后实际核数：`msnpureport config --get`，看 `Aic Coremask`/`Aiv Coremask`（bitmap 结构）。
2. 查询算子所需核数：Host 侧调试日志 `$HOME/ascend/log/debug/plog/plog-{pid}_*.log`，搜索 `core_num`。
3. 恢复：`--aic_switch 1 --coreid 0xFFFF`（或 `--aiv_switch 1 --coreid 0xFFFF`）。

## 应用进程运行异常

`config --set` 类命令在昇腾 AI 应用进程运行期间执行可能导致应用运行异常。**建议应用进程退出后再执行**。

## 配置未恢复导致的后续问题

定位类配置具有持久性，务必按下表恢复：

| 配置 | 影响 | 恢复 |
|------|------|------|
| `--accelerator_recover 0` | 影响执行性能，AI Core 业务异常后不再自动复位 | **必须重启运行环境** |
| `--singlecommit 1` | AI Core 内部多指令串行，性能下降 | 设为 `0`，或重启环境 |
| `--aic_switch 0` / `--aiv_switch 0` | 可用核数减少，算子可能失败 | 对应 switch 设为 `1` 且 `--coreid 0xFFFF`，或重启环境 |
| Device 日志级别调至 `debug` | 日志量激增，可能加速日志老化 | 改回原级别（先用 `config --get` 记录原值） |

## Device ID 指定错误

命令行 `-d/--device` 是**逻辑 ID**，不是 `npu-smi info` 显示的物理 ID。换算方法见 [config-set-log.md](config-set-log.md) 中「物理 ID 与逻辑 ID 换算」。

查询类命令不指定 `-d` 时默认只查 **Device 0**；导出类命令不指定 `-d` 时导出**所有 Device**。

## 导出文件无法解读

`stackcore/` 下的调用栈文件、`ub_info/` 下的 UB 维测信息需使用 **asys** 工具解析。不同产品型号文件名可能不同（`stackcore.slogd.xx` 或 `coretrace.slogd.xx`）。详见《故障处理》：<https://hiascend.com/document/redirect/CannCommunityasys>
