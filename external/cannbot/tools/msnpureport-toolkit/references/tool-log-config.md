# 配置 msnpureport 工具自身的维测日志

这两个配置作用于 **msnpureport 工具自身**产生的维测日志，不影响 Device 侧系统类日志（后者见 [config-set-log.md](config-set-log.md)）。

## 通用注意事项

- 均**仅支持 root 用户**执行。
- 容器场景下需添加 `--docker` 参数，并配置环境变量：

    ```sh
    export PATH=/usr/local/Ascend/driver/tools:$PATH
    ```

## 配置维测日志展示位置

### 命令功能

配置 msnpureport 工具维测日志展示位置。

### 命令格式

```sh
msnpureport report --print <value>
```

### 参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `--print <value>` | 必选 | `0`：打印在 Host 侧日志文件中，**默认是 0**。aarch64 架构为 `/var/log/messages`，x86_64 架构为 `/var/log/syslog`。<br>`1`：打屏展示 |

### 补充说明

- 该命令**仅控制 `msnpureport report [options]` 命令执行过程中**的日志展示位置。
- 由于 docker 中无法查看 syslog 日志，**建议 docker 中设置 `--print 1`**。

### 使用示例

```sh
msnpureport report --print 1
```

## 配置维测日志级别

### 命令功能

配置 msnpureport 工具维测日志级别。

### 命令格式

```sh
msnpureport report --log_level <value>
```

### 参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `--log_level <value>` | 必选 | `debug`、`info`（**默认**）、`warning`、`error` |

### 使用示例

```sh
msnpureport report --log_level info
```

## 查询帮助与版本

### 帮助信息

```sh
msnpureport help              # 总帮助
msnpureport config --help     # config 子命令帮助
msnpureport report --help     # report 子命令帮助
```

输出示例（因版本不同略有差异，以实际输出为准）：

```sh
Usage:  msnpureport <subcommand> [OPTIONS]
subcommand:
  config               configure settings and queries
  report               get device log
  help                 get help information
  version              get version information

get options information by msnpureport <subcommand> --help
Example:
        msnpureport config --help
```

### 版本信息

```sh
msnpureport version
```

输出示例：

```sh
msnpureport version: 1.1.0
```

> 帮助与版本查询无权限限制，无注意事项。
