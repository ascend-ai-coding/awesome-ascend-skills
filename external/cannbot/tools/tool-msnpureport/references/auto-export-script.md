# 连续导出日志和文件（脚本方式）

> **更推荐使用命令行方式**连续导出日志和文件，见 [export-permanent.md](export-permanent.md)。本文记录的是脚本方式，适用于需要自动容量管理与老化的场景。

## 适用场景

在模型推理/训练任务运行时，执行 msnpureport 命令导出 Device 侧日志文件或者 Host 驱动日志时，可能会出现日志导出不全的情况，比如：Device 出现异常导致无法导出 Device 侧日志、任务执行时间过长导致 Device 侧日志被清理掉。此时可以在任务执行之前，在 Host 侧运行 `msnpureport_auto_export.sh` 脚本连续导出 Device 侧的日志和文件。

## 注意事项

- 该操作**仅支持 root 用户**执行。且须确保 `<logAbsolutePath>` 指定的存储路径对普通用户无访问权限，否则普通用户也能操作导出的日志文件，存在恶意删除日志文件或泄露系统信息等安全风险。
- 脚本**不支持容器场景**，部署容器时禁止将 `msnpureport_auto_export.sh` 脚本映射到容器内。
- 脚本会持续采集所有 Device 侧日志，由于并发执行会导致采集内容重复、浪费系统资源，因此**不建议并发执行**。
- 在异常场景下，可能会出现连续导出失败的情况。
- 终止导出需使用 `Ctrl+C` 或 `kill -15 <pid>`。pid 可通过 `ps -elf | grep msnpureport_auto_export.sh` 查询。

## 操作步骤

1. 以 root 用户登录 Host 侧服务器。
2. 获取脚本。脚本在驱动 Driver 的安装目录下，路径为 `{Driver安装目录}/driver/tools/msnpureport_auto_export.sh`。
3. 在某个有执行权限的目录（如 `/home/work`）下执行如下命令运行脚本：

    ```sh
    {Driver安装目录}/driver/tools/msnpureport_auto_export.sh <timeInterval> <logAbsolutePathCapacity> <logAbsolutePath>
    ```

    命令示例：

    ```sh
    /usr/local/Ascend/driver/tools/msnpureport_auto_export.sh 2 10 /home/log/
    ```

### 加锁目录处理

在加锁的目录下（使用 `lsattr` 命令查看目录属性，有 `i` 选项的为加锁目录），用户没有权限运行该脚本。如需在该目录下运行，可通过 `chattr -i <加锁的目录>` 撤销目录的 `i` 选项，脚本运行完后建议重新加上。**为了安全起见，不建议在加锁目录中运行脚本。**

## 参数说明

| 参数 | 说明 |
|------|------|
| `<timeInterval>` | 导出日志和文件的间隔时间。取值为大于 0 的整数，单位是 s，如 2s |
| `<logAbsolutePathCapacity>` | 导出日志和文件的存储目录容量。取值为大于等于 2 的整数，单位是 G，如 10G |
| `<logAbsolutePath>` | 导出日志和文件的存储路径，**仅支持配置为绝对路径**，如 `/home/log/` |
| `<logClearFlag>` | 脚本启动时是否清空路径下的日志。取值：`1`-是，`0`-否，默认为 `1` |

## 输出目录说明

脚本运行成功后，日志和文件存储在指定的存储路径下（如 `/home/log/`），目录不存在时会自动创建。该目录下会自动创建如下子目录：

| 目录 | 说明 |
|------|------|
| `msnpureport_log_new` | 导出日志和文件的存储目录，包含 `hisi_logs`、`message`、`system_info`、`slog`、`stackcore` 等子目录 |
| `msnpureport_log_old` | 老化的日志和文件的存储目录 |

### 老化机制

- **追加去重类**：`hisi_logs` 目录下的 `history.log` 和 `message` 目录下的 `message.log` 在 Device 侧是同名文件内老化的，所以每次导出之后需要将导出内容追加到同一目录下的 `history_new.log` 和 `message_new.log` 中并去重，来获取所有日志。
- **覆盖类**：其他目录的日志文件在 Device 侧以时间戳命名，通过删除较早时间戳的文件进行老化，所以每次导出之后只需将导出内容拷贝覆盖即可获取所有日志文件。
- **容量轮转**：如果 `msnpureport_log_new` 存储的日志容量超过指定存储目录容量的**一半**（如 10/2=5G），将自动清空 `msnpureport_log_old` 下的日志，再将 `msnpureport_log_new` 下的内容全部移动到 `msnpureport_log_old`。

> 因此实际磁盘占用上限约为配置容量值，规划路径容量时需预留。
