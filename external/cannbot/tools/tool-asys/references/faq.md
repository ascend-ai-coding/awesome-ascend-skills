# FAQ

---

## 业务复跑报错

**问题现象**

先使用 `ctrl+z` 中止业务复跑 task，接着再次拉起业务复跑 task，屏显日志显示业务复跑 task 错误：

```
[INFO]: launch task start, running:
[WARNING]: task occurred error, output:
Segmentation fault (core dumped)
Process ForkServerPoolWorker-7:
Traceback (most recent call last):
  File "/usr/local/python3.7.5/lib/python3.7/multiprocessing/pool.py", line 127, in worker
    put((job, i, result))
......
BrokenPipeError: [Errno 32] Broken pipe
```

**可能原因**

执行 `ctrl+z` 操作等导致任务异常终止，但还存在任务进程残留（且还进行重定向写文件等操作），与后面新启动的 asys 复跑任务相互冲突，导致复跑异常。

**处理方法**

在 asys 复跑前查询是否存在运行中的推理/训练进程 ID，手工 kill 相关进程，然后再重新 asys 复跑。

---

## 实时堆栈导出超时报错

**问题现象**

使用 asys 导出实时堆栈时，部分场景下导出超时报错：

```
[ASYS] [ERROR]: Generating the stackcore bin file timeout. For details, see the related description in the document.
```

**可能原因与解决方法**

四种原因都通过 plog 日志的关键字区分。plog 默认路径为 `$HOME/ascend/log/run|debug/plog/plog-{pid}_*.log`。

| plog 关键字 | 原因 | 解决方法 |
|------------|------|---------|
| `attr init success` | 实时堆栈导出功能还未初始化完成 | 等待初始化完成，根据该关键字判断已完成后再尝试导出 |
| `close the signal capture function` | `ASCEND_COREDUMP_SIGNAL` 被设置为 `none`，关闭了部分信号集，导致实时堆栈导出功能不可用 | 设置 `ASCEND_COREDUMP_SIGNAL` 环境变量，打开信号集 |
| `unregister all signal handlers, can not capture signal` | 用户业务执行完成，实时堆栈导出功能相关的资源已释放 | 需再次执行用户业务，才可以导出实时堆栈信息 |
| 搜 `ERROR` 关键字 | 实时堆栈导出功能异常 | 查看具体报错信息，联系技术支持 |

排查顺序建议按上表自上而下：先确认初始化是否完成，再确认信号集是否被关闭，然后确认业务是否已结束，最后才当作功能异常上报。
