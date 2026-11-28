# SHMEM 编译运行调试

> **仓内路径**：下文 `examples/`、`docs/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

本文记录失败定位方法和调试开关。正确性用例如何生成、oracle 如何构造，应看 [correctness.md](../../shmem-ops-testcase-gen/references/correctness.md)。

## 参考资料

| 资料 | 用途 |
| --- | --- |
| [dump-debug.md](dump-debug.md) | AscendC DumpTensor/printf 调测 API 使用方法 |
| [log-debug.md](log-debug.md) | SHMEM 日志（摘要）；真源 `${SHMEM_REPO}/docs/debug/log_debug.md` |
| [shmem-repo-docs-index.md](shmem-repo-docs-index.md) | 仓内 `docs/`、`include/` API、examples 索引 |

---

## 1. 首个有效错误原则

失败后不要从日志末尾盲改。先找第一个能解释后续连锁错误的位置：

- compile：第一个 compiler error，而不是最后一个 `make` 失败
- link：第一个 `undefined reference` 或 `cannot find -l...`
- launch/runtime：第一个 ACL/SHMEM 返回非 0 的调用点
- correctness：第一个 mismatch 的 PE、case、index、actual、expected

记录 command、workdir、env、返回码和首个有效错误后再修。

---

## 2. 编译失败

常见原因：

| 信号 | 可能原因 | 检查 |
| --- | --- | --- |
| `Cannot find ASCEND_HOME_PATH` | 未 source CANN 环境 | 询问用户提供 `set_env.sh` 路径 |
| `bisheng: command not found` | CANN 编译器不可见 | `which bisheng`、`ASCEND_HOME_PATH` |
| `python: command not found` / `ModuleNotFoundError` | Python 环境或依赖不可见 | 询问用户提供 `python_cmd` 或激活命令 |
| 找不到 `shmem.h` | include 路径错 | 独立工程看 `$SHMEM_HOME_PATH/shmem/include`，example 看 `${PROJECT_SOURCE_DIR}/include` |
| Device API 在 Host 编译单元报错 | Host/Device 边界混淆 | Device kernel 用 CCE 编译选项；Host 只调用 wrapper |
| `aclshmem_add_*` 未定义 | 独立工程误用 examples 宏 | 该宏只在 shmem `examples/CMakeLists.txt` 中定义 |
| catcoc 头文件缺失 | `3rdparty/catlass` 不存在 | [custom-ops-entrypoints.md](custom-ops-entrypoints.md) §0 + SHMEM build -examples 会尝试拉取；无网时需先准备依赖 |
| RDMA target 不存在 | 未开启 RDMA | 使用 `bash scripts/build.sh -examples -enable_rdma` |

调试动作：

1. 确认构建模式是否选对
2. 对照相近 example 的 `CMakeLists.txt` 补 include、compile options 和 link target
3. 只改目标 example 或独立工程 CMake；不要先改全局 SHMEM CMake

---

## 3. 链接和加载失败

常见原因：

| 信号 | 可能原因 | 处理 |
| --- | --- | --- |
| `undefined reference to aclshmem...` | 未链接 `shmem` | `target_link_libraries(<target> PRIVATE shmem)` |
| `cannot find -lshmem` | link directory 错 | 指向 `$SHMEM_HOME_PATH/shmem/lib` |
| `libshmem.so: cannot open shared object file` | 运行时库不可见 | `source `${SHMEM_REPO}/install/set_env.sh` 或设置 `LD_LIBRARY_PATH` |
| bootstrap 插件加载失败 | 插件没和 `libshmem.so` 同目录 | 同步部署 `aclshmem_bootstrap_*.so` |

排查命令：

```bash
ldd <bin_or_so> | grep -E "shmem|not found"
```

---

## 4. 启动和初始化失败

`aclshmemx_init_attr` 返回非 0 时，**可参考**：

1. `export SHMEM_LOG_LEVEL=DEBUG SHMEM_LOG_TO_STDOUT=1` 或读 `${HOME}/shmem/log/aclshmem_*.log`
2. 定位 `SHMEM_REPO` 后 Read `${SHMEM_REPO}/docs/debug/log_debug.md` 对照 bootstrap/init 日志阶段
3. [shmem-repo-docs-index.md §5](shmem-repo-docs-index.md) 常见根因表

重点检查：

- 所有 PE 的 `n_pes`、rank、`ip_port` 一致
- 同一轮多进程使用同一个 `SHMEM_UID_SESSION_ID`
- 端口未被旧进程占用（含长期运行的 `torch_test_*.py`）
- 每个 PE 绑定的 device id 合理
- `local_mem_size` 在所有 PE 一致，且足够容纳 symmetric heap
- RDMA/SDMA engine 运行前已按要求开启构建选项

### 4.1 输出全 0 但进程仍 SUCCESS（高频假 FAIL）

**现象**：

- `result_`perf-workflow.md` §2 sweep / checker 报 FAILED
- 输出 `.bin` 全 0（`nz=0`），输入/golden 非零
- Host 仍打印 `[SUCCESS] ... completed on PE X`

**首要排查**（查 shmem 日志，默认 `${HOME}/shmem/log/aclshmem_*.log`）：

| 日志关键字 | 含义 | 处理 |
| --- | --- | --- |
| `address in use for bind listen on 127.0.0.1:xxxx` | tcp store 端口被占用 | 换 `IPPORT` 或结束占用进程 |
| `Memory Heap Not Initialized` | shmem init 失败但 Host 未检查返回值 | 修环境，**不要**先改 kernel |
| `AccStoreServer startup failed` | bootstrap 未就绪 | 检查 `IPPORT` + `SHMEM_UID_SESSION_ID` 是否冲突 |

**修复步骤**：

1. `pkill -f torch_test_.*\.py`（若有长期 Torch 测试）或换独立端口
2. `scripts/run.sh` 使用 `setup_shmem_dynamic_endpoints`（见 env-setup.snippet.md）
3. 重跑 `scripts/run.sh`；确认日志无上述 error 后再看 golden

**反模式**：看到 FAILED 就删 `SHMEMI_PROF` 或改 kernel put 逻辑 — 在 shmem 未初始化时无效。

常见恢复：

- 换一个 `ip_port` / `SHMEM_UID_SESSION_ID` 或清理旧进程后重跑
- 小 PE、小 shape、默认 engine 先跑通，再打开 RDMA/SDMA 或大 shape

---

## 5. 日志调试

详见 [log-debug.md](log-debug.md)。

### 环境变量

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `SHMEM_LOG_LEVEL` | 日志等级：DEBUG / INFO / WARN / ERROR / FATAL | ERROR |
| `SHMEM_LOG_TO_STDOUT` | 是否输出到控制台（0=关闭，1=开启） | 0（落盘到文件） |
| `SHMEM_LOG_PATH` | 日志保存路径 | `${HOME}/shmem/log` |

调试时建议：

```bash
export SHMEM_LOG_LEVEL=DEBUG
export SHMEM_LOG_TO_STDOUT=1
```

### 日志格式

每条日志包含：时间、日志级别、进程号、日志模块、日志文件、行号、日志信息。

### 关键日志阶段

| 阶段 | 日志标志 | 说明 |
| --- | --- | --- |
| bootstrap 初始化 | `bootstrap flag`、`SHMEM_UID_SESSION_ID` | 检查 flag 类型和 UID 设置 |
| 网络检测 | `remote address`、`netifaddr` | root 0 是否是回环地址（单机 vs 集群） |
| bootstrap 完成 | `bootstrap success` + PE 号 | 确认 PE 正确注册 |
| 初始化完成 | `init success` + PE 号 | SHMEM 就绪 |
| 去初始化 | `finalize success` + PE 号 | 正常退出 |

SHMEM 日志主要提供 Host 侧定位能力。Device 侧报错需配合 AscendC dump 工具定位（见 [dump-debug.md](dump-debug.md)）。

---

## 6. Device 调试（DumpTensor / printf）

详见 [dump-debug.md](dump-debug.md)。

### 启用方法

1. 在 kernel 函数入口增加 `#if defined(ENABLE_ASCENDC_DUMP)` 分支，调用 `AscendC::InitDump(false, dump, ALL_DUMPSIZE)`
2. 在需要调试的位置插入调测 API：

```cpp
// 打印变量
AscendC::printf("my_pe=%d block=%d offset=%d\n", my_pe, block_idx, offset);

// Dump Tensor 内容
AscendC::GlobalTensor<half> gmT;
gmT.SetGlobalBuffer((__gm__ half*)ptr, size);
AscendC::DumpTensor(gmT, size, 16);  // 每行 16 个元素
```

`ALL_DUMPSIZE` 和 `aclCheck` 等宏定义在 `examples/utils/debug.h`（默认 75MB，可自定义）。

### 编译和运行

```bash
bash scripts/build.sh -examples -enable_ascendc_dump
cd examples/<op_name>
bash scripts/run.sh 0,1
```

### 注意事项

- dump/printf 必须用 `#if defined(ENABLE_ASCENDC_DUMP)` 宏保护
- 性能采集前必须关闭 dump（不加 `-enable_ascendc_dump` 重新编译）
- 完整的代码插入示例见 `${SHMEM_REPO}/docs/debug/dump_debug.md`（先定位 `SHMEM_REPO`，见 [shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md)）

---

## 7. 通信与同步问题

优先检查这些高频问题：

| 问题 | 现象 | 检查 |
| --- | --- | --- |
| NBI 后无完成路径 | 偶发读旧值或不同 PE 结果不一致 | 是否有 event、quiet、barrier 等完成机制 |
| signal 旧值误唤醒 | repeats 后才失败 | signal 是否带 epoch/magic，复用前是否清零 |
| signal slot 冲突 | 多 core 多 peer 偶发错位 | slot 是否按 rank/core/phase 隔离 |
| symmetric allocation 不对称 | silent data corruption | 所有 PE 分配顺序和 size 是否一致 |
| offset 单位混用 | tail 或大 shape 错 | `offset_bytes` 和 `offset_elems` 是否混用 |
| barrier 完成域误用 | Host 等不到 Device 内通信完成 | Host/Device barrier、stream sync 语义是否匹配 |

---

## 8. 正确性失败定位

正确性失败只在本 skill 中做定位和分类，修复实现仍回 `shmem-ops-code-gen`。

定位顺序：

1. 确认 checker 本身输入、golden、dtype、shape、PE 数一致
2. 用 rank pattern exact case 排除通信错位
3. 检查首个 mismatch 是错 PE、错 chunk、错 tail、错 dtype cast 还是错同步
4. dump local compute 和通信 buffer
5. 若 final output 错但中间通信正确，定位 compute/finalize；若通信 buffer 已错，定位 RMA/sync/schedule

---

## 9. Profiling 辅助定位

正确性通过后才把 profiling 作为性能依据；正确性失败时 profiling 只用于辅助观察等待或未完成现象。

```bash
export SHMEM_CYCLE_PROF_PE=0
```

在 kernel 中：

```cpp
SHMEMI_PROF_START(frame_id);
// transport / wait / compute
SHMEMI_PROF_END(frame_id);
```

Host 侧调用 `aclshmemx_get_prof(nullptr, true)` 输出 cycles、count、avg time。

---

## 10. 停止条件

满足任一条件时停止本轮调试并上报：

- 失败来自 design contract 不可执行
- 需要修改 SHMEM 核心库，但 gap analysis 未授权
- 环境阻塞：CANN/Python 缺失已询问用户但无法补齐，或设备/权限/网络不可用
- 需要用户确认 PE 拓扑、端口、硬件或目标 SOC
