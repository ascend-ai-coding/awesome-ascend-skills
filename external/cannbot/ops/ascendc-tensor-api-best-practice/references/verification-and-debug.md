# 验证策略与调试方法论（无真机/异地开发）

本参考在实现、联调和功能精度验收节点按 implementation_route=tensor_api_custom 条件加载。

> 实证环境：x86 开发机（无 NPU、无 torch_npu）+ dav-3510 远程真机（torch_npu + 系统 CANN 9.2.0-beta.2）。

## 1. 无 torch_npu 环境的验证策略

x86 开发机装不了 torch_npu → torch extension 的 npu 模式本地不可建。用两条替代路径覆盖编译面：

| 路径 | 覆盖面 | 命令形态 |
|---|---|---|
| sim 模式全量编译 | 全部内核代码的编译正确性 | `build.sh --run-mode=sim` |
| npu 模式 DEMO_ONLY | kernel + host 侧 ASC 编译面（绕开 torch 依赖） | `cmake -D<OP>_DEMO_ONLY=ON -DCMAKE_ASC_RUN_MODE=npu` |

torch extension 的 C++ 部分是纯 Python.h 壳，版本敏感代码收敛在 kernel 侧；远程真机完成完整链路验证。**分层验证 = 本地 sim + 本地 DEMO_ONLY + 远程完整链路。**

## 2. 环境矩阵意识

| 环境 | 能跑什么 | 陷阱 |
|---|---|---|
| x86 开发机 + CANN toolkit | sim 编译、demo | 无 NPU 无 torch_npu，npu 模式不可建 |
| 远程 950 服务器 + 系统 CANN | 完整 npu 链路 + torch_npu | **torch_npu 只能在系统 CANN 环境跑**：单独 source 的 9.3.0 纯环境缺 libhccl 等，起不来 |
| 950 kernel-only 开发环境 | kernel 编译运行 | 无 opapi/torch_npu 配套，别指望端到端 |

## 3. 调试方法论

### 3.1 异常历史数据要实证证伪

曾发现"旧 so 20/20 PASS"的存档记录。排查：`nm -C <so> | grep ' U '` 找 undefined 符号（如 `ReportAscendProf`）→ 逐库 `nm` 查该符号定义是否存在 → 证明该 so 在任何 torch 环境无法 dlopen，**历史记录不可能成立**。不轻信存档，用工具复查。

### 3.2 构建产物 md5 不同 ≠ 功能不同

构建路径/时间戳会进 so。等价性验证用行为：import + callable + 全量 case 回归，不用 md5。

### 3.3 host 侧 tiling dump 开关

`<OP>_DUMP_TILING=1` 类环境变量让 host 侧打印 tiling 决策，比在 kernel 里 printf 高效（不改内核编译产物）。

### 3.4 demo BIT-EXACT 对照

demo 链路与 torch 链路输出 bit 级一致 = 构建无污染；不一致 = 先查构建（链接了错误头/库）再查算法。这是排除构建 bug 的第一手段。

### 3.5 远程执行纪律

- 远程 pip install 可能被 PEP 668 拦 → 测试脚本从工程目录直接 import（cwd 优先）
- 远程验证后清理临时目录、恢复正式测试数据集，避免污染后续轮次

## 4. 功能精度排障三因素（精度不达标先查这三项）

数值错（区别于 [precision-model.md](precision-model.md) 的硬件舍入极限，先看误差量级）时，按序排查：

### 4.1 同步缺失/错配

- **核内**：MTE/Cube/Vector/Scalar 多流水线间的数据依赖必须正确同步（Tensor API 侧事件 + `asc_sync_*`；VF 内 `mem_bar`）。漏同步 → 读到脏数据或结果未写回。
- **核间**：用多核同步时，逻辑核数 `num_blocks` 不得大于目标 AI Core 数，否则同步指令「卡死」（死等）。这是最常见的「运行不返回」根因之一。

### 4.2 偏移地址错误

多核时每核 GM 起始地址 = base + `block_idx` × 单核长度；单核长度由 Tiling 确定，偏移算错 → 数据错位（典型现象：首核对、后续核全错）。见 [tiling-strategy.md](tiling-strategy.md)。

### 4.3 浮点精度（与硬件舍入区分）

- 浮点不满足交换律/结合律，**计算顺序改变即精度差异**。并行 vs 串行、不同硬件都可能顺序不同，不期望 bit 级一致。
- 低精度 dtype（bf16/fp16）某 API 不支持时，先转高精度（float）算，再转回目标 dtype，防精度损失。
- 复合指令（乘加合一）与分步执行结果可能略有差异，属正常现象。

> 与 [precision-model.md](precision-model.md) 的 bit 级一致性纪律不冲突：**性能优化不改算法顺序 → 应 bit 级一致**；**并行/硬件差异改了顺序 → 不强求 bit 级一致**。二者前提不同。

### 4.4 禁止修改 Kernel 参数

- Kernel 入参指针本身不可改；输入参数指向内容不可改；输出参数指向内容可改。
- Tiling 数据（无论以何种形态传入 kernel）指针与内容都不可改（静态编译约束）。官方以传统 AscendC `tilingData` 为例，本路线同理适用。

## 5. 排障工具链

| 目的 | 工具 | 说明 |
|---|---|---|
| 功能调试 | printf / DumpTensor | dump 中间 tensor，定位数值错在哪一步 |
| 性能调试 | msOpProf + Task Duration | 见 [performance-analysis.md](performance-analysis.md) §2 |
| Tiling 调试 | Host 侧 gdb | Tiling 跑在 Host CPU，普通 gdb 即可 |
| Kernel 调试 | 孪生调试 / 上板调试 | Device 侧 kernel，走编程指南的 debug_and_tuning |
