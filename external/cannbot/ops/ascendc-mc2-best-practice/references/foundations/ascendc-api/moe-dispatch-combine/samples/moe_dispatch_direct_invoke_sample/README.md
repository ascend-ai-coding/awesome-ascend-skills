# MoE Dispatch 单算子直调工程

这个 sample 提供一个独立的 MoE Dispatch 算子，通过 `<<<>>>` 内核调用方式实现多卡 token 分发。

## 核心特性

- **Host 侧**：通过 `HcclCommInitAll` 初始化多卡 HCCL comm，用 `Mc2CcTilingConfig` 组 tiling，用 `HcclAllocComResourceByTiling` 创建通信资源 context
- **Kernel 侧**：解析 host 传入的 `mc2Context`，直接读取 tiling 中的 `epWorldSize`
- **通信方式**：走 MTE Win Area，不走 `aclnn`
- **状态同步**：使用按 `expert x srcRank` 分槽的 32B 状态区设计，并基于该二维统计收口 `epRecvCounts` 与 `expertTokenNums`

## 当前架构

当前 sample 可以按 4 层理解：

1. **Host 启动层**：`test/test_moe_dispatch.cpp` 负责建卡、建流、组 tiling、申请 `mc2Context`、准备输入输出并 launch kernel。
2. **通信上下文与兼容层**：`include/tiling_data.h` 和 `include/moe_dispatch_base_compat.h` 负责定义通信资源结构体和基础地址 helper 语义。
3. **通信搬运层**：`kernel/mte_dispatch_comm.h` 负责 window/status 布局、远端状态发布、本地状态等待。
4. **算子编排层**：`kernel/moe_dispatch.h` 负责路由、slot 计算、发窗、回搬和输出统计。

如果后续要基于 sample 改新算子，通常优先先改 host 与 comm 两层，再改 kernel 编排层。

## 简化约束

当前 sample 只保留最小可运行版本：

- `expertId` 直接等于 `dstRank`
- `moeExpertNum = epWorldSize`
- 每个 rank 只有 1 个 local expert
- `K=1`，无量化，无 shared expert
- 单 AIV 核处理完整数据路径
- 硬编码为单核处理，缺乏多核并行能力

## 为什么当前 sample 还不是真正的分核设计

当前 sample 虽然已经把 Host、Comm、Kernel 结构拆开，并且在代码里预留了 `GetBlockIdx()`、`InitBlockRange()`、`usedCores` 一类入口，但它仍然只是单核基线，不应直接视为完整的多核实现模板。

原因主要有四点：

1. 现在的分核控制主要还是围绕一个全局 token 区间展开，不能覆盖发送、状态、回搬、统计这些不同阶段的工作项差异。
2. 状态区处理天然更适合按 `expert` 或 `expert x srcRank` 分槽切分，而不是继续复用 token 维度的区间。
3. 输出回搬最终要服从 `expandX` / `expandIdx` 的输出顺序，真正做多核时通常还需要 prefix sum、偏移计算或按输出段切分，当前 sample 还没有补齐这部分设计。
4. 统计输出如 `sendCounts`、`recvCounts`、`expertTokenNums` 涉及核间合并，当前 sample 还没有实现“核内统计 + 核间归并”的完整闭环。

因此，当前 sample 更准确的定位是：

- 它是一个可以验证数据流、通信地址、状态协议和最小输出闭环的单核基线。
- 它可以作为多核改造的起点，但不能把“把 `usedCores` 改大”视为完成了真正的分核设计。

## 目录结构

```text
moe_dispatch_direct_invoke_sample/
├── CMakeLists.txt
├── README.md
├── build.sh
├── moe_dispatch.cpp              # Kernel 入口
├── run.sh
├── run_multi_data_test.sh
├── include/
│   ├── tiling_data.h             # Tiling 数据结构
│   └── utils.h
├── kernel/
│   ├── mte_dispatch_comm.h       # 通信辅助层
│   └── moe_dispatch.h            # Dispatch 算子实现
├── scripts/
│   └── verify_dispatch.py        # 输出验证脚本
└── test/
    └── test_moe_dispatch.cpp     # Host 测试程序
```

## 快速开始

### 1. 环境准备

```bash
export ASCEND_HOME_PATH=/path/to/cann
source ${ASCEND_HOME_PATH}/set_env.sh
```

### 2. 编译

```bash
bash build.sh
```

### 3. 运行

```bash
BS=4 H=16 RANK_SIZE=2 bash run.sh
```

## 状态区设计

### 布局

- 按 **先分 expert，再分 srcRank** 的方式组织
- 每个 slot 占用 32 字节（8 个 int32_t），满足硬件对齐要求
- slot 内容：`[flag, token_count, padding]`
  - flag: 0=未就绪，1=就绪
  - token_count: 该 srcRank 发给此 expert 的 token 数量
- `WaitRemoteStatus` 会先读取完整的 `expert x srcRank` 状态矩阵，再聚合出按 rank 的 `recvCounts`

### 同步流程

1. **SetRemoteStatus**：对每个 `dstRank x expert` slot 先写 `token_count`，最后写 `flag=1`
2. **WaitRemoteStatus**：轮询所有 `expert x srcRank` slot，全部就绪后生成 `expertRecvCounts` 和按 rank 聚合后的 `recvCounts`
3. **ClearLocalStatus**：等待成功后立即把本地状态区复位为 0，避免下一轮误读旧状态

## 输出文件

每个 rank 会在输出目录中生成：

- `input_rankX.bin`
- `expert_ids_rankX.bin`
- `expand_x_rankX.bin`
- `expand_idx_rankX.bin`
- `expert_token_nums_rankX.bin`
- `ep_recv_counts_rankX.bin`
- `summary_rankX.txt`

使用 `scripts/verify_dispatch.py` 验证输出结果。

## 扩展方向

当前 sample 支持以下扩展：

1. **多 local expert**：修改 `localExpertNum_`，状态区会自动按 expert 分区
2. **多 topK**：需要修改路由和 expandIdx 逻辑
3. **shared expert**：需要在状态区中增加 shared expert 分区
4. **多核并行**：需要把发送、状态、回搬、统计按阶段分别设计分核，不能只放大当前 sample 的单一 token 区间切分
