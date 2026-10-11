# 复杂 Cube+Vector 算子调试流程

适用于复杂 `Cube + Vector` 双核分离算子。下面的推荐点位以 Flash Attention 类算子为例。

## 核心原则

- **优先追踪 GM 上的 ND 数据**，尽量避免 L1/L0 的 NZ 格式（不便与 CPU golden 对比）。这里的 GM/global tensor、Workspace tensor 都可以直接 dump，不要误以为只能 dump LocalTensor。
- **分段验证**：先确认输入，再确认当前阶段输出，再确认下一阶段读入。不要看到 `S` 错了就只围着 `S` 猜；必须回查 `Q/K`，必要时回查更前面的搬运和同步。
- **追踪 Workspace 数据流**：关键中间数据通常落在 `Cube输出 / Vector输入 / Vector输出` 这几段，例如 S/P/O。
- **最小样本优先**：先看首核、首轮、首 slot；问题只在尾块或后续轮次时，再补 tail slot、最后一轮、指定 block。
- **同步是常见根因**：Cube/Vector 跨核、MTE2→V、V→MTE3 的数据依赖都要显式考虑。

## Flash Attention 数据流概览

```
[GM: Q/K/V] ──Cube──> [Workspace: S] ──Vector──> [Workspace: P]
                                            │
                                            └─> [Workspace: O_tmp]
                                            │
[Workspace: P] + [GM: V] ──Cube──> [Workspace: O_tmp] ──Vector──> [GM: Output]
```

## 通用调试步骤

先按下面顺序进入调试，不要把用例管理和 dump 流程拆开做。Lingxi 输入统一来自 `{task_dir}/model.py:get_input_groups()`，不要引入旧式的 `debug_case.csv` / `basic` / `general` 脚本入口。

1. 先从 `{task_dir}/evaluation_results.json` 或 `{task_dir}/.lingxi_verify_logs/latest.stdout.log` 找到最小失败 case；不要修改只读的 `{task_dir}/model.py` 或引入额外 case 文件。
2. DumpTensor 插桩先只覆盖一个失败 case 的首块路径：`bx=0`、`t=0`、`slot=0`，`dumpSize=8/16/32`。如果首样本正常，再追加 tail slot、最后一轮或指定 block。
3. 运行 Lingxi verify-runner 验证当前 task：

```bash
lingxi-ascendc verify-runner \
    --target ascendc \
    --workspace . \
    --action-id "agent-${task}-dumptensor-debug" \
    --task ${task}
```

4. DumpTensor 输出会进入 `{task_dir}/.lingxi_verify_logs/latest.stdout.log`。按 `desc` 过滤阅读：

```bash
grep -B1 "desc=300" {task_dir}/.lingxi_verify_logs/latest.stdout.log
```

5. 修复后必须用原始 `get_input_groups()` 回归一次 verify，并确认 `{task_dir}/evaluation_results.json.ok == true`。

## Flash Attention类算子推荐调试点位

### Point 1：验证输入数据

在 Cube 的 LoadQ/LoadK/LoadV 函数中，LoadNdGmToNzL1 之前：

```cpp
void LoadQ(int bz, int by, int bx) {
    uint64_t qOffset = ...;
    // Dump Q 的前几个元素（ND 格式，直接在 GM 上）
    DumpTensor(qGm_[qOffset], 100 + bx, 32);  // C输入
    // ... 原有 LoadNdGmToNzL1 逻辑
}
```

配合 CPU golden：在 PyTorch 参考实现中打印相同位置的 Q 数据。

### Point 2：验证 Cube 输出（Workspace S）

在 Cube 的 ComputeMM1 函数中，Fixpipe 写入后、CrossCoreSetFlag 前：

```cpp
void ComputeMM1(int bz, int by, int t, int slot) {
    // ... Mmad 和 Fixpipe 逻辑
    Fixpipe(wsSGm_[wsOffset], cL0, fixParams);
    
    // Dump S 矩阵（已写入 Workspace，ND 格式）
    uint32_t sOffset = slot * BLOCK_M * BLOCK_N;
    DumpTensor(wsSGm_[sOffset], 300 + slot, 32);  // C输出
    
    CrossCoreSetFlag<0x2, PIPE_FIX>(SIG_S_READY);
}
```

如果 `300 + slot` 的 `S` 异常，按这个顺序排查：

1. 同块 `Q` dump 是否正确。
2. 同块 `K` dump 是否正确。
3. `LoadNdGmToNzL1` / MMAD 参数 / scale 是否正确。
4. 生产 `S` 的写入是否在同步信号前完成。

### Point 3：验证 Vector 读入（Workspace S）

在 Vector 的 ComputeVec1 函数中，WaitFlag 后、DataCopy 读入后：

```cpp
void ComputeVec1(int slot, bool isFirst, bool isTailKV) {
    CrossCoreWaitFlag<0x2>(SIG_S_READY);
    
    // 读入 S
    DataCopy(sUb, wsSGm_[sOffset], tileSize);
    
    // Dump 读入的 S（验证是否与 Cube 输出一致）
    DumpTensor(sUb, 400 + slot, 32);  // V输入
    
    // ... 后续 Softmax 逻辑
}
```

`300 + slot` 正确但 `400 + slot` 错误时，优先怀疑：

1. 跨核同步缺失或时机错误。
2. slot/offset 错位。
3. DataCopy 读入参数错误。

### Point 4：验证 Softmax 输出（Workspace P）

在 ComputeVec1 函数中，Softmax 完成、写入 Workspace P 后：

```cpp
// Cast 和 DataCopy 写入 wsPGm 后
DataCopy(wsPGm_[pOffset], pHalf, tileSize);
outputQue1_.FreeTensor(pHalf);

// Dump P 矩阵（softmax 输出）
DumpTensor(wsPGm_[pOffset], 600 + slot, 32);  // V输出

CrossCoreSetFlag<0x2, PIPE_MTE3>(SIG_P_READY);
```

若 `P` 异常，不要直接认定是 Softmax API 问题。先确认：

1. `400 + slot` 的 `S` 是否正确。
2. tail mask 前后的 `S` 是否符合预期。
3. isFirst / 非首次迭代的状态输入是否正确。
4. UB 上计算与 MTE3 写 Workspace 前是否缺少必要同步。

### Point 5：验证 Cube MM2 输出（Workspace O_tmp）

在 Cube 的 ComputeMM2 函数中，Fixpipe 写入后：

```cpp
void ComputeMM2(int bz, int by, int t, int slot) {
    CrossCoreWaitFlag<0x2>(SIG_P_READY);
    
    // ... Mmad 和 Fixpipe 逻辑
    
    // Dump O_tmp（Cube MM2 输出）
    uint64_t wsBase = slot * BLOCK_M * dimAlign_;
    DumpTensor(wsOGm_[wsBase], 310 + slot, 32);  // C输出，第二段 C/V 链路用子号区分
    
    CrossCoreSetFlag<0x2, PIPE_FIX>(SIG_O_READY);
}
```

### Point 6：验证 Vector 最终输出

在 Vector 的 FinalizeOutputChunk 函数中，写 GM 前：

```cpp
void FinalizeOutputChunk(...) {
    // Cast 完成后
    Cast(outHalf, oUb, RoundMode::CAST_ROUND, dealRows * dim);
    
    // Dump 最终输出（写 GM 前）
    DumpTensor(outHalf, 600 + 50 + curBx_, 32);  // V输出，和前面的 P 输出错开
    
    // 写入 GM
    DataCopy(outGm_[outBase], outHalf, dealRows * actualDim);
}
```

## 编号约定

| 范围 | 位置 | 说明 |
|------|------|------|
| 100-199 | C输入 | 进入 Cube 的输入，如 Q/K/V、P/V |
| 200-299 | C中间 | Cube 内中间量，只有确实需要时才 dump |
| 300-399 | C输出 | Cube 写到 Workspace/GM 的结果，如 S、O_tmp |
| 400-499 | V输入 | Vector 从 Workspace/GM 读入后的数据 |
| 500-599 | V中间 | Softmax、mask、累加状态等 Vector 中间量 |
| 600-699 | V输出 | Vector 写到 Workspace/GM 前后的结果，如 P、最终 Output |

## 特殊场景调试

### 输出太多时的抽样策略

按下面顺序扩大范围，不要一上来全量 dump：

1. `bx=0, t=0, slot=0`，每个点先 dump 8 或 16 个元素。
2. 首核首轮正常时，改看出错 block 对应的 `bx/t/slot`。
3. 若只怀疑尾块，直接只保留最后一个 `slot` 和 tail 分支附近的 dump。
4. 若任务很多导致屏幕淹没，把整个运行输出重定向到 debug 文件，再按 `desc`、`slot`、`bx` grep。

### 尾块调试

只用一个触发尾块的 case（kvSeqLen 不对齐 BLOCK_N），重点检查：
- `tiling_.tailValid` 值是否正确
- 最后一个 slot 的 S 矩阵
- `isTailKV` 分支的 mask 处理（应将无效 KV 位置填 `-inf`）

示例：

```cpp
if (isTailKV) {
    // Dump mask 前的 S
    DumpTensor(sUb, 400 + slot, 32);
    
    // mask 处理
    LocalTensor<float> maskUb = maskBuf_.Get<float>();
    Duplicate(maskUb, SOFTMAX_NEG_INF, BLOCK_N);
    Duplicate(maskUb, 0.0f, tiling_.tailValid);
    // ... Add mask
    
    // Dump mask 后的 S
    DumpTensor(sUb, 500 + slot, 32);
}
```

### 首次 vs 非首次迭代

对比 isFirst=true 和 isFirst=false 的中间状态：

```cpp
if (isFirst) {
    // Dump 初始状态
    DumpTensor(softmaxMaxDefaultUb_, 500, 16);
    DumpTensor(softmaxSumDefaultUb_, 510, 16);
} else {
    // Dump 读取的前一轮状态
    DumpTensor(inStateUb, 520 + prevSlot, 16);
    DumpTensor(inSumUb, 530 + prevSlot, 16);
}
```

### 跨核同步问题

如果怀疑同步时机问题：

```cpp
// Cube 侧：SetFlag 前
void ComputeMM1(...) {
    Fixpipe(...);
    DumpTensor(wsSGm_[sOffset], 300 + slot, 32);  // C输出
    CrossCoreSetFlag<0x2, PIPE_FIX>(SIG_S_READY);
}

// Vector 侧：WaitFlag 后
void ComputeVec1(...) {
    CrossCoreWaitFlag<0x2>(SIG_S_READY);
    DataCopy(sUb, wsSGm_[sOffset], tileSize);
    DumpTensor(sUb, 400 + slot, 32);  // V输入
}
```

对比 desc `300+slot` 和 `400+slot` 的数据是否一致。

同时检查同一核内的流水依赖，不要只盯跨核 flag：

- `MTE2 -> V`：GM 搬到 UB 后，Vector 使用这块 UB 数据前是否完成必要同步。
- `V -> MTE3`：UB 上计算写回 GM/Workspace 前，MTE3 是否读取到了尚未完成写入的数据。
- `MTE2 -> MTE3`：若同一 UB 缓冲区存在搬入和搬出交叠使用，检查 buffer 生命周期和同步。

经验规则：生产端 dump 正常、消费端 dump 错误时，先查同步和地址；生产端已经错误时，再查公式、参数和输入。

## CPU Golden 对照

在 PyTorch 参考实现中，打印相同位置的数据：

```python
import torch

def flash_attention_reference(q, k, v, sm_scale):
    # Q/K/V shape: [batch, heads, seq_len, dim]
    batch, heads, q_len, dim = q.shape
    kv_len = k.shape[2]
    
    for bz in range(batch):
        for by in range(heads):
            for bx in range(q_len // BLOCK_M):
                # 对应 C输入 dump (desc 100+bx)
                q_block = q[bz, by, bx*BLOCK_M:(bx+1)*BLOCK_M, :]
                print(f"[CPU-100-{bx}] Q[0,:8]: {q_block[0,:8]}")
                
                for t in range(kv_len // BLOCK_N + 1):
                    slot = t
                    
                    # S = Q @ K^T
                    k_block = k[bz, by, t*BLOCK_N:(t+1)*BLOCK_N, :]
                    s = torch.matmul(q_block, k_block.transpose(-2, -1)) * sm_scale
                    
                    # 对应 C输出 dump (desc 300+slot)
                    print(f"[CPU-300-{slot}] S[0,:8]: {s[0,:8]}")
                    
                    # P = softmax(S)
                    p = torch.softmax(s, dim=-1)
                    print(f"[CPU-600-{slot}] P[0,:8]: {p[0,:8]}")
```

## 定位完成后

验证问题修复之后，移除所有 DumpTensor 调用，重新验证。
