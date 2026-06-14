> **原始文档路径**: asc-devkit/docs/api/context/c_api/sync/asc_sync_block_wait.md

# asc_sync_block_wait

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

和[asc_sync_block_arrive](asc_sync_block_arrive.md)配合使用（通过flagID关联），用于等待所有同步对象到达flagID对应的同步点。如果flagID所对应的计数器数值为0则阻塞不执行；如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。

## 函数原型

    ```cpp
    __aicore__ inline void asc_sync_block_wait(pipe_t pipe, uint8_t flag_id)
    __aicore__ inline void asc_sync_block_wait(pipe_t pipe, int64_t flag_id)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :---  | :--- | :--- |
| pipe | 输入 | 需要执行同步指令的流水类型。|
| flag_id | 输入 | 事件标号，取值范围[0,15]。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- flag_id的取值范围为0至15，每个flag_id的计数器范围为0至15。
- 必须保证每个flag_id的四位计数器不溢出，否则将引发异常。
- 必须保证相同的flag_id在同一时间仅被一条流水线使用。

## 调用示例

```cpp
int64_t flag_id = 1;
asc_sync_block_arrive(PIPE_S, flag_id);
asc_sync_block_wait(PIPE_S, flag_id);  
```
