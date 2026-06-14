> **原始文档路径**: asc-devkit/docs/api/context/c_api/sync/asc_sync_inter_wait.md

# asc_sync_inter_wait

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

核间同步寄存器同步指令，每个flagID均分配有计数器，执行asc_sync_inter_arrive后计数器将递增；执行asc_sync_inter_wait时如果对应的计数器数值为0则阻塞；如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。

## 函数原型

    ```cpp
    __aicore__ inline void asc_sync_inter_wait(pipe_t pipe, uint8_t flag_id)
    __aicore__ inline void asc_sync_inter_wait(pipe_t pipe, int64_t flag_id)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :---  | :--- | :--- |
| pipe | 输入 | 需要执行同步指令的流水类型。|
| flagID | 输入 | 事件标号，取值范围[0,15]。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- flagID的取值范围为0至15，每个flagID的计数器范围为0至15。
- 必须保证每个flagID的计数器不溢出，否则将引发异常。
- 必须保证相同的ID在同一时间仅被一条流水线等待。

## 调用示例

```cpp
int64_t flagID = 1;
asc_sync_inter_arrive(PIPE_S, flagID);
asc_sync_inter_wait(PIPE_S, flagID);  
```
