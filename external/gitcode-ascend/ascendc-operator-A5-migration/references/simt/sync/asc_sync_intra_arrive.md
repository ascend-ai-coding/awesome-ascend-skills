> **原始文档路径**: asc-devkit/docs/api/context/c_api/sync/asc_sync_intra_arrive.md

# asc_sync_intra_arrive

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

block内同步指令，与[asc_sync_intra_wait](asc_sync_intra_wait.md)配合使用，是block内不同Core的流水线同步指令，用于向每个Core的同步寄存器发送同步信号。每个sync_id对应Vector Core与Cube Core中的一个初始值为0的计数器。asc_sync_intra_arrive指令后其他Core对应的计数器增加1；执行asc_sync_intra_wait时如果对应的计数器数值为0则阻塞不执行；如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。

## 函数原型

    ```cpp
    __aicore__ inline void asc_sync_intra_arrive(pipe_t pipe, uint8_t sync_id)
    __aicore__ inline void asc_sync_intra_arrive(pipe_t pipe, uint64_t sync_id)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :---  | :--- | :--- |
| pipe | 输入 | 该指令的执行流水线。 |
| sync_id | 输入 | 事件标号。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 需要保证每一个ID的计数器不会溢出，否则会出现异常。
- 需要保证相同的ID只会被一个流水线控制。
- 每个ID对应一个计数器，特定CORE的同步ID计数器可通过其他CORE发起的asc_sync_intra_arrive递增；也可通过本身发起的asc_sync_intra_wait递减。 
- 三个CORE的ID映射规则如下：
    - ID为0~15的VECC0RE0的asc_sync_intra_arrive指令会被映射到CUBECORE的0~15的ID。 
    - ID为0~15的VECC0RE1的asc_sync_intra_arrive指令会被映射到CUBECORE的16~31的ID。 
    - ID为0~15的CUBECORE的asc_sync_intra_arrive指令会被映射到VECC0RE0的0~15的ID。 
    - ID为16~31的CUBECORE的asc_sync_intra_arrive指令会被映射到VECC0RE0的0~15的ID。 

## 调用示例

```cpp
//设置同步指令事件标号为1
int64_t sync_id = 1;
asc_sync_intra_arrive(PIPE_S, sync_id);
asc_sync_intra_wait(PIPE_S, sync_id);  
```
