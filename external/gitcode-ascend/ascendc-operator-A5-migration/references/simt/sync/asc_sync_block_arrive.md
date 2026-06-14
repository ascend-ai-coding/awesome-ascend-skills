> **原始文档路径**: asc-devkit/docs/api/context/c_api/sync/asc_sync_block_arrive.md

# asc_sync_block_arrive

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √ </cann-filter>|

## 功能说明

和[asc_sync_block_wait](asc_sync_block_wait.md)配合使用（通过flagID关联），用于向核间同步寄存器发送同步信号，标识当前pipeline已到达flagID对应的同步点。每个flagID对应一个初始值为0的计数器，执行asc_sync_block_arrive后flagID对应的计数器增加1；执行asc_sync_block_wait时如果对应的计数器数值为0则阻塞不执行；如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。

## 函数原型

    ```cpp
   __aicore__ inline void asc_sync_block_arrive(pipe_t pipe, int64_t flag_id)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :---  | :--- | :--- |
| pipe | 输入 | 设置这条指令所在的流水类型。|
| flag_id | 输入 | 事件标号，用于标识同一组同步信号。取值范围为[0,10]。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 使用该同步接口时，需要按照如下规则设置Kernel类型：
  - 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
  - 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。
- 不建议开发者同时使用该接口和Matmul高阶API，否则会有flagID冲突的风险。
- 同一flagID的计数器最多设置15次。

## 调用示例

```cpp
// Host侧调用接口aclrtGetHardwareSyncAddr获取核间同步基地址ffts_addr
uint64_t config = *(__gm__ uint64_t*)ffts_addr;
asc_set_ffts_base_addr(config);
int64_t flag_id = 1;
asc_sync_block_arrive(PIPE_S, flag_id);
asc_sync_block_wait(PIPE_S, flag_id);  
```
