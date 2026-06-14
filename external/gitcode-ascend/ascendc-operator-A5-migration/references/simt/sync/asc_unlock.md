> **原始文档路径**: asc-devkit/docs/api/context/c_api/sync/asc_unlock.md

# asc_unlock
## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

用于AI Core内部异步通道同步的指令，用于释放由asc_lock指令申请的缓存。

## 函数原型

    ```cpp
    enum ascMutexExecuteMode {
        ASC_LOCK_BLOCK = 0,
        ASC_LOCK_NON_BLOCK = 1
    };
    
    __aicore__ inline void asc_unlock(pipe_t pipe, uint8_t mutex_id)
    __aicore__ inline void asc_unlock(pipe_t pipe, uint8_t mutex_id, const ascMutexExecuteMode mode)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :---  | :--- | :--- |
| pipe | 输入 | 该指令的执行流水线。 |
| mutex_id | 输入 | buffer标号。取值范围[0, 31]。|
| mode | 输入 | 可选参数，用于指定是否阻塞流水线，默认值为ASC_LOCK_BLOCK。<br>&bull; ASC_LOCK_BLOCK：该指令等待pipe所对应的流水线中所有前置指令完成后执行。<br>&bull; ASC_LOCK_NON_BLOCK：该指令等待pipe所对应的流水线中所有前置指令完成且相同mutex_id的所有asc_unlock指令执行完成后执行。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 必须与asc_lock搭配使用，且需要使用相同的mutex_id与mode。此外，asc_unlock必须始终插在对应的asc_lock指令之后，否则会出现未定义的硬件行为。
- 具有相同mutex_id的asc_lock与asc_unlock组合，无论pipe与mode是否相同，均不得在编程顺序中嵌套，否则硬件行为将不可预测。
- 对于程序顺序中连续出现的，具有相同pipe与mutex_id的指令对，后一个asc_lock将不再阻塞流水线运行，若需实现同一流水线的依赖关系，则必须使用指令asc_sync_pipe。

## 调用示例

```cpp
//buffer标号为1
uint8_t mutex_id = 1;
//等待PIPE_S中所有前置指令完成后释放标号为1的缓存
asc_lock(PIPE_S, mutex_id);
asc_unlock(PIPE_S, mutex_id); 
```
