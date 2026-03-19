# Docker Notes

当前版本只用于发现现有容器环境，并在用户明确确认容器名后执行 benchmark。

## 选择规则

1. 优先使用用户提供的容器名。
2. 如果用户没有提供容器名，也没有给出 torch / CANN 版本要求，则寻找同时具备 torch 和 CANN 的容器，返回容器名、torch 版本、torch_npu 版本、CANN 版本给用户确认是否使用。
3. 如果用户给出了 torch 或 CANN 版本要求，则按版本要求筛选容器，再把命中的容器信息返回给用户确认。
4. 不要默认复用未知容器，也不要在未确认前直接执行 benchmark。

## CANN 检查路径

判断容器中的 CANN 版本时，需要同时检查两类路径：

- `CANN < 8.5` 常见路径：`/usr/local/Ascend/ascend-toolkit/latest`
- `CANN >= 8.5` 常见路径：`/usr/local/Ascend/cann/latest`

版本判断细则按 [cann](cann.md) 执行。

## 容器检查重点

- `pip list | grep -E "^(torch|torch-npu|torch_npu)"`
- `ls -ld /usr/local/Ascend/ascend-toolkit/latest`
- `ls -ld /usr/local/Ascend/cann/latest`
- 容器内对应 CANN 根目录下的版本目录列表

## 返回要求

返回候选容器时至少包含：

- 容器名
- torch 版本
- torch_npu 版本
- 当前引用的 CANN 版本或候选版本目录
- 是否满足用户提出的版本要求
