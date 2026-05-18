# Prompt 示例

下面这些 prompt 可用于向使用者展示“这个 skill 能帮什么、适合怎么提问”。实际回答时，不需要一次性全部给出，按用户当前场景挑 3 到 5 条最贴近的即可。

如果用户是第一次使用，优先参考更短的精选版：

- [prompt-examples-beginner.md](prompt-examples-beginner.md)

## 通用接入

- `帮我把这个 Ascend C 自定义算子接到 PyTorch 里，优先在算子实现工程内轻量接入。`
- `这个算子要接入 torch_npu，你先帮我判断该走 Pattern A、B 还是 C。`
- `这是一个已有实现的 NPU 算子，帮我补 PyTorch host wrapper，并注册成 torch.ops.npu.xxx。`
- `帮我给这个算子搭一个最小可运行的接入目录，包含 setup.py、加载逻辑和 test。`

## 基于已有图算子复用

- `这个算子在 xpu_kernel 里已经实现了，帮我走 OpCommand 方式接入，不要重复写 kernel。`
- `帮我从 op_def 里解析图算子名、输入输出名，然后补 Pattern C 的接入代码。`
- `这个算子已经有 CANN / xpu_kernel 图算子实现，帮我只做轻量接入和冒烟测试。`

## 自研 kernel 接入

- `我这里只有 AscendC kernel，没有 PyTorch 接入层，帮我按正式路线补齐接入。`
- `这个算子需要 workspace 和 tiling，帮我按 Pattern B 生成 host、CMake 和测试。`
- `这是一个无 workspace 的简单算子，帮我按 Pattern A 接到 torch.library。`

## Plan A / Plan B 选择

- `先帮我判断，这个算子接入应该放在算子实现工程里，还是放到 op-plugin 里。`
- `优先按 Plan A 做接入；如果不适合，再告诉我为什么要切到 Plan B。`
- `如果用 op-plugin 接入，帮我说明 .cpp、example、test 分别该放哪。`

## 容器场景

- `环境在容器里，但代码在宿主机，帮我按容器内构建、容器外驱动的方式完成接入。`
- `请按 docker exec 的方式帮我完成这个算子的预检查、构建、安装和测试。`
- `帮我检查一下这个容器场景下，挂载路径、Python 环境、set_env.sh 应该怎么处理。`

## 快速验证 / Pybind

- `我现在不想正式接入，只想快速验证 C++ 函数能不能从 Python 调起来，帮我走 Pybind 最小闭环。`
- `先用 Pybind 把这个函数跑通，后面我再迁移到 torch.library。`

## 构建与测试

- `帮我把这个算子接入后，补一个最小冒烟测试，验证 load_library 和 torch.ops.npu.xxx 能正常调用。`
- `帮我检查这个接入流程里构建、安装、测试是否都能闭环。`
- `如果这个算子没有 CPU reference，帮我至少把注册检查和输出 shape 测试补上。`

## 排障

- `这个算子接入时报 graph/types.h 找不到，帮我定位是 include 还是环境问题。`
- `我这边 load_library 后没有 torch.ops.npu.xxx，帮我检查注册链路哪里断了。`
- `这个扩展在源码目录里能 import，在安装后反而有问题，帮我排查是不是源码目录遮蔽了 site-packages。`
- `当前环境没有 TorchConfig.cmake，帮我改成更稳妥的构建方式。`

