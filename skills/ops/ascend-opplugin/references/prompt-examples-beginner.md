# Prompt 示例（新手版 5 条精选）

如果使用者是第一次接触这个 skill，可以优先给出下面这 5 条示例。它们覆盖了最常见的几类需求。

- `帮我把这个 Ascend C 自定义算子接到 PyTorch 里，优先在算子实现工程内轻量接入。`
- `这个算子要接入 torch_npu，你先帮我判断该走 Pattern A、B 还是 C。`
- `这个算子在 xpu_kernel 里已经实现了，帮我走 OpCommand 方式接入，不要重复写 kernel。`
- `环境在容器里，但代码在宿主机，帮我按容器内构建、容器外驱动的方式完成接入。`
- `帮我把这个算子接入后，补一个最小冒烟测试，验证 load_library 和 torch.ops.npu.xxx 能正常调用。`

如果用户的问题更偏某个具体方向，再继续从完整示例文档中补充：

- [prompt-examples.md](prompt-examples.md)

