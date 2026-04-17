# 常见 PyTorch 迁移替换接口

> 来源: [昇腾文档 - 常见PyTorch迁移替换接口](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/PT_LMTMOG_0070.html)
> 适用版本: Ascend Extension for PyTorch 7.3.0

用户需要替换原生 PyTorch 框架的接口，才能使用昇腾 PyTorch 框架。在进行网络迁移时，用户需要将部分接口转换成适配昇腾 AI 处理器后的接口。更多接口请参见《Ascend Extension for PyTorch 自定义 API 参考》。

## 设备接口替换表

| PyTorch 原始接口 | 适配昇腾 AI 处理器后的接口 | 说明 |
| --- | --- | --- |
| `torch.cuda.is_available()` | `torch_npu.npu.is_available()` | 判断当前环境中是否存在可用的 NPU 设备（返回 True 表示至少有一块可用的 NPU） |
| `torch.cuda.current_device()` | `torch_npu.npu.current_device()` | 获取当前正在使用的 device |
| `torch.cuda.device_count()` | `torch_npu.npu.device_count()` | 获取当前环境上的设备数量 |
| `torch.cuda.set_device()` | `torch_npu.npu.set_device()` | 设置当前正在使用的 device |
| `torch.tensor([1,2,3]).is_cuda` | `torch.tensor([1,2,3]).is_npu` | 判断某个 tensor 是否是 CUDA/NPU 设备上的格式 |
| `torch.tensor([1,2,3]).cuda()` | `torch.tensor([1,2,3]).npu()` | 将某个 tensor 转换成 CUDA/NPU 格式 |
| `torch.tensor([1,2,3]).to("cuda")` | `torch.tensor([1,2,3]).to('npu')` | 将某个 tensor 转换成 CUDA/NPU 格式 |
| `torch.cuda.synchronize()` | `torch_npu.npu.synchronize()` | 同步等待事件完成 |
| `torch.cuda.device` | `torch_npu.npu.device` | 生成一个 device 类，可以执行 device 相关操作 |
| `torch.cuda.Stream(device)` | `torch_npu.npu.Stream(device)` | 生成一个 stream 对象 |
| `torch.cuda.stream(Stream)` | `torch_npu.npu.stream(Stream)` | 多用于作用域限定 |
| `torch.cuda.current_stream()` | `torch_npu.npu.current_stream()` | 获取当前 stream |
| `torch.cuda.default_stream()` | `torch_npu.npu.default_stream()` | 获取默认 stream |
| `device = torch.device("cuda:0")` | `device = torch.device("npu:0")` | 指定一个设备 |
| `torch.autograd.profiler.profile(use_cuda=True)` | `torch.autograd.profiler.profile(use_npu=True)` | 指定执行 profiler 过程中使用 CUDA/NPU |
| `torch.cuda.Event()` | `torch_npu.npu.Event()` | 返回某个设备上的事件 |

## Tensor 创建接口替换

用户在构建网络或进行网络迁移时，需要创建指定数据类型的 tensor。在昇腾 AI 处理器上创建的部分 tensor 如下：

| GPU tensor | 适配昇腾 AI 处理器后的接口 |
| --- | --- |
| `torch.tensor([1,2,3], dtype=torch.long, device='cuda')` | `torch.tensor([1,2,3], dtype=torch.long, device='npu')` |
| `torch.tensor([1,2,3], dtype=torch.int, device='cuda')` | `torch.tensor([1,2,3], dtype=torch.int, device='npu')` |
| `torch.tensor([1,2,3], dtype=torch.half, device='cuda')` | `torch.tensor([1,2,3], dtype=torch.half, device='npu')` |
| `torch.tensor([1,2,3], dtype=torch.float, device='cuda')` | `torch.tensor([1,2,3], dtype=torch.float, device='npu')` |
| `torch.tensor([1,2,3], dtype=torch.bool, device='cuda')` | `torch.tensor([1,2,3], dtype=torch.bool, device='npu')` |
| `torch.cuda.BoolTensor([1,2,3])` | `torch.npu.BoolTensor([1,2,3])` |
| `torch.cuda.FloatTensor([1,2,3])` | `torch.npu.FloatTensor([1,2,3])` |
| `torch.cuda.IntTensor([1,2,3])` | `torch.npu.IntTensor([1,2,3])` |
| `torch.cuda.LongTensor([1,2,3])` | `torch.npu.LongTensor([1,2,3])` |
| `torch.cuda.HalfTensor([1,2,3])` | `torch.npu.HalfTensor([1,2,3])` |

## 替换规则总结

核心替换模式：

- **模块前缀**: `torch.cuda` → `torch_npu.npu`
- **设备字符串**: `"cuda"` / `"cuda:0"` → `"npu"` / `"npu:0"`
- **Tensor 方法**: `.cuda()` → `.npu()`，`.is_cuda` → `.is_npu`
- **Tensor 类型**: `torch.cuda.*Tensor` → `torch.npu.*Tensor`
- **Profiler 参数**: `use_cuda=True` → `use_npu=True`
