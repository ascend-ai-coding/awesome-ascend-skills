# torch_npu.erase_stream

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- tensor通过record_stream在内存池上添加标记后，该标记已被stream使用，可以通过该接口移除该标记。

- 多流之间的内存可以复用，默认通过record_stream标记内存池，防止复用的内存被提前还给内存池而出现踩踏行为。内存池在每次内存申请时，通过query device上的event来确定算子是否已经被执行完，可以安全释放了。但是这种host和device结合的机制会出现副作用：当host下发比device执行快很多时，可能导致峰值内存被推高，原因是host在query时device还没执行完。

- 当前接口提供了一种erase_stream的能力，通过主动地在event wait之后擦除并free内存实现内存池提前归还。由于后续算子一定是在event wait后才执行，因此这块被提前释放回内存池的内存不会被后续的算子踩踏。

## 函数原型

```python
torch_npu.erase_stream(tensor, stream) -> None
```

## 参数说明

- **tensor** (`Tensor`)：必选参数，需要进行标记移除的Tensor。
- **stream** (`torch_npu.npu.Stream`)：必选参数，被移除标记所属的stream。

## 返回值说明

`None`

无返回值

## 约束说明

该接口需要结合event wait来使用，保证算子执行完成后才对标记进行移除，避免出现内存踩踏行为。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> stream1 = torch_npu.npu.Stream()
>>> stream2 = torch_npu.npu.Stream()
>>> event = torch_npu.npu.Event()
>>> with torch_npu.npu.stream(stream2):
...     matrix1 = torch.ones(1000, 1000, device='npu')
...     matrix2 = torch.ones(1000, 1000, device='npu')
...     tensor1 = torch.matmul(matrix1, matrix2)
...     data_ptr1 = tensor1.data_ptr()
...     print(data_ptr1)
...     tensor1.record_stream(stream1)
...     event.record(stream2)  # 记录事件
...     event.wait(stream1)    # 等待事件完成，确保算子执行完毕
...     torch_npu.erase_stream(tensor1, stream1)
...     del tensor1
...     tensor2 = torch.ones(1000, 1000, device='npu')
...     print(tensor2.data_ptr())
...
20616943637504
20616943637504
```
