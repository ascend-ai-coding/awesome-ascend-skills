# AscendC 算子调测 API（DumpTensor / printf）

> **仓内路径**：下文 `examples/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

AscendC 算子调测 API 是 AscendC 提供的调试能力，可进行 kernel 内部的打印、Tensor 内容的查看（Dump）。

关于 kernel 调测 API 的详细介绍，可参考 [DumpTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0192.html) 和 [printf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0193.html)。

## 插入调试代码

1. 修改使用该功能的核函数入口和相关调用代码，增加开启调测功能（`#if defined(ENABLE_ASCENDC_DUMP)`）的编译时代码，具体可参考 `examples/allgather/main.cpp`。
2. 在想进行调试的层级，增加调测 API 调用，如下所示：

```diff
// examples/allgather/main.cpp
#if defined(ENABLE_ASCENDC_DUMP)
__global__ __aicore__
void ShmemAllGatherKernel(
   uint64_t fftsAddr,
   GM_ADDR gmInput, GM_ADDR gmOutput, GM_ADDR gmSymmetric, GM_ADDR dump)
{
   AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
__global__ __aicore__
void ShmemAllGatherKernel(
   uint64_t fftsAddr,
   GM_ADDR gmInput, GM_ADDR gmOutput, GM_ADDR gmSymmetric)
{
#endif
   // Set FFTS address
   AscendC::SetSyncBaseAddr(fftsAddr);

+  AscendC::printf("fftsAddr is %d\n", fftsAddr);
+  AscendC::GlobalTensor<ElementB> gmT;
+  gmT.SetGlobalBuffer((__gm__ Element*)gmInput, count);
+  AscendC::DumpTensor(gmT, n, 16);
   ...
}
```

注意：`ALL_DUMPSIZE` 及 `aclCheck` 等宏和接口定义位于文件 `examples/utils/debug.h` 中，其中 `ALL_DUMPSIZE` 默认为 75MB，用户可根据需要进行自定义修改。

## 编译运行

1. 打开工具的编译开关 `-enable_ascendc_dump`，使能 AscendC 算子调测 API 编译算子样例：

```bash
bash scripts/build.sh -examples -enable_ascendc_dump
```

2. 在 examples 目录执行 demo：

```bash
cd examples/<op_name>
bash scripts/run.sh 0,1
```

## 注意事项

- dump/printf 必须用 `#if defined(ENABLE_ASCENDC_DUMP)` 宏保护
- 性能采集前必须关闭 dump（不加 `-enable_ascendc_dump` 重新编译）
- `ALL_DUMPSIZE` 默认 75MB，大 tensor dump 时可能需要调大
