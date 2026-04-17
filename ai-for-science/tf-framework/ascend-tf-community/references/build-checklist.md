# Ascend TF Community Build Checklist

## 编译前

- 确认机器架构，aarch64 通常需要源码编译。
- 确认 GCC、Bazel、Python 版本满足文档要求。
- 确认 CANN 与 tfplugin 版本组合可用。

## 编译中

- 明确设置 ABI。
- 按需处理 `nsync` 等补丁。
- 如镜像下载慢，提前准备可替换的源码源。

## 编译后

- 先单独验证 `import tensorflow`。
- 再安装并验证 `npu_device`。
- 最后再把业务模型接入验证。

## 常见故障点

- ABI 不一致。
- Bazel 版本不匹配。
- 安装完 TF 之后，缺少运行时 Python 包。
