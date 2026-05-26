# third_party/tensor_api/ —— 外部依赖（dav-3510 硬件 API）

本目录刻意留空。`tensor_api/` 不在本 skill 仓内提交，需从外部仓库拉取：

```bash
# 从工程目录运行
git clone --depth 1 https://gitcode.com/cann/ops-tensor.git /tmp/ops-tensor
cp -r /tmp/ops-tensor/include/tensor_api ./tensor_api
rm -rf /tmp/ops-tensor
```

拉取后目录应该是：
```
third_party/tensor_api/
    ├── impl/tensor_api/   {algorithm, arch, atom, tensor, utils}
    └── include/tensor_api/ {algorithm, arch, atom, tensor, utils, tensor.h}
```

完整说明：见 `references/matmul_pattern.md` §0.5.0「准备 tensor_api 依赖」。
