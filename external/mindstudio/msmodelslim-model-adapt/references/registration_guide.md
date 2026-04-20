# 适配器注册指南

在 `config/config.ini` 中注册模型与入口。

## 示例

```ini
[ModelAdapter]
my_model = MyModel-7B, MyModel-13B

[ModelAdapterEntryPoints]
my_model = msmodelslim.model.my_model.model_adapter:MyModelAdapter
```

注册完成后，务必执行 `bash install.sh` 安装更新。
