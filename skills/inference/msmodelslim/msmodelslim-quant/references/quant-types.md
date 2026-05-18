# QuantType Table
# 量化类型参考表

从 `msmodelslim/core/const.py` 读取：

| QuantType | 说明 |
|----------|------|
| W4A8 | 权重INT4量化，激活值INT8量化 |
| W4A8C8 | 权重INT4量化，激活值INT8量化，KVCache INT8量化 |
| W8A16 | 权重INT8量化，激活值不量化 |
| W8A8 | 权重INT8量化，激活值INT8量化 |
| W8A8S | 权重INT8稀疏量化，激活值INT8量化 |
| W8A8C8 | 权重INT8量化，激活值INT8量化，KVCache INT8量化 |
| W16A16S | 权重浮点稀疏 |

**支持的量化类型**：W4A8, W4A8C8, W8A16, W8A8, W8A8S, W8A8C8, W16A16S
