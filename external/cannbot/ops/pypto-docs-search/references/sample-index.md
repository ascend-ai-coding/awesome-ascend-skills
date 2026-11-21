# PyPTO 算子参考实现 / golden 索引

算子参考实现与 golden 只在缓存里，无文档站形态。按模型 / 算子名定位，直接 `Read` 取全文。

## 算子参考实现（照着写的权威范本）

常见路径：`$PYPTO_DEVKIT_DIR/ops/pypto_tensor/<模型>/<算子>/<算子>_impl.py`，层级和文件名以实际目录为准。`<模型>` / `<算子>` 用命令取，不写死（随版本增删）：

```bash
ls "$PYPTO_DEVKIT_DIR/ops/pypto_tensor"                 # 列全部模型
find -L "$PYPTO_DEVKIT_DIR/ops/pypto_tensor" -ipath "*<算子名>*"    # 按算子名跨模型定位（如 rms_norm / attention / moe）
grep -RIl "<符号>" "$PYPTO_DEVKIT_DIR/ops/pypto_tensor"            # 按 API/符号找哪些实现用到
```

例：`$PYPTO_DEVKIT_DIR/ops/pypto_tensor/<模型>/rms_norm/rms_norm_impl.py`；attention / matmul 类算子用上方命令按名称定位（部分含 `BWD/FWD`、`quant` 等变体子目录）。

## golden / 测试

路径：`$PYPTO_DEVKIT_DIR/tests/<模型>/`（算子子目录可有可无），含 `*_golden*.py`（参考实现）与 `test_*.py`。写 golden 时先在此找同类算子的现成 golden 对照：

```bash
grep -RIl --exclude-dir=pypto_pro "<算子名>" "$PYPTO_DEVKIT_DIR/tests"
```
