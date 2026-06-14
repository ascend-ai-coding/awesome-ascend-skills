# Gate 5 全量测试

## 输入

- Gate 4 通过结果
- `ATB_<OpName>_gen.yaml`
- `generator_<op>.py`
- `node.yaml` / `node_perf.yaml`
- 可选：`<op>.csv`

## 命令

执行 `atk task` 前确认 `node.yaml`、`node_perf.yaml` 各自至少有一条 `backend: atb`（否则不会走 ATB 精度/性能路径）。

```bash
atk case -f ATB_<OpName>_gen.yaml -p ./generator_<op>.py -dt 200 -l error
atk task -n node.yaml -c result/.../all_....json -p ./ --task accuracy -ap $atb_path/common -l error
atk task -n node_perf.yaml -c result/.../all_....json -p ./ --task performance_device -ap $atb_path/common -mt 10 -sp --save_data profile -l error
```

可选 CSV 联动：

```bash
cd "$ATB_REPO_PATH/tests/framework/python/CsvOpsTestTool"
python3 atb_csv_ops_test.py -i <csv_path> -ps new -ll info
```

## 通过标准

- [ ] 全量 case 已生成
- [ ] 全量精度通过
- [ ] 全量性能完成
- [ ] CSV 联动结果已记录（如适用）

## 失败回流

- 全量 case 生成失败：回流 Gate 2 修复生成约束
- 精度失败：回流 Gate 3 / Gate 4 修复 Golden 或代表集问题
- 性能失败或 OOM：降并发/拆批后重跑 Gate 5
