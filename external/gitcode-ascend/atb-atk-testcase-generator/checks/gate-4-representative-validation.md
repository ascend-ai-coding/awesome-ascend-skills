# Gate 4 代表用例验证

## 输入

- Gate 3 通过结果
- `ATB_<OpName>_gen.yaml`
- `generator_<op>.py`
- `node.yaml`

## 命令

执行 `atk task` 前确认 `node.yaml` 中至少有一条 `backend: atb`（否则不会走 ATB 精度路径）。

```bash
atk case -f ATB_<OpName>_gen.yaml -p ./generator_<op>.py -dt 10 -l info
atk task -n node.yaml -c result/.../all_....json -p ./ --task accuracy -ap $atb_path/common -l info
```

## 通过标准

- [ ] 每 dtype 代表用例已生成
- [ ] 精度通过率为 100%
- [ ] 无关键错误日志
- [ ] 用户确认 Gate 4 通过

## 失败回流

- 代表用例生成失败：回流 Gate 2 检查 YAML/Generator
- 精度不通过：回流 Gate 3 修复 Golden 实现
- 用户未确认：停止在 Gate 4，等待确认后再进入 Gate 5
