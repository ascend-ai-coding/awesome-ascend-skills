# 功耗测试参数说明

> 来源: MindCluster 7.3.0 开发文档 - 昇腾社区

---

## 主要参数

| 参数 | 说明 | 是否必填 | 默认值 | 范围 |
|-----|------|---------|-------|------|
| `-p, --power` | 启用功耗测试 | 是 | - | - |
| `-t, --type` | 算子运算类型 | 否 | fp16 | fp16/int8 |
| `-pt, --pressure-type` | 压力测试类型 | 否 | tdp | edp/tdp |
| `--dur, --duration` | 运行时间 | 否 | 600 秒 | [60, 604800] 秒 |
| `--it, --interval-times` | 屏幕刷新间隔 | 否 | 5 秒 | [1, 5] 秒 |
| `--skip-check` | 跳过健康检查 | 否 | 检查 | - |
| `-pm, --print-mode` | 打印模式 | 否 | refresh | refresh/history |
| `-q, --quiet` | 跳过确认提示 | 否 | 提示确认 | - |
| `-fmt, --format` | 输出格式 | 否 | normal | normal/json |

---

## 数据类型 (-t) 说明

| 类型 | 支持产品 |
|-----|---------|
| `fp16` | 全系列 |
| `int8` | 非 A2/A3 系列 |

**注意**：Atlas A2/A3 训练/推理系列只支持 fp16。

---

## 压力测试类型 (-pt) 说明

| 类型 | 说明 | 支持产品 |
|-----|------|---------|
| `tdp` | Thermal Design Power，热设计功耗 | A2/A3 系列 |
| `edp` | Estimated Design Power，估计设计功耗 | A2/A3 系列 |

**默认**：tdp

---

## 打印模式 (-pm) 说明

| 模式 | 说明 | 注意事项 |
|-----|------|---------|
| `refresh` | 每次打印清除历史 | 芯片多时需要调小字体，避免显示异常 |
| `history` | 保留历史打印信息 | - |

---

## 常用参数组合

### 基础测试

```bash
# 默认 FP16 TDP 测试，600 秒
ascend-dmi -p -q

# INT8 测试
ascend-dmi -p -t int8 -q
```

### 指定时长和间隔

```bash
# 测试 60 秒，每 5 秒刷新
ascend-dmi -p --dur 60 --it 5 --pm refresh -q
```

### EDP 测试

```bash
# EDP 功耗压力测试（A2/A3 系列）
ascend-dmi -p -pt edp -q
```

---

## 输出字段

| 字段 | 说明 | 单位 |
|-----|------|-----|
| Card/Type | 卡型号 | - |
| NPU Count | NPU 数量 | 个 |
| Power | 实时功耗 | W |
| Chip Name | 芯片名称 | - |
| Health | 健康状态 | - |
| Temperature | 芯片温度 | °C |
| AI Core Usage | AI Core 利用率 | % |
| Voltage | 芯片电压 | V |
| Frequency | 芯片频率 | MHz |

---

## 温度监控

- 测试时监控温度变化
- 如温度 > 90°C，可能触发降频
- 建议在温度稳定时开始测试
