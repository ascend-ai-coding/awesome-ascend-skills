# 排错速查表（现象 → 阶段 → 根因 → 对策）

| 现象 / 报错 | 阶段 | 根因 | 对策 |
| --- | --- | --- | --- |
| `CompressInt4ToInt8 Value -1 out of range [0,15]` | omg | 权重非网格（原始浮点含负值）+ UINT4 打包 | 用 dopt torch 导出（`s3_export`，decoder）；拆分用 `s5_grid_weights.py` 搬整网网格权重。见 `platform-9030.md` 铁律 2 |
| `type MatMul don't support!` | omg | ① 按 INT4 打包（9030 不支持）② 容器 AscendC 工具链未激活 | ① `quant_param_2=True`（UINT4）② 容器内 `source install.sh + set_ascendc_env.sh`，确认 `which ccec` 非空 |
| omg 在宿主直接崩 / glibc 报错 | omg | 宿主 glibc 2.17 跑不了 9030 omg | 必须在 glibc ≥ 2.35 容器内跑。见 `platform-9030.md` 铁律 1 |
| `GatherV2D Input[1] DataType INT64 is wrong` | omg | Gather 索引被转成 int64 | `s3_rewrite.py` 删掉把索引转 int64 的 Cast，索引保 int32 |
| `Inputs of Max must have same dtype. Got Long and Int` | s3_export 导出 | arange / 常量 int64 vs offset int32 | 模型定义里让 arange / max 等常量 dtype 跟随 `offset.dtype` |
| omg pre-check 报 `xxx is not supported`（如 LSTM / LogSoftmax） | omg | 算子不被平台支持 | 拆算子（LogSoftmax → Softmax + Log）；架构级不支持（LSTM）需换结构 |
| 量化参数贴不上（节点无量化效果） | omg | 节点名与 `quant_param_file` key 不对齐 | 检查图改写阶段的 MatMul 重命名是否与量化参数 key 一致 |
| omg 编译 `kernelfunc not found` | omg | 纯权重量化（无激活量化），平台不支持 WxA32 | 必须 W4A16 / W8A8，激活也要量化 |
| 端侧输出 Inf / NaN / 乱码 | 端侧 | 激活 scale 退化导致 fp16 溢出 | 两遍标定，退化层强制 float（见 `model-strategies.md` punc） |
| 端侧识别异常 / 插字 | 端侧 | offset / att_mask 按 float 导出破坏整数语义 | encoder 用 int32 导出 + 图改写转 float（`model-strategies.md`） |
| 标点不插 / 标点错 | 端侧 | 标点模型版本不对，或校准分布与口语不一致 | 用端侧匹配的内置 embed 单模型版；校准数据贴近口语 |
| 端侧退化但 CPU / ORT 正常 | 端侧 | NPU 激活量化执行问题 | 收集证据（输入、各层输出对比）反馈平台方 |

---

## 排错通用心法

1. **先定位是「转换期」还是「端侧运行期」**：omg 报错属转换期，乱码 / 插字属端侧运行期，两类根因与修复点完全不同。
2. **9030 报 UINT4 / MatMul 类错，先核两条铁律**（`platform-9030.md`）：容器工具链激活了吗？`quant_param_2=True` 吗？权重落网格了吗？
3. **量化参数贴不上**多半是节点名问题：对照 `quant_param_file` 的 key 与图里 MatMul 名。
4. **端侧精度问题**优先做 CPU/ORT 对照与逐层输出对比，定位是哪一层、哪种量化（激活 scale 退化最常见）。
