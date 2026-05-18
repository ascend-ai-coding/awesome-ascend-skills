# 历史安全问题经验库 (Lessons Learned)

> 本节记录推理服务引擎中实际发生过的安全问题及修复经验，用于指导审查时排查同类问题。
> 由 SKILL.md 按需引用。

---

## 经验 1：服务化配置参数组合导致 OOM — 内存放大系数被忽视

**问题编号：** SEC-EXP-001
**严重级别：** CRITICAL
**影响范围：** 推理服务引擎 HTTP 服务端
**问题类型：** CWE-400 / CWE-770（资源耗尽）

**问题描述：**

推理服务的 HTTP 服务端配置了以下参数：

| 配置项 | 值 | 含义 |
|--------|-----|------|
| `maxReqs` | 10,000 | 最大并发请求数 |
| `bodyLimit` | 10 MB | 单个请求体大小上限 |
| `HEADER_LIMIT` | 8 KB | 单个请求头大小上限 |

每个参数单独看都在合理范围内。但被忽视的关键因素是：三方组件 **nlohmann::json** 将字符串请求解析为 JSON 结构体时，内存占用放大约 **33 倍**。

**放大原因分析：**
- nlohmann::json 的每个 JSON 节点是一个 `basic_json` 对象，包含：`json_value` union（8 字节）+ type tag（1 字节）+ padding（对齐到 8/16 字节）
- 每个 JSON 字符串值内部使用 `std::string`，即使短字符串也需 32 字节 SSO 缓冲区
- JSON 对象使用 `std::map<std::string, basic_json>`，每个键值对需要红黑树节点开销（3 个指针 + color bit ≈ 32 字节）
- JSON 数组使用 `std::vector<basic_json>`，有 capacity 预留和指针开销
- 综合来看：一个紧凑的 JSON 字符串（如 `{"a":"b"}`，7 字节）在内存中占用约 200+ 字节

**最坏情况计算：**
```
峰值内存 = maxReqs × bodyLimit × JSON放大系数
         = 10,000 × 10 MB × 33
         = 3,300,000 MB
         ≈ 3.3 TB
```
而部署 Pod 的可用内存通常为 64~512 GB，**差距达 6~50 倍**。

**攻击场景：**
攻击者无需发送恶意请求，只需构造大量**合法但体积接近 bodyLimit 上限**的推理请求（如包含超长 prompt 或大量 few-shot 示例的请求），即可触发 OOM Killed。这是一种**合法流量的 DoS 攻击**，传统 WAF 无法识别。

**根本原因：**
1. **配置参数各自为政**：maxReqs、bodyLimit、headerLimit 由不同模块/不同开发者设置，没有统一的资源预算视角
2. **忽视反序列化放大系数**：将 `bodyLimit` 等同于内存占用，未考虑 JSON 解析后的实际内存膨胀
3. **缺少启动时校验**：服务启动时未验证配置参数组合后的峰值内存是否在安全范围内
4. **缺少运行时保护**：无内存水位监控，无请求准入控制，直到 OOM Killed 才发现问题

**修复方案：**

1. **启动时校验**：服务启动时计算 `maxReqs × bodyLimit × JSON放大系数`，若超过可用内存的 60% 则拒绝启动并打印告警
2. **参数联动约束**：根据部署环境内存反推 maxReqs 的安全上限：`safeMaxReqs = availableMem × 0.6 / (bodyLimit × jsonAmplification)`
3. **运行时准入控制**：监控 `/proc/meminfo` 中的 `MemAvailable`，内存使用率超过 80% 时返回 HTTP 503 拒绝新请求
4. **考虑低放大系数的替代方案**：RapidJSON DOM 模式放大约 4~8x，simdjson on-demand 模式几乎无放大（流式解析）

**排查同类问题的审查清单：**

```
- [ ] 检查服务所有资源上限配置参数（并发数、请求体/头大小、超时时间、队列深度等）
- [ ] 计算各参数组合后的最坏情况峰值资源消耗（内存、CPU、文件描述符、网络带宽等）
- [ ] 确认反序列化库（JSON/XML/Protobuf/YAML 等）的内存放大系数已纳入计算
- [ ] 确认峰值资源消耗 < 部署目标的物理资源上限 × 安全比例（建议 60%）
- [ ] 检查是否存在启动时配置参数合理性校验
- [ ] 检查是否存在运行时资源使用率监控和过载保护（背压/熔断/限流）
- [ ] 检查 OOM / 资源耗尽场景的降级策略（优雅拒绝 vs 进程崩溃）
```

**各 JSON 库内存放大系数参考表：**

| JSON 库 | 语言 | 解析模式 | 放大系数 | 说明 |
|---------|------|---------|---------|------|
| **nlohmann::json** | C++ | DOM (全量) | **20~40x** | std::map + std::string + type tag 开销大 |
| **RapidJSON** | C++ | DOM (全量) | **4~8x** | 自有分配器，紧凑内存布局 |
| **RapidJSON** | C++ | SAX (流式) | **~0x** | 事件驱动，不构建完整树 |
| **simdjson** | C++ | On-demand | **~0x** | 流式按需解析，几乎无额外内存 |
| **cJSON** | C | DOM (全量) | **8~15x** | 链表结构，每节点固定开销 |
| **json (stdlib)** | Python | DOM (全量) | **4~10x** | Python 对象头开销 |
| **orjson** | Python | DOM (全量) | **2~5x** | Rust 实现，内存更紧凑 |
| **Jackson** | Java | DOM (全量) | **5~15x** | Java 对象头 + 引用开销 |
| **Gson** | Java | DOM (全量) | **8~20x** | 反射 + 装箱开销更大 |

---

## 经验 2：多模态 Token 未配对导致数组越界 DoS — `<|begin_of_image|>` 无 `<|end_of_image|>`

**问题编号：** SEC-EXP-002
**严重级别：** CRITICAL
**影响范围：** 推理服务引擎多模态推理路径
**问题类型：** CWE-129 (Improper Validation of Array Index) / CWE-248 (Uncaught Exception) / CWE-20 (Improper Input Validation)

**问题描述：**

多模态模型处理图片输入时，代码通过特殊 Token `<|begin_of_image|>` 和 `<|end_of_image|>` 标识图片数据的起止位置。处理逻辑使用 NumPy 查找这两个 Token 的位置：

```python
# 问题代码
boi_positions = np.where(np.equal(input_ids, self.config.boi_token_id))[0]
eoi_positions = np.where(np.equal(input_ids, self.config.eoi_token_id))[0]
# ... 后续直接用硬索引取 eoi 位置
eoi_pos = eoi_positions[0]  # ← 若 eoi 不存在，IndexError
```

**攻击方式：**

攻击者发送多模态推理请求时，只发送 `<|begin_of_image|>` 而**不发送** `<|end_of_image|>`：

```json
{
  "prompt": "<|begin_of_image|>fake_image_data_without_end_token",
  "model": "vision-model-v1"
}
```

**崩溃链路：**

1. Tokenizer 将 prompt 转为 `input_ids`，其中包含 `boi_token_id` 但**不包含** `eoi_token_id`
2. `np.where(np.equal(input_ids, self.config.eoi_token_id))[0]` 返回**空数组** `[]`
3. 后续代码执行 `eoi_positions[0]` 或 `eoi_positions[i]` 触发 `IndexError: index 0 is out of bounds for axis 0 with size 0`
4. 该异常未被任何 try/except 捕获，**沿调用栈一路传播到进程顶层**
5. 推理进程崩溃退出 → 服务不可用 → **DoS 达成**

**根本原因：**

1. **隐式假设**：代码假设 `<|begin_of_image|>` 和 `<|end_of_image|>` 一定成对出现、数量相等，但用户输入不受此约束
2. **无输入校验**：未在使用前验证 `eoi_positions` 数组非空且长度与 `boi_positions` 匹配
3. **无异常兜底**：推理请求处理链路上无 `try/except IndexError` 的框架层保护

**修复方案：**

1. **Token 配对校验**：在索引访问前，验证 begin/end Token 数量相等、位置有效（参见 Python §10 `validate_special_token_pairs()`）
2. **框架层兜底**：在推理请求 handler 中添加 `except (IndexError, ValueError, KeyError)` 捕获，返回 HTTP 400 而非进程崩溃
3. **防御性编程**：所有 `np.where()` / `np.equal()` 结果在索引访问前检查 `.size > 0`

**排查同类问题的关键模式：**

```
# 搜索以下代码模式：
np.where(...)[0][N]       # 硬索引取位置，N 可能越界
np.equal(input_ids, token_id)  # 特殊 Token 查找
positions[i]              # 循环中按索引取配对 Token 位置
# 任何假设"某 Token 一定存在于 input_ids 中"的代码
```

---

## 经验 3：多模态 Token 序列格式假设被打破导致 DoS — `<|vision_start|><|video_pad|><|vision_end|>`

**问题编号：** SEC-EXP-003
**严重级别：** CRITICAL
**影响范围：** 推理服务引擎多模态推理路径
**问题类型：** CWE-129 (Improper Validation of Array Index) / CWE-248 (Uncaught Exception) / CWE-20 (Improper Input Validation)

**问题描述：**

多模态模型处理视觉输入时，代码假设所有以 `<|vision_start|>` 开头的 Token 序列都遵循**内部私有协议格式**（例如：`<|vision_start|>` 后紧跟 image_count、width、height 等元数据字段，然后是 image_pad Token 序列，最后以 `<|vision_end|>` 结束）。

**攻击方式：**

攻击者构造非预期的合法 Token 组合：

```json
{
  "prompt": "<|vision_start|><|video_pad|><|vision_end|>Please describe this image.",
  "model": "vision-model-v1"
}
```

发送 `<|vision_start|>` + `<|video_pad|>` + `<|vision_end|>` 序列，而非代码预期的 `<|vision_start|>` + image 元数据 + `<|image_pad|>` 序列。

**崩溃链路：**

1. 代码检测到 `<|vision_start|>` Token，进入视觉序列处理分支
2. 按照内部协议格式，代码直接按**固定偏移**读取后续 Token：
   - `input_ids[start_pos + 1]` → 预期为 image_count，实际为 `video_pad_id`（语义错误）
   - `input_ids[start_pos + 2]` → 预期为 width，实际可能为 `vision_end_id` 或已越界
   - `input_ids[start_pos + 3]` → 预期为 height，**数组越界**
3. 偏移计算全部基于错误的值，导致后续切片操作 `input_ids[start+4 : start+4+total_patches]` 的 `total_patches` 值为随机大数或负数
4. 触发 `IndexError` 或内存访问异常
5. **框架侧没有对应的异常捕获机制**，异常直接导致推理 worker 进程崩溃 → 服务不可用

**根本原因：**

1. **隐式协议假设**：代码将 `<|vision_start|>` 视为"私有协议"的起始标记，假设后续 Token 必定遵循特定格式（image 元数据 + image_pad），但用户可以注入任意 Token 组合
2. **混合 Token 类型未处理**：`<|video_pad|>` 出现在预期 `<|image_pad|>` 的位置，代码无分派逻辑区分不同子类型
3. **硬编码偏移**：使用 `start_pos + 1/2/3` 等固定偏移取值，而非基于实际序列内容动态解析
4. **框架层无兜底**：推理 handler 没有 `try/except` 保护，单个请求的异常可以杀死整个进程

**修复方案：**

1. **序列内容白名单校验**：在解析前验证 `vision_start..vision_end` 之间的所有 Token 是否属于合法集合（参见 Python §10 `parse_vision_sequence()`、C++ §11 `parseVisionSequenceSafe()`）
2. **按内容分派，不硬假设格式**：检查实际 Token 类型（image_pad / video_pad / audio_pad），根据类型走不同解析分支，而非按固定偏移读取
3. **框架层兜底捕获**：在推理 handler 中添加 `IndexError`/`out_of_range` 的 catch，返回 HTTP 400
4. **边界检查**：所有基于偏移的数组访问前，先验证 `start_pos + offset < input_ids.size()`

---

## 经验 2 和经验 3 的共性总结

| 维度 | 共性模式 | 防御策略 |
|------|---------|---------|
| **输入信任** | 假设用户输入的 Token 序列遵循内部预期格式 | **零信任原则**：对所有用户可控的 Token 序列做显式校验 |
| **配对假设** | 假设 begin/end Token 一定成对且数量匹配 | 先验证配对完整性，再按对处理 |
| **格式假设** | 假设特定 Token 后的内容遵循私有协议 | 白名单校验序列内容，按实际类型分派 |
| **异常处理** | 框架层无兜底，单请求异常杀死进程 | 推理 handler 必须有 IndexError/out_of_range catch |
| **攻击面** | 特殊 Token 对外暴露，可被用户直接注入 | 入口处（tokenize 之后、forward 之前）做统一校验 |

**排查同类问题的关键模式：**

```
# 搜索以下代码模式：
input_ids[start_pos + N]   # 基于固定偏移的 Token 访问，N 可能越界
positions[i]               # 假设配对 Token 数量一致的索引访问
if token_id == vision_start_id:  # 进入特定处理分支后是否做了格式校验
# 检查所有 except/catch 块：推理路径上是否有 IndexError/out_of_range 的兜底
```
