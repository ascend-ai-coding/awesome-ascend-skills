# Python 安全审查详细参考

> 本文件包含 Python 安全审查的完整代码示例和审查要点。由 SKILL.md 按需引用。

## 目录

1. [代码注入](#1-代码注入)
2. [SQL 注入](#2-sql-注入)
3. [反序列化漏洞](#3-反序列化漏洞)
4. [路径遍历](#4-路径遍历)
5. [敏感信息泄露](#5-敏感信息泄露)
6. [assert 误用](#6-assert-误用)
7. [临时文件安全](#7-临时文件安全)
8. [正则表达式拒绝服务 (ReDoS)](#8-正则表达式拒绝服务-redos)
9. [JSON 请求嵌套深度校验](#9-json-请求嵌套深度校验)
10. [特殊 Token 注入与多模态输入校验](#10-特殊-token-注入与多模态输入校验)

---

## 1. 代码注入

```python
# ❌ 不安全：eval/exec 执行用户输入
user_input = request.args.get("expr")
result = eval(user_input)  # 任意代码执行

# ✅ 安全：使用 ast.literal_eval 或白名单
import ast
result = ast.literal_eval(user_input)  # 仅解析字面量
```

```python
# ❌ 不安全：subprocess 使用 shell=True
import subprocess
subprocess.run(f"grep {user_input} /var/log/app.log", shell=True)  # 命令注入

# ✅ 安全：使用列表参数，避免 shell=True
subprocess.run(["grep", user_input, "/var/log/app.log"], shell=False)
```

## 2. SQL 注入

```python
# ❌ 不安全：字符串拼接 SQL
cursor.execute(f"SELECT * FROM users WHERE name = '{name}'")

# ✅ 安全：参数化查询
cursor.execute("SELECT * FROM users WHERE name = %s", (name,))
```

## 3. 反序列化漏洞

```python
# ❌ 不安全：pickle 加载不受信任的数据
import pickle
data = pickle.loads(untrusted_bytes)  # 任意代码执行

# ❌ 不安全：yaml.load 无 Loader
import yaml
config = yaml.load(untrusted_yaml)  # 任意代码执行

# ✅ 安全：使用 safe_load
config = yaml.safe_load(untrusted_yaml)

# ✅ 安全：使用 JSON 替代 pickle
import json
data = json.loads(untrusted_string)
```

## 4. 路径遍历

```python
# ❌ 不安全：直接拼接用户输入的路径
file_path = os.path.join("/data/uploads", user_filename)
with open(file_path) as f:  # ../../../etc/passwd
    content = f.read()

# ✅ 安全：验证解析后的路径在允许范围内
import os
base_dir = os.path.realpath("/data/uploads")
file_path = os.path.realpath(os.path.join(base_dir, user_filename))
if not file_path.startswith(base_dir):
    raise ValueError("Path traversal detected")
with open(file_path) as f:
    content = f.read()
```

## 5. 敏感信息泄露

```python
# ❌ 不安全：硬编码密钥
API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "admin123"

# ❌ 不安全：日志中记录敏感信息
logger.info(f"User login: password={password}")
logger.debug(f"API response: {response.json()}")  # 可能含敏感数据

# ✅ 安全：从环境变量读取，统一处理缺失和空值
import os
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required and must be non-empty")

# ✅ 安全：日志脱敏
logger.info(f"User login: user={username}")
logger.debug(f"API response status: {response.status_code}")
```

## 6. assert 误用

```python
# ❌ 不安全：用 assert 做运行时检查（python -O 会跳过 assert）
assert user.is_admin, "Unauthorized"

# ✅ 安全：用显式条件判断
if not user.is_admin:
    raise PermissionError("Unauthorized")
```

## 7. 临时文件安全

```python
# ❌ 不安全：可预测的临时文件名
with open("/tmp/myapp_data.txt", "w") as f:  # 竞态条件 + 符号链接攻击
    f.write(data)

# ✅ 安全：使用 tempfile
import tempfile
with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    f.write(data)
```

## 8. 正则表达式拒绝服务 (ReDoS)

```python
# ❌ 不安全：嵌套量词导致指数级回溯
import re
pattern = re.compile(r"(a+)+$")  # ReDoS
pattern.match("a" * 30 + "!")  # 极慢

# ✅ 安全：使用原子组或限制输入长度
pattern = re.compile(r"a+$")
if len(user_input) > 1000:
    raise ValueError("Input too long")
```

## 9. JSON 请求嵌套深度校验

```python
# ❌ 不安全：直接解析未限制深度的 JSON 请求
import json

def handle_request(raw_body: str):
    data = json.loads(raw_body)  # 无深度限制，恶意嵌套可导致栈溢出或 CPU/内存耗尽
    process(data)

# ❌ 不安全：递归遍历 JSON 无深度保护
def walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            walk(v)  # 深层嵌套导致 RecursionError / 栈溢出
    elif isinstance(obj, list):
        for item in obj:
            walk(item)
```

```python
# ✅ 安全：解析前检查 JSON 嵌套深度
import json

MAX_JSON_DEPTH = 32  # 根据业务需求设置合理上限

def check_json_depth(raw_body: str, max_depth: int = MAX_JSON_DEPTH) -> int:
    """在解析前通过字符扫描快速检测嵌套深度（O(n) 时间，O(1) 空间）"""
    depth = 0
    max_seen = 0
    in_string = False
    escape = False
    for ch in raw_body:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            depth += 1
            if depth > max_depth:
                raise ValueError(f"JSON nesting depth {depth} exceeds maximum allowed {max_depth}")
            max_seen = max(max_seen, depth)
        elif ch in ('}', ']'):
            depth -= 1
    return max_seen

def handle_request(raw_body: str):
    check_json_depth(raw_body)  # 先检查深度，再解析
    data = json.loads(raw_body)
    process(data)
```

```python
# ✅ 安全：使用 json.JSONDecoder 自定义解析深度限制（Python 3.x）
import json

class DepthLimitedDecoder(json.JSONDecoder):
    """限制 JSON 嵌套深度的解码器"""
    MAX_DEPTH = 32

    def __init__(self, *args, **kwargs):
        self._depth = 0
        super().__init__(*args, **kwargs)

    def decode(self, s, **kwargs):
        self._check_depth(s)
        return super().decode(s, **kwargs)

    def _check_depth(self, s: str):
        depth = 0
        in_string = False
        escape = False
        for ch in s:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                depth += 1
                if depth > self.MAX_DEPTH:
                    raise json.JSONDecodeError(
                        f"JSON nesting depth exceeds {self.MAX_DEPTH}", s, 0
                    )
            elif ch in ('}', ']'):
                depth -= 1

# 使用示例
data = json.loads(raw_body, cls=DepthLimitedDecoder)
```

```python
# ✅ 安全：Web 框架中间件统一拦截（以 Flask 为例）
from flask import Flask, request, abort
import json

app = Flask(__name__)
MAX_JSON_DEPTH = 32

@app.before_request
def check_json_nesting_depth():
    if request.is_json:
        try:
            check_json_depth(request.get_data(as_text=True), MAX_JSON_DEPTH)
        except ValueError:
            abort(400, description="JSON nesting depth exceeds allowed limit")
```

**审查要点：**
- 搜索所有 `json.loads()`、`json.load()`、`request.get_json()`、`request.json` 调用
- 检查解析入口是否存在嵌套深度限制
- 检查递归遍历 JSON 数据结构的函数是否有递归深度保护
- 对于 HTTP/gRPC/WebSocket 等外部请求入口，嵌套深度限制应在**解析层/中间件**统一实施
- 建议最大深度：API 请求 ≤ 32 层，配置文件 ≤ 64 层（根据业务实际调整）
- 相关漏洞标准：CWE-674 (Uncontrolled Recursion)、CWE-400 (Uncontrolled Resource Consumption)

## 10. 特殊 Token 注入与多模态输入校验

```python
# ❌ 不安全：假设特殊 Token 必定成对出现，用硬索引取配对 Token 位置
import numpy as np

def process_multimodal_input(input_ids, config):
    # 查找 <|begin_of_image|> 和 <|end_of_image|> 的位置
    boi_positions = np.where(np.equal(input_ids, config.boi_token_id))[0]
    eoi_positions = np.where(np.equal(input_ids, config.eoi_token_id))[0]

    for i, boi_pos in enumerate(boi_positions):
        eoi_pos = eoi_positions[i]  # ← 致命：若 eoi 缺失/数量不匹配，数组越界 → DoS
        image_tokens = input_ids[boi_pos:eoi_pos]
        process_image(image_tokens)
```

```python
# ❌ 不安全：假设特定起始 Token 后的序列一定遵循内部私有协议格式
def process_vision_input(input_ids, config):
    vision_start_positions = np.where(np.equal(input_ids, config.vision_start_id))[0]

    for start_pos in vision_start_positions:
        # 假设 vision_start 后一定跟 image_pad 序列，直接按协议偏移取值
        image_count = input_ids[start_pos + 1]    # ← 可能越界
        width  = input_ids[start_pos + 2]          # ← 可能越界
        height = input_ids[start_pos + 3]          # ← 可能越界
        total_patches = width * height * image_count
        patches = input_ids[start_pos + 4 : start_pos + 4 + total_patches]  # ← 可能越界
        # 如果用户发送 <|vision_start|><|video_pad|><|vision_end|> 而非预期的 image_pad 序列
        # 上述偏移计算全部错误，导致数组越界 → 未捕获异常 → 进程崩溃
```

```python
# ✅ 安全：严格校验特殊 Token 的配对完整性和序列格式

def validate_special_token_pairs(input_ids: np.ndarray,
                                  begin_token_id: int,
                                  end_token_id: int,
                                  token_name: str = "special") -> list:
    """校验特殊 Token 严格成对出现且不交叉嵌套"""
    begin_positions = np.where(np.equal(input_ids, begin_token_id))[0].tolist()
    end_positions = np.where(np.equal(input_ids, end_token_id))[0].tolist()

    # 校验 1：数量必须相等
    if len(begin_positions) != len(end_positions):
        raise ValueError(
            f"Mismatched {token_name} tokens: "
            f"found {len(begin_positions)} begin vs {len(end_positions)} end"
        )

    # 校验 2：每对 begin 必须在对应 end 之前
    pairs = []
    for i, (begin_pos, end_pos) in enumerate(zip(begin_positions, end_positions)):
        if end_pos <= begin_pos:
            raise ValueError(
                f"{token_name} token pair {i}: end position {end_pos} "
                f"is not after begin position {begin_pos}"
            )
        pairs.append((begin_pos, end_pos))

    # 校验 3：相邻对不交叉
    for i in range(len(pairs) - 1):
        if pairs[i][1] > pairs[i + 1][0]:
            raise ValueError(
                f"{token_name} token pairs {i} and {i+1} overlap"
            )

    return pairs

def process_multimodal_input(input_ids, config):
    try:
        pairs = validate_special_token_pairs(
            input_ids, config.boi_token_id, config.eoi_token_id, "image"
        )
        for boi_pos, eoi_pos in pairs:
            image_tokens = input_ids[boi_pos:eoi_pos]
            process_image(image_tokens)
    except ValueError as e:
        logger.warning(f"Invalid multimodal input rejected: {e}")
        raise InvalidRequestError(str(e))  # 返回 4xx，不崩溃
```

```python
# ✅ 安全：校验私有协议格式序列，不信任 Token 序列的隐式假设

def parse_vision_sequence(input_ids: np.ndarray,
                           start_pos: int,
                           end_pos: int,
                           config) -> dict:
    """安全解析 vision_start..vision_end 之间的序列"""
    seq = input_ids[start_pos + 1 : end_pos]  # 取 start/end 之间的内容

    # 校验 1：序列非空
    if len(seq) == 0:
        raise ValueError(f"Empty vision sequence at position {start_pos}")

    # 校验 2：校验序列中的 Token 类型是否合法（白名单）
    allowed_tokens = {config.image_pad_id, config.video_pad_id, config.audio_pad_id}
    actual_tokens = set(seq.tolist())
    illegal_tokens = actual_tokens - allowed_tokens
    if illegal_tokens:
        raise ValueError(
            f"Illegal tokens in vision sequence: {illegal_tokens}, "
            f"allowed: {allowed_tokens}"
        )

    # 校验 3：按实际内容类型分派，不硬假设格式
    if config.image_pad_id in actual_tokens:
        return {"type": "image", "pad_count": int(np.sum(seq == config.image_pad_id))}
    elif config.video_pad_id in actual_tokens:
        return {"type": "video", "pad_count": int(np.sum(seq == config.video_pad_id))}
    else:
        raise ValueError(f"Unrecognized vision sequence content at position {start_pos}")

def process_vision_input_safe(input_ids, config):
    try:
        # 第一步：校验 vision_start / vision_end 成对
        pairs = validate_special_token_pairs(
            input_ids, config.vision_start_id, config.vision_end_id, "vision"
        )
        # 第二步：逐对解析，每对都做格式校验
        for start_pos, end_pos in pairs:
            info = parse_vision_sequence(input_ids, start_pos, end_pos, config)
            dispatch_vision_processing(info)
    except ValueError as e:
        logger.warning(f"Invalid vision input rejected: {e}")
        raise InvalidRequestError(str(e))
```

```python
# ✅ 安全：框架层统一异常捕获，防止未预期异常导致进程崩溃

def inference_request_handler(request):
    """推理请求入口 — 框架层兜底"""
    try:
        input_ids = tokenize(request.prompt)
        validate_multimodal_tokens(input_ids, model_config)
        result = model.forward(input_ids)
        return SuccessResponse(result)
    except InvalidRequestError as e:
        return ErrorResponse(400, f"Bad request: {e}")  # 客户端错误，优雅拒绝
    except (IndexError, KeyError, ValueError) as e:
        # 兜底：捕获所有可能由恶意输入触发的数组越界/键缺失/值错误
        logger.error(f"Input validation gap detected: {type(e).__name__}: {e}")
        return ErrorResponse(400, "Malformed input")
    except Exception as e:
        # 最终兜底：任何未预期异常都不应导致进程退出
        logger.error(f"Unexpected error in inference handler: {e}", exc_info=True)
        return ErrorResponse(500, "Internal server error")
```

**审查要点：**
- 搜索所有 `np.where`、`np.equal`、`torch.where`、`torch.eq` + 数组索引操作（`[0]`、`[i]`、`positions[n]`），检查是否假设结果非空或长度匹配
- 搜索所有特殊 Token ID 的查找和配对逻辑（`boi`/`eoi`、`vision_start`/`vision_end`、`audio_start`/`audio_end`、`begin_of_image`/`end_of_image` 等），检查是否验证了成对完整性
- 检查多模态处理代码是否对 Token 序列格式做了**隐式假设**（如"start 后面一定跟 image_pad"），评估用户发送非预期 Token 组合时是否会触发越界
- 检查推理请求处理路径上是否存在 `IndexError`/`ValueError`/`KeyError` 的**框架层兜底捕获**，确保单个请求的异常不会导致整个服务进程崩溃
- 校验逻辑应在 **tokenize 之后、model.forward 之前** 执行，作为输入预处理的一部分
- 相关漏洞标准：CWE-129 (Improper Validation of Array Index)、CWE-248 (Uncaught Exception)、CWE-20 (Improper Input Validation)

---

## Python 安全工具

| 工具 | 用途 | 命令 |
|------|------|------|
| **bandit** | 静态安全分析 | `bandit -r src/` |
| **safety / pip-audit** | 依赖漏洞扫描 | `pip-audit` |
| **pylint** | 代码质量 + 部分安全规则 | `pylint src/` |
| **mypy** | 类型检查，防止类型混淆 | `mypy src/` |
| **semgrep** | 自定义安全规则 | `semgrep --config=p/python` |
