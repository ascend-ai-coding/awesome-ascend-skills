# Skill: Python 常用设计模式应用

## 触发条件

当用户要求"应用设计模式"、"用模式重构"、"design pattern refactoring"，或代码中出现明显可用模式优化的结构时触发。

## 目标

在重构过程中识别适合引入设计模式的场景，选择最 Pythonic 的实现方式，避免过度设计。

## 核心原则

- 只在模式能明确简化代码时才引入，不为用模式而用模式
- 优先使用 Python 语言特性（装饰器、上下文管理器、生成器等）而非照搬 Java 风格的 GoF 实现
- 引入模式后代码行数不应显著增加，如果增加了，重新评估是否值得

---

## 1. 创建型模式

### 1.1 工厂模式（Factory）

**适用场景：** 根据输入动态创建不同类型的对象，存在 if-elif 链创建实例。

**Pythonic 实现：** 用字典注册表替代工厂类。

```python
# Bad: if-elif 工厂
def create_serializer(format_type):
    if format_type == "json":
        return JsonSerializer()
    elif format_type == "xml":
        return XmlSerializer()
    elif format_type == "csv":
        return CsvSerializer()

# Good: 注册表工厂
_SERIALIZERS: dict[str, type[Serializer]] = {
    "json": JsonSerializer,
    "xml": XmlSerializer,
    "csv": CsvSerializer,
}

def create_serializer(format_type: str) -> Serializer:
    cls = _SERIALIZERS.get(format_type)
    if cls is None:
        raise ValueError(f"Unsupported format: {format_type}")
    return cls()
```

**进阶：** 用装饰器实现自注册工厂。

```python
_REGISTRY: dict[str, type[Serializer]] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

@register("json")
class JsonSerializer(Serializer):
    ...
```

### 1.2 单例模式（Singleton）

**适用场景：** 全局唯一资源（连接池、配置管理器）。

**Pythonic 实现：** 优先使用模块级实例，而非单例类。

```python
# Good: 模块级单例（最简方案）
# config.py
_config = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
```

**备选：** `functools.lru_cache` 实现惰性单例。

```python
@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config.from_env()
```

### 1.3 建造者模式（Builder）

**适用场景：** 构造参数超过 5 个，且有多种合法组合。

**Pythonic 实现：** 用 `dataclass` + 类方法替代 Builder 类。

```python
@dataclass
class QueryConfig:
    table: str
    columns: list[str] = field(default_factory=lambda: ["*"])
    where: str | None = None
    order_by: str | None = None
    limit: int | None = None

    @classmethod
    def simple(cls, table: str) -> "QueryConfig":
        return cls(table=table)

    @classmethod
    def paginated(cls, table: str, page: int, size: int = 20) -> "QueryConfig":
        return cls(table=table, limit=size, offset=page * size)
```

---

## 2. 结构型模式

### 2.1 装饰器模式（Decorator）

**适用场景：** 为函数/方法添加横切关注点（日志、缓存、重试、权限）。

**Pythonic 实现：** 直接使用 Python 装饰器语法。

```python
import functools
import time

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_exc
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    ...
```

### 2.2 适配器模式（Adapter）

**适用场景：** 对接第三方库或遗留接口，接口不匹配。

**Pythonic 实现：** 用函数适配或轻量包装类。

```python
# 函数适配器（简单场景）
def adapt_legacy_response(legacy_data: dict) -> StandardResponse:
    return StandardResponse(
        id=legacy_data["ID"],
        name=legacy_data["user_name"],
        created_at=datetime.fromisoformat(legacy_data["create_time"]),
    )

# 类适配器（需要维持状态时）
class NewStorageAdapter:
    def __init__(self, legacy_storage: LegacyStorage):
        self._storage = legacy_storage

    def get(self, key: str) -> bytes:
        return self._storage.read_blob(key)

    def put(self, key: str, data: bytes) -> None:
        self._storage.write_blob(key, data)
```

### 2.3 上下文管理器模式

**适用场景：** 资源获取/释放、临时状态切换、事务管理。

**Pythonic 实现：** `contextlib.contextmanager` 或 `__enter__/__exit__`。

```python
from contextlib import contextmanager

@contextmanager
def temporary_env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[key]
        else:
            os.environ[key] = old
```

---

## 3. 行为型模式

### 3.1 策略模式（Strategy）

**适用场景：** 同一操作有多种算法/策略，运行时可切换。

**Pythonic 实现：** 用 `Callable` 或 `Protocol` 替代策略接口类。

```python
from typing import Callable

# 函数式策略
SortStrategy = Callable[[list], list]

def process_data(data: list, sort_strategy: SortStrategy) -> list:
    sorted_data = sort_strategy(data)
    return sorted_data

# 使用
result = process_data(data, sort_strategy=sorted)
result = process_data(data, sort_strategy=lambda x: sorted(x, reverse=True))
```

**复杂策略用 Protocol：**

```python
from typing import Protocol

class CompressionStrategy(Protocol):
    def compress(self, data: bytes) -> bytes: ...
    def decompress(self, data: bytes) -> bytes: ...

class GzipCompression:
    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data)
    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)
```

### 3.2 观察者模式（Observer / Event）

**适用场景：** 对象状态变化需通知多个依赖方，解耦发布者和订阅者。

**Pythonic 实现：** 轻量回调列表。

```python
from typing import Callable

class EventEmitter:
    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}

    def on(self, event: str, callback: Callable) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event: str, *args, **kwargs) -> None:
        for callback in self._listeners.get(event, []):
            callback(*args, **kwargs)
```

### 3.3 模板方法模式（Template Method）

**适用场景：** 多个类共享相同流程骨架，但某些步骤实现不同。

**Pythonic 实现：** ABC + 抽象方法，或直接用函数参数注入可变步骤。

```python
from abc import ABC, abstractmethod

class DataPipeline(ABC):
    def run(self, source: str) -> None:
        raw = self.extract(source)
        cleaned = self.transform(raw)
        self.load(cleaned)

    @abstractmethod
    def extract(self, source: str) -> list[dict]: ...

    @abstractmethod
    def transform(self, data: list[dict]) -> list[dict]: ...

    @abstractmethod
    def load(self, data: list[dict]) -> None: ...
```

### 3.4 单分发泛函（singledispatch）

**适用场景：** 根据第一个参数的类型执行不同逻辑，替代 isinstance if-elif 链。

**Pythonic 实现：** `functools.singledispatch`（函数级）或 `singledispatchmethod`（方法级）。

```python
from functools import singledispatch

@singledispatch
def serialize(obj) -> str:
    raise TypeError(f"Cannot serialize {type(obj)}")

@serialize.register
def _(obj: int) -> str:
    return str(obj)

@serialize.register
def _(obj: list) -> str:
    return json.dumps(obj)

@serialize.register
def _(obj: datetime) -> str:
    return obj.isoformat()
```

**方法级用 `singledispatchmethod`：**

```python
from functools import singledispatchmethod

class Formatter:
    @singledispatchmethod
    def format(self, value):
        return str(value)

    @format.register
    def _(self, value: float):
        return f"{value:.2f}"

    @format.register
    def _(self, value: datetime):
        return value.strftime("%Y-%m-%d")
```

### 3.5 迭代器模式（Iterator）

**适用场景：** 惰性处理大数据集、流式数据、分页 API。

**Pythonic 实现：** 生成器函数。

```python
def paginated_fetch(url: str, page_size: int = 100):
    page = 0
    while True:
        response = requests.get(url, params={"page": page, "size": page_size})
        items = response.json()["items"]
        if not items:
            break
        yield from items
        page += 1

# 使用：惰性消费，内存友好
for item in paginated_fetch("/api/users"):
    process(item)
```

---

## 4. 反模式警示

以下情况不应引入设计模式：

| 场景 | 错误做法 | 正确做法 |
|------|----------|----------|
| 只有一种实现 | 创建 Interface + Factory | 直接用具体类 |
| 简单条件分支（2-3 个） | 策略模式 | if-elif 即可 |
| 无需运行时切换 | 抽象工厂 | 直接实例化 |
| 只有一个观察者 | 完整事件系统 | 直接调用 |

---

## 执行流程

1. 分析代码中的结构性问题（重复分支、硬编码依赖、紧耦合）
2. 判断是否有合适的模式可以简化问题
3. 选择最 Pythonic 的实现方式（函数 > Protocol > ABC > 完整类层次）
4. 实施重构，确保接口兼容
5. 验证重构后代码行数未显著膨胀，复杂度确实降低
