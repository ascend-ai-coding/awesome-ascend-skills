# Pytest Fixtures 完整指南

## 基本用法

### 定义fixture

```python
import pytest

@pytest.fixture
def sample_data():
    return {"name": "test", "value": 42}

def test_with_fixture(sample_data):
    assert sample_data["value"] == 42
```

### Fixture作用域

```python
@pytest.fixture(scope="function")  # 默认，每个测试函数执行一次
def function_fixture():
    return "function scope"

@pytest.fixture(scope="class")  # 每个测试类执行一次
def class_fixture():
    return "class scope"

@pytest.fixture(scope="module")  # 每个模块执行一次
def module_fixture():
    return "module scope"

@pytest.fixture(scope="session")  # 整个测试会话执行一次
def session_fixture():
    return "session scope"
```

### 使用yield进行清理

```python
@pytest.fixture
def database():
    conn = create_connection()
    yield conn  # 提供给测试使用
    conn.close()  # 测试后执行清理
```

### 带参数的fixtures

```python
@pytest.fixture(params=["mysql", "postgresql", "sqlite"])
def db_type(request):
    return request.param

def test_database(db_type):
    assert db_type in ["mysql", "postgresql", "sqlite"]
```

## 高级模式

### Fixture依赖

```python
@pytest.fixture
def config():
    return {"host": "localhost", "port": 5432}

@pytest.fixture
def database(config):
    return create_connection(config["host"], config["port"])

def test_with_dependent_fixture(database):
    assert database is not None
```

### 使用conftest.py共享fixtures

```python
# conftest.py
@pytest.fixture
def shared_resource():
    return "shared across all tests in this directory"
```

### 自动使用fixtures

```python
@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(level=logging.INFO)
    yield
    logging.shutdown()
```

### Fixture工厂模式

```python
@pytest.fixture
def make_user():
    def _make_user(name, email):
        return User(name=name, email=email)
    return _make_user

def test_user_creation(make_user):
    user = make_user("Alice", "alice@example.com")
    assert user.name == "Alice"
```

### 带名称的fixtures

```python
@pytest.fixture(name="custom_name")
def long_fixture_name():
    return "value"

def test_with_custom_name(custom_name):
    assert custom_name == "value"
```

## 最佳实践

1. **使用适当的作用域**：根据资源创建成本选择合适的作用域
2. **使用yield进行清理**：确保资源被正确释放
3. **在conftest.py中组织共享fixtures**：避免重复定义
4. **使用描述性的fixture名称**：提高代码可读性
5. **避免过度使用autouse**：只在真正需要时使用
