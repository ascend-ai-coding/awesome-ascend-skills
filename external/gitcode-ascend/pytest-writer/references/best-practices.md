# Pytest 测试最佳实践

## 测试组织

### 目录结构

```
project/
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── auth.py
│       └── database.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_auth.py
    ├── test_database.py
    └── integration/
        ├── __init__.py
        └── test_api.py
```

### 按功能组织测试

- 每个模块对应一个测试文件
- 相关的测试放在同一个文件中
- 集成测试放在单独的目录中

## 测试命名

### 描述性的测试名称

```python
def test_user_login_with_valid_credentials():
    pass

def test_user_login_with_invalid_password():
    pass

def test_user_login_with_nonexistent_account():
    pass
```

### 遵循命名约定

- 测试文件：`test_*.py` 或 `*_test.py`
- 测试函数：`test_*`
- 测试类：`Test*`

## 测试结构

### AAA模式

```python
def test_calculate_total_with_discount():
    # Arrange - 准备测试数据和设置
    cart = Cart(items=[Item(price=100), Item(price=50)])
    discount = 0.1
    
    # Act - 执行被测试的功能
    total = cart.calculate_total(discount)
    
    # Assert - 验证结果
    assert total == 135.0
```

### 单一职责

每个测试应该只测试一个功能点：

```python
def test_user_creation():
    user = User(name="Alice", email="alice@example.com")
    assert user.name == "Alice"
    assert user.email == "alice@example.com"

def test_user_validation():
    with pytest.raises(ValueError):
        User(name="", email="alice@example.com")
```

## 测试隔离

### 独立性

每个测试应该独立运行，不依赖其他测试：

```python
def test_create_user():
    user = create_user("Alice")
    assert user.id is not None

def test_delete_user():
    user = create_user("Bob")
    delete_user(user.id)
    assert get_user(user.id) is None
```

### 使用fixtures确保隔离

```python
@pytest.fixture
def clean_database():
    db = create_test_database()
    yield db
    db.cleanup()

def test_with_clean_database(clean_database):
    user = create_user("Alice")
    assert user is not None
```

## 测试覆盖率

### 测试正常路径

```python
def test_successful_operation():
    result = perform_operation(valid_input)
    assert result.success
```

### 测试边界条件

```python
@pytest.mark.parametrize("value,expected", [
    (-1, False),
    (0, True),
    (1, True),
    (99, True),
    (100, False),
])
def test_boundary_values(value, expected):
    assert is_valid(value) == expected
```

### 测试异常情况

```python
def test_invalid_input():
    with pytest.raises(ValueError):
        perform_operation(invalid_input)
```

## 测试数据管理

### 使用fixtures管理测试数据

```python
@pytest.fixture
def sample_user():
    return User(name="Alice", email="alice@example.com")

def test_user_operations(sample_user):
    assert sample_user.name == "Alice"
```

### 使用参数化测试减少重复

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_uppercase(input, expected):
    assert input.upper() == expected
```

### 使用工厂函数创建测试数据

```python
def create_user(name="Alice", email="alice@example.com"):
    return User(name=name, email=email)

def test_user_creation():
    user = create_user()
    assert user.name == "Alice"
```

## 测试性能

### 避免不必要的设置

```python
@pytest.fixture(scope="module")
def expensive_resource():
    return create_expensive_resource()

def test_1(expensive_resource):
    pass

def test_2(expensive_resource):
    pass
```

### 使用mock避免外部依赖

```python
from unittest.mock import patch

@patch('myapp.external_api_call')
def test_with_mock(mock_api):
    mock_api.return_value = {"status": "ok"}
    result = my_function()
    assert result == "ok"
```

## 测试文档

### 添加文档字符串

```python
def test_user_authentication():
    """
    测试用户认证功能
    
    验证：
    - 有效的用户名和密码可以成功登录
    - 错误的密码会导致认证失败
    """
    user = authenticate("alice", "password")
    assert user.is_authenticated
```

### 使用注释解释复杂逻辑

```python
def test_complex_calculation():
    # 准备测试数据
    data = {
        "values": [1, 2, 3, 4, 5],
        "weights": [0.1, 0.2, 0.3, 0.2, 0.2]
    }
    
    # 执行计算
    result = weighted_average(data)
    
    # 验证结果
    assert result == pytest.approx(3.0, rel=1e-2)
```

## 测试维护

### 定期重构测试

- 删除重复的测试代码
- 提取共同的测试逻辑到fixtures
- 保持测试代码简洁

### 保持测试更新

- 当功能变更时更新测试
- 删除不再相关的测试
- 添加新功能的测试

## 最佳实践总结

1. **保持测试简单**：每个测试应该只测试一个功能点
2. **使用描述性的名称**：测试名称应该清楚地表达测试意图
3. **遵循AAA模式**：Arrange、Act、Assert
4. **确保测试隔离**：每个测试应该独立运行
5. **使用fixtures管理资源**：避免重复的设置代码
6. **测试边界条件**：不要只测试正常路径
7. **使用参数化测试**：减少重复代码
8. **保持测试更新**：确保测试与代码同步
9. **使用mock避免外部依赖**：提高测试速度和可靠性
10. **添加适当的文档**：帮助理解测试目的
