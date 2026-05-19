# Pytest 断言技巧指南

## 基本断言

### 简单断言

```python
def test_equality():
    assert 1 + 1 == 2

def test_inequality():
    assert 1 + 1 != 3

def test_comparison():
    assert 5 > 3
    assert 5 >= 5
    assert 3 < 5
    assert 3 <= 3
```

### 布尔断言

```python
def test_boolean():
    assert True
    assert not False
    assert is_valid()
```

### 成员断言

```python
def test_membership():
    assert 1 in [1, 2, 3]
    assert "key" in {"key": "value"}
    assert "substring" in "full string"
```

## 高级断言

### 字典断言

```python
def test_dict_comparison():
    expected = {"a": 1, "b": 2}
    actual = {"a": 1, "b": 2}
    assert actual == expected
```

pytest会显示详细的差异信息。

### 列表断言

```python
def test_list_comparison():
    expected = [1, 2, 3, 4]
    actual = [1, 2, 3, 5]
    assert actual == expected
```

### 近似值断言

```python
def test_float_comparison():
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert 1.0001 == pytest.approx(1.0, rel=1e-4)
    assert 1.0 == pytest.approx(1.0, abs=0.1)
```

### 异常断言

```python
def test_exception():
    with pytest.raises(ValueError) as exc_info:
        int("invalid")
    assert "invalid literal" in str(exc_info.value)
```

### 警告断言

```python
import warnings

def test_warning():
    with pytest.warns(UserWarning):
        warnings.warn("This is a warning")
```

## 自定义断言消息

### 添加自定义消息

```python
def test_with_message():
    result = calculate()
    assert result == expected, f"Expected {expected}, got {result}"
```

### 使用上下文信息

```python
def test_with_context():
    user = create_user()
    assert user.is_active(), f"User {user.id} should be active"
```

## 断言技巧

### 使用pytest.approx处理浮点数

```python
def test_float_operations():
    result = 0.1 + 0.2
    assert result == pytest.approx(0.3)
```

### 使用pytest.raises检查异常类型

```python
def test_specific_exception():
    with pytest.raises(ValueError) as exc_info:
        raise ValueError("error message")
    assert str(exc_info.value) == "error message"
```

### 使用pytest.warns检查警告

```python
def test_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        old_function()
```

### 使用pytest.raises检查异常属性

```python
def test_exception_attributes():
    with pytest.raises(CustomError) as exc_info:
        raise CustomError(code=404, message="Not found")
    assert exc_info.value.code == 404
```

## 最佳实践

1. **使用描述性的断言**：断言应该清楚地表达测试意图
2. **避免复杂的断言逻辑**：保持断言简单直接
3. **使用pytest.approx处理浮点数**：避免浮点数精度问题
4. **提供有用的错误消息**：当断言失败时，错误消息应该帮助调试
5. **测试一个条件**：每个断言应该测试一个条件
6. **使用pytest.raises测试异常**：不要使用try-except块

## 常见模式

### AAA模式的断言

```python
def test_calculate_total():
    # Arrange
    items = [Item(price=100), Item(price=50)]
    discount = 0.1
    
    # Act
    total = calculate_total(items, discount)
    
    # Assert
    assert total == 135.0
```

### 多重断言

```python
def test_multiple_assertions():
    user = create_user("Alice", "alice@example.com")
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    assert user.is_active()
```

### 条件断言

```python
def test_conditional_assertion():
    result = process_data()
    if result is not None:
        assert result.success
        assert result.data is not None
```
