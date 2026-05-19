# Pytest 参数化测试指南

## 基本参数化

### 简单参数化

```python
@pytest.mark.parametrize("input,expected", [
    (3, 4),
    (5, 6),
    (10, 11),
])
def test_increment(input, expected):
    assert input + 1 == expected
```

### 单参数

```python
@pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
def test_positive_numbers(value):
    assert value > 0
```

## 高级模式

### 多参数组合

```python
@pytest.mark.parametrize("x,y,expected", [
    (1, 2, 3),
    (2, 3, 5),
    (3, 4, 7),
])
def test_addition(x, y, expected):
    assert x + y == expected
```

### 参数化fixtures

```python
@pytest.fixture(params=["mysql", "postgresql", "sqlite"])
def db_connection(request):
    return create_connection(request.param)

def test_database_operations(db_connection):
    assert db_connection.is_connected()
```

### 多个参数化装饰器

```python
@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("y", [2, 3])
def test_combinations(x, y):
    assert x + y > 0
```

### 使用ids自定义测试ID

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
], ids=["lowercase", "uppercase"])
def test_uppercase(input, expected):
    assert input.upper() == expected
```

### 动态生成测试ID

```python
def id_func(val):
    return f"test_{val}"

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
], ids=id_func)
def test_double(input, expected):
    assert input * 2 == expected
```

## 实用模式

### 测试边界值

```python
@pytest.mark.parametrize("value,expected", [
    (-1, False),
    (0, True),
    (1, True),
    (99, True),
    (100, False),
])
def test_in_range(value, expected):
    assert is_valid(value) == expected
```

### 测试异常情况

```python
@pytest.mark.parametrize("input,exception", [
    ("", ValueError),
    ("abc", ValueError),
    ("-1", ValueError),
])
def test_invalid_input(input, exception):
    with pytest.raises(exception):
        parse_number(input)
```

### 参数化测试类

```python
@pytest.mark.parametrize("a,b", [(1, 2), (3, 4)])
class TestMath:
    def test_addition(self, a, b):
        assert a + b > a
    
    def test_multiplication(self, a, b):
        assert a * b >= a
```

### 使用字典参数

```python
test_data = [
    {"input": "hello", "expected": "HELLO"},
    {"input": "world", "expected": "WORLD"},
]

@pytest.mark.parametrize("data", test_data)
def test_uppercase(data):
    assert data["input"].upper() == data["expected"]
```

## 最佳实践

1. **使用有意义的参数值**：避免使用模糊的测试数据
2. **提供清晰的测试ID**：使用ids参数提高测试报告可读性
3. **组合相关测试**：使用多个参数化装饰器时注意组合爆炸
4. **使用fixtures管理复杂设置**：对于复杂的测试设置，使用fixtures而不是参数化
5. **保持测试独立性**：每个参数化测试应该独立运行
