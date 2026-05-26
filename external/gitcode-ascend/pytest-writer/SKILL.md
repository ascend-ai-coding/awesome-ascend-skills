---
name: external-gitcode-ascend-pytest-writer
description: 专业的pytest测试用例编写助手，用于创建、编写和优化Python测试用例。当需要编写测试文件、创建测试代码、重构优化测试、调试失败测试、使用fixtures、参数化测试、断言技巧、测试覆盖率分析时使用此技能。
original-name: pytest-writer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# Pytest Writer

## Overview

使用pytest框架编写高质量、可维护的Python测试用例。支持简单断言、自动发现、fixtures、参数化测试等pytest核心特性。

## Quick Start

### 基本测试结构

```python
# test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 4
```

运行测试：
```bash
pytest
```

### 测试文件命名规范

- 测试文件：`test_*.py` 或 `*_test.py`
- 测试函数：`test_*`
- 测试类：`Test*`

## Core Capabilities

### 1. 编写基本测试

使用简单的assert语句，pytest提供详细的失败信息：

```python
def test_addition():
    assert 1 + 1 == 2

def test_string_operations():
    text = "hello"
    assert text.upper() == "HELLO"
    assert len(text) == 5
```

### 2. 使用Fixtures

Fixtures用于管理测试资源和依赖关系：

```python
import pytest

@pytest.fixture
def sample_data():
    return {"name": "test", "value": 42}

def test_with_fixture(sample_data):
    assert sample_data["value"] == 42
```

**高级fixtures用法**：参见 [references/fixtures.md](references/fixtures.md)

### 3. 参数化测试

使用@pytest.mark.parametrize减少重复代码：

```python
@pytest.mark.parametrize("input,expected", [
    (3, 4),
    (5, 6),
    (10, 11),
])
def test_increment(input, expected):
    assert input + 1 == expected
```

**更多参数化模式**：参见 [references/parametrize.md](references/parametrize.md)

### 4. 断言技巧

利用pytest的断言内省功能：

```python
def test_dict_comparison():
    expected = {"a": 1, "b": 2}
    actual = {"a": 1, "b": 3}
    assert actual == expected  # pytest会显示详细的差异
```

**断言最佳实践**：参见 [references/assertions.md](references/assertions.md)

### 5. 测试组织

#### 按功能组织

```
tests/
├── test_auth.py
├── test_database.py
├── test_api.py
└── conftest.py  # 共享fixtures
```

#### 使用conftest.py

在conftest.py中定义共享fixtures：

```python
# conftest.py
@pytest.fixture
def db_connection():
    conn = create_connection()
    yield conn
    conn.close()
```

### 6. 异常测试

使用pytest.raises测试异常：

```python
def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0

def test_value_error():
    with pytest.raises(ValueError) as exc_info:
        int("invalid")
    assert "invalid literal" in str(exc_info.value)
```

### 7. 跳过和预期失败

```python
@pytest.mark.skip(reason="功能未实现")
def test_not_implemented():
    pass

@pytest.mark.skipif(sys.version_info < (3, 8), reason="需要Python 3.8+")
def test_python38_feature():
    pass

@pytest.mark.xfail
def test_known_failure():
    assert False  # 预期失败
```

## Testing Best Practices

### 测试命名

- 使用描述性的测试名称：`test_user_login_with_valid_credentials`
- 遵循AAA模式：Arrange（准备）→ Act（执行）→ Assert（断言）

```python
def test_calculate_total_with_discount():
    # Arrange
    cart = Cart(items=[Item(price=100), Item(price=50)])
    discount = 0.1
    
    # Act
    total = cart.calculate_total(discount)
    
    # Assert
    assert total == 135
```

### 测试隔离

- 每个测试应该独立运行
- 使用fixtures确保测试间的隔离
- 避免测试间的依赖关系

**更多最佳实践**：参见 [references/best-practices.md](references/best-practices.md)

## Running Tests

### 基本命令

```bash
# 运行所有测试
pytest

# 运行特定文件
pytest tests/test_auth.py

# 运行特定测试函数
pytest tests/test_auth.py::test_login

# 运行匹配模式的测试
pytest -k "login"

# 显示详细输出
pytest -v

# 显示print语句输出
pytest -s

# 在第一个失败时停止
pytest -x

# 运行上次失败的测试
pytest --lf
```

### 测试覆盖率

```bash
# 生成覆盖率报告
pytest --cov=src

# 生成HTML报告
pytest --cov=src --cov-report=html
```

## Debugging Failed Tests

### 查看详细错误信息

```bash
pytest -v --tb=long
```

### 使用pdb调试

```python
def test_debugging():
    import pdb; pdb.set_trace()
    result = complex_calculation()
    assert result == expected
```

或在命令行：
```bash
pytest --pdb
```

## Resources

### references/

详细参考文档：

- **[fixtures.md](references/fixtures.md)** - fixtures的完整用法、作用域、参数化fixtures
- **[parametrize.md](references/parametrize.md)** - 参数化测试的各种模式和技巧
- **[assertions.md](references/assertions.md)** - 断言技巧、自定义断言消息
- **[best-practices.md](references/best-practices.md)** - 测试组织、命名规范、测试隔离等最佳实践

### assets/

- **[test_template.py](assets/test_template.py)** - 标准测试文件模板，包含常用模式和结构
