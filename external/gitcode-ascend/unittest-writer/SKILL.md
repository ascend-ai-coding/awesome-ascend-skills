---
name: external-gitcode-ascend-unittest-writer
description: Python unittest 框架的专业测试用例编写助手。用于创建、编写和优化 Python 单元测试，包括测试用例结构、断言方法、测试组织、setUp/tearDown
  模式以及命令行执行。当需要编写测试文件、创建测试代码、重构优化测试、调试失败测试时使用此技能。
original-name: unittest-writer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# Unittest Writer

## Quick Start

创建基本测试用例：

```python
import unittest

class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
```

## Basic Structure

### Test Class

继承 `unittest.TestCase` 创建测试类：

```python
import unittest

class TestMyFunction(unittest.TestCase):
    def test_method_name(self):
        pass
```

### Test Methods

- 测试方法必须以 `test_` 开头
- 每个测试方法应该独立运行
- 使用断言方法验证结果

### setUp and tearDown

在测试前准备环境，测试后清理资源：

```python
class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database()
        self.db.connect()

    def tearDown(self):
        self.db.disconnect()

    def test_query(self):
        result = self.db.query('SELECT * FROM users')
        self.assertIsNotNone(result)
```

## Common Assertions

### Equality

```python
self.assertEqual(a, b)           # a == b
self.assertNotEqual(a, b)        # a != b
```

### Boolean

```python
self.assertTrue(x)              # bool(x) is True
self.assertFalse(x)             # bool(x) is False
```

### Comparison

```python
self.assertGreater(a, b)        # a > b
self.assertGreaterEqual(a, b)   # a >= b
self.assertLess(a, b)           # a < b
self.assertLessEqual(a, b)      # a <= b
```

### Membership

```python
self.assertIn(item, container)  # item in container
self.assertNotIn(item, container)
```

### Identity

```python
self.assertIs(a, b)             # a is b
self.assertIsNot(a, b)
```

### None

```python
self.assertIsNone(x)            # x is None
self.assertIsNotNone(x)
```

### Exceptions

```python
with self.assertRaises(ValueError):
    raise ValueError('error message')

with self.assertRaises(TypeError) as cm:
    invalid_operation()
self.assertEqual(str(cm.exception), 'expected message')
```

### Floating Point

```python
self.assertAlmostEqual(a, b, places=7)
self.assertNotAlmostEqual(a, b)
```

## Test Organization

### Test Discovery

unittest 自动发现测试：

```bash
python -m unittest discover
python -m unittest discover -s tests -p 'test_*.py'
```

### Test Modules

按模块组织测试：

```
project/
├── mymodule.py
└── tests/
    ├── __init__.py
    ├── test_mymodule.py
    └── test_other.py
```

### Test Suites

手动组织测试套件：

```python
def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestStringMethods('test_upper'))
    suite.addTest(TestStringMethods('test_split'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
```

## Running Tests

### Command Line

```bash
# 运行所有测试
python -m unittest

# 运行特定模块
python -m unittest test_module

# 运行特定类
python -m unittest test_module.TestClass

# 运行特定方法
python -m unittest test_module.TestClass.test_method

# 详细输出
python -m unittest -v test_module

# 停止在第一个失败
python -m unittest -f test_module
```

### Within Script

```python
if __name__ == '__main__':
    unittest.main()
```

## Best Practices

1. **One assertion per test**: 每个测试方法专注于一个行为
2. **Descriptive names**: 使用描述性的测试方法名
3. **Independent tests**: 测试之间不应有依赖
4. **Arrange-Act-Assert**: 组织测试代码结构
5. **Test edge cases**: 测试边界条件和异常情况

## Advanced Features

### Skipping Tests

```python
@unittest.skip('reason')
def test_feature(self):
    pass

@unittest.skipIf(condition, 'reason')
def test_feature(self):
    pass

@unittest.skipUnless(condition, 'reason')
def test_feature(self):
    pass
```

### Expected Failures

```python
@unittest.expectedFailure
def test_feature(self):
    pass
```

### Test Context Managers

```python
def test_context_manager(self):
    with self.assertLogs('logger', level='INFO') as cm:
        logging.getLogger('logger').info('message')
    self.assertIn('message', cm.output)
```

## Resources

### Detailed Assertion Reference

See [assertions.md](references/assertions.md) for complete list of all assertion methods with examples.

### Test Patterns and Best Practices

See [patterns.md](references/patterns.md) for:
- Testing design patterns
- Boundary testing strategies
- Exception testing patterns
- Mock and fixture usage

### Advanced Features

See [advanced.md](references/advanced.md) for:
- Mock objects and patching
- Test fixtures and setup inheritance
- Parameterized tests
- Test coverage integration
- Custom test runners
