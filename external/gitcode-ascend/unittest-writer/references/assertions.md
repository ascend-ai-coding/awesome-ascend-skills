# Unittest 断言方法完整参考

## 目录

- [相等性断言](#相等性断言)
- [布尔断言](#布尔断言)
- [比较断言](#比较断言)
- [成员断言](#成员断言)
- [同一性断言](#同一性断言)
- [None 断言](#none-断言)
- [异常断言](#异常断言)
- [警告断言](#警告断言)
- [日志断言](#日志断言)
- [正则表达式断言](#正则表达式断言)
- [多类型断言](#多类型断言)
- [字符串断言](#字符串断言)
- [集合断言](#集合断言)
- [字典断言](#字典断言)
- [浮点数断言](#浮点数断言)

## 相等性断言

### assertEqual(a, b, msg=None)

验证 `a == b`

```python
self.assertEqual(1 + 1, 2)
self.assertEqual('hello', 'hello')
self.assertEqual([1, 2], [1, 2])
```

### assertNotEqual(a, b, msg=None)

验证 `a != b`

```python
self.assertNotEqual(1, 2)
self.assertNotEqual('hello', 'world')
```

## 布尔断言

### assertTrue(expr, msg=None)

验证 `bool(expr) is True`

```python
self.assertTrue(5 > 3)
self.assertTrue([1, 2, 3])
self.assertTrue('hello')
```

### assertFalse(expr, msg=None)

验证 `bool(expr) is False`

```python
self.assertFalse(5 < 3)
self.assertFalse([])
self.assertFalse('')
```

## 比较断言

### assertGreater(a, b, msg=None)

验证 `a > b`

```python
self.assertGreater(5, 3)
self.assertGreater('b', 'a')
```

### assertGreaterEqual(a, b, msg=None)

验证 `a >= b`

```python
self.assertGreaterEqual(5, 5)
self.assertGreaterEqual(5, 3)
```

### assertLess(a, b, msg=None)

验证 `a < b`

```python
self.assertLess(3, 5)
self.assertLess('a', 'b')
```

### assertLessEqual(a, b, msg=None)

验证 `a <= b`

```python
self.assertLessEqual(3, 3)
self.assertLessEqual(3, 5)
```

## 成员断言

### assertIn(member, container, msg=None)

验证 `member in container`

```python
self.assertIn(2, [1, 2, 3])
self.assertIn('a', 'abc')
self.assertIn('key', {'key': 'value'})
```

### assertNotIn(member, container, msg=None)

验证 `member not in container`

```python
self.assertNotIn(4, [1, 2, 3])
self.assertNotIn('d', 'abc')
```

## 同一性断言

### assertIs(expr1, expr2, msg=None)

验证 `expr1 is expr2`

```python
a = [1, 2, 3]
b = a
self.assertIs(a, b)

self.assertIs(None, None)
self.assertIs(True, True)
```

### assertIsNot(expr1, expr2, msg=None)

验证 `expr1 is not expr2`

```python
a = [1, 2, 3]
b = [1, 2, 3]
self.assertIsNot(a, b)
```

## None 断言

### assertIsNone(expr, msg=None)

验证 `expr is None`

```python
self.assertIsNone(None)
self.assertIsNone(function_returning_none())
```

### assertIsNotNone(expr, msg=None)

验证 `expr is not None`

```python
self.assertIsNotNone(0)
self.assertIsNotNone('')
self.assertIsNotNone([])
```

## 异常断言

### assertRaises(exception, callable, *args, **kwargs)

验证调用 `callable(*args, **kwargs)` 时抛出指定异常

```python
def raises_value_error():
    raise ValueError('error message')

self.assertRaises(ValueError, raises_value_error)

self.assertRaises(ZeroDivisionError, lambda: 1 / 0)
```

### assertRaisesRegex(exception, regex, callable, *args, **kwargs)

验证抛出异常且异常消息匹配正则表达式

```python
def raises_value_error():
    raise ValueError('error code: 123')

self.assertRaisesRegex(ValueError, r'error code: \d+', raises_value_error)
```

### assertRaises 作为上下文管理器

```python
with self.assertRaises(ValueError) as cm:
    raise ValueError('error message')

self.assertEqual(str(cm.exception), 'error message')
```

## 警告断言

### assertWarns(warning, callable, *args, **kwargs)

验证调用时抛出指定警告

```python
import warnings

def raises_warning():
    warnings.warn('warning message', UserWarning)

self.assertWarns(UserWarning, raises_warning)
```

### assertWarnsRegex(warning, regex, callable, *args, **kwargs)

验证抛出警告且消息匹配正则表达式

```python
def raises_warning():
    warnings.warn('warning code: 123', UserWarning)

self.assertWarnsRegex(UserWarning, r'warning code: \d+', raises_warning)
```

## 日志断言

### assertLogs(logger=None, level=None)

验证日志消息被记录

```python
import logging

def test_logging(self):
    with self.assertLogs('my_logger', level='INFO') as cm:
        logging.getLogger('my_logger').info('test message')

    self.assertEqual(cm.output, ['INFO:my_logger:test message'])
    self.assertEqual(cm.records[0].getMessage(), 'test message')
```

## 正则表达式断言

### assertRegex(text, regex, msg=None)

验证 `regex` 匹配 `text`

```python
self.assertRegex('hello world', r'hello.*world')
self.assertRegex('test@example.com', r'[\w.]+@[\w.]+')
```

### assertNotRegex(text, regex, msg=None)

验证 `regex` 不匹配 `text`

```python
self.assertNotRegex('hello world', r'goodbye')
```

## 多类型断言

### assertMultiLineEqual(a, b, msg=None)

比较多行字符串，显示差异

```python
a = '''line 1
line 2
line 3'''

b = '''line 1
line 2
line 3'''

self.assertMultiLineEqual(a, b)
```

### assertSequenceEqual(seq1, seq2, msg=None, seq_type=None)

比较两个序列

```python
self.assertSequenceEqual([1, 2, 3], [1, 2, 3])
self.assertSequenceEqual((1, 2, 3), (1, 2, 3))
```

### assertListEqual(list1, list2, msg=None)

比较两个列表

```python
self.assertListEqual([1, 2, 3], [1, 2, 3])
```

### assertTupleEqual(tuple1, tuple2, msg=None)

比较两个元组

```python
self.assertTupleEqual((1, 2, 3), (1, 2, 3))
```

### assertSetEqual(set1, set2, msg=None)

比较两个集合（不考虑顺序）

```python
self.assertSetEqual({1, 2, 3}, {3, 2, 1})
```

## 字符串断言

### assertCountEqual(first, second, msg=None)

验证两个序列包含相同元素（不考虑顺序）

```python
self.assertCountEqual([1, 2, 3], [3, 2, 1])
self.assertCountEqual('abc', 'cba')
```

## 集合断言

### assertDictEqual(d1, d2, msg=None)

比较两个字典

```python
self.assertDictEqual({'a': 1, 'b': 2}, {'a': 1, 'b': 2})
```

## 字典断言

### assertDictContainsSubset(subset, dictionary, msg=None)

验证 `subset` 中的所有键值对都在 `dictionary` 中

```python
self.assertDictContainsSubset({'a': 1}, {'a': 1, 'b': 2})
```

## 浮点数断言

### assertAlmostEqual(a, b, places=7, msg=None, delta=None)

验证 `round(a - b, 7) == 0`

```python
self.assertAlmostEqual(1.0000001, 1.0)
self.assertAlmostEqual(0.12345678, 0.12345679, places=7)
```

使用 `delta` 参数：

```python
self.assertAlmostEqual(1.0, 1.1, delta=0.2)
```

### assertNotAlmostEqual(a, b, places=7, msg=None, delta=None)

验证 `round(a - b, 7) != 0`

```python
self.assertNotAlmostEqual(1.0, 1.1)
```

## 自定义消息

所有断言方法都支持 `msg` 参数，用于自定义失败消息：

```python
self.assertEqual(1, 2, msg='1 should equal 2')
self.assertTrue(False, msg='This should be True')
```

## 断言别名

以下断言方法有别名：

- `assertEqual` ≡ `assertLongEqual`
- `assertNotEqual` ≡ `assertNotLongEqual`
- `assertRaisesRegex` ≡ `assertRaisesRegexp`
- `assertRegex` ≡ `assertRegexpMatches`
- `assertNotRegex` ≡ `assertNotRegexpMatches`
