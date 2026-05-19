# Unittest 测试模式和最佳实践

## 目录

- [测试设计原则](#测试设计原则)
- [Arrange-Act-Assert 模式](#arrange-act-assert-模式)
- [边界测试策略](#边界测试策略)
- [异常测试模式](#异常测试模式)
- [测试隔离](#测试隔离)
- [测试数据管理](#测试数据管理)
- [Mock 和 Fixture 使用](#mock-和-fixture-使用)
- [测试命名规范](#测试命名规范)
- [测试组织结构](#测试组织结构)

## 测试设计原则

### 1. 单一职责

每个测试方法只验证一个行为：

```python
class TestCalculator(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(add(2, 3), 5)

    def test_subtraction(self):
        self.assertEqual(subtract(5, 3), 2)
```

### 2. 独立性

测试之间不应有依赖关系：

```python
class TestCounter(unittest.TestCase):
    def setUp(self):
        self.counter = Counter()

    def test_increment(self):
        self.counter.increment()
        self.assertEqual(self.counter.value, 1)

    def test_decrement(self):
        self.counter.decrement()
        self.assertEqual(self.counter.value, -1)
```

### 3. 可重复性

测试应该在任何环境下都能重复运行：

```python
class TestFileOperation(unittest.TestCase):
    def setUp(self):
        self.test_file = tempfile.NamedTemporaryFile(delete=False)
        self.test_file.close()

    def tearDown(self):
        os.unlink(self.test_file.name)

    def test_write_and_read(self):
        with open(self.test_file.name, 'w') as f:
            f.write('test content')

        with open(self.test_file.name, 'r') as f:
            content = f.read()

        self.assertEqual(content, 'test content')
```

### 4. 快速执行

测试应该快速运行，避免不必要的等待：

```python
class TestAPI(unittest.TestCase):
    @unittest.mock.patch('requests.get')
    def test_api_call(self, mock_get):
        mock_get.return_value.status_code = 200
        response = call_api('http://example.com')
        self.assertEqual(response.status_code, 200)
```

## Arrange-Act-Assert 模式

将测试分为三个清晰的阶段：

```python
def test_user_authentication(self):
    # Arrange - 准备测试数据和环境
    user = User(username='testuser', password='password123')
    db.add(user)
    db.commit()

    # Act - 执行被测试的操作
    result = authenticate('testuser', 'password123')

    # Assert - 验证结果
    self.assertTrue(result)
```

### 复杂示例

```python
def test_order_processing(self):
    # Arrange
    customer = Customer(name='John Doe', email='john@example.com')
    product = Product(name='Widget', price=10.0, stock=100)
    order = Order(customer=customer)

    # Act
    order.add_item(product, quantity=5)
    order.process()

    # Assert
    self.assertEqual(order.total, 50.0)
    self.assertEqual(product.stock, 95)
    self.assertEqual(order.status, 'processed')
```

## 边界测试策略

### 数值边界

```python
class TestRangeValidator(unittest.TestCase):
    def test_minimum_boundary(self):
        self.assertTrue(validate_range(0, min=0, max=100))

    def test_maximum_boundary(self):
        self.assertTrue(validate_range(100, min=0, max=100))

    def test_below_minimum(self):
        self.assertFalse(validate_range(-1, min=0, max=100))

    def test_above_maximum(self):
        self.assertFalse(validate_range(101, min=0, max=100))
```

### 字符串边界

```python
class TestStringValidator(unittest.TestCase):
    def test_empty_string(self):
        self.assertFalse(validate_non_empty(''))

    def test_single_character(self):
        self.assertTrue(validate_non_empty('a'))

    def test_whitespace_only(self):
        self.assertFalse(validate_non_empty('   '))
```

### 集合边界

```python
class TestCollectionValidator(unittest.TestCase):
    def test_empty_list(self):
        self.assertFalse(validate_non_empty_list([]))

    def test_single_element(self):
        self.assertTrue(validate_non_empty_list([1]))

    def test_multiple_elements(self):
        self.assertTrue(validate_non_empty_list([1, 2, 3]))
```

## 异常测试模式

### 基本异常测试

```python
def test_division_by_zero(self):
    with self.assertRaises(ZeroDivisionError):
        1 / 0
```

### 异常消息验证

```python
def test_invalid_input_exception(self):
    with self.assertRaises(ValueError) as cm:
        process_data('invalid')

    self.assertEqual(str(cm.exception), 'Invalid input data')
```

### 异常属性验证

```python
def test_custom_exception_attributes(self):
    with self.assertRaises(CustomError) as cm:
        raise CustomError(code=404, message='Not found')

    self.assertEqual(cm.exception.code, 404)
    self.assertEqual(cm.exception.message, 'Not found')
```

### 多种异常类型

```python
def test_multiple_exception_types(self):
    with self.assertRaises((ValueError, TypeError)):
        risky_operation()
```

## 测试隔离

### 使用 setUp 和 tearDown

```python
class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.connection = Database.connect(':memory:')
        self.connection.create_tables()

    def tearDown(self):
        self.connection.close()

    def test_insert_user(self):
        self.connection.insert('users', {'name': 'John'})
        result = self.connection.query('SELECT * FROM users')
        self.assertEqual(len(result), 1)
```

### 使用 setUpClass 和 tearDownClass

```python
class TestAPIClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = APIClient()
        cls.client.authenticate()

    @classmethod
    def tearDownClass(cls):
        cls.client.logout()

    def test_get_user(self):
        user = self.client.get_user(1)
        self.assertIsNotNone(user)
```

### 使用上下文管理器

```python
class TestFileProcessing(unittest.TestCase):
    def test_process_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test data')
            temp_path = f.name

        try:
            result = process_file(temp_path)
            self.assertEqual(result, 'processed')
        finally:
            os.unlink(temp_path)
```

## 测试数据管理

### 使用测试数据集

```python
class TestDataProcessing(unittest.TestCase):
    test_data = [
        ('input1', 'expected1'),
        ('input2', 'expected2'),
        ('input3', 'expected3'),
    ]

    def test_process_data(self):
        for input_data, expected in self.test_data:
            with self.subTest(input_data=input_data):
                result = process(input_data)
                self.assertEqual(result, expected)
```

### 使用参数化测试（通过循环）

```python
class TestMathOperations(unittest.TestCase):
    def test_addition(self):
        test_cases = [
            (1, 2, 3),
            (0, 0, 0),
            (-1, 1, 0),
            (100, 200, 300),
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b):
                self.assertEqual(add(a, b), expected)
```

### 使用工厂函数创建测试数据

```python
class TestUserManagement(unittest.TestCase):
    def create_test_user(self, **kwargs):
        defaults = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'password123'
        }
        defaults.update(kwargs)
        return User(**defaults)

    def test_user_creation(self):
        user = self.create_test_user(username='custom')
        self.assertEqual(user.username, 'custom')
```

## Mock 和 Fixture 使用

### Mock 函数调用

```python
import unittest.mock

class TestExternalAPI(unittest.TestCase):
    @unittest.mock.patch('requests.get')
    def test_fetch_data(self, mock_get):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_get.return_value = mock_response

        result = fetch_data('http://api.example.com/data')

        self.assertEqual(result, {'data': 'test'})
        mock_get.assert_called_once_with('http://api.example.com/data')
```

### Mock 类方法

```python
class TestDatabase(unittest.TestCase):
    @unittest.mock.patch('myapp.db.Database.connect')
    def test_database_connection(self, mock_connect):
        mock_connection = unittest.mock.Mock()
        mock_connect.return_value = mock_connection

        db = Database()
        db.connect()

        mock_connect.assert_called_once()
```

### 使用 Mock 对象

```python
class TestPaymentProcessor(unittest.TestCase):
    def test_payment_processing(self):
        mock_gateway = unittest.mock.Mock()
        mock_gateway.charge.return_value = {'status': 'success', 'transaction_id': '12345'}

        processor = PaymentProcessor(mock_gateway)
        result = processor.process_payment(amount=100, card='4111...')

        self.assertTrue(result['success'])
        mock_gateway.charge.assert_called_once_with(amount=100, card='4111...')
```

### 使用 Property Mock

```python
class TestConfiguration(unittest.TestCase):
    @unittest.mock.patch.object(Config, 'api_key', new_callable=unittest.mock.PropertyMock)
    def test_api_key_property(self, mock_api_key):
        mock_api_key.return_value = 'test-api-key'

        config = Config()
        self.assertEqual(config.api_key, 'test-api-key')
```

## 测试命名规范

### 描述性测试名称

```python
class TestUserAuthentication(unittest.TestCase):
    def test_user_can_authenticate_with_valid_credentials(self):
        pass

    def test_user_cannot_authenticate_with_invalid_password(self):
        pass

    def test_user_cannot_authenticate_with_nonexistent_username(self):
        pass
```

### 测试场景命名

```python
class TestOrderProcessing(unittest.TestCase):
    def test_order_processing_succeeds_with_valid_payment(self):
        pass

    def test_order_processing_fails_with_insufficient_stock(self):
        pass

    def test_order_processing_fails_with_expired_payment_method(self):
        pass
```

## 测试组织结构

### 按功能模块组织

```
tests/
├── __init__.py
├── test_auth.py
├── test_users.py
├── test_orders.py
└── test_products.py
```

### 按测试类型组织

```
tests/
├── __init__.py
├── unit/
│   ├── test_auth.py
│   └── test_users.py
├── integration/
│   ├── test_order_flow.py
│   └── test_payment_flow.py
└── e2e/
    └── test_user_journey.py
```

### 使用测试套件

```python
def create_test_suite():
    suite = unittest.TestSuite()

    # 添加特定测试
    suite.addTest(TestAuth('test_login'))
    suite.addTest(TestAuth('test_logout'))

    # 添加整个测试类
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUsers))

    # 添加整个模块
    suite.addTests(unittest.TestLoader().loadTestsFromModule(test_orders))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(create_test_suite())
```

## 性能测试模式

### 使用 timeit

```python
import timeit

class TestPerformance(unittest.TestCase):
    def test_function_performance(self):
        execution_time = timeit.timeit(
            lambda: expensive_function(),
            number=100
        )

        self.assertLess(execution_time, 1.0)
```

### 设置超时

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError()

class TestTimeout(unittest.TestCase):
    def test_function_timeout(self):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        try:
            long_running_function()
        finally:
            signal.alarm(0)
```

## 测试覆盖率最佳实践

### 覆盖关键路径

```python
class TestBusinessLogic(unittest.TestCase):
    def test_happy_path(self):
        result = process_order(valid_order)
        self.assertEqual(result.status, 'completed')

    def test_error_paths(self):
        with self.assertRaises(ValidationError):
            process_order(invalid_order)
```

### 测试边界条件

```python
class TestValidation(unittest.TestCase):
    def test_minimum_value(self):
        self.assertTrue(validate(0))

    def test_maximum_value(self):
        self.assertTrue(validate(100))

    def test_below_minimum(self):
        self.assertFalse(validate(-1))

    def test_above_maximum(self):
        self.assertFalse(validate(101))
```
