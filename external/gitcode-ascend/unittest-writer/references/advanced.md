# Unittest 高级特性

## 目录

- [Mock 对象和 Patching](#mock-对象和-patching)
- [测试 Fixtures 和 Setup 继承](#测试-fixtures-和-setup-继承)
- [参数化测试](#参数化测试)
- [测试覆盖率集成](#测试覆盖率集成)
- [自定义测试运行器](#自定义测试运行器)
- [测试发现和加载](#测试发现和加载)
- [测试结果处理](#测试结果处理)
- [并行测试执行](#并行测试执行)
- [测试资源管理](#测试资源管理)

## Mock 对象和 Patching

### 基础 Mock 对象

```python
import unittest.mock

class TestMockBasics(unittest.TestCase):
    def test_mock_creation(self):
        mock = unittest.mock.Mock()
        mock.method()
        mock.method.assert_called_once()

    def test_mock_return_value(self):
        mock = unittest.mock.Mock(return_value=42)
        result = mock()
        self.assertEqual(result, 42)

    def test_mock_side_effect(self):
        mock = unittest.mock.Mock(side_effect=[1, 2, 3])
        self.assertEqual(mock(), 1)
        self.assertEqual(mock(), 2)
        self.assertEqual(mock(), 3)

    def test_mock_side_effect_exception(self):
        mock = unittest.mock.Mock(side_effect=ValueError('error'))
        with self.assertRaises(ValueError):
            mock()
```

### Patch 装饰器

```python
class TestPatching(unittest.TestCase):
    @unittest.mock.patch('module.function')
    def test_patch_function(self, mock_function):
        mock_function.return_value = 'patched'
        result = module.function()
        self.assertEqual(result, 'patched')

    @unittest.mock.patch.object(Class, 'method')
    def test_patch_method(self, mock_method):
        mock_method.return_value = 'patched'
        obj = Class()
        result = obj.method()
        self.assertEqual(result, 'patched')

    @unittest.mock.patch.dict('os.environ', {'TEST_VAR': 'test_value'})
    def test_patch_dict(self):
        self.assertEqual(os.environ['TEST_VAR'], 'test_value')
```

### Patch 上下文管理器

```python
class TestPatchContextManager(unittest.TestCase):
    def test_patch_with_context_manager(self):
        with unittest.mock.patch('module.function') as mock_func:
            mock_func.return_value = 'patched'
            result = module.function()
            self.assertEqual(result, 'patched')

        # patch 在上下文结束后自动恢复
        self.assertNotEqual(module.function, 'patched')
```

### Mock 属性和方法

```python
class TestMockAttributes(unittest.TestCase):
    def test_mock_attributes(self):
        mock = unittest.mock.Mock()
        mock.attribute = 'value'
        self.assertEqual(mock.attribute, 'value')

    def test_mock_method_calls(self):
        mock = unittest.mock.Mock()
        mock.method1(1, 2)
        mock.method2(a=3, b=4)

        self.assertEqual(mock.method1.call_count, 1)
        mock.method1.assert_called_with(1, 2)
        mock.method2.assert_called_once_with(a=3, b=4)

    def test_mock_call_args(self):
        mock = unittest.mock.Mock()
        mock(1, 2, a=3)

        self.assertEqual(mock.call_args, ((1, 2), {'a': 3}))
        self.assertEqual(mock.call_args[0], (1, 2))
        self.assertEqual(mock.call_args[1], {'a': 3})
```

### MagicMock 和 Mock 的区别

```python
class TestMagicMock(unittest.TestCase):
    def test_magicmock_supports_magic_methods(self):
        mock = unittest.mock.MagicMock()
        self.assertEqual(len(mock), 0)
        self.assertEqual(mock[0], mock())
        self.assertEqual(iter(mock), iter([]))

    def test_mock_does_not_support_magic_methods(self):
        mock = unittest.mock.Mock()
        with self.assertRaises(TypeError):
            len(mock)
```

### Spec 和 Spec_set

```python
class TestMockSpec(unittest.TestCase):
    def test_mock_with_spec(self):
        class RealClass:
            def method1(self):
                pass
            def method2(self):
                pass

        mock = unittest.mock.Mock(spec=RealClass)
        mock.method1()
        mock.method2()

        with self.assertRaises(AttributeError):
            mock.method3()  # 不在 spec 中

    def test_mock_with_spec_set(self):
        class RealClass:
            attribute = 'value'

        mock = unittest.mock.Mock(spec_set=RealClass)
        mock.attribute = 'new_value'

        with self.assertRaises(AttributeError):
            mock.new_attribute = 'value'  # 不在 spec_set 中
```

## 测试 Fixtures 和 Setup 继承

### 多层 Setup

```python
class BaseTest(unittest.TestCase):
    def setUp(self):
        self.base_resource = create_base_resource()

    def tearDown(self):
        cleanup_base_resource(self.base_resource)


class DerivedTest(BaseTest):
    def setUp(self):
        super().setUp()
        self.derived_resource = create_derived_resource()

    def tearDown(self):
        cleanup_derived_resource(self.derived_resource)
        super().tearDown()

    def test_with_both_resources(self):
        self.assertIsNotNone(self.base_resource)
        self.assertIsNotNone(self.derived_resource)
```

### SetupClass 和 TearDownClass

```python
class TestWithClassLevelSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared_resource = create_shared_resource()

    @classmethod
    def tearDownClass(cls):
        cleanup_shared_resource(cls.shared_resource)

    def test_1(self):
        self.assertIsNotNone(self.shared_resource)

    def test_2(self):
        self.assertIsNotNone(self.shared_resource)
```

### SetupModule 和 TearDownModule

```python
def setUpModule():
    global module_resource
    module_resource = create_module_resource()

def tearDownModule():
    cleanup_module_resource(module_resource)

class TestInModule(unittest.TestCase):
    def test_access_module_resource(self):
        self.assertIsNotNone(module_resource)
```

## 参数化测试

### 使用 subTest

```python
class TestParameterized(unittest.TestCase):
    def test_addition(self):
        test_cases = [
            (1, 2, 3),
            (0, 0, 0),
            (-1, 1, 0),
            (100, 200, 300),
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, expected=expected):
                result = add(a, b)
                self.assertEqual(result, expected)
```

### 使用数据驱动测试

```python
class TestDataDriven(unittest.TestCase):
    test_data = [
        {'input': 'test1', 'expected': 'result1'},
        {'input': 'test2', 'expected': 'result2'},
        {'input': 'test3', 'expected': 'result3'},
    ]

    def test_process_data(self):
        for data in self.test_data:
            with self.subTest(data=data):
                result = process(data['input'])
                self.assertEqual(result, data['expected'])
```

### 使用测试生成器（高级）

```python
def generate_test_cases():
    test_cases = [
        ('case1', 1, 2, 3),
        ('case2', 0, 0, 0),
        ('case3', -1, 1, 0),
    ]

    for name, a, b, expected in test_cases:
        def test_func(self, a=a, b=b, expected=expected):
            self.assertEqual(add(a, b), expected)

        test_func.__name__ = f'test_{name}'
        setattr(TestGenerated, f'test_{name}', test_func)

class TestGenerated(unittest.TestCase):
    pass

generate_test_cases()
```

## 测试覆盖率集成

### 使用 coverage.py

```bash
# 安装 coverage
pip install coverage

# 运行测试并收集覆盖率
coverage run -m unittest discover

# 生成覆盖率报告
coverage report
coverage html
```

### 在测试中使用 coverage

```python
import coverage

class TestWithCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cov = coverage.Coverage()
        cls.cov.start()

    @classmethod
    def tearDownClass(cls):
        cls.cov.stop()
        cls.cov.save()
        cls.cov.report()
```

### 覆盖率配置

```ini
# .coveragerc
[run]
source = myapp
omit =
    */tests/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## 自定义测试运行器

### 自定义 TextTestRunner

```python
class CustomTestRunner(unittest.TextTestRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stream = sys.stdout

    def run(self, test):
        result = super().run(test)
        print(f"\nTests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        return result

if __name__ == '__main__':
    unittest.main(testRunner=CustomTestRunner)
```

### 自定义 TestResult

```python
class CustomTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)

    def printSummary(self):
        print(f"\nSuccesses: {len(self.successes)}")
        print(f"Failures: {len(self.failures)}")
        print(f"Errors: {len(self.errors)}")

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    runner = unittest.TextTestRunner(resultclass=CustomTestResult)
    result = runner.run(suite)
    result.printSummary()
```

## 测试发现和加载

### 自定义测试加载

```python
class CustomTestLoader(unittest.TestLoader):
    def loadTestsFromName(self, name, module=None):
        if name.startswith('slow_'):
            return unittest.TestSuite()
        return super().loadTestsFromName(name, module)

if __name__ == '__main__':
    loader = CustomTestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    runner = unittest.TextTestRunner()
    runner.run(suite)
```

### 按标签过滤测试

```python
def load_tests_by_tag(tag):
    loader = unittest.TestLoader()
    all_tests = loader.discover('tests')

    filtered_suite = unittest.TestSuite()
    for test_group in all_tests:
        for test in test_group:
            if hasattr(test, '_testMethodName'):
                test_method = getattr(test, test._testMethodName)
                if hasattr(test_method, 'tags') and tag in test_method.tags:
                    filtered_suite.addTest(test)

    return filtered_suite

def test_tag(*tags):
    def decorator(func):
        func.tags = tags
        return func
    return decorator

class TestTagged(unittest.TestCase):
    @test_tag('slow', 'integration')
    def test_slow_integration(self):
        pass

    @test_tag('fast', 'unit')
    def test_fast_unit(self):
        pass
```

## 测试结果处理

### 自定义测试结果格式化

```python
class JSONTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.results = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.results.append({
            'test': str(test),
            'status': 'success'
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.results.append({
            'test': str(test),
            'status': 'failure',
            'error': str(err[1])
        })

    def addError(self, test, err):
        super().addError(test, err)
        self.results.append({
            'test': str(test),
            'status': 'error',
            'error': str(err[1])
        })

    def toJSON(self):
        import json
        return json.dumps(self.results, indent=2)
```

### 生成 HTML 报告

```python
class HTMLTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.test_results = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_results.append({
            'name': str(test),
            'status': 'PASS',
            'message': ''
        })

    def generate_report(self, filename='test_report.html'):
        html = """
        <html>
        <head><title>Test Report</title></head>
        <body>
        <h1>Test Results</h1>
        <table>
        <tr><th>Test</th><th>Status</th><th>Message</th></tr>
        """

        for result in self.test_results:
            html += f"""
            <tr>
                <td>{result['name']}</td>
                <td>{result['status']}</td>
                <td>{result['message']}</td>
            </tr>
            """

        html += """
        </table>
        </body>
        </html>
        """

        with open(filename, 'w') as f:
            f.write(html)
```

## 并行测试执行

### 使用 multiprocessing

```python
import multiprocessing

def run_test_suite(test_suite):
    runner = unittest.TextTestRunner(stream=open(os.devnull, 'w'))
    result = runner.run(test_suite)
    return result

class ParallelTestRunner:
    def __init__(self, num_processes=None):
        self.num_processes = num_processes or multiprocessing.cpu_count()

    def run(self, test_suite):
        suites = list(test_suite)
        chunk_size = len(suites) // self.num_processes

        chunks = []
        for i in range(self.num_processes):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_processes - 1 else len(suites)
            chunk = unittest.TestSuite(suites[start:end])
            chunks.append(chunk)

        with multiprocessing.Pool(self.num_processes) as pool:
            results = pool.map(run_test_suite, chunks)

        return self._merge_results(results)

    def _merge_results(self, results):
        merged = unittest.TestResult()
        for result in results:
            merged.testsRun += result.testsRun
            merged.failures.extend(result.failures)
            merged.errors.extend(result.errors)
            merged.skipped.extend(result.skipped)
        return merged
```

## 测试资源管理

### 资源池

```python
class ResourcePool:
    def __init__(self, max_resources=5):
        self.pool = []
        self.max_resources = max_resources
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            if len(self.pool) < self.max_resources:
                resource = create_resource()
                self.pool.append(resource)
                return resource
            return self.pool.pop()

    def release(self, resource):
        with self.lock:
            self.pool.append(resource)

class TestWithResourcePool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resource_pool = ResourcePool()

    def test_with_resource(self):
        resource = self.resource_pool.acquire()
        try:
            result = perform_operation(resource)
            self.assertIsNotNone(result)
        finally:
            self.resource_pool.release(resource)
```

### 临时文件管理

```python
class TestWithTempFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_with_temp_file(self):
        temp_file = os.path.join(self.temp_dir, 'test.txt')
        with open(temp_file, 'w') as f:
            f.write('test content')

        result = process_file(temp_file)
        self.assertEqual(result, 'processed')
```

### 数据库测试工具

```python
class DatabaseTestHelper:
    def __init__(self, db_url=':memory:'):
        self.db_url = db_url
        self.connection = None

    def __enter__(self):
        self.connection = sqlite3.connect(self.db_url)
        self._setup_schema()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

    def _setup_schema(self):
        cursor = self.connection.cursor()
        cursor.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')
        self.connection.commit()

class TestWithDatabase(unittest.TestCase):
    def test_database_operations(self):
        with DatabaseTestHelper() as db:
            cursor = db.cursor()
            cursor.execute("INSERT INTO users (name) VALUES ('John')")
            db.commit()

            cursor.execute("SELECT * FROM users")
            result = cursor.fetchall()
            self.assertEqual(len(result), 1)
```

## 测试隔离和并发控制

### 使用锁保护共享资源

```python
import threading

class TestWithSharedResource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared_resource = SharedResource()
        cls.lock = threading.Lock()

    def test_concurrent_access(self):
        with self.lock:
            result = self.shared_resource.access()
            self.assertIsNotNone(result)
```

### 使用信号量限制并发

```python
import threading

class TestWithSemaphore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semaphore = threading.Semaphore(3)  # 最多3个并发

    def test_limited_concurrency(self):
        with self.semaphore:
            result = perform_limited_operation()
            self.assertIsNotNone(result)
```

## 测试超时控制

### 使用装饰器设置超时

```python
import signal
import functools

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(f'Test timed out after {seconds} seconds')

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)

            return result
        return wrapper
    return decorator

class TestWithTimeout(unittest.TestCase):
    @timeout(5)
    def test_slow_operation(self):
        result = slow_operation()
        self.assertIsNotNone(result)
```

## 测试数据生成

### 使用 Faker 生成测试数据

```python
from faker import Faker

class TestWithFaker(unittest.TestCase):
    def setUp(self):
        self.faker = Faker()

    def test_with_generated_data(self):
        user_data = {
            'name': self.faker.name(),
            'email': self.faker.email(),
            'address': self.faker.address()
        }

        result = process_user_data(user_data)
        self.assertTrue(result['valid'])
```

### 使用 hypothesis 进行属性测试

```python
from hypothesis import given, strategies as st

class TestWithHypothesis(unittest.TestCase):
    @given(st.integers(), st.integers())
    def test_commutative_property(self, a, b):
        self.assertEqual(add(a, b), add(b, a))

    @given(st.text(min_size=1))
    def test_string_properties(self, s):
        self.assertEqual(len(s * 2), len(s) * 2)
```
