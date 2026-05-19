"""
Pytest测试文件模板

这是一个标准的pytest测试文件模板，包含了常用的测试模式和结构。
根据实际需求修改和扩展此模板。
"""

import pytest


class TestBasicFunctionality:
    """基本功能测试类"""

    def test_simple_assertion(self):
        """简单断言测试"""
        result = 1 + 1
        assert result == 2

    def test_string_operations(self):
        """字符串操作测试"""
        text = "hello"
        assert text.upper() == "HELLO"
        assert len(text) == 5


class TestWithFixtures:
    """使用fixtures的测试类"""

    @pytest.fixture
    def sample_data(self):
        """fixture示例"""
        return {"name": "test", "value": 42}

    def test_with_fixture(self, sample_data):
        """使用fixture的测试"""
        assert sample_data["value"] == 42
        assert sample_data["name"] == "test"


class TestParametrized:
    """参数化测试类"""

    @pytest.mark.parametrize("input,expected", [
        (3, 4),
        (5, 6),
        (10, 11),
    ])
    def test_increment(self, input, expected):
        """参数化测试示例"""
        assert input + 1 == expected


class TestExceptions:
    """异常测试类"""

    def test_zero_division(self):
        """测试除零异常"""
        with pytest.raises(ZeroDivisionError):
            1 / 0

    def test_value_error(self):
        """测试值错误异常"""
        with pytest.raises(ValueError) as exc_info:
            int("invalid")
        assert "invalid literal" in str(exc_info.value)


class TestAAA:
    """AAA模式测试类"""

    def test_calculate_total_with_discount(self):
        """使用AAA模式的测试示例"""
        # Arrange - 准备测试数据和设置
        items = [{"price": 100}, {"price": 50}]
        discount = 0.1

        # Act - 执行被测试的功能
        total = sum(item["price"] for item in items) * (1 - discount)

        # Assert - 验证结果
        assert total == 135.0


class TestSkipAndXfail:
    """跳过和预期失败测试类"""

    @pytest.mark.skip(reason="功能未实现")
    def test_not_implemented(self):
        """跳过测试示例"""
        pass

    @pytest.mark.xfail
    def test_known_failure(self):
        """预期失败测试示例"""
        assert False


class TestFloatComparison:
    """浮点数比较测试类"""

    def test_float_approx(self):
        """使用pytest.approx进行浮点数比较"""
        result = 0.1 + 0.2
        assert result == pytest.approx(0.3)

    def test_float_with_tolerance(self):
        """带容差的浮点数比较"""
        assert 1.0001 == pytest.approx(1.0, rel=1e-4)


class TestBoundaryValues:
    """边界值测试类"""

    @pytest.mark.parametrize("value,expected", [
        (-1, False),
        (0, True),
        (1, True),
        (99, True),
        (100, False),
    ])
    def test_in_range(self, value, expected):
        """边界值测试示例"""
        def is_valid(val):
            return 0 <= val <= 99
        assert is_valid(value) == expected


class TestDictAndList:
    """字典和列表测试类"""

    def test_dict_comparison(self):
        """字典比较测试"""
        expected = {"a": 1, "b": 2, "c": 3}
        actual = {"a": 1, "b": 2, "c": 3}
        assert actual == expected

    def test_list_comparison(self):
        """列表比较测试"""
        expected = [1, 2, 3, 4, 5]
        actual = [1, 2, 3, 4, 5]
        assert actual == expected

    def test_dict_subset(self):
        """字典子集测试"""
        data = {"a": 1, "b": 2, "c": 3}
        assert "a" in data
        assert data["a"] == 1


class TestSetupTeardown:
    """设置和清理测试类"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """自动使用的fixture，用于设置和清理"""
        print("\nSetup: 准备测试环境")
        yield
        print("Teardown: 清理测试环境")

    def test_with_setup_teardown(self):
        """使用设置和清理的测试"""
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
