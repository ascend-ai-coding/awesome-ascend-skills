# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""属性值解析单测：_resolve_kwargs（stage10 oracle kwargs）+ _coerce_attr_value（synthesize 属性字面量）。

锁定两条新逻辑：
  * _resolve_kwargs：单个 attr 占位符保留原生类型（int/float），复合串仍 stringify
  * _coerce_attr_value：list/tuple 字面量采纳，数值型字符串属性不误转
"""
from __future__ import annotations

from evaluators.formula_oracle_equiv import _resolve_kwargs
from evaluators.stages import _coerce_attr_value


# ---------- _resolve_kwargs -------------------------------------------------

def test_resolve_kwargs_single_int_attr_keeps_native_type():
    """单个 int 属性占位符 → 原生 int（非字符串），否则 torch.softmax(dim="-1") 会 TypeError。"""
    spec = {"attributes": [{"name": "dim", "default": -1}]}
    oracle = {"kwargs": {"dim": "${attr.dim}"}}
    out = _resolve_kwargs(spec, oracle, {})
    assert out == {"dim": -1}
    assert isinstance(out["dim"], int) and not isinstance(out["dim"], bool)


def test_resolve_kwargs_composite_placeholder_stringified():
    """占位符嵌入复合串 → 仍 stringify（无法保留原生类型）。"""
    spec = {"attributes": [{"name": "a", "default": 3}]}
    oracle = {"kwargs": {"k": "pre_${attr.a}_post"}}
    assert _resolve_kwargs(spec, oracle, {}) == {"k": "pre_3_post"}


def test_resolve_kwargs_float_attr_native():
    spec = {"attributes": [{"name": "scale", "default": 1.5}]}
    oracle = {"kwargs": {"scale": "${attr.scale}"}}
    out = _resolve_kwargs(spec, oracle, {})
    assert out == {"scale": 1.5}
    assert isinstance(out["scale"], float)


def test_resolve_kwargs_missing_default_falls_back():
    spec = {"attributes": [{"name": "dim"}]}   # 无 default
    oracle = {"kwargs": {"dim": "${attr.dim}"}}
    out = _resolve_kwargs(spec, oracle, {})
    assert out == {"dim": 0}   # 单占位符兜底为原生 0


# ---------- _coerce_attr_value ---------------------------------------------

def test_coerce_list_literal_adopted():
    """list 属性在 synthesize 以字符串 '[2,2]' 书写 → 解析为原生 list（split_size 场景）。"""
    v = _coerce_attr_value("[2,2]")
    assert v == [2, 2]
    assert isinstance(v, list)


def test_coerce_numeric_string_preserved():
    """数值型字符串属性不得误转为 int（字符串属性 '123' 应保持 str）。"""
    v = _coerce_attr_value("123")
    assert v == "123"
    assert isinstance(v, str)


def test_coerce_non_literal_string_preserved():
    assert _coerce_attr_value("fast") == "fast"
    assert _coerce_attr_value("NCHW") == "NCHW"


def test_coerce_scalar_passthrough():
    assert _coerce_attr_value(0) == 0
    assert _coerce_attr_value(1.5) == 1.5
    assert _coerce_attr_value(True) is True
