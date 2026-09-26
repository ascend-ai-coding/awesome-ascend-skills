#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Bind optional positional and keyword values for stable script APIs."""

MISSING = object()


def bind_extra(args, kwargs, names, defaults):
    """Keep existing call forms while grouping related optional arguments."""
    if len(args) > len(names):
        raise TypeError("too many positional arguments")
    values = dict(zip(names, args))
    for name, value in kwargs.items():
        if name not in names:
            raise TypeError(f"unexpected keyword argument: {name}")
        if name in values:
            raise TypeError(f"multiple values for argument: {name}")
        values[name] = value
    for name, default in zip(names, defaults):
        if name not in values:
            if default is MISSING:
                raise TypeError(f"missing required argument: {name}")
            values[name] = default
    return tuple(values[name] for name in names)
