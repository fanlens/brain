#!/usr/bin/env python
# -*- coding: utf-8 -*-
import typing
from collections import defaultdict


def _output_list(xs: typing.List[str]) -> typing.List[str]:
    return xs


def _output_str(xs: typing.List[str]) -> typing.AnyStr:
    return ' '.join(xs)


def _output_dict(xs: typing.List[str]) -> typing.Dict[str, int]:
    output = defaultdict(int)
    for token in xs:
        output[token] += 1
    return output


_type_to_fun = {
    list: _output_list,
    str: _output_str,
    dict: _output_dict
}


def output(type: typing.Union[list, str, dict], xs: typing.List[str]) -> typing.Union[
    typing.List[str], typing.AnyStr, typing.Dict[str, int]]:
    assert type in _type_to_fun
    return _type_to_fun[type](xs)
