#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Home to different feature extraction modules"""
from typing import Iterable, List, Dict, Union, Type, DefaultDict, Any, cast


def _output_list(xs: Iterable[str]) -> List[str]:
    return list(xs)


def _output_str(xs: Iterable[str]) -> str:
    return ' '.join(xs)


def _output_dict(xs: Iterable[str]) -> Dict[str, int]:
    output_dict = DefaultDict(int)  # type: DefaultDict[str, int]
    for x in xs:
        output_dict[x] += 1
    return output_dict


TOutputType = Union[Type[list], Type[str], Type[dict]]
TOutput = Union[List[str], str, Dict[str, int]]


def is_output_type(output_type: TOutputType) -> bool:
    """
    :param output_type: output type to check
    :return: a valid output type?
    """
    return output_type in (list, str, dict)


def to_output_type(output_type: Any) -> TOutputType:
    """
    Transform untyped variable (e.g. an `Any` from a dict) into a `TOutputType`
    :param output_type: untyped output type
    :return: correctly typed output type
    :raises ValueError: if output type is invalid
    """
    if not is_output_type(output_type):
        raise ValueError('Not a valid output type')
    return cast(TOutputType, output_type)


def output(output_type: TOutputType, xs: Iterable[str]) -> TOutput:
    """
    :param output_type: type of output to use
    :param xs: list of output values
    :return: output transformed to desired type
    """
    # no singledispatch since type of list etc is always 'type'
    if output_type is list:
        return _output_list(xs)
    elif output_type is str:
        return _output_str(xs)
    elif output_type is dict:
        return _output_dict(xs)
    else:
        raise NotImplementedError('Not a valid output type')


if __name__ == "__main__":
    print(output(list, ['hello', 'world']))
