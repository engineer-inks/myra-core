"""Stream conformity functions.

Declares our main :code:`conform` function and its usual conformity
sub-functions. These are useful when trying to conform inconsistent
input user data.

"""
import re
from typing import Union, List, Optional, Callable

from pyspark.sql import functions as F, DataFrame

from ..utils import to_list, valid_or_default

O = Union[Callable, str]
O = Optional[Union[O, List[O]]]


# region conforming functions

def sort_cols(df: DataFrame) -> DataFrame:
    """Sort the columns in a spark DataFrame to increase consistency.

    :param df: the non-normalized DataFrame being read.
    :return: The frame with its columns sorted in alphabetic order.

    Examples
    --------
    .. jupyter-execute:: /examples/io/stream/sort_cols.py
    """
    return df.select(*sorted(df.columns))


def lower_cols(df: DataFrame) -> DataFrame:
    """Lower-case all columns in the first-level of a frame.

    :param df: the processing frame.
    :return: A frame in which all columns are lower-case.

    Examples
    --------
    .. jupyter-execute:: /examples/io/stream/lower_cols.py
    """
    cols = [c.lower() for c in df.columns]
    assert_same_number_of_cols(df.columns, cols, 'lower_cols')

    return df.select([F.col(c).alias(r) for c, r in zip(df.columns, cols)])


def simple_cols(df: DataFrame) -> DataFrame:
    """Simplify the columns in the first-level of a frame.

    All non-alphanumeric characters are replaced by "_" in this operation.
    Any leading/trailing underscores are removed.

    :param df: the processing frame.
    :return: A frame with simple column names.

    Examples
    --------
    .. jupyter-execute:: /examples/io/stream/lower_cols.py
    """
    r = re.compile('[^0-9a-zA-Z]+')
    cols = [r.sub('_', c).strip('_') for c in df.columns]
    assert_same_number_of_cols(df.columns, cols, 'simple_cols')

    return df.select([F.col(c).alias(r) for c, r in zip(df.columns, cols)])


# endregion


# region validators

def assert_same_number_of_cols(a: List[str],
                               b: List[str],
                               op_name: str):
    """Assert the number of columns is the same.

    Useful to make sure the frame was not tempered with by
    the application of a possibly-destructive operation.

    :param a: the first (old) collection of columns.
    :param b: the second (new) collection of columns.
    :param op_name: The name of the operation being performed.

    Examples
    --------
    .. jupyter-execute:: /examples/io/stream/assert_same_number_of_cols.py
        :raises: ValueError
    """
    a, b = set(a), set(b)

    if len(a) != len(b):
        raise ValueError(f'Illegal operation `{op_name}` performed, as the '
                         f'number of columns in the frame would change from '
                         f'{len(a)} to {len(b)}. Difference:\n'
                         f'  A - B: {a - b}\n'
                         f'  B - A: {b - a}.')


# endregion


CONFORMING_OPS = {
    'sort_cols': sort_cols,
    'simple_cols': simple_cols,
    'lower_cols': lower_cols,
}
"""Dict: Map to all available conforming functions."""

DEFAULT_OPS = ('lower_cols', 'simple_cols', 'sort_cols')
"""Tuple[str]: Tuple of default operations executed in :code:`conform`."""


def adapter(op: Union[str, Callable]) -> Callable:
    """Retrieve a known conforming operation if its name is passed.
    Otherwise, this will work as the identity function.

    :param op: the operation's name or function of interest.
    :return: The operation referenced.
    """
    global CONFORMING_OPS

    if callable(op):
        return op

    if isinstance(op, str):
        if op not in CONFORMING_OPS:
            raise ValueError(
                f'Conforming operation `{op}` not found in `CONFORMING_OPS`. '
                f'Available options are: {CONFORMING_OPS.items()}')
        return CONFORMING_OPS[op]

    raise ValueError(f'Cannot infer an appropriate adapter for operation `{op}`. '
                     f'Valid arguments are callables or {list(CONFORMING_OPS.keys())}.')


def conform(dfs: Union[DataFrame, List[DataFrame]],
            ops: O = DEFAULT_OPS) -> List[DataFrame]:
    """Conform {DataFrames} according to some operation.

    :param dfs: frame or list of frames to be conformed.
    :param ops: str, function or list of str/functions
        Operation or list of operations used to conform the frames.
        Defaults to all conforming operations :code:`CONFORMING_OPS`.

    :return: The sames frames passed as arguments, but conformed
        according to the selected functions.

    Examples
    --------
    .. jupyter-execute:: /examples/io/stream/conform.py
    """
    if not dfs:
        return dfs

    dfs = to_list(dfs)
    ops = to_list(valid_or_default(ops, DEFAULT_OPS))

    for op in ops:
        op = adapter(op)
        dfs = [op(d) for d in dfs]

    return dfs
