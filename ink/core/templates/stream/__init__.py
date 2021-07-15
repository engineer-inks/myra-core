from functools import reduce, partial
from typing import Union, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from .conforming import conform
from ..base import adapter, split_protocol_path
from ..utils import to_list, unpack, is_list

S = Union[str, F.DataFrame]
S = Union[S, List[S]]


_ADAPTERS_AVAILABLE = {
    'mem': ('ink.core.templates.stream.adapters.memory', 'MemoryStreamAdapter'),
    'file': ('ink.core.templates.stream.adapters.file', 'FileStreamAdapter'),
    'bigquery': ('ink.core.templates.stream.adapters.big_query', 'BigQueryStreamAdapter')
}
_ADAPTERS = {}

adapter = partial(adapter,
                  adapters=_ADAPTERS,
                  adapters_available=_ADAPTERS_AVAILABLE)


def read(sources: S,
         **options: Dict[str, Any]) -> F.DataFrame:
    """Read heterogeneous sources into spark frames.

    Paths, frames, lists, pandas and numpy arrays are supported
    as well and will be converted into dataframes.

    Parameters
    ----------
    sources: str, list-like of str
        Path or list of paths to sources.
    options: \**kwargs
        Used as options during the read operation.
        It will vary according to the type of source being read.
        If the data is already loaded in memory (pandas, numpy, DataFrame),
        these options are ignored.

    Returns
    -------

    DataFrame
        The input stream read, encapsulated into a spark dataframe.

    Examples
    --------
      >>> C.io.stream.read(['gs://bucket/data.csv', 'file:///base.parquet'])
      [DataFrame, DataFrame]
      >>> C.io.stream.read('file:///e.json')
      DataFrame
      >>> C.io.stream.read(pd.read_csv(...))
      DataFrame

    """
    many = is_list(sources)
    sources = to_list(sources)
    loaded = []

    for s in sources:
        if isinstance(s, (F.DataFrame, pd.DataFrame, np.ndarray)):
            frame = adapter('mem').read(s, **options)
        else:
            protocol, filename = split_protocol_path(s)
            frame = adapter(protocol).read(s, **options)

        loaded.append(frame)

    return loaded if many else unpack(loaded)


def write(df: F.DataFrame,
          filename: str,
          **options: Dict[str, Any]):
    """Write a spark dataframe into a stream.

    Parameters
    ----------
    df: DataFrame
        frame that will be saved

    filename: str
        location for the stream, a file topic or table name

    options: \**kwargs
        Used as options during the write operation.
        It will vary according to the type of source being written.

    Examples
    --------
    >>> C.io.stream.write(C.datasets.iris(), 'gs://bucket/iris.parquet')

    """
    protocol, _ = split_protocol_path(filename)
    return adapter(protocol).write(df, filename, **options)


def merge(dfs: List[F.DataFrame]) -> Optional[F.DataFrame]:
    """Merge multiple frames into a single one.
    """
    return (reduce(F.DataFrame.unionByName, dfs)
            if dfs
            else None)


__all__ = ['read', 'write', 'merge', 'conform']
