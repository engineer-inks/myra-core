import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from .stream_adapter import StreamAdapter
from ...configs import spark


def _warn_unused_filename(filename):
    if filename is not None:
        logging.warning(
            f'Filename value "{filename}" passed to `stream#write` method '
            f'is ignored, as the distributed dataframe is being simply '
            f'converted into a pandas DataFrame.')


class MemoryStreamAdapter(StreamAdapter):
    def read(self,
             filename: Union[str, F.DataFrame, pd.DataFrame, np.ndarray],
             columns: List[str] = None) -> F.DataFrame:
        if isinstance(filename, F.DataFrame):
            return filename

        if isinstance(filename, np.ndarray):
            if not columns:
                columns = [f'_c{i}' for i in range(filename.shape[1])]
            filename = pd.DataFrame(filename, columns=columns)

        if isinstance(filename, pd.DataFrame):
            return spark().createDataFrame(filename)

    def write(self,
              df: F.DataFrame,
              filename: str,
              mode: str = 'overwrite',
              coalesce: Optional[int] = None,
              partitions: List[str] = None):
        _warn_unused_filename(filename)
        return df.toPandas()
