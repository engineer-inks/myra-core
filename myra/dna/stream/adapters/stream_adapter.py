import abc
from typing import Dict, Any

from pyspark.sql import DataFrame


class StreamAdapter(metaclass=abc.ABCMeta):
    """Abstract stream environment adapter."""

    def read(self,
             filename: str,
             **options: Dict[str, Any]) -> DataFrame:
        """Reads a file from a stream.

        Parameters
        ----------
        filename : str

        options : dict
            Used as options during the read operation.
            It will vary according to the type of source being read.
            If the data is already loaded in memory (pandas, numpy,
            DataFrame), these options are ignored.

        Returns
        -------
        pyspark.sql.DataFrame

        Raises
        ------
        FileNotFoundError
            If file does not exist in storage.

        """
        raise NotImplementedError

    def write(self,
              filename: str,
              df: DataFrame,
              **options: Dict[str, Any]):
        """Writes a spark dataframe into a stream.

        Parameters
        ----------
        filename : str
            the location of the stream, can be a file, topic, table etc
        df: DataFrame
            frame to be written into the stream
        options : dict
            Used as options during the write operation. It will
            vary according to the type of source being written.

        """
        raise NotImplementedError
