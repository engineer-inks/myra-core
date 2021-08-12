import os
from typing import Dict, Any, List, Optional

from pyspark.sql import functions as F

from .stream_adapter import StreamAdapter
from ink.core.templates.configs import spark

ALLOWED_INPUT_FILE_FORMATS = ('.csv', '.json', '.parquet')
DEFAULT_READING_OPTIONS = {
    'csv': {'header': True},
    'json': {},
    'parquet': {}
}


class FileStreamAdapter(StreamAdapter):
    def read(self,
             filename: str,
             **options: Dict[str, Any]) -> F.DataFrame:
        _, ext = os.path.splitext(filename)

        if ext not in ALLOWED_INPUT_FILE_FORMATS:
            raise ValueError(f'Format {ext} is not permitted as input. '
                             f'Allowed values are: {ALLOWED_INPUT_FILE_FORMATS}')

        reader = spark().read
        options = options or DEFAULT_READING_OPTIONS[ext[1:]]

        for k, v in options.items():
            reader = reader.option(k, v)

        return reader.format(ext[1:]).load(filename)

    def write(self,
              df: F.DataFrame,
              filename: str,
              mode: str = 'overwrite',
              coalesce: Optional[int] = None,
              partitions: List[str] = None):
        if coalesce:
            df = df.coalesce(coalesce)

        writer = df.write
        writer = writer.mode(mode)

        if partitions:
            writer.partitionBy(partitions)

        writer.parquet(filename)
