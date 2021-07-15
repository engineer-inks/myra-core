from pyspark.sql import functions as F


def anonymize(col: F.Column,
              token: str = '',
              bits: int = 256,
              length: int = None) -> F.Column:
    col = F.col(col).cast('string')
    col = F.concat(col, F.lit(token)) if token else col
    col = F.sha2(col, bits)
    col = F.substring(col, 0, length) if length else col

    return col
