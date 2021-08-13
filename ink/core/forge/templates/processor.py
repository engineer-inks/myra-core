from pyspark.sql import DataFrame, functions as F

import ink.core.forge.joins.core as C


class Rawing(C.processors.Rawing):
    """NAME_SIMPLE Rawing Processor.

    transient → raw

    Adds ingestion timestamp to the transient data
    and append it to the raw data.

    """
    SAVING_OPTIONS = {'mode': 'append'}

    def call(self, x: DataFrame):
        x = x.withColumn('ingested_at', F.current_timestamp())

        return x


class Trusting(C.processors.Trusting):
    """NAME_SIMPLE Trusting Processor.

    raw → trusted

    Discard samples without :code:`ids` and remove
    entry duplicates by removing the older ones.

    """
    SAVING_OPTIONS = {'mode': 'overwrite'}

    def call(self, x: DataFrame):
        x = x.where(x.id.isNotNull())
        x = self.discard_duplicates(x, 'id', 'created_at')

        return x


class Refining(C.processors.Refining):
    """NAME_SIMPLE Refining Processor.

    trusted → refined

    Sort data according to its :code:`id` and
    :code:`created_at` to improve reading performance.

    """
    SAVING_OPTIONS = {'mode': 'overwrite'}

    def call(self, x: DataFrame):
        x = x.orderBy('id', 'created_at')

        return x
