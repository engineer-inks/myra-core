import logging
from typing import Union, Optional, Iterable
from unittest import TestCase

from pyspark.sql import SparkSession, DataFrame, Column, functions as F


from ink.core.forge.joins.core import configs
from ink.core.forge.joins.core.configs import Config
from .utils import to_list


class SparkTestCase(TestCase):
    """A unit test case with spark setup.

    Examples
    --------
    .. code-block:: python

        import dextra.dna.core as C

        class SimpleTestCase(C.testing.SparkTestCase):
            def test_sanity(self):
                p = SimpleProcessor(..., config=self.config)
                y = p.call(x)

                self.assertIsNotNone(y)

            def test_expected_output(self):
                p = SimpleProcessor(..., config=self.config)
                y = p.call(x)

                self.assertSchemaMatches(y, {
                  'name': 'str',
                  'age': 'int',
                  'aliases': 'array<str>',
                  'comments': {
                    'title': 'str',
                    'created_at': 'timestamp'
                  }
                })

    """
    spark: SparkSession
    config: Config

    @classmethod
    def setUpClass(cls):
        cls.setUpSpark()

    @classmethod
    def tearDownClass(cls):
        cls.tearDownSpark()

    @classmethod
    def setUpSpark(cls):
        logging.getLogger('py4j').setLevel(logging.WARN)

        cls.config: Config = configs.Test()
        cls.spark: SparkSession = cls.config.spark

    @classmethod
    def tearDownSpark(cls):
        cls.spark.stop()

    def assertSchemaMatches(self, f: DataFrame, schema):
        kinds = dict(f.dtypes)

        for field, kind in schema.items():
            with self.subTest(f'testing integrity of {field}:{kind}'):
                self.assertIn(field, kinds, msg=f'{field} is not a field of the frame.')

                if isinstance(kind, str):  # int, string, array<string>
                    self.assertEqual(kinds[field], kind, msg=f'Field {field}\'s type does not match.')

                if isinstance(kind, dict):  # {'name': 'string', 'age': 'int'}
                    self.assertSchemaMatches(f.select(f'{field}.*'), kind)

                if isinstance(kind, list):  # [{'name': 'string', 'age': 'int'}]
                    self.assertSchemaMatches(f.select(F.explode(field)).select('col.*'), kind[0])


class DataConsistencyTestCase:
    """Util for input data consistency checking.

    Parameters
    ----------
    inputs: frames or list of frames
        input data being tested
    config: Config
        current configuration, usually coming from the job or a property of
        processor

    """
    def __init__(self, inputs=(), config: Config = None):
        self.inputs = inputs
        self.config = config

    def assert_contains(self, x: DataFrame, columns: Iterable):
        """Assert that a DataFrame contains at least the given columns.

        Parameters
        ----------
        x: DataFrame
           the DataFrame being tested
        columns: list of strings
            the expected columns within the frame
        """
        expected = set(columns)
        actual = set(x.columns)

        assert not expected - actual, f'The following columns are missing: {expected - actual}. Available: {actual}.'

    def assert_no_null_values(self,
                              x: DataFrame,
                              column: Union[str, Column, Iterable[str]]):
        """Assert that a DataFrame's column does not contain any null values.

        Parameters
        ----------
        x: DataFrame
           the DataFrame being tested
        column: str or list of strings
            the column being tested. If a list of columns is passed, they are
            interpreted as a single entry and the composition will be different than null
            if any of its parts is different
        """
        column = (F.col(f) if isinstance(f, str) else f
                  for f in to_list(column))

        for c in column:
            x = x.filter(c.isNull())

        assert not x.count(), f'{column} column contains null values.'

    def assert_parsable_time(self,
                             x: DataFrame,
                             column: Union[str, Column],
                             fmt: Optional[str] = None):
        """Assert a timestamp string column can be parsed using a given format.

        Parameters
        ----------
        x: DataFrame
           the DataFrame being tested
        column: str or Column
            the column being tested
        fmt: str
            the expected format for {column}'s values
        """
        if isinstance(column, str):
            column = F.col(column)

        invalids = x.where(column.isNotNull() & F.to_timestamp(column, fmt).isNull()).count()

        assert not invalids, f'{column} contains invalid formatted dates.'


__all__ = ['SparkTestCase', 'DataConsistencyTestCase']
