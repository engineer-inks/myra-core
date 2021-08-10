import os
from logging import warning
from typing import Dict, Union, Optional

import yaml
from pyspark.sql import SparkSession
from pyspark.sql.types import Row

from . import utils
import findspark

def spark(mode='default') -> SparkSession:
    """Retrieve current spark session.

    :param mode: str
        The session mode. Should be either "default" or "test".

    :return: SparkSession
    """
    if mode == 'default':
        return SparkSession.builder.getOrCreate()
    elif mode == 'test':
        return (SparkSession.builder
                .master('local[2]')
                .appName('my-local-testing-pyspark-context')
                .enableHiveSupport()
                .getOrCreate())
    elif mode == 'deltalake':
        findspark.init()
        return SparkSession.builder \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:0.7.0") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.databricks.delta.schema.autoMerge.enabled", "true") \
            .getOrCreate()
    else:
        raise ValueError(f'Illegal value "{mode}" mode parameter. '
                         'It should be either "default", "test" or "deltalake".')


class Config:
    """Encapsulates all config properties for an execution.

    Parameters
    ----------
    env: str, default=DEFAULT_ENV
        environment
    config_dir: str, optional
        path to config folder containing desired configuration.
        Whenever passed, the :func:`load` method will be invoked
        and all yaml config files within that folder will be loaded
        and added as attributes to the calling object.

    Examples
    --------
    .. jupyter-execute::

        from dextra.dna.core import Config
        config = Config('local')
        config.lakes.transient

    """

    DEFAULT_ENV = 'local'
    SPARK_MODE = 'default'
    CONFIGS_FOLDER = os.environ.get('CONFIGS_DIR', 'config')
    CONFIG_EXTENSIONS = ('.yml', '.yaml')

    lakes: Row
    logging: Row
    security: Row

    def __init__(self,
                 env: str = None,
                 config_dir: Optional[Union[str, bool]] = None,
                 spark_mode: str = None):
        self.env = (env
                    or os.environ.get('ENV')
                    or self.DEFAULT_ENV)
        self.config_dir = (config_dir
                           or os.environ.get('CONFIG_DIR')
                           or os.path.join(self.CONFIGS_FOLDER, self.env))
        if spark_mode:
            self.SPARK_MODE = spark_mode

        if self.config_dir:
            self.load(raise_errors=False)

    @property
    def spark(self) -> SparkSession:
        return spark(self.SPARK_MODE)

    def load(self, raise_errors: bool = True) -> 'Config':
        """Load configuration from a yaml file.

        Parameters
        ----------
        raise_errors: bool
            whether errors should be raised or ignored when
            reading the configuration items
        """

        if not os.path.exists(self.config_dir):
            m = f'Attempting to load config from "{self.config_dir}", but no configuration was found.'

            if raise_errors:
                raise ValueError(m)
            else:
                warning(m)

            return self

        for f in os.listdir(self.config_dir):
            if not self.is_config_file(f):
                continue

            with open(os.path.join(self.config_dir, f)) as fp:
                g = fp.read()

                try:
                    g = g.format_map(os.environ)
                except ValueError:
                    ...

                for k, v in yaml.safe_load(g).items():
                    setattr(self, k, utils.as_row(v))

        return self

    def is_config_file(self, f) -> bool:
        """Check if a file is a plausible configuration file.
        """
        _, ext = os.path.splitext(f)
        return ext in self.CONFIG_EXTENSIONS

    def __repr__(self):
        return (f'<{self.__module__}.{type(self).__name__} at {hex(id(self))}, '
                f'logging: {self.logging}, lakes: {self.lakes}>')


class Test(Config):
    """Test configuration commonly used during test executions.

    """
    DEFAULT_ENV = 'test'
    SPARK_MODE = 'test'
