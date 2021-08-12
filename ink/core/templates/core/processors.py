"""Describe The Different Processor Classes.
"""

import abc
import logging
from typing import List, Optional, Union, Dict

from pyspark.sql import functions as F

from ink.core.templates.core import models, io, security
from ink.core.templates import configs
from .utils import to_list, stopwatch, NamedEntityMixin

I = Union[str, F.DataFrame, Dict[str, F.DataFrame]]


class Processor(NamedEntityMixin, metaclass=abc.ABCMeta):
    """Base Processor Class for Spark DataFrames.

    Parameters
    ----------
    inputs: I
        input stream or dictionary of input streams
    outputs: str
        saving location for the DataFrame processed
    config: Config
        the configuration under which this processor should run

    Attributes
    ----------
    loaded: DataFrame or map of frames
        Input frame (or map of frames) that store the loaded
        frames referenced by the paths passed in :code:`inputs`.
    processed: DataFrame, optional
        :code:`pyspark.sql.DataFrame` containing the result of
        the operations applied by this processor.
        This is set internally and should only be passed when testing.
    """

    SAVING_OPTIONS = {'mode': 'overwrite'}
    """Options passed to the :func:`io.stream.write` function.
    Through this option, one can change important aspects of how the data
    is persisted, such as partitioning schema or writing mode.
    """

    def __init__(self,
                 inputs: I,
                 outputs: str,
                 config: configs.Config = None,
                 loaded: Optional[I] = None,
                 processed: Optional[F.DataFrame] = None,
                 watches: Dict[str, stopwatch] = None):
        self.inputs = inputs
        self.outputs = outputs

        self.loaded = loaded
        self.processed = processed
        self.watches = watches or {}
        self.config = config

    @property
    def spark(self):
        """Retrieve the current spark session.

        Returns
        -------
        pyspark.sql.sessions.SparkSession
            Alias for the spark session contained within
            this processor's :code:`config`
        """
        return self.config.spark

    def setup(self, config=None) -> 'Processor':
        """Setup the processor for utilization.

        All custom setup should be done by re-implementing this method.
        """
        if config:
            self.config = config
        return self

    def perform(self) -> 'Processor':
        """Load the input data, if necessary, and apply the operations defined
           in the :func:`call` method to them. Finally, attributes the result
           to the :code:`processed` property.
        """
        logging.debug(f'run proc {self.fullname()}')

        with stopwatch(mode='silent') as w:
            self.load()

        self.watches['load'] = w

        with stopwatch(mode='silent') as w:
            if len(self.loaded) == 1:
                # Single inputs are passed as positional arguments.
                self.processed = self.call(*self.loaded.values())
            else:
                self.processed = self.call(**self.loaded)
        
        self.watches['proc'] = w

        return self

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abc.abstractmethod
    def call(self, *args, **kwargs) -> F.DataFrame:
        """Defer to actual logic for data stream processing.

        This method must be implemented by the user in order to have any effect.
        The input is automatically unpacked in this method, so its override
        should either:

        1. have exactly one positional argument if it
           will be fed with a single input
        2. have arguments that match exactly the input dict keys

        Returns
        -------
        DataFrame
            the processed data stream
        """
        raise NotImplementedError

    def describe(self,
                 mode: str = 'simple') -> 'Processor':
        """Describe this processor with respect to its inputs and output.

        Parameters
        ----------
        mode: str
            the mode in which the description happens. Options are
            'simple' or 'explain'
        """

        describe_t = (self.config.logging.processors.describe
                      if 'describe' in self.config.logging.processors
                      else '%(processor)s')
        profile_t = (self.config.logging.processors.profile
                     if 'profile' in self.config.logging.processors
                     else 'load, proc and save time: %(load_t)s %(proc_t)s %(save_t)s')
        describe_c = {'processor': self.fullname(), 'inputs': self.inputs,
                      'proc': self.processed or '?', 'outputs': self.outputs}
        profile_c = {'load_t': '?', 'proc_t': '?', 'save_t': '?',
                     'started_at': (self.watches['load'].started_at
                                    if 'load' in self.watches
                                    else '?')}
        profile_c.update({f'{k}_t': round(v.elapsed, 3) for k, v in self.watches.items()})

        print(describe_t % describe_c, '',
              profile_t % profile_c,
              sep='\n', end='\n\n')

        if mode == 'explain' and self.processed:
            self.processed.explain()

        return self

    def load(self) -> 'Processor':
        """Load a processor's input into its space.

        :code:`Dict[str, DataFrame]` and :code:`DataFrame` are
        normalized into a single dictionary representation.
        """
        inputs = (self.inputs
                if isinstance(self.inputs, dict)
                else {'inputs': self.inputs})
        self.loaded = {k: io.stream.read(v)
                    for k, v in inputs.items()}

        return self

    def save(self) -> 'Processor':
        """Save the processed data into the path described by :code:`outputs`.
        """
        self._ensure_processed()
        
        with stopwatch(f'{self.fullname()} saving', mode='silent') as w:
            io.stream.write(self.processed,
                            self.outputs,
                            **self.SAVING_OPTIONS)
        
        self.watches['save'] = w

        return self

    def teardown(self) -> 'Processor':
        """Tear down processor after it's ran and its result is saved.

        By default, it does not do anything. The user must override this
        method and implement the appropriate logic when convenient.
        """
        return self

    def _ensure_processed(self):
        if self.processed is None:
            raise ValueError('The action you are attempting to perform '
                             'requires the data to be already processed '
                             'by this processor. Call the method '
                             '`Processor#perform` beforehand.')


class Rawing(Processor, metaclass=abc.ABCMeta):
    """Processor concerned with consolidating transient user data
    into a raw persisting bucket.

    Usually associated with the transient → raw transition.

    This processor passes :code:`mode='append'` to the spark writer when
    saving its results, which indicates input data will be accumulated.

    """

    SAVING_OPTIONS = {'mode': 'append'}

    def hash_sensible_info(self,
                           x: F.DataFrame,
                           columns: List[str]) -> F.DataFrame:
        """Hash data using an irreversible injective function.

        Parameters
        ----------
        x: DataFrame
            frame containing sensitive information
        columns: column or list of columns
            list of columns that could potentially be used to identify users

        Returns
        -------
        DataFrame
            A DataFrame without identifying features.
        """
        bits = self.config.security.encryption.bits
        length = self.config.security.encryption.length
        token = self.config.security.encryption.token

        for i in to_list(columns):
            x = x.withColumn(i, security.functions.anonymize(i, token, bits, length))

        return x


class Trusting(Processor, metaclass=abc.ABCMeta):
    """Processor concerned with validating data.

    Usually associated with the raw → trusted transition.

    """

    def discard_duplicates(self,
                           x: F.DataFrame,
                           id_field: Union[str, List[str]] = 'id',
                           date_field: str = None):
        """Erase duplicate records based on an identifying field.

        If {date_field} is passed, the records are sorted by most recent before
        dropping duplicates, prioritizing the discard of old entries.

        Parameters
        ----------
        x: DataFrame
           the frame possibly containing duplicates
        id_field: str, list
           column in the input frame containing the identifying data
        date_field: str, optional
           column in the input frame containing the date reference of each record

        Returns
        -------
        DataFrame
            A frame without duplicated rows.
        """
        if date_field:
            x = x.orderBy(F.desc(date_field))

        return x.dropDuplicates(to_list(id_field))


class Refining(Processor, metaclass=abc.ABCMeta):
    """Processor concerned with enriching consolidated data.

    Usually associated with the trusted → refined transition.

    """
    ML_DRIVER = models.drivers.Spark
    _ml_driver = None

    @property
    def ml(self):
        if not self._ml_driver:
            self._ml_driver = self.ML_DRIVER(
                config=self.config,
                weights_prefix=self.fullname().lower())
        return self._ml_driver


class Job(NamedEntityMixin):
    """Encapsulate a job's execution.

    Easily compose processors and their connection within a single object.

    Parameters
    ----------
    processors: list of Processor objects
        processors that must be sequentially executed in this job.
    inputs: I
        data stream that can be passed to the processors.
        This parameter is ignored if the user has not overridden the
        :func:`Job.build` method and passed it themselves.
    outputs: str
        path in which the job results are saved.
        This parameter is ignored if the user has not overridden the
        :func:`Job.build` method and passed it themselves.
    config: Config
        the configuration under which this job should run.
        It will be passed to the underlying processors during
        the setup stage.

    Examples
    --------
    .. jupyter-execute:: /examples/processors/pipeline.py
    """

    def __init__(self,
                 *processors: Processor,
                 inputs: Optional[I] = None,
                 outputs: Optional[str] = None,
                 config: configs.Config = None):
        self.processors = processors
        self.inputs = inputs
        self.outputs = outputs
        self.config = config

        self.staged = 0

    def setup(self, config: Optional[configs.Config] = None) -> 'Job':
        """Build and ensure all processors are within the same environment.

        Parameters
        ----------
        config: Config
                custom configuration in which this pipeline should run
        """
        logging.debug(f'setup {self.fullname()}')

        if config:
            self.config = config

        self.build()

        for p in self.processors:
            p.setup(config=self.config)

        return self

    def build(self):
        """Build the processing workflow, if necessary.

        By default, nothing is done and the processors passed during
        construction are used. However, one can overwrite this method to build
        more complicated associations between processors, as well as to
        encapsulate their links in order to perform integration testing.
        """
        logging.debug(f'build {self.fullname()}')

    def describe(self) -> 'Job':
        """Describe all processors within this pipeline.
        """
        print(type(self).__name__,
              f'  config: {self.config and self.config.env}',
              f'  staged: {self.staged}',
              f'  processors:',
              *(f'    - {p.fullname()}' for p in self.processors),
              sep='\n')

        return self

    def perform(self) -> 'Job':
        """Perform all operations within a pipeline and commit them.

        The methods :func:`Processor.perform`, :func:`Processor.describe` and
        :func:`Processor.save` are sequentially invoked for each processor
        within this pipeline.

        The :code:`stage` counter is increased at each committed step.
        """
        logging.debug(f'run job {self.fullname()}')

        for ix, p in enumerate(self.processors):
            p.perform().describe().save()
            self.staged = ix + 1

        return self

    def teardown(self) -> 'Job':
        """Teardown all processors within this pipeline by calling
        each individual :code:`teardown` method.
        """
        logging.debug(f'teardown {self.fullname()} ')

        for ix, p in enumerate(self.processors):
            p.teardown()

        return self


__all__ = ['Processor', 'Rawing', 'Trusting', 'Refining', 'Job']
