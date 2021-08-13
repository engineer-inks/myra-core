import abc
import logging
import os
from typing import Optional, Union, List

from pyspark.sql import functions as F

from ...analysis.utils import correlatable_columns
from ink.core.forge.joins.core.configs import Config
from ...utils import valid_or_default, NamedEntityMixin

FS = Optional[Union[str, List[str]]]


def extract_features(x: F.DataFrame,
                     target: str,
                     features: FS):
    """Extract features (if missing) by analyzing the DataFrame.

    If :code:`features` is passed (not None), then this will work as the
    identity function. Otherwise, the columns that contain learnable patterns
    are identified through the :func:`dextra.dna.core.analaysis.utils.correlatable_columns`
    function.

    Parameters
    ----------
    x: DataFrame
        the frame onto which the model will be applied
    target: str
        the name of the column containing the target data, if any
    features: str or list of strings
        a column of a list of columns containing features

    Returns
    -------
    str or list of strings
        :code:`features`, if passed. Otherwise, returns a list of column names
        that contain patterns (according to the analysis module)

    """
    if features is None:
        features = [c for c in correlatable_columns(x) if c != target]

    if not features:
        raise ValueError('Cannot infer features from columns. '
                         'Try specifying it manually.')

    return features


class Driver(NamedEntityMixin,
             metaclass=abc.ABCMeta):
    """Orchestrates Machine Learning Models.

    Can be used to train, apply and store machine learning models throughout
    data pipeline's executions.

    Parameters
    ----------
    config: Config
        the current configuration held by the project.
        Should be passed by the caller (usually a job or Processor).

    caching: bool
        whether models trained and loaded should be cached or not.
        This is an abstract directive. Subclasses of Driver are
        responsible for caching whenever possible.

    """

    CACHING: bool = True

    def __init__(self,
                 config: Config,
                 caching: bool = None,
                 weights_prefix: Optional[str] = ''):
        self.config = config
        self.caching = valid_or_default(caching, self.CACHING)
        self.weights_prefix = weights_prefix

        self.cached_models = {}

    @abc.abstractmethod
    def load(self, weights: str):
        """Load The Model From A Name Passed.

        It's the driver's job to save the model, as an driver
        abstracts which kind of model is being handled. Nevertheless,
        :code:`config.lakes.models` is probably a good place to look for it.

        Returns
        -------
        ML model
            The model object loaded.
        """

    @abc.abstractmethod
    def save(self, model, weights: str):
        """Save a Trained Model to a Location Inferred from The Weights Passed.
        :code:`config.lakes.models` is probably a good place to look for it.

        Parameters
        ----------
        model: ML Model
            the model being saved
        weights: str
            the path into which the model should be saved

        Returns
        -------
        The ML model object loaded.
        """

    @abc.abstractmethod
    def learn(self,
              x: F.DataFrame,
              features: FS = None,
              target: str = 'target',
              weights: str = None):
        """Learn To Estimate a Target Based on Features Passed.

        This method should probably also save the model once it's trained.

        Parameters
        ----------
        x: DataFrame
            contains the training data
        features: str
            columns containing the features for the model
        target: str
            column containing the variable being estimated
        weights: str
            identifier for persisting the model. Whenever not available
            (:code:`None`), this should be inferred from the target.
        """

    @abc.abstractmethod
    def evaluate(self,
                 x: F.DataFrame,
                 features: FS = None,
                 target: str = 'target',
                 weights: str = None):
        """Evaluate Metrics For a Trained Model Over a Validation Dataset.

        Parameters
        ----------
        x: DataFrame
            contains the validation data
        features: str or list of strings
            column containing the variable being estimated
        target: str
            column containing the variable being estimated
        weights: str
            identifier for persisting the model. Whenever not available
            (:code:`None`), this should be inferred from the target.
        """

    @abc.abstractmethod
    def infer(self,
              x: F.DataFrame,
              features: FS = None,
              target: str = 'target',
              weights: str = None,
              on_errors: str = 'raise') -> F.DataFrame:
        """Infer A Specific Target Value from Available Features and a Trained Model.

        Parameters
        ----------
        x: DataFrame
            contains the validation data
        features: str or list of strings
            column containing the variable being estimated
        target: str
            column containing the variable being estimated
        weights: str
            identifier for persisting the model. Whenever not available
            (:code:`None`), this should be inferred from the target.
        on_errors: str ('raise', 'ignore')
            whether errors should be raised or ignored
            when inferring. If this is the case, the returned DataFrame should contain
            only null values in the the target column.
            Common errors are: incompatible features, missing model.

        Returns
        -------
        DataFrame
            The input DataFrame :code:`x`, plus the column :code:`target`
            containing the inferred values.

        """

    def trained(self, weights: str):
        """Check If a Trained Model Exists.

        This method should be overridden to produce more efficient checking strategies
        whenever they are available.

        Returns
        -------
        bool
            True if it exists, False otherwise.
        """
        try:
            self.load(weights)
        except FileNotFoundError:
            return False

        return True

    def location_from_target(self, target: str) -> str:
        """Determine a model's location from the target column.

        We append the suffix :code:`.p` here in order to represent
        a :code:`pickle` file.

        Parameters
        ----------
        target : The target column in which the data is trained over.

        Returns
        -------
        str
            The full location of the model.
        """
        if self.weights_prefix:
            return os.path.join(self.config.lakes.models,
                                self.weights_prefix,
                                target)

        return os.path.join(self.config.lakes.models,
                            target)

    def cache(self, weights: str, model=None):
        """Cache a model into memory or retrieve a cached one.

        Useful before persisting a model, holding it into memory for
        subsequential calls to :func:`infer` or :func:`evaluate` methods.

        Parameters
        ----------
        weights: str
            the path used when persisting the model,
            which uniquely identifies a trained model
        model: spark or sklearn model, optional
            a trained model, when setting cache, otherwise None

        Returns
        -------
        ML Model or None
            A trained ML model that has been previously cached,
            or None if :code:`model` parameter was not passed.
        """
        if not model:  # get
            logging.debug(f'driver cache get {weights}')
            return self.cached_models.get(weights)

        if self.caching:
            logging.debug(f'driver cache set {weights} {model}')
            self.cached_models[weights] = model

    def uncache(self, weights: str):
        """Clear model from cache and remove it from memory.

        This method is useful whenever a model consumes a large amount of
        memory, and it will no longer be necessary for the pipeline's
        execution.

        Parameters
        ----------
        weights: str
            the path used when persisting the model,
            which uniquely identifies a trained model
        """
        logging.debug(f'driver uncache {weights}')
        model = self.cached_models.pop(weights, None)

        if model is not None:
            del model

        return self

    def _balanced(self,
                  x: F.DataFrame,
                  target: str = 'target',
                  unbalance_rate: float = 1.5):
        """Balance a dataframe according to a target column through
        random undersampling.

        Parameters
        ----------
        x: DataFrame
            the frame onto which the model will be applied
        target: str
            the name of the column containing the target data, if any
        unbalance_rate: float in [1, inf)
            the maximum balance rate of :code:`target` column after this
            method is applied. Lower numbers create fully-balanced datasets,
            but might result in an aggressive reduction of the original set.

        """
        samples = dict(x.groupBy(target).count().collect())
        min_freq = min(samples.values())
        rates = {c: min(unbalance_rate * min_freq / f, 1) for c, f in samples.items()}

        return x.sampleBy(target, rates)
