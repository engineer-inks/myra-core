import logging
import uuid
from typing import Callable, Optional, Tuple, List, Union

import joblib
import joblibspark
import numpy as np
from pyspark import ml
from pyspark.sql import functions as F
from sklearn import metrics
from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils import parallel_backend

from .base import Driver
from .. import errors, functions
from ..base import logistic_regressor_gs
from ... import io
from ...configs import spark, Config
from ...utils import valid_or_default, is_list

E = Callable[[str], BaseEstimator]
FE = Union[str, List[str]]
XY = Tuple[np.ndarray, np.ndarray]

joblibspark.register_spark()


class SkLearn(Driver):
    """Scikit-learn Models Driver.

    When used as is, this driver will serve logistic regression classifiers,
    trained using GridSearch 5-2 cross-validation strategy.

    Arguments
    ---------
    config: Config
        Reference to current run's configuration.
    caching: bool
        Whether RAM-level caching should be kept.
    estimator_fn: Callable
        Building function of the serving model. By Default, uses a
        :class:`GridSearchCV` wrapped around a
        :class:`LogisticRegression`.
        (dextra.dna.core.models.logistic_regression_gs).
    scoring: str
        String representing the scoring method for the model.
        Ignored if :func:`~SkLearn.evaluate` is not called.
    broadcasting: bool
        Whether to broadcast the model across all cluster nodes.

    """

    ESTIMATOR_FN: E = staticmethod(logistic_regressor_gs)
    SCORING: str = 'balanced_accuracy'
    BROADCASTING: bool = True

    def __init__(self,
                 config: Config,
                 caching: Optional[bool] = None,
                 estimator_fn: Optional[E] = None,
                 scoring: Optional[str] = None,
                 broadcasting: Optional[bool] = None,
                 weights_prefix: str = ''):
        super().__init__(config=config, caching=caching, weights_prefix=weights_prefix)

        self.estimator_fn = valid_or_default(estimator_fn, self.ESTIMATOR_FN)
        self.scoring = valid_or_default(scoring, self.SCORING)
        self.broadcasting = valid_or_default(broadcasting, self.BROADCASTING)

    def build(self):
        logging.debug(f'driver build {self.fullname()}')
        return self.estimator_fn(scoring=self.scoring)

    def load(self, weights):
        logging.debug(f'driver load {weights}')
        return self.cache(weights) or self.load_from_storage(weights)

    def load_from_storage(self, weights: str) -> BaseEstimator:
        """Load model from the storage backend.

        Parameters
        ----------
        weights: str
            the path into which the model should be saved

        Returns
        -------
        BaseEstimator
            The loaded sklearn model

        """
        logging.debug(f'driver load model from storage {weights}')
        protocol, location = io.storage.split_protocol_path(weights)

        if protocol != 'file':
            location = f'/tmp/{uuid.uuid4()}'

            try:
                io.storage.copy(source=weights, target=location)
            except FileNotFoundError as error:
                raise errors.TrainingNotFound(error)

        try:
            model = joblib.load(location)
            self.cache(weights, model)

        except FileNotFoundError as error:
            raise errors.TrainingNotFound(error)

        return model

    def save(self,
             model: BaseEstimator,
             weights: str) -> 'SkLearn':
        """Save a scikit-learn model in the storage backend.

        This method first saves the model in a temporary location so it can
        then be copied into the running storage backend (e.g. GCS).

        Parameters
        ----------
        model: ML Model
            the model being saved
        weights: str
            the path into which the model should be saved
        """
        logging.debug(f'driver save {weights} {model}')
        self.cache(weights, model)

        location = f'/tmp/{uuid.uuid4()}'
        joblib.dump(model, location)
        io.storage.copy(source=location, target=weights)

        return self

    def learn(self,
              x: F.DataFrame,
              features: FE = 'features',
              target: str = 'target',
              weights: str = None) -> 'SkLearn':
        logging.debug(f'driver learn {weights}')

        model = self.build()

        x = self.normalize_features(x, features)
        x = x.select('features', F.col(target).alias('target'))
        x = self._balanced(x)
        x, y = self.extract_arrays(x)

        try:
            with parallel_backend('spark', n_jobs=3):
                model.fit(x, y)
        except ValueError as error:
            raise errors.Training(error)

        self.save(model, weights or self.location_from_target(target))

        return self

    def evaluate(self,
                 x: F.DataFrame,
                 features: FE = 'features',
                 target: str = 'target',
                 weights: str = None):
        weights = weights or self.location_from_target(target)
        logging.debug(f'driver evaluate {weights}')

        model = self.load(weights)

        x = self.normalize_features(x, features)
        x = x.select('features', F.col(target).alias('target'))
        z, y = self.extract_arrays(x)

        scorer = metrics.get_scorer(self.scoring)
        score = scorer(model, z, y)

        if is_classifier(model):
            logging.debug(f'classification report over all data ({len(y)} samples):')
            logging.debug(metrics.classification_report(y, model.predict(z)))

        logging.info(f'{self.scoring} score: {score}')

        return score

    def infer(self,
              x: F.DataFrame,
              features: FE = 'features',
              target: str = 'target',
              weights: str = None,
              on_errors: str = 'raise') -> F.DataFrame:
        weights = weights or self.location_from_target(target)
        logging.debug(f'driver infer {weights}')

        try:
            model = self.load(weights)

            if self.broadcasting:
                model = spark().sparkContext.broadcast(model)

            x = self.normalize_features(x, features)
            return x.withColumn(target, functions.predict('features', model))

        except errors.TrainingNotFound:
            if on_errors == 'raise':
                raise
            return x.withColumn(target, F.lit(None))

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
        return super().location_from_target(target) + '.p'

    def normalize_features(self, x: F.DataFrame, features: FE) -> F.DataFrame:
        """Collect feature columns into a single Vector column,
        if it hasn't happened yet.

        Parameters
        ----------
        x: DataFrame
            the frame onto which the model will be applied
        features: str
            columns containing the features for the model

        Returns
        -------
        DataFrame
            A frame containing the 'features' column in it

        """
        if is_list(features):
            assembler = ml.feature.VectorAssembler(inputCols=features,
                                                   outputCol='features')
            x = assembler.transform(x)

        if isinstance(features, str) and features != 'features':
            return x.withColumn('features', F.col(features))

        return x

    def extract_arrays(self, x: F.DataFrame) -> XY:
        """Extract the DataFrame features as numpy arrays.

        Parameters
        ----------
        x: DataFrame
            the frame onto which the model will be applied

        Returns
        -------
        tuple(ndarray, ndarray)
            The numpy array pair (x, y) used during training or evaluation.
        """
        x_ = x.select('features', 'target').collect()
        z = np.asarray([r.features.toArray() for r in x_])
        y = np.asarray([d.target for d in x_])

        return z, y

    def __repr__(self):
        return (f'<{self.__module__}.{type(self).__name__} at {hex(id(self))}, '
                f'estimator fn: {self.estimator_fn.__name__}, '
                f'scoring: {self.scoring}, '
                f'config: {self.config.env}>')


class BatchedSkLearn(SkLearn):
    """Driver for Scikit-learn models that leverage vectorized user functions.
    It currently supports only structured data.
    """
    def infer(self,
              x: F.DataFrame,
              features: List[str] = None,
              target: str = 'target',
              weights: str = None,
              on_errors: str = 'raise') -> F.DataFrame:
        weights = weights or self.location_from_target(target)
        logging.debug(f'driver infer {weights}')

        self._assert_features_list(features)

        try:
            model = self.load(weights)

            if self.broadcasting:
                model = spark().sparkContext.broadcast(model)

            return x.withColumn(target, functions.predict_batch(features, model))

        except errors.TrainingNotFound:
            if on_errors == 'raise':
                raise
            return x.withColumn(target, F.lit(None))

    def normalize_features(self, x: F.DataFrame, features: FE) -> F.DataFrame:
        self._assert_features_list(features)
        return super().normalize_features(x, features)

    def _assert_features_list(self, features: FE):
        if not features or not isinstance(features, List):
            raise ValueError(
                f'`BatchedSkLearn` driver can only predict using a list of '
                f'structured features, but {features.__class__.__name__} '
                f'`{features}` was passed. Check the docs for more on the '
                f'limitations of this driver.')
