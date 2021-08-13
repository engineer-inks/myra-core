import logging

from py4j.protocol import Py4JJavaError
from pyspark import ml
from pyspark.sql import functions as F

from .base import FS, Driver, extract_features
from .. import errors
from ...analysis.utils import is_class
from ink.core.forge.joins.core.configs import Config
from ...utils import is_list, valid_or_default


class Spark(Driver):
    """Orchestrates Spark Machine Learning Models.

    Parameters
    ----------

    config: Config
        the current configuration held by the project.
        Should be passed by the caller (usually a job or Processor).

    caching: bool
        whether models trained and loaded should be cached or not.
        This is an abstract directive. Subclasses of Driver are
        responsible for caching whenever possible.
        When not specified, defaults to :code:`Driver.CACHING`.

    scoring: str
        the metric being tracked during evaluation.
        When not specified, defaults to :code:`Spark.CACHING`.

    """
    SCORING: str = 'accuracy'
    TREES: int = 20
    KEEP_INTERMEDIATE: bool = False
    EVALUATOR = ml.evaluation.MulticlassClassificationEvaluator

    def __init__(self,
                 config: Config,
                 caching: bool = None,
                 scoring: str = None,
                 weights_prefix: str = '',
                 keep_intermediate: bool = None):
        super().__init__(config=config, caching=caching, weights_prefix=weights_prefix)

        self.scoring = valid_or_default(scoring, self.SCORING)
        self.keep_intermediate = valid_or_default(keep_intermediate, self.KEEP_INTERMEDIATE)

    def _indexing(self, x, target, target_is_class):
        if not target_is_class:
            return x, [], target

        si = ml.feature.StringIndexer(inputCol=target, outputCol=f'_{target}_indexed').fit(x)

        return x, [si], f'_{target}_indexed'

    def build(self,
              training: F.DataFrame,
              features: FS = None,
              target='target',
              **kwargs) -> ml.pipeline.Pipeline:
        logging.debug(f'driver build {self.fullname()}')

        target_is_class = is_class(training, target)

        training, stages, label_col = self._indexing(training, target, target_is_class)

        features = extract_features(training, target, features)
        logging.info('the following columns will be used as features:', features)

        if is_list(features):
            stages += [ml.feature.VectorAssembler(inputCols=features, outputCol='features')]

        stages += [
            ml.classification.RandomForestClassifier(
                featuresCol='features',
                labelCol=label_col,
                predictionCol=f'_{target}_prediction',
                probabilityCol=f'{target}_probability',
                numTrees=kwargs.get('trees', self.TREES)),
        ]

        if target_is_class:
            stages += [
                ml.feature.IndexToString(
                    inputCol=f'_{target}_prediction',
                    outputCol=f'{target}_prediction',
                    labels=stages[0].labels)
            ]

        return ml.pipeline.Pipeline(stages=stages)

    def load(self, weights: str) -> ml.pipeline.PipelineModel:
        logging.debug(f'driver load {weights}')
        return self.cache(weights) or self.load_from_storage(weights)

    def load_from_storage(self, weights: str) -> ml.pipeline.PipelineModel:
        """Load model from the storage backend.

        Parameters
        ----------
        weights: str
            the path into which the model should be saved

        Returns
        -------
        PipelineModel
            The loaded spark model
        """
        logging.debug(f'driver load model from storage {weights}')

        try:
            model = ml.pipeline.PipelineModel.load(weights)

        except Py4JJavaError as error:
            raise errors.TrainingNotFound(error)

        self.cache(weights, model)

        return model

    def save(self, model, weights) -> 'Spark':
        logging.debug(f'driver save {weights} {model}')
        self.cache(weights, model)

        model.write().overwrite().save(weights)

        return self

    def learn(self,
              x: F.DataFrame,
              features: FS = None,
              target: str = 'target',
              weights: str = None) -> 'Spark':
        logging.debug(f'driver learn {weights}')

        x = self._balanced(x, target)
        model = self.build(x, features, target).fit(x)

        location = self.location_from_target(target)
        self.save(model, weights or location)

        return self

    def evaluate(self,
                 x: F.DataFrame,
                 features: FS = None,
                 target: str = 'target',
                 weights: str = None,
                 balanced: bool = False):
        weights = weights or self.location_from_target(target)
        logging.debug(f'driver evaluate {weights}')

        model = self.load(weights)

        y = x if not balanced else self._balanced(x, target)
        y = model.transform(y)
        e = self.EVALUATOR(
            predictionCol=f'_{target}_prediction',
            labelCol=f'_{target}_indexed',
            metricName=self.scoring)

        return e.evaluate(y)

    def infer(self,
              x: F.DataFrame,
              features: FS = None,
              target: str = 'target',
              weights: str = None,
              on_errors: str = 'raise') -> F.DataFrame:
        weights = weights or self.location_from_target(target)
        logging.debug(f'driver infer {weights}')

        try:
            model = self.load(weights or self.location_from_target(target))

            y = (model
                 .transform(x)
                 .withColumnRenamed(model.stages[-1].getOutputCol(), target))

            y = self._cleanup(x, y, target)

            return y

        except errors.TrainingNotFound:
            if on_errors == 'raise': raise
            return x.withColumn(target, F.lit(None))

    def _cleanup(self, x, y, target):
        if self.keep_intermediate:
            return y

        intermediate = set(y.columns) - set(x.columns) - {target}
        return y.drop(*intermediate)

    def _warn_features_ignored_after_training(self, features):
        if features:
            logging.warning(
                f'The parameter `features` is ignored after the training of a '
                f'Spark model.  Value passed: `{features}`')
