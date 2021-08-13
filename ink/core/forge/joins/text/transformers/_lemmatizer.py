from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import functions as F

from ..utils import load_spacy_model, lemmatize
from ink.core.forge.joins.core.configs import spark

class SPACY:
    MODEL = 'pt_core_news_sm'


class SpacyTransformer(Transformer,
                       HasInputCol,
                       HasOutputCol,
                       DefaultParamsWritable,
                       DefaultParamsReadable):
    """Spacy base transformer.

    Holds a spacy language model within and can process
    spark dataframes using it.

    """
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, languageModel=None):
        super().__init__()
        self.languageModel = Param(self, 'languageModel', 'Name for Spacy model used during lemmatization')
        self._setDefault(languageModel=None)
        self._setDefault(outputCol='lemma')
        self._set(**self._input_kwargs)

    def setLanguageModel(self, value):
        return self._set(languageModel=value)

    def getLanguageModel(self) -> 'spacy.Language':
        lm = self.getOrDefault(self.languageModel)
        lm = load_spacy_model(lm or SPACY.MODEL)

        return lm


class Lemmatizer(SpacyTransformer):
    """Lemmatizer Transformer.

    Parameters
    ----------
    inputCol : str or col
        Column or column name that will be lemmatized.
    outputCol : str or col
        Output column which will store :code:`inputCol` lemmatized values.
    languageModel : str,
        Name of Spacy language model. Defaults to ``pt_core_news_sm``.

    Examples
    --------
    .. jupyter-execute:: /examples/text/transformers/lemmatizer.py
    """

    def _transform(self, dataset):
        model = spark.spark().sparkContext.broadcast(self.getLanguageModel())
        lemma_fn = F.udf(lambda t: lemmatize(t, model.value))
        return dataset.withColumn(self.getOutputCol(), lemma_fn(self.getInputCol()))
