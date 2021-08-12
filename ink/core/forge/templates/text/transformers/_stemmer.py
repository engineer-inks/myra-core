from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import functions as F, types as T

from ink.core.forge.templates.configs import spark
from ..utils import stem


class Stemmer(Transformer, HasInputCol, HasOutputCol, DefaultParamsWritable, DefaultParamsReadable):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, model=None):
        super().__init__()
        self.model = Param(self, 'model', 'NLTK model used in the stemming process')
        self._setDefault(model=None)
        self._setDefault(outputCol='stem')
        self._set(**self._input_kwargs)

    def setModel(self, value):
        return self._set(model=value)

    def getModel(self):
        from nltk.stem.snowball import PortugueseStemmer
        return self.getOrDefault(self.model) or PortugueseStemmer()

    def _transform(self, dataset):
        model = spark.spark().sparkContext.broadcast(self.getModel())
        stem_fn = F.udf(lambda text: stem(text, model.value),
                        returnType=T.ArrayType(T.StringType()))
        return dataset.withColumn(self.getOutputCol(), stem_fn(self.getInputCol()))
