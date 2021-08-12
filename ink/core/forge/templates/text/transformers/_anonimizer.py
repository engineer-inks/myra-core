import os
from functools import reduce
from typing import Callable, List, Dict, Union

from pyspark import keyword_only
from pyspark.ml import Model, Estimator, Transformer
from pyspark.ml.param import Param, Params
from pyspark.ml.util import (MLReadable, MLWritable, MLReader, MLWriter,
                             DefaultParamsReader, DefaultParamsWriter,
                             DefaultParamsReadable, DefaultParamsWritable)
from pyspark.sql import DataFrame, functions as F, types as T
from pyspark.sql.utils import AnalysisException

from .. import functions, datasets, utils


class _ReversibleIdMapReader(MLReader):
    def load(self, path: str) -> 'ReversibleIdMap':
        return ReversibleIdMap(self.sparkSession.read.parquet(path))


class _ReversibleIdMapWriter(MLWriter):
    instance: 'ReversibleIdMap'

    def __init__(self, instance: 'ReversibleIdMap'):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path):
        self.instance._assoc.write.parquet(path)


class ReversibleIdMap(MLWritable, MLReadable):
    """A distributed key-value pair set with two-way mapping capabilities.

    See Also
    --------
    Anonimizer
    Deanonimizer

    """

    def __init__(self, other: Union['ReversibleIdMap', DataFrame] = None):
        self._column_name = None
        if other is None:
            self._assoc = None
        elif isinstance(other, ReversibleIdMap):
            self._assoc = other._assoc
        elif isinstance(other, DataFrame):
            self._assoc = other
        else:
            raise ValueError(f"other should be of type ReversibleIdMap or pyspark.sql.DataFrame, not {type(other)}")

    @classmethod
    def read(cls):
        return _ReversibleIdMapReader()

    @property
    def write(self):
        return _ReversibleIdMapWriter(self)

    @property
    def _last_id(self):
        if self._assoc is None:
            return -1
        return self._assoc.agg({self._value_column(): 'max'}).collect()[0][0]

    def _key_column(self, reverse: bool = False):
        return '_anonimized' if reverse else '_source'

    def _value_column(self, reverse: bool = False):
        return self._key_column(reverse=not reverse)

    def fit(self, df: DataFrame, column_name: str) -> 'ReversibleIdMap':
        """Fits this map to a dataframe column.

        This function generates a unique identifier for every unique value in the column and stores it for mapping. It
        must always be called before ``apply``.

        This function is incremental, which means it will not generate new values when called twice with the same
        dataframe, but will create new entries on the map when called with a new one with different values. The columns
        do not necessarily have to have the same name every time this function is called.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The dataframe.
        column_name : str
            The name of the column to be mapped.

        """
        self._column_name = column_name
        old = self._assoc
        offset = self._last_id + 1
        df = df.select(F.col(self._column_name).alias(self._key_column()))
        df = df.distinct()
        anonimized = F.monotonically_increasing_id() + offset
        if old is not None:
            self._assoc = df.join(old, [self._key_column()], how='full')
            self._assoc = self._assoc.orderBy(self._key_column())
            self._assoc = self._assoc.withColumn(self._value_column(), F.coalesce(self._value_column(), anonimized))
        else:
            df = df.orderBy(self._column_name)
            self._assoc = df.withColumn(self._value_column(), anonimized)
        return self

    def _fetch_anonimized(self, df: DataFrame, reverse: bool = False):
        return df.join(self._assoc, on=[df[self._column_name] == self._assoc[self._key_column(reverse)]])

    def apply(self, df: DataFrame, column_name: str = None, reverse: bool = False) -> DataFrame:
        """Applies map on a dataframe column.

        See Also
        --------
        fit : Fits this map to a dataframe column. Must be called at least once before ``apply``.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The dataframe.
        column_name : str, optional
            The name of the column to be mapped. If None (default), the last ``column_name`` passed to ``fit`` will be
            assumed.
        reverse : bool, optional
            Whether to reverse the anonimization process (Default: False).

        Returns
        -------
        pyspark.sql.DataFrame
            Dataframe with the specified column mapped.

        """
        if column_name:
            self._column_name = column_name
        df = self._fetch_anonimized(df, reverse=reverse)
        df = df.drop(self._column_name)
        df = df.withColumnRenamed(self._value_column(reverse), self._column_name)
        df = df.drop(self._key_column(reverse))
        return df

    def revert(self, df: DataFrame, column_name: str = None) -> DataFrame:
        """Reverts a map application on a dataframe column.

        See Also
        --------
        apply : Applies map on a dataframe column.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The dataframe.
        column_name : str, optional
            The name of the column to be mapped. If None (default), the last ``column_name`` passed to ``fit`` will be
            assumed.

        Returns
        -------
        pyspark.sql.DataFrame
            Dataframe with the specified column mapped.

        """
        return self.apply(df, column_name=column_name, reverse=True)


class _AnonimizerBaseReader(DefaultParamsReader):
    def load(self, path: str) -> MLReadable:
        instance: AnonimizerBase = super().load(path)
        if instance.getOrDefault('reversible'):
            idmaps = {}
            for col in instance.getOrDefault('idColumns'):
                try:
                    idmaps[col] = ReversibleIdMap.load(os.path.join(path, 'id_maps', col))
                except AnalysisException:
                    ...
            instance.set(instance.idMaps, idmaps)
        return instance


class _AnonimizerBaseWriter(DefaultParamsWriter):
    instance: 'AnonimizerBase'

    def saveImpl(self, path: str) -> None:
        if self.instance.getOrDefault('reversible'):
            for column, idmap in self.instance.getIdMaps().items():
                idmap.write.save(os.path.join(path, 'id_maps', column))
        DefaultParamsWriter.saveMetadata(
            self.instance,
            path,
            self.sc,
            paramMap={
                p.name: self.instance._paramMap[p]
                for p in self.instance._paramMap if p.name != 'idMaps'})


class AnonimizerBase(Params, DefaultParamsWritable, DefaultParamsReadable):

    @classmethod
    def read(cls) -> MLReader:
        return _AnonimizerBaseReader(cls)

    def write(self) -> MLWriter:
        return _AnonimizerBaseWriter(self)

    def _setup(self,
               idColumns=(),
               textColumns=(),
               reversible=False,
               patterns=None,
               idMaps={}):
        self.idColumns = Param(self, "idColumns", "Array de colunas que são apenas um identificador e devem ser "
                                                  "anonimizadas")
        self.textColumns = Param(self, "textColumns", "Array de colunas que contém textos e devem ser anonimizadas "
                                                      "através dos matchers")
        self.reversible = Param(self, "reversible", "Se for True, anonimiza as colunas de identificador de forma "
                                                    "reversível (padrão: False)")
        self.patterns = Param(self, "patterns", "Dicionário de expressões regulares e substituições para substituir no"
                                                " texto")
        self.idMaps = Param(self, "idMaps", "Dicionário de identificadores para valores anonimizados")
        self.set(self.idColumns, idColumns)
        self.set(self.textColumns, textColumns)
        self.set(self.reversible, reversible)
        self.set(self.patterns, utils.patterns_as_masks(patterns or datasets.all_patterns()))
        self.set(self.idMaps, idMaps)

    def getIdColumns(self) -> List[str]:
        return self.getOrDefault(self.idColumns) or []

    def getTextColumns(self) -> List[str]:
        return self.getOrDefault(self.textColumns) or []

    def isReversible(self) -> bool:
        return self.getOrDefault(self.reversible)

    def getIdMaps(self) -> Dict[str, ReversibleIdMap]:
        return self.getOrDefault(self.idMaps) or {}

    def getMap(self, column: str) -> ReversibleIdMap:
        return self.getIdMaps()[column]

    def getPatterns(self) -> Dict[str, str]:
        return self.getOrDefault(self.patterns)


class AnonimizerModel(Model, AnonimizerBase):
    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setup(**kwargs)

    def _build_hasher(self) -> Callable[[DataFrame, str], DataFrame]:
        def hash_id_column(df: DataFrame, column: str) -> DataFrame:
            return df.withColumn(column, F.sha2(F.col(column).cast(T.StringType()), 256))

        def hash_id_column_reversible(df: DataFrame, column: str) -> DataFrame:
            return self.getMap(column).apply(df, column_name=column)

        return hash_id_column_reversible if self.isReversible() else hash_id_column

    def _build_text_anonimizer(self) -> Callable[[DataFrame, str], DataFrame]:
        def match_and_replace(x: DataFrame, column: str):
            for pattern, replacement in self.getPatterns().items():
                x = x.withColumn(column, functions.replace(column, pattern, replacement))
            return x

        return match_and_replace

    def _transform(self, df: DataFrame) -> DataFrame:
        hash_id = self._build_hasher()
        df = reduce(hash_id, self.getIdColumns(), df)

        text_anonimizer = self._build_text_anonimizer()
        df = reduce(text_anonimizer, self.getTextColumns(), df)
        return df


class Anonimizer(Estimator, AnonimizerBase):
    """Anonimization Transformer.

    Parameters
    ----------
    idColumns : list of str, optional
        List of identifier columns that will be anonimized using a id map. Default is empty.
    textColumns : list of str, optional
        List of text columns that will have sensitive information replaced with neutral text. Default is empty.
    reversible : bool
        Whether the id anonymization should be reversible or not. Please notice text anonimization can not be reversed.
    patterns : dict of str to str, optional
        Dictionary in format ``{ regex: replacement }`` with patterns to replace in text columns.
    idMaps : dict of str to ReversibleIdMap
        Dictionary in format ``{ column: id_map }`` with reversible id maps to be applied to columns when reversible
        is ``True``. Pass this parameter to ensure consistency when using different ``Anonimizer`` objects.

    Note
    ----
    The parameter ``reversible`` SHOULD NOT be used carelessly, since anonimization is essentially pointless if it is
    easily reversible. When it is set to ``True``, make sure you do not store the id map anywhere acessible by anyone
    that should not be able to access it.

    Examples
    --------
    Basic usage:

    .. jupyter-execute:: /examples/text/transformers/anonimizer/anonimizer.py

    It is also possible to reuse id maps between instances and ensure consistency of mappings on different dataframes:

    .. jupyter-execute:: /examples/text/transformers/anonimizer/anonimizer_consistency.py

    See Also
    --------
    Deanonimizer : for examples of reversible anonimization.

    """

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setup(**kwargs)

    def _fit(self, df: DataFrame) -> AnonimizerModel:
        kwargs = self._input_kwargs.copy()
        if self.isReversible():
            id_maps = self.getIdMaps()
            for column in self.getIdColumns():
                id_map = ReversibleIdMap(other=id_maps[column] if column in id_maps else None)
                id_maps[column] = id_map.fit(df, column)
            kwargs['idMaps'] = id_maps
        return AnonimizerModel(**kwargs)


class Deanonimizer(Transformer):
    """Deanonimization Transformer. Reverts processes made by Anonimizer where applicable.

    Parameters
    ----------
    columns : list of str
        List of columns that should be transformed.
    anonimizer : AnonimizerBase or dict of str to ReversibleIdMap
        If an ``Anonimizer`` or an ``AnonimizerModel`` are passed, their id maps are used to revert the anonimization
        on columns that were anonimized in a reversible way.
        If a dict of ``str`` to ``ReversibleIdMap`` is passed, it should be formatted as ``{column: id_map}`` and each
        column in ``columns`` will be reversed using the given ``id_map``.

    See Also
    --------
    Anonimizer
    ReversibleIdMap

    Examples
    --------
    .. jupyter-execute:: /examples/text/transformers/anonimizer/deanonimizer.py

    Note
    ----
    Anonimization is pointless if deanonimizing data is easy. This class and the ``reversible`` parameter on
    ``Anonimizer`` should be used cautiously.

    """

    @keyword_only
    def __init__(self, columns=(), anonimizer=None):
        super().__init__()
        self.columns = Param(self, "columns", "Colunas que devem ser mapeadas.")
        self.anonimizer = Param(self, "anonimizer", "Anonimizer usado para fazer a anonimização, ou um dict com os"
                                                    "mapas de ID para cada coluna desejada.")

        if isinstance(anonimizer, dict):
            anonimizer = Anonimizer(idmaps=anonimizer)
        elif not isinstance(anonimizer, AnonimizerBase):
            raise ValueError(f'anonimizer should be of type dict or AnonimizerBase, not {type(anonimizer)}')

        self.set(self.columns, columns)
        self.set(self.anonimizer, anonimizer)

    def _transform(self, df: DataFrame) -> DataFrame:
        idmaps = self.getOrDefault(self.anonimizer).getIdMaps()
        for column, id_map in idmaps.items():
            df = id_map.revert(df)
        return df


__all__ = ['ReversibleIdMap', 'AnonimizerBase', 'Anonimizer', 'AnonimizerModel', 'Deanonimizer']
