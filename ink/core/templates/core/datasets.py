from pyspark.sql import DataFrame

from . import io


def sklearn_ds(name: str, *args, **kwargs) -> DataFrame:
    """Load Scikit-learn dataset from function name.

    Parameters
    ----------
    name: Name of the function within :code:`sklearn.datasets`
    args: Arguments passed to the building function
    \**kwargs
        Key arguments passed to the building function

    Examples
    --------
    .. code-block:: python

        import dextra.dna.core as C
        C.datasets.iris().limit(5).toPandas()

    """
    import pandas as pd
    from sklearn import datasets

    x = getattr(datasets, name)(*args, **kwargs)
    d = pd.DataFrame(x.data, columns=x.feature_names)

    d['target'] = (x.target_names[x.target].astype(str)
                   if hasattr(x, 'target_names')
                   else x.target)

    d = io.stream.read(d)
    d = io.stream.conform(d)
    d = io.stream.merge(d)

    return d


def wine() -> DataFrame:
    """Load the wine dataset from :code:`sklearn.datasets` as spark dataframe.

    Examples
    --------
    .. jupyter-execute::

        import dextra.dna.core as C
        C.datasets.wine().limit(2).toPandas()

    """
    return sklearn_ds('load_wine')


def iris() -> DataFrame:
    """Load the iris dataset from :code:`sklearn.datasets` as spark dataframe.

    Examples
    --------
    .. jupyter-execute::

        import dextra.dna.core as C
        C.datasets.iris().limit(2).toPandas()

    """
    return sklearn_ds('load_iris')


def digits() -> DataFrame:
    """Load the digits dataset from :code:`sklearn.datasets` as spark dataframe.

    Examples
    --------
    .. jupyter-execute::

        import dextra.dna.core as C
        C.datasets.digits().limit(2).toPandas()

    """
    return sklearn_ds('load_digits')


def boston() -> DataFrame:
    """Load the iris dataset from :code:`sklearn.datasets` as spark dataframe.

    Examples
    --------
    .. jupyter-execute::

        import dextra.dna.core as C
        C.datasets.boston().limit(2).toPandas()

    """
    return sklearn_ds('load_boston')
