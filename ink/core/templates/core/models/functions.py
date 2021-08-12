from typing import Union

import pandas as pd
from pyspark import Broadcast
from pyspark.sql import types as T, functions as F
from pyspark.sql.column import Column
from sklearn.base import BaseEstimator

from .utils import as_model, as_features


# region sample prediction functions

def predict_likeness(col: Union[str, Column],
                     model: BaseEstimator,
                     label: int = 1) -> Column:
    """Predict Likeness of a Given Class.

    Parameters
    ----------
    col :
        Column describing features fed to the model.
        This column is commonly named 'features'.
    model :
        A scikit-learn inference model.
    label :
        The label of interest, over which the probability
        is extracted. If `None` is passed, then the highest
        probability is returned.

    Returns
    -------
    pyspark.sql.udf
        The spark function representing the operation that applied to model
        to the frame's column.
    """
    return F.udf(lambda x: float(as_model(model).predict_proba(as_features(x))[0].max()
                                 if label is None
                                 else as_model(model).predict_proba(as_features(x))[0, label]),
                 T.FloatType())(col)


def predict(col: Union[str, Column], model: Union[BaseEstimator, Broadcast]) -> Column:
    """Apply the model#predict method over a column.

    Parameters
    ----------
    col :
        Column describing features fed to the model.
        This column is commonly named 'features'.
    model :
        A scikit-learn inference model.

    Returns
    -------
    pyspark.sql.udf
        The spark function representing the operation that applied to model
        to the frame's column.
    """
    return F.udf(lambda x: str(as_model(model).predict(as_features(x))[0]))(col)


def transform(col: Union[str, Column], model: BaseEstimator) -> Column:
    """Apply the model#transform method over a column.

    Parameters
    ----------
    col :
        Column describing features fed to the model.
        This column is commonly named 'features'.
    model :
        A scikit-learn inference model.

    Returns
    -------
    pyspark.sql.udf
        The spark function representing the operation that applied to model
        to the frame's column.
    """
    return F.udf(lambda x: str(as_model(model).transform(as_features(x))[0]))(col)


# endregion

# region batching

def predict_likeness_batch(cols, model, label=1) -> Column:
    def _predict_likeness_batch(*x):
        return pd.Series(as_model(model).predict_proba(pd.concat(x, axis=1)).max(axis=1)
                         if label is None else
                         as_model(model).predict_proba(pd.concat(x, axis=1))[:, label])

    return F.pandas_udf(_predict_likeness_batch, 'double')(*cols)


def predict_batch(cols, model) -> Column:
    return F.pandas_udf(lambda *x: pd.Series(as_model(model)
                                             .predict(pd.concat(x, axis=1))),
                        'string')(*cols)

# endregion
