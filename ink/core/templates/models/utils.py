from typing import Union, List

import numpy as np
from pyspark import Broadcast
from pyspark.ml.linalg import DenseVector
from sklearn.base import BaseEstimator


def as_model(model: Union[BaseEstimator, Broadcast]) -> BaseEstimator:
    """Unpack model if it has been previously wrapped in a Broadcast object.

    Parameters
    ----------
    model: The model (or broadcast wrapper containing it) to be returned.

    Returns
    -------
    A scikit-learn estimator.
    """
    return (model.value
            if isinstance(model, Broadcast)
            else model)


def as_features(x: Union[DenseVector, List[float], np.ndarray]):
    """Convert a non-normalized input in a sklearn-friendly feature array.

    Parameters
    ----------
    x: The non-conforming data source representation.

    Returns
    -------
    Numpy array of exactly one sample, multiple features.
    """
    if isinstance(x, DenseVector):
        x = x.toArray()

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    return x.reshape(1, -1)
