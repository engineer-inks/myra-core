"""This module contains data analysis and visualization routines.

"""
import logging
from typing import Union, List, Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F, types as T

from .utils import correlatable_columns
from .. import models


def _get_reducer(r):
    from sklearn import manifold, decomposition

    reducers = {
        'tsne': manifold.TSNE,
        'pca': decomposition.PCA,
    }

    return (reducers[r.lower()]
            if isinstance(r, str)
            else r)


def distribution(df: DataFrame,
                 label: Optional[str] = None,
                 subset: Union[str, List[str]] = 'numeric',
                 sampling: Union[float, int] = 4096,
                 reducer_method: str = 'tsne',
                 cluster_method: str = 'optics',
                 workers: int = 1,
                 **kwargs):
    import seaborn as sns

    cols = sorted(correlatable_columns(df, subset=subset))
    if label: cols += [label]
    df = df.select(*cols)

    count = df.count()

    if isinstance(sampling, int):
        sampling = sampling / count

    if sampling < 1:
        logging.warning(f'The frames contains {count} samples. Data will be sampled.')
        df = df.sample(sampling)

    x = np.asarray(df.collect())

    if label:
        x, y = x[:, :-1], x[:, -1]
    else:
        m = models.discover_structure(x, cluster_method, workers)
        y = m.fit_predict(x)

    r = _get_reducer(reducer_method)(n_components=2)
    z = r.fit_transform(x)

    return sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=y, **kwargs)


def sankey(df: DataFrame,
           sequence_col: str = 'journey',
           backend: str = 'bokeh',
           **kwargs):
    """Plot a Sankey diagram given a DataFrame with journey data.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe. Each row should represent a journey.
    sequence_col : str, optional
        Name of the column containing the sequence of events for an user.
    backend : str, optional
        Holoviews backend to be used. ``None`` when no backend should be loaded (default).
    \**kwargs
        Keyword arguments to be passed to the sankey diagram plotting library.

        See `holoviews.Sankey <https://holoviews.org/reference/elements/bokeh/Sankey.html>`_
        for further information on what arguments are accepted.

    See Also
    --------
    dextra.dna.core.analysis.utils.journey : for further explanation on how to
        create a frame of sequence of events.

    Examples
    --------
    .. jupyter-execute:: /examples/analysis/visualization/sankey.py

    """
    import holoviews as hv

    if backend:
        hv.extension(backend)

    seq_fn = F.udf(lambda seq: list(zip(range(len(seq)), seq, seq[1:])),
                   returnType=T.ArrayType(T.ArrayType(T.StringType())))

    from_to = (df.withColumn('_st', F.explode(seq_fn(sequence_col)))
               .groupBy('_st')
               .count())

    df_sankey = pd.DataFrame([(f"{int(r['_st'][0]) + 1} {r['_st'][1]}",
                               f"{int(r['_st'][0]) + 2} {r['_st'][2]}",
                               r['count']) for r in from_to.collect()],
                             columns=['source', 'target', 'value'])
    plot = hv.Sankey(df_sankey)
    return plot.opts(**kwargs)
