from typing import Union, Optional, List

import numpy as np
import pandas as pd
from pyspark.ml.pipeline import PipelineModel
from sklearn import cluster
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, BaseCrossValidator, RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline as SKLPipeline
from sklearn.preprocessing import Normalizer

SUPPORTED_ESTIMATORS = (BaseEstimator,)
SUPPORTED_CLUSTERS = ('kmeans', 'optics')


def logistic_regressor_gs(params: dict = None,
                          grid_searching: bool = True,
                          scoring: Optional[str] = None,
                          cv: Union[int, BaseCrossValidator] = 3,
                          workers: int = 1):
    model = LogisticRegression(solver='lbfgs', multi_class='auto', class_weight='balanced', max_iter=400)
    params = params or grid_searching and {'C': np.logspace(-3, 2, 10)}

    return (GridSearchCV(model, params, cv=cv, n_jobs=workers, scoring=scoring)
            if grid_searching
            else model)


def score_the_shit_out_of(model, x, y,
                          cv: Union[int, BaseCrossValidator] = None,
                          scoring: Optional[str] = None,
                          workers: int = 1,
                          verbose: int = 0):
    if not isinstance(model, SUPPORTED_ESTIMATORS):
        raise ValueError(f'Model {model} of class {model.__class__} is not supported. '
                         f'Use one of the following: {SUPPORTED_ESTIMATORS}.')

    if isinstance(model, BaseEstimator):
        # 5-2-fold cross-validation over the entire dataset.
        return (cross_val_score(model, x, y,
                                cv=cv or RepeatedKFold(n_splits=2, n_repeats=5),
                                scoring=scoring,
                                n_jobs=workers,
                                verbose=verbose)
                .mean())


def explain(model,
            feature_names: Union[PipelineModel, List[str]],
            features_used: int = 16,
            **kwargs):
    """Generate a report that explains the model's behavior.

    :param model:         the model that should be explained.
    :param feature_names: a list of feature names.
    :param features_used: the number of features shown during the explanation.
    :param \**kwargs:  see below.

    :Keyword Arguments:
        * *svd* -- SVD model used for unsupervised learning with KMeans

    :return: str, a report of the model
    """
    report = ''
    estimator = model.best_estimator_ if isinstance(model, GridSearchCV) else model

    if hasattr(estimator, 'feature_importances_'):
        g = estimator.feature_importances_
        most_important = np.argsort(g)[::-1][:features_used]

        report += f'\n### Most important features:\n'
        report += (pd.DataFrame([[feature_names[i], g[i]] for i in most_important],
                                columns=['term', 'weight'])
                   .set_index('term').T.to_string())
        report += '\n'

    elif isinstance(estimator, LogisticRegression):
        is_binary = len(estimator.coef_) == 1

        if is_binary:
            g = np.vstack((-estimator.coef_, estimator.coef_))
        else:
            g = estimator.coef_

        for label in estimator.classes_:
            c, = np.where(estimator.classes_ == label)[0]
            most_important = np.argsort(np.abs(g[c]))[::-1][:features_used]

            report += f'\n### Most contributing factors for the classification of label "{label}":\n'
            report += (pd.DataFrame([[feature_names[i], g[c, i]] for i in most_important],
                                    columns=['term', 'weight'])
                       .set_index('term').T.to_string())
            report += '\n'
    elif isinstance(estimator, cluster.KMeans):
        report += "Top terms per cluster:\n"

        if 'svd' in kwargs:
            svd = kwargs['svd']
            original_space_centroids = svd.inverse_transform(model.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = model.cluster_centers_.argsort()[:, ::-1]

        for i in range(model.cluster_centers_.size):
            report += "Cluster %d:" % i
            for ind in order_centroids[i, :features_used]:
                report += ' %s' % feature_names[ind]
            report += '\n'
    else:
        raise ValueError(f'We don\'t know how to explain the model {type(model)} yet. '
                         f'Try logistic regression machines or kmeans model.')

    return report


def _calculate_wcss(x, min_clusters, max_clusters):
    return [
        (cluster
         .MiniBatchKMeans(n_clusters=n, init='k-means++', n_init=1)
         .fit(x).inertia_)
        for n in range(min_clusters, max_clusters)
    ]


def _optimal_number_of_clusters(wcss, min_clusters, max_clusters):
    x1, y1 = min_clusters, wcss[0]
    x2, y2 = max_clusters, wcss[-1]

    distances = []
    for i, inertia in enumerate(wcss):
        x0 = i + min_clusters
        y0 = inertia
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** (1 / 2)
        distances.append(numerator / denominator)

    return np.argmax(distances) + min_clusters


def latent_semantic_analysis(n_components):
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    return make_pipeline(svd, normalizer)


def discover_structure(x,
                       method: str = 'optics',
                       min_clusters: int = 2,
                       max_clusters: int = 32,
                       workers: int = 1):
    if method not in SUPPORTED_CLUSTERS:
        raise ValueError(f'Unknown method {method} passed. '
                         f'Available options are: {SUPPORTED_CLUSTERS}')

    if method == 'optics':
        return cluster.OPTICS(n_jobs=workers)

    wcss = _calculate_wcss(x, min_clusters, max_clusters)
    num_clusters = _optimal_number_of_clusters(wcss, min_clusters, max_clusters)
    return cluster.MiniBatchKMeans(n_clusters=num_clusters,
                                   init='k-means++',
                                   n_init=1)


def get_svd(model: SKLPipeline) -> TruncatedSVD:
    model = next(s for s in model.named_steps.values() if isinstance(s, TruncatedSVD))

    if model is None:
        raise ValueError('This model does not seem to have a svd transform.')

    return model
