"""Correlation for large-scale distributed data.

Contains multiple functions to infer relationship between binary, categorical
and continuous features.

"""
import warnings
from math import sqrt
from typing import Union, Tuple, Callable, List

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from scipy import stats

from . import utils as AU


class ChiSquareSmallFrequencyWarning(UserWarning):
    """Warned when chi-squared's result cannot be guaranteed
    because of data instability.
    """


class CategoricalCorrelationSingleClassWarning(UserWarning):
    """Warned when a feature/columns contains a single categorical value.

    In this case, aggregations cannot be produced in order to find
    significant differences between groups.
    """


class UncorrelatableError(ValueError):
    """Warned when an appropriate hypotheses-test cannot be drawn
    for the specific data type.
    """


def cramers_v(df: DataFrame,
              column_a: str,
              column_b: str,
              how: str = None,
              bias_correction: bool = True) -> Tuple[float, float]:
    """Calculates Cramér's V between categorical variables.

    See `Cramér's V <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`_ for further information.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe.
    column_a, column_b : str
        Names of the columns to be analysed.
    how : str, optional
        Allows a statistic other than Pearson's chi-squared to be used. See
        `scipy.stats.chi2_contingency <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html#scipy-stats-chi2-contingency>`_
        for more details. Defaults to None.
    bias_correction : bool, optional
        Whether to apply bias correction on the Cramér's V calculation. See
        `Cramér's V#Bias correction <https://en.wikipedia.org/wiki/Cram%C3%A9r's_V#Bias_correction>`_
        for details.

    Returns
    -------
    v : float
        Cramér's V between columns ``column_a`` and ``column_b``.
    p : float
        The p-value of the test.

    Examples
    --------
    .. jupyter-execute:: /examples/analysis/correlation/cramer_v.py

    Warns
    -----
    ChiSquareSmallFrequencyWarning
        If any of the contingency table's entries is less than 5. See
        `Pearson's_chi-squared_test#problems <https://en.wikipedia.org/wiki/Pearson's_chi-squared_test#Problems>`_
        for more information.
    CategoricalCorrelationSingleClassWarning
        If any of the columns has only one class, in which case it makes no sense to calculate correlation and the
        returned value is (0.0, 0.0).

    """
    contingency = df.crosstab(column_a, column_b).toPandas()

    contingency = contingency.set_index(contingency.columns[0])
    if (contingency.values < 5).any():
        warnings.warn("Some of the observed frequencies are less than 5, the Chi-Squared test may not be valid. "
                      "See https://en.wikipedia.org/wiki/Pearson's_chi-squared_test#Problems for more information.",
                      ChiSquareSmallFrequencyWarning)

    chi2, p, _, expected = stats.chi2_contingency(contingency, lambda_=how)

    if (expected < 5).any() and how != 'log-likelihood':
        warnings.warn("Some of the expected frequencies are less than 5, a likelihood ratio-based test statistic "
                      "may be better fit for this sample. Consider setting how=\"likelihood\". "
                      "See https://en.wikipedia.org/wiki/Pearson's_chi-squared_test#Problems for more "
                      "information.",
                      ChiSquareSmallFrequencyWarning)
    n = df.count()
    k = contingency.shape[0]
    r = contingency.shape[1]

    if k == 1 or r == 1:
        warnings.warn("Categorical column has only one class. Correlation cannot be calculated.",
                      CategoricalCorrelationSingleClassWarning)
        return 0.0, 0.0

    phi2 = chi2 / n

    if bias_correction:
        phi2 = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        k = k - (k - 1) ** 2 / (n - 1)
        r = r - (r - 1) ** 2 / (n - 1)

    v = sqrt(phi2 / (min(k, r) - 1))

    return v, p


def z_test(df: DataFrame, column_a: str, column_b: str) -> Tuple[float, float]:
    """Calculates z-score and p-value of the distributions of a numeric variable over a categorical variable.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe.
    column_a, column_b : str
        The name of the columns. One must be discrete and the other must be numeric.

    Returns
    -------
    z_score : float
        The z-score calculated for the distributions of ``num_column`` when grouped by ``cat_column``.
    p_value : float
        The p-value of the test.

    """
    try:
        num_column = next(c for c in (column_a, column_b) if AU.is_numeric(df, c))
        cat_column = next(c for c in (column_a, column_b) if AU.is_class(df, c, cat_threshold=None))
    except StopIteration:
        raise ValueError("One of `column_a` and `column_b` must be a class and the other must be numeric.")

    sigma, mu = df.agg(F.stddev_pop(num_column), F.mean(num_column)).collect()[0]
    m = df.groupBy(cat_column).agg(F.mean(num_column).alias('_mean')).collect()[0]['_mean']

    se = sigma / sqrt(df.count())

    zscore = (m - mu) / se
    pvalue = stats.norm.sf(abs(zscore)) * 2

    return zscore, pvalue


def pearson(df: DataFrame, column_a: str, column_b: str) -> Tuple[float, float]:
    """Calculates a Pearson correlation coefficient and the p-value for
    linear relationship between two datasets.

    See `scipy.stats.pearsonr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html>`.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe.
    column_a, column_b : str
        Names of the columns to be analysed.

    Returns
    -------
    coefficient : float
        Pearson's correlation coefficient between the columns
    p_value : float
        The 2 tailed p-value of the test.

    Examples
    --------
    .. jupyter-execute:: /examples/analysis/correlation/pearson.py

    """
    series = np.array(df.select(column_a, column_b).collect()).astype(float)
    return stats.pearsonr(*series.T)


def _correlation_test(df: DataFrame,
                      column_a: str,
                      column_b: str,
                      cat_threshold: Union[int, float, None] = .1) \
        -> Tuple[str, Callable[[DataFrame, str, str], Tuple[float, float]]]:
    if AU.is_class(df, column_a, cat_threshold=cat_threshold):
        if AU.is_class(df, column_b, cat_threshold=cat_threshold):
            return 'cramers_v', cramers_v
        elif AU.is_numeric(df, column_b):
            return 'z_test', z_test
    elif AU.is_numeric(df, column_a):
        if AU.is_class(df, column_b, cat_threshold=cat_threshold):
            return 'z_test', z_test
        elif AU.is_numeric(df, column_b):
            return 'pearson', pearson
    raise UncorrelatableError(f'Columns `{column_a}` and `{column_b}` on dataframe are not correlatable')


def one_to_many(df: DataFrame,
                column: str,
                subset: Union[str, List[str]] = 'auto',
                cat_threshold: Union[int, float, None] = .1) -> pd.DataFrame:
    """Calculates correlation between a single column and every other column in a subset of the dataframe.

    There are essentially three types of correlation, each calculated using a different method:

    - **Numeric-Numeric:** Between two numeric quantities. This is calculated using
      `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_,
      and the statistic ranges from -1.0 to 1.0, where -1.0 means strongly *negatively* correlated, 1.0 means
      strongly *positively* correlated and 0.0 means not correlated at all.

    - **Categorical-Categorical:** Between two nominal quantities. This is calculated using
      `Cramér's V <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`_, and the statistic ranges
      from 0.0 to 1.0, where 0.0 means no relationship and 1.0 means strong relationship.

    - **Categorical-Numeric:** Between a categorical and a nominal quantity. This is calculated by grouping data by
      the categorical column and applying a `Z-test <https://en.wikipedia.org/wiki/Z-test>`_ to determine whether
      the distribution of the numeric column is the same between all categories (in which case the columns are not
      correlated). This test does not yield a correlation measure, but a z-score, which measures distance between
      distributions in terms of population standard deviations of the numeric column.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe
    column : str
        Column name
    subset : {'auto', 'categorical', 'numeric'} or list of str, optional
        Determines which columns to analyze for correlation with `column`.
    cat_threshold : int or float or None, optional
        Maximum proportion between unique values and total that is considered to be categorical. Default is 0.1.

    See Also
    --------
    dextra.dna.core.analysis.utils.columns : for further explanation on ``subset``.
    dextra.dna.core.analysis.utils.is_categorical : for a detailed explanation on ``cat_threshold``.

    Returns
    -------
    pandas.DataFrame
        A Pandas DataFrame with columns ``test``, ``statistic`` and ``pvalue`` and one row for each column in the
        specified subset.

    Raises
    ------
    ValueError
        If ``column`` is neither numeric nor categorical.

    See Also
    --------
    correlation_matrix : Calculate correlation of every column to every other.

    Examples
    --------
    .. jupyter-execute:: /examples/analysis/correlation/one_to_many.py

    """
    subset = AU.correlatable_columns(df, subset, cat_threshold=cat_threshold) - {column} if isinstance(subset,
                                                                                                       str) else subset
    corr = {}
    for other in subset:
        name, test = _correlation_test(df, column, other, cat_threshold=cat_threshold)
        corr[other] = (name, *test(df, column, other))
    return (pd.DataFrame(corr, index=['test', 'statistic', 'pvalue']).T
            .astype({'test': str, 'statistic': float, 'pvalue': float}))


def correlation_matrix(
        df: DataFrame,
        subset: Union[str, List[str]] = 'auto',
        cat_threshold: Union[int, float, None] = .1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates a correlation matrix for columns in ``subset``.

    Since z-score works on a different scale than pearson coefficient and cramér's v, it is not shown in the
    correlation matrix. To analyze numeric-categorical correlation, use ``z_test_matrix``.

    See Also
    --------
    z_test_matrix : Check for correlation between each categorical-numeric pair of columns on a dataframe.
    dextra.dna.core.analysis.utils.is_categorical : for a detailed explanation on ``cat_threshold``.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe
    subset : {'auto', 'categorical', 'numeric'} or list of str, optional
        Determines which columns to analyze for correlation.
    cat_threshold : int or float or None, optional
        Maximum proportion between unique values and total that is considered to be categorical. Default is 0.1.

    See Also
    --------
    dextra.dna.core.analysis.utils.columns : for further explanation on ``subset``.

    Returns
    -------
    cm : pandas.DataFrame
        Correlation matrix as a pandas DataFrame.
    pvalues : pandas.DataFrame
        Matrix with p-values for the given correlation values.

    Examples
    --------
    .. jupyter-execute:: /examples/analysis/correlation/correlation_matrix.py

    """
    if isinstance(subset, str):
        subset = list(AU.correlatable_columns(df, subset, cat_threshold=cat_threshold))

    subset = [c for c in subset if AU.is_correlatable(df, c, cat_threshold=cat_threshold)]

    cm = pd.DataFrame(index=subset, columns=subset, dtype=float)
    pvalues = pd.DataFrame(index=subset, columns=subset, dtype=float)
    for i, column in enumerate(subset):
        for _, row in one_to_many(df, column, subset=subset[i:], cat_threshold=cat_threshold).iterrows():
            if row.test != 'z_test':
                cm.loc[column][row.name] = cm.loc[row.name][column] = row.statistic
            pvalues.loc[column][row.name] = pvalues.loc[row.name][column] = row.pvalue
    return cm, pvalues


def z_test_matrix(df: DataFrame,
                  alpha: float = 0.05,
                  subset: Union[str, List[str]] = 'auto',
                  cat_threshold: Union[int, float, None] = .1):
    """Calculates a matrix of 0's and 1's indicating columns that are probably related in ``subset``.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe.
    alpha : float
        Significance level (default: 0.05).
    subset : {'auto', 'categorical', 'numeric'} or list of str, optional
        Determines which columns to analyze for correlation.
    cat_threshold : int or float or None, optional
        Maximum proportion between unique values and total that is considered to be categorical. Default is 0.1.

    See Also
    --------
    dextra.dna.core.analysis.utils.columns : for further explanation on ``subset``.
    dextra.dna.core.analysis.utils.is_categorical : for a detailed explanation on ``cat_threshold``.

    Returns
    -------
    cm : pandas.DataFrame
        Boolean matrix as a pandas DataFrame. True values indicate a significance > 1 - alpha.
    pvalues : pandas.DataFrame
        Matrix with p-values for the given correlation values.

    Examples
    --------
    .. jupyter-execute:: /examples/analysis/correlation/z_test_matrix.py

    """
    subset = (list(AU.correlatable_columns(df, subset, cat_threshold=cat_threshold))
              if isinstance(subset, str)
              else subset)

    clazz = AU.correlatable_columns(df.select(subset), subset='class', cat_threshold=cat_threshold)
    numeric = AU.correlatable_columns(df.select(subset), subset='numeric')

    cm = pd.DataFrame(index=clazz, columns=numeric)
    pvalues = pd.DataFrame(index=clazz, columns=numeric)

    for i, column in enumerate(clazz):
        for _, row in one_to_many(df, column, subset=numeric, cat_threshold=cat_threshold).iterrows():
            cm.loc[column][row.name] = row.pvalue < alpha
            pvalues.loc[column][row.name] = row.pvalue
    return cm, pvalues
