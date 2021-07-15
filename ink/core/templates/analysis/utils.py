import operator
from typing import List, Union, Any, Callable

from pyspark.sql import DataFrame, functions as F, types as T, Window, Column

from .. import utils as U


# region column identifiers
def is_binary(df: DataFrame, column: str) -> bool:
    """Checks whether a given column in a dataframe is binary.

    Parameters
    ----------
    df : DataFrame
        The dataframe.
    column : str
        Column name.

    Return
    ------
    bool
        True if ``column`` has two possible values, False otherwise.

    """
    return isinstance(df.schema[column].dataType, T.BooleanType)


def is_categorical(df: DataFrame,
                   column: str,
                   cat_threshold: Union[int, float, None] = .9) -> bool:
    """Checks whether a given column in a dataframe is categorical.

    See Also
    --------
    is_binary   : Checks whether a given column in a dataframe is binary.
    is_discrete : Check whether a given column is either categorical or binary.

    Parameters
    ----------
    df : DataFrame
        The dataframe.
    column : str
        Column name.
    cat_threshold : int or float or None, optional
        Maximum proportion between unique values and total that is considered to be categorical. Default is .9, but
        this should be set to whatever value is sensible for your dataset.

        Values between 0 and 1 will be considered proportions, while integer values greater than 2 will be considered
        absolute quantities. Absolute value 1 is not allowed, as a single class is not useful for any calculations.

        If set to None, any checks regarding unique values proportion are ignored. This is not recommended unless the
        dataset is either very small or the column is known to be probably categorical.

        This parameter aims to avoid using id columns as categorical and end up doing expensive calculations (e.g.
        correlation) on too many classes.

    Returns
    -------
    bool
        True if ``column`` is categorical, False otherwise.

    """
    if df.schema[column].dataType != T.StringType():
        return False
    if cat_threshold is None:
        return True

    c, d = df.select(F.count(column), F.countDistinct(column)).collect()[0]

    if cat_threshold > 1:
        if not isinstance(cat_threshold, int):
            raise ValueError('`cat_threshold` should be a float between 0 and 1 representing the '
                             'distinct percentage or integer representing the count of distinct values.')
        return d <= cat_threshold

    coefficient = d / c
    return coefficient <= cat_threshold


def is_numeric(df: DataFrame, column: str):
    """Checks whether a given column in a dataframe is numeric.

    Parameters
    ----------
    df : DataFrame
        The dataframe.
    column : str
        Column name.

    Returns
    -------
    bool
        True if ``column`` is of some numeric type, False otherwise.

    """
    return isinstance(df.schema[column].dataType, T.NumericType)


def is_class(df: DataFrame, column: str, cat_threshold: Union[int, float, None] = .9):
    """Checks whether a given column in a dataframe is a classification (a category or some binary value).

    Parameters
    ----------
    df : DataFrame
        The dataframe.
    column : str
        Column name.
    cat_threshold : int or float or None, optional
        Maximum proportion between unique values and total that is considered to be categorical. Default is .9.

    See Also
    --------
    is_categorical : for a detailed explanation on ``cat_threshold``.

    Returns
    -------
    bool
        True if ``column`` is either categorical or binary, False otherwise.

    """
    return is_categorical(df, column, cat_threshold) or is_binary(df, column)


def is_correlatable(df: DataFrame,
                    column: str,
                    cat_threshold: Union[int, float, None] = .9):
    """Checks whether a given column in a dataframe can be used to calculate correlation.

    Parameters
    ----------
    df : DataFrame
        The dataframe.
    column : str
        Column name.
    cat_threshold : int or float or None, optional
        Maximum proportion between unique values and total that is considered to be categorical. Default is .9.

    See Also
    --------
    is_categorical : for a detailed explanation on ``cat_threshold``.

    Returns
    -------
    bool
        True if ``column`` is either numeric, categorical or binary. False otherwise.

    """
    return (is_binary(df, column) or
            is_numeric(df, column) or
            is_categorical(df, column, cat_threshold))


def correlatable_columns(df: DataFrame,
                         subset: Union[str, List[str]] = 'correlatable',
                         cat_threshold: Union[int, float, None] = .9):
    """Selects a subset of correlatable columns in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The dataframe.
    subset : str or list of str, optional
        Can be either a list of column names or one of:

        - ``'correlatable'``: Select all correlatable columns (default)
        - ``'binary'``: Retrieve only binary columns
        - ``'categorical'``: Retrieve only categorical (nominal) columns
        - ``'class'``: Retrieve categorical and binary columns
        - ``'numeric'``: Retrieve only numeric columns
    cat_threshold : int or float or None, optional
        Maximum proportion between unique values and total that is considered to be categorical. Default is .9.

    See Also
    --------
    is_binary      : Checks whether a given column in a dataframe is binary.
    is_categorical : Checks whether a given column in a dataframe is categorical. Also includes a detailed explanation
                     on``cat_threshold``.
    is_numeric     : Checks whether a given column in a dataframe is numeric.

    Returns
    -------
        list of str
            A list with the selected columns' namess
    """
    if subset in ['auto', 'correlatable']:
        subset = (c for c in df.columns if
                  is_correlatable(df, c, cat_threshold=cat_threshold))
    elif subset in ['binary', 'boolean', 'bool', 'b']:
        subset = (c for c in df.columns if is_binary(df, c))
    elif subset in ['categorical', 'cat', 'c']:
        subset = (c for c in df.columns if
                  is_categorical(df, c, cat_threshold=cat_threshold))
    elif subset == 'class':
        subset = (c for c in df.columns if is_class(df, c, cat_threshold=cat_threshold))
    elif subset in ['numeric', 'num', 'n']:
        subset = (c for c in df.columns if is_numeric(df, c))
    else:
        subset = U.to_list(subset)
    return set(subset)


# end region

# region sequences

def sessionize(df: DataFrame,
               threshold: Any,
               id_col: Union[str, Column] = 'id',
               ordering_col: Union[str, Column] = 'created_at',
               interval_function: Callable[[Column, Column], Any] = operator.sub,
               output_col: str = 'session',
               interval_col: str = None,
               rank_col: str = None) -> DataFrame:
    """Determines sessions for an user event dataframe.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe.
    threshold : Any
        Maximum interval between events such that they are considered to be in the same session.
    id_col : pyspark.sql.Column or str, optional
        Name of the user identifying column, used to group events from a user into a single row. (Default: id)
    ordering_col : pyspark.sql.Column or str, optional
        Name of the column used to order the events. (Default: created_at)
    interval_function : callable, optional
        Lambda function used to evaluate intervals between events. This should receive two ``pyspark.sql.Column`` and
        return an object comparable with ``session_thresold``. Defaults to ``operator.sub``, that is, usual subtraction.
    output_col : str, optional
        Name of the output column (default: ``session``).
    rank_col, interval_column : str, optional
        Names of the columns to receive event rank (number) within the session and interval between events. None to
        ignore (default).

    Returns
    -------
    pyspark.sql.DataFrame
        The given dataframe with an additional column containing session information.

    Examples
    --------
    .. jupyter-execute:: /examples/analysis/utils/sessionize.py

    """
    window = Window.partitionBy(id_col).orderBy(ordering_col)
    df = df.withColumn('_lag', interval_function(F.col(ordering_col),
                                                 F.lag(ordering_col).over(window)))
    df = df.withColumn('_session_change',
                       F.when(F.col('_lag') > F.lit(threshold), 1).otherwise(0))
    if interval_col:
        df = df.withColumnRenamed('_lag', interval_col)
    else:
        df = df.drop('_lag')
    df = df.withColumn(output_col, F.sum('_session_change').over(window))
    if rank_col:
        window = window.partitionBy(id_col, output_col)
        df = df.withColumn(rank_col, F.rank().over(window))
    return df.drop('_session_change')


def sequentiate(df: DataFrame,
                id_col: Union[str, Column] = 'id',
                ordering_col: Union[str, Column] = 'created_at',
                event_col: Union[str, Column] = 'event',
                session_col: Union[str, Column] = None,
                output_col: str = 'journey',
                partial_output_col: str = None,
                rank_output_col: str = None,
                max_depth: int = None) -> DataFrame:
    """Determines the sequence of events for each user during each session given a user event dataframe.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe.
    id_col : pyspark.sql.Column or str, optional
        User identifying column, used to group events from a user into a single row. (Default: id)
    ordering_col : pyspark.sql.Column or str, optional
        Column used to order the events. (Default: created_at)
    event_col : pyspark.sql.Column or str, optional
        Column with the event information. (Default: event)
    session_col : str, optional
        Name of the column with the session identifier. Set to ``None`` for no session grouping (default).
    output_col : str, optional
        Name of the output column (default: ``journey``).
    partial_output_col : str, optional
        Name of the column for partial output (default: None).
        Partial output for this function is the sequence of event up until (and including) the current event.
    rank_output_col: str, optional
        Name of the column for rank output (default: None).
        Rank is the position of some event on the full sequence of events.
    max_depth : int, optional
        Maximum event sequence length. None for unlimited (default).

    See Also examples/
    --------
    sessionize : Determine sessions for an user event dataframe.

    Returns
    -------
    pyspark.sql.DataFrame
        The given dataframe with an additional column containing event sequence information.

    Examples
    --------
    .. jupyter-execute:: /examples/analysis/utils/journey.py

    """
    if session_col:
        window = Window.partitionBy(id_col, session_col)
    else:
        window = Window.partitionBy(id_col)
    window = window.orderBy(ordering_col)

    df = df.withColumn('_rank', F.rank().over(window))
    if max_depth:
        df = df.filter(F.col('_rank') <= max_depth)

    if partial_output_col:
        df = df.withColumn(output_col, F.collect_list(event_col).over(window))

    if output_col:
        df = df.withColumn(output_col, F.collect_list(event_col).over(
            window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))

    if rank_output_col:
        df = df.withColumnRenamed("_rank", rank_output_col)
    else:
        df = df.drop("_rank")

    identity = [id_col, session_col] if session_col else [id_col]
    df = df.drop_duplicates(identity)

    return df

# end region
