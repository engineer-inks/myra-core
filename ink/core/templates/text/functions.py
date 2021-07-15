"""Functions applicable to unstructured text spark columns.

.. jupyter-execute::
    :hide-code:

    import ink.core.templates.text as T

    x = T.datasets.newsgroups20()

"""

import re
from typing import Union, List, Iterable, Optional, Tuple

import numpy as np
from pyspark.sql import types as T, functions as F
from pyspark.sql.column import Column

from .. import utils


def clean(col: Union[Column, str]) -> Column:
    """Clean text using :func:`dextra.dna.text.utils.clean`.

    Parameters
    ----------
    col: str or Column
        column containing unstructured text that will be cleaned
    
    Returns
    -------
    Column of str
        The cleaned column ``col``.
    
    Examples
    --------
    .. jupyter-execute::

        (x.withColumn('text_cleaned', T.functions.clean('text'))
          .select('text', 'text_cleaned')
          .limit(5)
          .toPandas())

    """
    return F.udf(utils.clean)(col)


def contains(col: Union[Column, str],
             terms: Iterable[str]) -> Column:
    """Checks if any of the terms are contained in the column.

    The function :func:`dextra.dna.text.utils.contains` is used underneath.

    Parameters
    ----------
    col: str or Column
        column containing unstructured text that will be checked
    
    Returns
    -------
    Column of int
        The column containing ``1`` or ``0``.

    Examples
    --------
    .. jupyter-execute::

        words = ['washington', 'purdue']
        (x.withColumn('washington_or_purdue', T.functions.contains('text', words))
          .select('text', 'washington_or_purdue')
          .limit(5)
          .toPandas())

    """
    terms = utils.to_list(terms)
    terms = r'|'.join(terms)

    if not terms:
        raise ValueError(f'Illegal empty terms passed: {terms}.')

    return F.udf(lambda x: int(utils.contains(x, pattern=terms)),
                 returnType=T.IntegerType())(col)


def replace(col: Union[Column, str],
            expressions: Union[str, List[str], Tuple[str]],
            replacement: str) -> Column:
    """Replace occurrences of any of the expressions in a column of free text.

    Parameters
    ----------
    col: str or Column
        column containing unstructured text that will be processed
    expressions: str or list of strings
        expression or list of expressions that should be replaced
    replacement: str
        string that will replace the occurrences in ``col``

    Returns
    -------
    Column of str
        The column ``col`` with all ``expressions`` occurrences
        replaced by ``replacement``.

    Examples
    --------
    .. jupyter-execute::

        email_pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',

        (x.withColumn('text', T.functions.replace('text', email_pattern, '{email}'))
          .limit(5)
          .toPandas())

    """
    expressions = utils.to_list(expressions)
    expressions = [r for r in expressions if r.strip()]
    expressions.sort(key=lambda s: -len(s))
    matching = r'|'.join(expressions)

    if not matching:
        raise ValueError('`expressions` must be a non empty list of strings that represent a regex rule. '
                         'Furthermore, it should not include expressions containing only white spaces, '
                         'tabs and line breaks.')

    return F.regexp_replace(col, matching, replacement)


def extract(col: Union[Column, str],
            expressions: Union[str, List[str]],
            flags: int = 0) -> Column:
    """Extract the first occurrence of any of the expressions
    in a column of free text.

    Parameters
    ----------
    col: str or Column
        column containing unstructured text that will be processed
    expressions: str or list of strings
        expression or list of expressions that should be extracted
    flags: int
        int containing any regex matching flags

    Returns
    -------
    Column of str
        A column containing the values extracted from ``col``.

    Examples
    --------
    .. jupyter-execute::

        email_pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',

        (x.withColumn('text', T.functions.extract('text', email_pattern))
          .limit(5)
          .toPandas())

    """
    expressions = utils.to_list(expressions)
    matching = r'|'.join(expressions)

    return F.udf(lambda d: utils.extract(d, matching, flags))(col)


def argmax(col: Union[Column, str],
           vocabulary: List[str],
           mask: Optional[np.ndarray] = None):
    mask = (np.asarray(mask)
            if mask is not None
            else np.ones(len(vocabulary)))

    return F.udf(lambda y: (vocabulary[np.argmax(y * mask).item()]
                            if ((y.toArray() * mask) > 0).any()
                            else None))(col)
