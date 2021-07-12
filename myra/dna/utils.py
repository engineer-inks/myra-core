import logging
import time
from datetime import datetime
from typing import Any, List, Optional, Union, Dict


def is_list(x: Any) -> bool:
    """Check if input is a list or list-like.

    Parameters
    ----------
    x: any
       the object being checked, usually a frame/list of
       frames or record in the processing pipeline

    Returns
    -------
    bool
        Whether the input is a list-like object or not.

    """
    return isinstance(x, (list, tuple))


def to_list(x: Any) -> List[Any]:
    """Ensure the parameter is a list by wrapping it if necessary.

    Parameters
    ----------
    x: any
       the entry being evaluated, usually a frame
       or data record in the processing stream

    Returns
    -------
    list
        The input :code:`x` if it is a list. Otherwise, :code:`[x]` is passed.

    """
    return x if is_list(x) else [x]


def unpack(x: List[Any], on_many: str = 'warn') -> Any:
    """Undoes the :func:`to_list` function by extracting the first element.

    If the list has more than one element, :code:`on_many` parameter will
    dictate the behavior of this function.

    Parameters
    ----------
    x: list
        the list to be unpacked
    on_many: str
        what to do whenever the input list :code:`x` contains more
        than one element. "warn" is the default, the first element is unpacked
        and a warning is signaled stating about possible data loss. If "keep"
        is passed, unpacking is skipped

    Returns
    -------
    any
        The input unwrapped.

    """
    if not is_list(x):
        return x

    if len(x) <= 1:
        return next(iter(x))

    if on_many == 'warn':
        logging.warning(f'Multiple values to unpack from {x}. '
                        f'Some information may be lost.')

        return next(iter(x))

    if on_many == 'keep':
        return x

    raise ValueError(f'on_many parameter "{on_many}" is illegal for unpack function. '
                     f'Try "warn" to unpack the first element regardless the input size '
                     f'or "keep" to prevent unpacking if list has more than one element.')


def valid_or_default(a: Optional[Any], b: Any) -> Any:
    """Returns the first parameter a if it's valid.

    Parameters
    ----------
    a: any
       the parameter checked
    b: any
       the second choice in case :code:`a` is invalid

    Returns
    -------
    any
        :code:`a` if it is valid. :code:`b` otherwise.

    """
    return a if a is not None else b


def as_row(obj: Union[Dict, List[Dict]]):
    """Convert a dict into a :code:`pyspark.sql.Row`.

    If a list of dicts is passed, then this function is recursively
    applied to each element of the list.

    Parameters
    ----------
    obj: dict or list of dictionaries
         the object that will be converted into a Row

    Returns
    -------
    :class:`pyspark.sql.Row`
        A Row object containing all of the information within
        :code:`obj`.

    """
    from pyspark.sql import Row

    if isinstance(obj, dict):
        d = {k: as_row(v) for k, v in obj.items()}
        return Row(**d)

    if is_list(obj):
        return [as_row(v) for v in obj]

    return obj


class NamedEntityMixin:
    @classmethod
    def fullname(cls) -> str:
        """Retrieve a fully-representative name for this entity.
        """
        m = cls.__module__
        if m is None or m == str.__class__.__module__ or str(m) == '__main__':
            return cls.__name__  # Avoid reporting __builtin__
        return m + '.' + cls.__name__


class stopwatch:
    """Measure elapsed time for python-operations.

    This will consider blocked wall time as well, as much of the operations
    are committed during :code:`spark.DataFrame#write`.

    Parameters
    ----------
    label: str
        title used in report

    mode: str
        type of report produced.
        Options are 'short', 'long' or 'silent'

    Examples
    --------
    .. jupyter-execute::

        import dextra.dna.core as C

        with C.utils.stopwatch('Loading iris'):
            print('Samples loaded:', C.datasets.iris().count())

    """

    def __init__(self, label: str = 'execution time', mode: str = 'long'):
        if mode not in ('short', 'long', 'silent'):
            raise ValueError(f'report parameter ({mode}) should be '
                             f'either "short", "long" or "silent".')

        self.label = label
        self.mode = mode

        self.started_at = None
        self.ended_at = None
        self.perf_start = None

    @property
    def elapsed(self):
        """Time, in seconds, between start and end of the window
           profiled by the :code:`stopwatch`.
        """
        return self.perf_end - self.perf_start

    @property
    def _indent(self):
        return ' ' * (len(self.label) - len(self.label.lstrip()) + 2)

    def __enter__(self):
        self.started_at = datetime.now()
        self.perf_start = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ended_at = datetime.now()
        self.perf_end = time.perf_counter()

        if self.mode == 'short':
            print(f'{self.label}: {self.elapsed:.3f} seconds')
        elif self.mode == 'long':
            print(self.label)
            print(f'{self._indent}started at: {self.started_at}')
            print(f'{self._indent}ended at:   {self.ended_at}')
            print(f'{self._indent}elapsed:    {self.elapsed:.3f} seconds')
