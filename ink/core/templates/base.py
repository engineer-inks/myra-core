import importlib
import logging
from typing import Dict, Any, Union, Optional, Type, Tuple

PROTOCOL_SEP = '://'


def _assert_known_protocol(protocol, known_protocols):
    if protocol not in known_protocols:
        available = ', '.join(known_protocols)
        raise ValueError(f'Protocol scheme `{protocol}` is not supported. '
                         f'Available protocols are: {available}.')


def adapter(protocol: str,
            adapters: Dict[str, Any],
            adapters_available: Dict[str, str]):
    """Retrieve the appropriate adapter for a given protocol.

    If the adapter is not found in the :attr:`adapters` pool, its class will
    be fetched from :attr:`adapters_available`, instantiated and added to the pool.
    No adapter (nor its module) will be imported into the program's memory unless
    necessary. This allows us to clip dependencies for sources we are not using,
    as no direct references are pointed.
    Example: a project that only runs locally can safely remove the
    `google-cloud-storage` dependency and still work, as the GoogleStorageAdapter
    will never be imported.

    Parameters
    ----------
    protocol: str
        the protocol of interest
    adapters: dict of adapters started
    adapters_available: dict of paths to adapters registered

    Returns
    -------
    Adapter
        Storage or stream adapter
    """
    if protocol not in adapters:
        _assert_known_protocol(protocol, adapters_available)

        module, cls = adapters_available[protocol]
        module = importlib.import_module(module, __package__)
        cls = getattr(module, cls)
        adapters[protocol] = cls()
    return adapters[protocol]


def split_protocol_path(location: str) -> Tuple[str, str]:
    """Separates and returns protocol and path from a location.

    Parameters
    ----------
    location : str
        A valid URI with format ``[protocol://]path``. If ``protocol://`` is not present, ``file://`` is inferred.

    Returns
    -------
    protocol : str
    path : str

    """
    parts = location.split(PROTOCOL_SEP)

    if len(parts) > 2:
        raise ValueError(f'Malformed location "{location}".')

    if len(parts) == 1:
        return 'file', location

    protocol, path = parts
    return protocol, path


def without_protocol(location: str) -> str:
    """Returns a location without its associated protocol.

    Parameters
    ----------
    location : str
        A valid URI with format ``[protocol://]path``

    Returns
    -------
    str

    """
    return split_protocol_path(location)[-1]


def join_protocol_path(protocol: str, location: str) -> str:
    """Joins a protocol and some location.

    Parameters
    ----------
    protocol : str
    location : str

    Returns
    -------
    str
        A location with format ``protocol://path``, where ``path`` is the given location with its protocol removed, if
        it has any.

    """
    return f'{protocol}://{without_protocol(location)}'


# region error handling

ON_ERROR_HANDLERS = ('raise', 'log')


def handle_error(error: Union[Exception, str],
                 on_error: str = 'raise',
                 wrapper_cls: Optional[Type[Exception]] = None):
    """Raises or logs occurring errors based on the caller choice.

    :param error: the error that occurred
    :type error: Exception, str
    :param on_error: how to handle the error (raise, log or ignore)
    :type on_error: str
    :param wrapper_cls: an exception class to wrap around the error.
                        This is useful whenever you want to merge
                        errors from multiple backends into a single
                        base error.
    :type wrapper_cls: Type[Exception]
    """
    if on_error == 'raise':
        raise wrapper_cls(error) if wrapper_cls else error

    if on_error == 'log':
        return logging.error(error)

    if on_error == 'ignore':
        return

    if on_error not in ON_ERROR_HANDLERS:
        raise ValueError(f'`handle_error` function cannot understand '
                         f'on_error="{on_error}". It should assume '
                         f'one of the following values: {ON_ERROR_HANDLERS}.')

# endregion
