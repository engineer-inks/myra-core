import io
from functools import partial
from typing import List
from typing import Optional, Union

from .base import (adapter,
                    split_protocol_path,
                    without_protocol,
                    join_protocol_path,
                    handle_error)
from . import security


_ADAPTERS_AVAILABLE = {
    'file': ('config.local.class_config.adapters.file_system', 'FileSystemStorageAdapter'),
    'gs': ('config.local.class_config.adapters.google_cloud', 'GoogleCloudStorageAdapter')
}
_ADAPTERS = {}

adapter = partial(adapter,
                  adapters=_ADAPTERS,
                  adapters_available=_ADAPTERS_AVAILABLE)


def open(filename: str, mode: str = 'rb', encoding='utf-8', on_error: str = 'raise'):
    """Open files from multiple sources.

    # TODO: Implement cryptography for sources.
    """
    if not 3 >= len(mode) >= 1 or len(mode) > 1 and mode[1] not in {'+', 'b'}:
        return handle_error('invalid mode: ' + mode, on_error, ValueError)

    both = len(mode) > 1 and mode[1] == '+'
    binary = mode[-1] == 'b'

    protocol, filename = split_protocol_path(filename)
    try:
        ap = adapter(protocol)
        mode = '+' if both else mode[0]

        if mode[0] == '+':
            stream = io.BufferedRWPair(ap.reader(filename), ap.writer(filename))
        elif mode[0] == 'r':
            stream = ap.reader(filename)
        elif mode[0] in ('w', 'a'):
            stream = ap.writer(filename, append=mode[0] == 'a')
        else:
            return handle_error('invalid mode: ' + mode, on_error, ValueError)

        if not binary:
            stream = io.TextIOWrapper(stream, encoding)

        return stream
    except FileNotFoundError as error:
        handle_error(error, on_error)


def read(filename: str, encoding='utf-8', crypto: security.Crypto = None,
         on_error: str = 'raise') -> str:
    """Reads a file from storage.

    Parameters
    ----------
    filename : str
    crypto : security.Crypto
        Cryptography agent to be used for decryption.
    encoding : str, optional
        Encoding to read the file into (default: ``'utf-8'``).
    on_error : {'raise', 'log'}, optional
        Behavior in case errors happen during the process (default: ``'raise'``).

    Returns
    -------
    str

    Raises
    ------
    FileNotFoundError
        If file does not exist in storage.

    """
    protocol, filename = split_protocol_path(filename)
    try:
        data = adapter(protocol).read(filename)
        if crypto:
            data = crypto.decrypt(data)
        return data.decode(encoding)
    except FileNotFoundError as error:
        handle_error(error, on_error)


def write(filename: str, data: Union[bytes, str], encoding='utf-8', crypto: security.Crypto = None):
    """Writes data to a file on storage.

    Parameters
    ----------
    filename : str
    data : bytes or str
    encoding : str, optional
        Encoding to use when ``data`` is of type ``str`` (default: ``'utf-8'``).
    crypto : security.Crypto
        Cryptography agent to be used for encryption.

    """
    protocol, filename = split_protocol_path(filename)
    if isinstance(data, str):
        data = data.encode(encoding)
    if crypto:
        data = crypto.encrypt(data)
    adapter(protocol).write(filename, data)


def delete(filename: str, on_error: str = 'raise'):
    """Deletes a file from storage.

    Parameters
    ----------
    filename : str
    on_error : {'raise', 'log'}, optional
        Behavior in case errors happen during the process (default: ``'raise'``).

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.

    """
    protocol, filename = split_protocol_path(filename)
    try:
        adapter(protocol).delete(filename)
    except FileNotFoundError as error:
        handle_error(error, on_error)


def copy(source: str, target: str, on_error: str = 'raise', move: bool = False):
    """Copies a file internally or between storages.

    Parameters
    ----------
    source, target : str
        Source and target file names.
    on_error : {'raise', 'log'}, optional
        Behavior in case errors happen during the process (default: ``'raise'``).
    move : bool, optional
        Whether to delete the file after copying (default: False).

    Raises
    ------
    FileNotFoundError
        If source file does not exist.

    """
    source_protocol, source_filename = split_protocol_path(source)
    target_protocol, target_filename = split_protocol_path(target)
    try:
        adapter(source_protocol).copy(source_filename, target_filename,
                                      other=adapter(target_protocol), move=move)
    except FileNotFoundError as error:
        handle_error(error, on_error)


def move(source: str, target: str, on_error: str = 'raise'):
    """Moves a file internally or between storages.

    Parameters
    ----------
    source, target : str
        Source and target file names.
    on_error : {'raise', 'log'}, optional
        Behavior in case errors happen during the process (default: ``'raise'``).

    Raises
    ------
    FileNotFoundError
        If source file does not exist.

    """
    copy(source, target, on_error, move=True)


def listdir(dirname: str, matching: Optional[str] = None, on_error: str = 'raise') -> List[str]:
    """Lists contents of a directory on storage.

    Parameters
    ----------
    dirname : str
        Name of the directory.
    matching : str
        Regular expression to filter files with.
    on_error : {'raise', 'log'}, optional
        Behavior in case errors happen during the process (default: ``'raise'``).

    Returns
    -------
    list of str
        A list with the names of the files in the directory.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.

    """
    protocol, dirname = split_protocol_path(dirname)
    try:
        return adapter(protocol).listdir(dirname, matching)
    except FileNotFoundError as error:
        handle_error(error, on_error)


def isdir(path: str, on_error: str = 'raise') -> List[str]:
    """Determines whether a given path refers to an existing directory.

    Parameters
    ----------
    path : str
    on_error : {'raise', 'log'}, optional
        Behavior in case errors happen during the process (default: ``'raise'``).

    Returns
    -------
    bool

    """
    protocol, path = split_protocol_path(path)
    try:
        return adapter(protocol).isdir(path)
    except FileNotFoundError as error:
        handle_error(error, on_error)


def isfile(path: str, on_error: str = 'raise') -> List[str]:
    """Determines whether a given path refers to an existing file.

    Parameters
    ----------
    path : str
    on_error : {'raise', 'log'}, optional
        Behavior in case errors happen during the process (default: ``'raise'``).

    Returns
    -------
    bool

    """
    protocol, path = split_protocol_path(path)
    try:
        return adapter(protocol).isfile(path)
    except FileNotFoundError as error:
        handle_error(error, on_error)


def exists(filename: str) -> bool:
    """Checks whether a file exists in storage

    Parameters
    ----------
    filename : str

    Returns
    -------
    bool

    """
    protocol, filename = split_protocol_path(filename)
    return adapter(protocol).exists(filename)


__all__ = [
    'split_protocol_path', 'without_protocol', 'join_protocol_path',
    'adapter',
    'open', 'read', 'write', 'delete', 'copy', 'move',
    'listdir', 'isdir', 'isfile', 'exists',
]
