import abc
import re
import io
import os
from typing import List, Optional, Union, BinaryIO


class StorageAdapter(metaclass=abc.ABCMeta):
    """Abstract storage environment adapter."""

    @abc.abstractmethod
    def reader(self, filename: str) -> BinaryIO:
        """Opens a read stream for a file from storage.

        Parameters
        ----------
        filename : str

        Returns
        -------
        file-like object

        Raises
        ------
        FileNotFoundError
            If file does not exist in storage.

        """
        ...

    def read(self, filename: str) -> bytes:
        """Reads a file from storage.

        Parameters
        ----------
        filename : str

        Returns
        -------
        bytes

        Raises
        ------
        FileNotFoundError
            If file does not exist in storage.

        """
        with self.reader(filename) as file:
            return file.read()

    @abc.abstractmethod
    def writer(self, filename: str, append: bool = False) -> BinaryIO:
        """Opens a write stream for a file on storage.

        Parameters
        ----------
        filename : str
        append : bool, optional
            Whether to open file in append mode (default: False).

        Returns
        -------
        file-like object

        Raises
        ------
        FileNotFoundError
            If file does not exist in storage.

        """
        ...

    def write(self, filename: str, data: Union[bytes, BinaryIO], buffer_size: Optional[int] = io.DEFAULT_BUFFER_SIZE):
        """Writes data to a file on storage.

        Parameters
        ----------
        filename : str
        data : bytes or str
        buffer_size : int, optional
            Number of bytes to write at a time. Used only when data is a file-like object.

        """
        with self.writer(filename) as file:
            if not isinstance(data, (bytes, bytearray)):
                for chunk in iter(lambda: data.read(buffer_size), b''):
                    file.write(chunk)
            else:
                file.write(data)

    @abc.abstractmethod
    def isdir(self, path: str) -> bool:
        """Determines whether a given path refers to an existing directory.

        Parameters
        ----------
        path : str

        Returns
        -------
        bool

        """
        ...

    @abc.abstractmethod
    def isfile(self, path: str) -> bool:
        """Determines whether a given path refers to an existing file.

        Parameters
        ----------
        path : str

        Returns
        -------
        bool

        """
        ...

    def listdir(self, dirname: str, matching: Optional[str] = None) -> List[str]:
        """Lists contents of a directory on storage.

        Parameters
        ----------
        dirname : str
            Name of the directory.
        matching : str
            Regular expression to filter files with.

        Returns
        -------
        list of str
            A list with the names of the files in the directory.

        Raises
        ------
        FileNotFoundError
            If the specified directory does not exist.
        NotADirectoryError
            If the path does not refer to a directory.

        """
        files = self._listdir(dirname)
        if isinstance(matching, str):
            matching = re.compile(matching)
            return [f for f in files if not matching or matching.search(f)]
        else:
            return files

    @abc.abstractmethod
    def _listdir(self, dirname: str) -> List[str]:
        """Lists contents of a directory on storage.

        Parameters
        ----------
        dirname : str
            Name of the directory.

        Returns
        -------
        list of str
            A list with the names of the files in the directory.

        Raises
        ------
        FileNotFoundError
            If the specified directory does not exist.

        """
        ...

    @abc.abstractmethod
    def delete(self, filename: str):
        """Deletes a file from storage.

        Parameters
        ----------
        filename : str

        Raises
        ------
        FileNotFoundError
            If the specified directory does not exist.

        """
        ...

    @abc.abstractmethod
    def exists(self, filename: str) -> bool:
        """Checks whether a file exists in storage

        Parameters
        ----------
        filename : str

        Returns
        -------
        bool

        """
        ...

    @abc.abstractmethod
    def makedirs(self, filename: str):
        """Create a tree os directories if it does not yet exist.

        Parameters
        ----------
        filename : str
        """

    def _pre_copy(self, source: str, target: str, other: 'StorageAdapter'):
        if not self.exists(source):
            raise FileNotFoundError()

        (other or self).makedirs(target)

    def copy(self, source: str, target: str, other: Optional['StorageAdapter'] = None, move: bool = False):
        """Copies a file internally or between storages.

        Parameters
        ----------
        source, target : str
            Source and target file names.
        other : StorageAdapter, optional
            ``StorageAdapter`` for target storage environment.
        move : bool, optional
            Whether to delete the file after copying (default: False).

        Raises
        ------
        FileNotFoundError
            If source file does not exist.

        """
        if other is None:
            other = self

        self._pre_copy(source, target, other)

        if self.isfile(source):
            if other.isdir(target):
                target = os.path.join(target, os.path.basename(source))
            other.write(target, self.reader(source))

        elif self.isdir(source):
            if other.isfile(target):
                raise NotADirectoryError(f"Trying to copy a directory but target {target} is not a directory")
            for sub in self.listdir(source):
                self.copy(os.path.join(source, sub), os.path.join(target, sub))
        
        else:
            raise FileNotFoundError(f"{source} is not a file or a directory.")

        if move:
            self.delete(source)

    def move(self, source: str, target: str, other: Optional['StorageAdapter'] = None):
        """Moves a file internally or between storages.

        Parameters
        ----------
        source, target : str
            Source and target file names.
        other : StorageAdapter, optional
            ``StorageAdapter`` for target storage environment.

        Raises
        ------
        FileNotFoundError
            If source file does not exist.

        """
        self.copy(source, target, other, move=True)
