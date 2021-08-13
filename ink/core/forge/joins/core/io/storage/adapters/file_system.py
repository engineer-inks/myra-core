import os

import shutil
from typing import Optional, BinaryIO

from .storage_adapter import StorageAdapter


class FileSystemStorageAdapter(StorageAdapter):
    def reader(self, filename: str) -> BinaryIO:
        return open(filename, 'rb')

    def writer(self, filename: str, append: bool = False) -> BinaryIO:
        return open(filename, 'ab' if append else 'wb')

    def makedirs(self, filename: str):
        dirname = os.path.dirname(filename.rstrip('/'))
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def write(self, filename: str, *args):
        self.makedirs(filename)
        super().write(filename, *args)

    _listdir = staticmethod(os.listdir)
    isdir = staticmethod(os.path.isdir)
    isfile = staticmethod(os.path.isfile)

    def delete(self, filename: str):
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        else:
            os.remove(filename)

    def copy(self, source: str, target: str, other: Optional['StorageAdapter'] = None, move: bool = False):
        self._pre_copy(source, target, other)

        if not isinstance(other, FileSystemStorageAdapter):
            return super().copy(source, target, other, move=move)

        if move:
            return shutil.move(source, target)

        if os.path.isdir(source):
            return shutil.copytree(source, target)

        shutil.copy(source, target)

    def exists(self, filename: str) -> bool:
        return os.path.exists(filename)
