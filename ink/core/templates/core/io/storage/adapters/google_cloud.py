import io
import os
from typing import Tuple, List, Optional, Union, BinaryIO

import google
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage as gcs
from google.resumable_media.requests import ResumableUpload

from .file_system import FileSystemStorageAdapter
from .storage_adapter import StorageAdapter


class IntermediateBuffer(io.BufferedIOBase, BinaryIO):
    def __init__(self):
        self._buffer = b''
        self._offset = 0
        self._length = 0

    def write(self, data: Union[bytes, bytearray]) -> int:
        size = len(data)
        self._buffer += data
        self._length += size
        del data
        return size

    def read1(self, size: Optional[int] = -1) -> bytes:
        if size == -1:
            return self.readall()
        self._offset += min(self.size - self._offset, size)
        data, self._buffer = self._buffer[:size], self._buffer[size:]
        return data

    def readall(self):
        self._offset = self._length
        return self._buffer

    read = read1

    def tell(self):
        return self._offset

    def __len__(self):
        return self._length

    size = property(__len__)

    def seekable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


class GoogleCloudStorageIO(io.BufferedIOBase, BinaryIO):
    BUFFER_SIZE = 8 * 1024 * 1024

    def __init__(self,
                 blob: gcs.Blob,
                 resumable_upload: ResumableUpload = None,
                 output_buffer: IntermediateBuffer = None):
        super().__init__()
        self._blob = blob
        self._offset = 0

        self._resumable_upload = resumable_upload
        self._output_buffer = output_buffer
        self._transport = None
        self._output_buffer_bytes = 0

        self._input_buffer = None

    @property
    def size(self):
        if self._blob.size is None:
            self._blob.reload()
        return self._blob.size

    @property
    def _buffered_bytes(self):
        return self._input_buffer.size - self._input_buffer.tell()

    @property
    def _ended(self):
        return self._offset >= self.size - 1

    def _feed_buffer(self, size: int):
        self._input_buffer = self._input_buffer or IntermediateBuffer()
        while self._buffered_bytes < size and not self._ended:
            end = min(self._offset + max(self.BUFFER_SIZE, size), self.size) - 1
            self._blob.download_to_file(self._input_buffer,
                                        start=self._offset,
                                        end=end)
            self._offset = end

    def read1(self, size: Optional[int] = -1) -> bytes:
        try:
            if size == -1:
                return self.readall()
            else:
                if self._offset >= self.size or size == 0:
                    return b''
                self._feed_buffer(size)
                data = self._input_buffer.read(size)
                return data
        except google.api_core.exceptions.NotFound as error:
            raise FileNotFoundError(error)

    read = read1

    def readinto(self, b: bytearray) -> Optional[int]:
        try:
            if self._offset >= self.size:
                return 0
            with io.BytesIO(b) as stream:
                start = stream.tell()
                self._blob.download_to_file(stream, start=self._offset)
                self._offset = self.size
                return stream.tell() - start
        except google.api_core.exceptions.NotFound as error:
            raise FileNotFoundError(error)

    def readall(self) -> bytes:
        try:
            if self._offset >= self.size:
                return b''
            with io.BytesIO() as stream:
                self._blob.download_to_file(stream, start=self._offset)
                self._offset = self.size
                return stream.getvalue()
        except google.api_core.exceptions.NotFound as error:
            raise FileNotFoundError(error)

    def _initiate_upload(self):
        if self._resumable_upload is None:
            self._resumable_upload = ResumableUpload(
                upload_url=f'https://www.googleapis.com/upload/storage/v1/b/'
                        f'{self._blob.bucket.name}/o?uploadType=resumable',
                chunk_size=self.BUFFER_SIZE)
            self._transport = AuthorizedSession(credentials=self._blob.client._credentials)
        
            self._output_buffer = self._output_buffer or IntermediateBuffer()
            self._resumable_upload.initiate(self._transport,
                                            stream=self._output_buffer,
                                            metadata={'name': self._blob.name},
                                            content_type='application/octet-stream',
                                            stream_final=False)

    def _transmit_next_chunk(self):
        try:
            self._resumable_upload.transmit_next_chunk(self._transport)
        except google.resumable_media.common.InvalidResponse:
            self._resumable_upload.recover(self._transport)
        self._output_buffer_bytes -= min(self._output_buffer_bytes, self.BUFFER_SIZE)

    def write(self, b: Union[bytes, bytearray, BinaryIO]) -> int:
        self._initiate_upload()
        if isinstance(b, (bytes, bytearray)):
            self._output_buffer.write(b)
            self._output_buffer_bytes += len(b)
            while self._output_buffer_bytes >= self.BUFFER_SIZE:
                self._transmit_next_chunk()
        else:
            for chunk in iter(lambda: b.read(self.BUFFER_SIZE), b''):
                self.write(chunk)

    def flush(self):
        while self._output_buffer_bytes > 0:
            self._transmit_next_chunk()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._offset = offset
        elif whence == io.SEEK_CUR:
            self._offset += offset
        elif whence == io.SEEK_END:
            self._offset = self.size + offset
        else:
            raise ValueError("Invalid value for `whence`")
        return self._offset

    def tell(self) -> int:
        return self._offset

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        if self._output_buffer is not None:
            self.flush()
            del self._output_buffer
        if self._input_buffer is not None:
            del self._input_buffer
        super().close()


class GoogleCloudStorageAdapter(StorageAdapter):
    def __init__(self, client: Optional[gcs.Client] = None):
        super().__init__()
        self._client = client

    def duplex(self, filename: str) -> GoogleCloudStorageIO:
        return GoogleCloudStorageIO(self._blob(filename))

    reader = duplex

    def writer(self, filename: str, append: bool = False) -> GoogleCloudStorageIO:
        self.makedirs(filename)
        stream = self.duplex(filename)
        if append:
            stream.seek(0, os.SEEK_END)
        return stream

    @staticmethod
    def _split_path(filename: str) -> Tuple[str, str]:
        bucket, *file = filename.split('/')
        return bucket, '/'.join(file)

    def _blob(self, filename: str) -> gcs.Blob:
        bucket, file = self._split_path(filename)
        bucket = self.client.get_bucket(bucket)
        return bucket.get_blob(file) or bucket.get_blob(file + '/') or bucket.blob(file)

    @property
    def client(self):
        if self._client is None:
            self._client = gcs.Client()
        return self._client

    def isdir(self, path: str) -> bool:
        return (self.exists(path) and
                (not self._split_path(path)[1] or  # when bucket itself
                 self._blob(path).name.endswith('/')))

    def isfile(self, path: str) -> bool:
        return self.exists(path) and not self.isdir(path)

    def _listdir(self, dirname: str) -> List[str]:
        if not dirname.endswith('/'):
            dirname += '/'

        bucket_name, path = self._split_path(dirname)
        bucket = self.client.get_bucket(bucket_name)

        iterator = bucket.list_blobs(prefix=path, fields='items/name', delimiter='/')
        iterator.extra_params['includeTrailingDelimiter'] = True
        blobs = {b.name for b in iterator if b.name != path}

        iterator = bucket.list_blobs(prefix=path, fields='prefixes', delimiter='/')
        iterator.extra_params['includeTrailingDelimiter'] = True
        iterator._items_key = 'prefixes'
        iterator.item_to_value = lambda it, x: x
        blobs.update({b for b in iterator})

        try:
            return [b[len(path):].strip('/') for b in blobs]
        except google.api_core.exceptions.NotFound as error:
            raise FileNotFoundError(error)

    def delete(self, filename: str):
        try:
            if self.isdir(filename):
                for sub in self.listdir(filename):
                    self.delete(os.path.join(filename, sub))
            self._blob(filename).delete()
        except google.api_core.exceptions.NotFound as error:
            raise FileNotFoundError(error)

    def makedirs(self, path: str):
        path_ = ''
        for p in path.split('/')[:-1]:
            path_ += p + '/'
            if not self.exists(path_):
                self._blob(path_).upload_from_string(b'')

    def copy(self, source: str, target: str, other: Optional[StorageAdapter] = None, move: bool = False):
        super()._pre_copy(source, target, other)

        if isinstance(other, FileSystemStorageAdapter):
            return self._blob(source).download_to_filename(target)

        if self.isdir(source) or not isinstance(other, GoogleCloudStorageAdapter):
            return super().copy(source, target, other, move=move)

        try:
            source_bucket, source_file = self._split_path(source)
            target_bucket, target_file = self._split_path(target)
            blob = self._blob(source)
            if move and source_bucket == target_bucket:
                self.client.get_bucket(source_bucket).rename_blob(blob, target_file)
            else:
                self.client.get_bucket(source_bucket).copy_blob(blob,
                                                                self.client.get_bucket(target_bucket),
                                                                target_file)
                if move:
                    blob.delete()
        except google.api_core.exceptions.NotFound as error:
            raise FileNotFoundError(error)

    def exists(self, path: str) -> bool:
        bucket, filename = self._split_path(path)
        if not filename:
            return self.client.bucket(bucket).exists()
        return self._blob(path).exists()
