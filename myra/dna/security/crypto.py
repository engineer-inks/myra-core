from typing import Union

from .. import io


class NoPrivateKeyError(ValueError):
    pass


class Crypto:
    """Asymmetric encryption manager."""

    def __init__(self,
                 public_key: Union[str, bytes, 'RSAPublicKey'] = None,
                 private_key: Union[str, bytes, 'RSAPrivateKey'] = None):
        """
        If neither {public_key} nor {private_key} are passed, they'll be generated automatically.
        Decryption is only possible when {private_key} is passed.

        :param public_key: RSA public key in PEM format (default: None). Ignored when {privateKey} is not {None}.
        :param private_key: RSA private key in PEM format (default: None). If it is passed, {publicKey} will be derived
                           from this key.
        """
        if private_key is not None:
            self._init_with_private_key(private_key)
        elif public_key is not None:
            self._init_with_public_key(public_key)
        else:
            self._private_key = self.generate_private_key()
            self._public_key = self._private_key.public_key()

    def _init_with_private_key(self, private_key):
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

        if isinstance(private_key, RSAPrivateKey):
            self._private_key = private_key
        else:
            # TODO: Permitir senha na `privateKey`?
            self._private_key = serialization.load_pem_private_key(
                private_key,
                backend=default_backend(),
                password=None)
        self._public_key = self._private_key.public_key()

    def _init_with_public_key(self, public_key):
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

        if isinstance(public_key, RSAPublicKey):
            self._public_key = public_key
        else:
            self._public_key = serialization.load_pem_public_key(public_key, backend=default_backend())
        self._private_key = None

    @classmethod
    def generate_private_key(cls):
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa

        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

    @property
    def _private_bytes(self):
        from cryptography.hazmat.primitives import serialization

        return self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

    @property
    def _public_bytes(self):
        from cryptography.hazmat.primitives import serialization

        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    @classmethod
    def load(cls, public_file: str = None, private_file: str = None) -> 'Crypto':
        """Loads a key with some name and creates an instance of Crypto.

        :param public_file: public key PEM file name
        :param private_file: private key PEM file name
        :return: encrypt-only Crypto set up with the loaded public key.
        """
        if private_file:
            return Crypto(private_key=bytes(io.storage.read(private_file), encoding='utf8'))

        if public_file:
            return Crypto(public_key=bytes(io.storage.read(public_file), encoding='utf8'))

        raise ValueError(
            '`Crypto.load` needs at least one valid key file to be passed. '
            'Either point to a public or private key path when calling it.')

    def encrypt(self, data: Union[str, bytes, bytearray]) -> bytes:
        """Takes some data and outputs it as an encrypted byte sequence.

        :param data: data to be encrypted.
        :return: encrypted data.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        if isinstance(data, str):
            data = bytes(data, encoding='utf8')

        return self._public_key.encrypt(data,
                                        padding=padding.OAEP(
                                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                            algorithm=hashes.SHA256(),
                                            label=None
                                        ))

    def decrypt(self, data: bytes) -> bytes:
        """Takes some encrypted data and outputs it decrypted as a byte sequence.

        :param data: encrypted data to be decrypted.
        :return: decrypted data.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        if self._private_key is None:
            raise NoPrivateKeyError('No private key was set. Decrypting is not possible.')
        return self._private_key.decrypt(data, padding=padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        ))

    def save(self, public_file, private_file=None):
        """Saves cryptographic keys to the appropriate bucket in PEM format.

        :param public_file: filename to which the public key will be saved
        :param private_file: filename to which the private key will be saved
        """
        if self._private_key is not None and private_file is None:
            raise ValueError('Crypto instance contains a private key. The `private_file`'
                             'parameter must be passed in order to store it.')

        io.storage.write(public_file, self._public_bytes)

        if self._private_key is not None:
            io.storage.write(private_file, self._private_bytes)

    def __getstate__(self):
        if self._private_key is None:
            return {'public': self._public_bytes}
        else:
            return {'private': self._private_bytes}

    def __setstate__(self, state):
        if 'public' in state:
            self._init_with_public_key(state['public'])
        else:
            self._init_with_private_key(state['private'])
