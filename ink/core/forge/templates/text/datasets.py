"""Describes common text collections used across multiple applications.
Examples are stop words and common names of the Portuguese language.

"""
import os
import logging

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from sklearn.datasets._base import get_data_home, RemoteFileMetadata, _fetch_remote

from ink.core.forge.templates.core.io.stream import read, conform, merge

from .. import utils

NEGATIVE_WORDS = [
    'revoltada',
    'revoltado',
    'abuso',
    'absurdo',
    'nao aceito',
    'nao concordo',
    'impossivel',
    'nao quero mais',
    'quero cancelar',
    'vou cancelar',
    'nunca mais',
    'muito caro',
    'carissimo',
    'muito ruim',
    'pessimo',
    'horrivel',
    'uma merda',
    'esse lixo',
    'idiota',
    'besta',
    'porcaria',
    'o concorrente',
    'droga']

HIGH_FRICTION_WORDS = [
    'procon',
    'reclam',
    'ouvidoria',
    'advogado',
    'advogada',
    'justica',
    'enganacao',
    'enganad',
    'enganos']

RECALL_WORDS = [
    'segunda vez',
    'duas vezes',
    'terceira vez',
    'ja falei antes',
    'nao e a primeira vez',
    'nao e o primeiro contato',
    'nao consigo resolver',
    'nao recebo contato',
    'ja liguei duas vezes',
    'ja liguei varias vezes',
    'quarta vez',
    'tres vezes',
    'ninguem resolve',
    'ninguem soluciona',
    'nao resolve']

NO_SERVICE_WORDS = [
    'central via telefone',
    'central de relacionamento',
    'central de relacionamento nos numeros',
    'solicitar nos telefones',
    'central alelo',
    'por aqui nao conseguem',
    'por aqui nao consigo',
    'central via voz',
    'voce tem que solicitar pela central',
    'solicitar pela central',
    'consegue verificar na central',
    'somente na central',
    'atendimento especializado via voz',
    'entre em contato com os numeros',
    'enviar o contrato para central',
    'solicitar nos numeros',
    'nao realizamos',
    'central de e commerce',
    'por aqui o que posso fazer',
    'contato com a nossa central',
    'somente com a nossa central',
    'sistema encerra',
    'resolver de outra forma',
    'sistema esta em manutencao',
    'retorne as 8',
    'ligar no atendimento',
    'nao poderei te auxiliar',
    'atendimento exclusivo']

GENERO_NOMES = RemoteFileMetadata(
    filename='genero-nomes.2.csv',
    url='https://docs.google.com/uc?export=download&id=1jmtTSahvt_LY7db_s1LnZkfSb8d7clep',
    checksum='b23c31486bb0bda1f71393e4063d2c17207c901fdbfca2e69bc4cae49285729a')


def newsgroups20(random_state=42) -> DataFrame:
    """Load 20NewsGroups dataset as a spark DataFrame.

    Parameters
    ----------

    subset:
        Subset of the dataset to be loaded.
        Allowed values are 'train' or 'test'.

    random_state:
        Random state or integer used to load the set.
        Default is 42.

    Examples
    --------
    .. jupyter-execute::

        import dextra.dna.text as T

        T.datasets.newsgroups20().limit(5).toPandas()
    """
    from sklearn import datasets

    def load_part(subset):
        x = datasets.fetch_20newsgroups(
            subset=subset, random_state=random_state)
        d = pd.DataFrame(x.data, columns=['text'])
        d['target'] = np.asarray(x.target_names)[x.target].astype(str)
        d['subset'] = subset
        return d[['text', 'subset', 'target']]

    return read(load_part('train')
                            .append(load_part('test')))


def genero_nomes(data_home=None,
                 download_if_missing=True,
                 only_names=False,
                 most_frequent=None) -> pd.DataFrame:
    """Load the GeneroNomes IBGE dataset into a pandas DataFrame.

    Parameters
    ----------
    data_home: str, optional
        data containing folder. The user's home will be used by default.
    download_if_missing: bool
        raises an error if the data is not already in disk
    only_names: bool
        when true, drops all columns except by `first_name`
    most_frequent: int, optional
        limit by the `n` most frequent names

    Returns
    -------
    pandas.DataFrame
        The dataframe containing the most frequent names in Brazil.

    Examples
    --------
    .. jupyter-execute::

        import myra.dna.text as T

        T.datasets.genero_nomes().head()

    """
    data_home = get_data_home(data_home=data_home)
    os.makedirs(data_home, exist_ok=True)

    filepath = os.path.join(data_home, GENERO_NOMES.filename)
    if not os.path.exists(filepath):
        if not download_if_missing:
            raise IOError('Data not found and `download_if_missing` is False')

        logging.debug(f'downloading `Genero Nomes` from {GENERO_NOMES.url} to {data_home}')
        _ = _fetch_remote(GENERO_NOMES, dirname=data_home)

    x = pd.read_csv(filepath, usecols=[
                    'first_name', 'frequency_total'] if only_names else None)

    if most_frequent:
        # This dataset is already sorted by most frequent names.
        x = x[:most_frequent]

    return x[['first_name']] if only_names else x


NUMBERS = ('um dois tres quatro cinco seis sete oito nove dez onze doze '
           'treze quatorze quinze dezesseis dezesete dezoito dezenove '
           'vinte trinta quarenta cinquenta sessenta setenta oitenta '
           'noventa cem duzentos trezentos quatrocentos quinhentos '
           'seiscentos setecentos oitocentos novecentos mil milhao'
           .split())

COMMON_PATTERNS = {
    'cnpj': r'\d{2}([\s.])?\d{3}([\s.])?\d{3}([\s.])?\d{4}([\s-])?\d{2}',
    'cpf': r'\d{3}([\s.])?\d{3}([\s.])?\d{3}([\s-])?\d{2}',
    'rg': r'\d{2}([\s.])?\d{3}([\s.])?\d{3}([\s-])?(\d{1}|x|X)',

    'phone': r'(\(?\d{2}(\s)?\)?)?\d{4,5}([\s-])?\d{4}',
    'service-phone': r'\d{4}([\s-])?\d{3}([\s-])?\d{4}',
}


def all_patterns():
    """Load all data patterns.

    This function loads the IBGE dataset and merges it with the
    already existing ``COMMON_PATTERNS`` map.

    Returns
    -------
    dict
        A dictionary of (entity, mask) to be hashed in free text.
    """
    n = r'(?i)' + r'|'.join(r'\b'
                            + genero_nomes(only_names=True, most_frequent=4000)
                            .first_name
                            .apply(utils.clean) + r'\b')

    return {'name': n, **COMMON_PATTERNS}

def sklearn_ds(name: str, *args, **kwargs) -> DataFrame:
    """Load Scikit-learn dataset from function name.

    Parameters
    ----------
    name: Name of the function within :code:`sklearn.datasets`
    args: Arguments passed to the building function
    \**kwargs
        Key arguments passed to the building function

    Examples
    --------
    .. code-block:: python

        import dextra.dna.core as C
        C.datasets.iris().limit(5).toPandas()

    """
    import pandas as pd
    from sklearn import datasets

    x = getattr(datasets, name)(*args, **kwargs)
    d = pd.DataFrame(x.data, columns=x.feature_names)

    d['target'] = (x.target_names[x.target].astype(str)
                   if hasattr(x, 'target_names')
                   else x.target)

    d = read(d)
    d = conform(d)
    d = merge(d)

    return d


def wine() -> DataFrame:
    """Load the wine dataset from :code:`sklearn.datasets` as spark dataframe.

    Examples
    --------
    .. jupyter-execute::

        import myra.dna.core as C
        C.datasets.wine().limit(2).toPandas()

    """
    return sklearn_ds('load_wine')


def iris() -> DataFrame:
    """Load the iris dataset from :code:`sklearn.datasets` as spark dataframe.

    Examples
    --------
    .. jupyter-execute::

        import myra.dna.core as C
        C.datasets.iris().limit(2).toPandas()

    """
    return sklearn_ds('load_iris')


def digits() -> DataFrame:
    """Load the digits dataset from :code:`sklearn.datasets` as spark dataframe.

    Examples
    --------
    .. jupyter-execute::

        import dextra.dna.core as C
        C.datasets.digits().limit(2).toPandas()

    """
    return sklearn_ds('load_digits')


def boston() -> DataFrame:
    """Load the iris dataset from :code:`sklearn.datasets` as spark dataframe.

    Examples
    --------
    .. jupyter-execute::

        import myra.dna.core as C
        C.datasets.boston().limit(2).toPandas()

    """
    return sklearn_ds('load_boston')
