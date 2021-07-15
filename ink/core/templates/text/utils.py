import html
import logging
import re
import unicodedata
from typing import List

import numpy as np

from ..utils import *


def names_distance(a: str, b: str) -> int:
    """Calculate names distance.

    Parameters
    ----------
    a : str, required
        name to be compared with ``b``
    b : str, required
        name to be compared with ``a``

    Returns
    -------
    int
        Number that represents the distance between ``a`` and ``b``.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/names_distance.py
    """
    if a is None and b is None:
        return 15
    if a is None:
        return len(b)
    if b is None:
        return len(a)

    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous_row = range(len(b) + 1)
    for i, column1 in enumerate(a):
        current_row = [i + 1]
        for j, column2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (column1 != column2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def are_similar_names(a: str, b: str) -> bool:
    """Check if names are similar using ``names_distance()`` method.

    Parameters
    ----------
    a : str, required
        name to be compared with ``b``
    b : str, required
        name to be compared with ``a``

    Returns
    -------
    bool
        Boolean flag that represents if they are similar or not.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/are_similar_names.py
    """
    if not a or not b:
        return None
    if a == b:
        return True
    return names_distance(a, b) < 15


def simplify(text: str) -> str:
    """This method will normalize any accented character to its standard form.

    Parameters
    ----------
    text : str, required
        text to be simplified

    Returns
    -------
    str
        Simplified text.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/simplify.py
    """
    if not text:
        return ''

    return str(unicodedata
               .normalize('NFD', text)
               .encode('ascii', 'ignore')
               .decode("utf-8"))


def clean(text: str) -> str:
    """This method will remove special characters and normalize the text using ``simplify()`` method.

    Parameters
    ----------
    text : str, required
        text to be cleaned

    Returns
    -------
    str
        Cleaned text.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/clean.py
    """
    if not text:
        return ''

    text = re.sub('[,.;*!?:"-/]', ' ', text.lower())
    text = re.sub('\\s+', ' ', text)

    return simplify(text).strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from the text.

    Parameters
    ----------
    text : str, required
        text to have HTML tags removed

    Returns
    -------
    str
        Cleaned text.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/remove_html_tags.py
    """
    text = re.sub(r'(&#*[a-z|A-Z|\\d]+),', '\\g<1>;', text)
    text = html.unescape(text)

    return re.sub('<[^<]+?>', ' ', text)


# region Spacy-based utils

def load_spacy_model(lm: Union[str, 'spacy.language.Language'],
                     download_if_missing: bool = True) -> 'spacy.language.Language':
    import spacy

    if isinstance(lm, spacy.language.Language):
        return lm

    try:
        logging.debug(f'loading Spacy model {lm}')
        return spacy.load(lm)

    except OSError:
        if not download_if_missing:
            raise

        spacy.cli.download(lm)
        return load_spacy_model(lm, download_if_missing=False)


def analyze_with_spacy(text: str, language_model: 'spacy.language.Language') -> Optional[List[str]]:
    """Applies a model to a text.

    Parameters
    ----------
    text : str, required
        text to be analyzed
    language_model : spacy.language.Language, required
        spacy language model to be used

    Returns
    -------
    List[str]
        List of features extracted from the text analyzed.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/analyze_with_spacy.py
    """
    if not text:
        return

    doc = language_model(text)

    return [(t.text, t.lemma_, t.pos_, t.tag_, t.is_stop) for t in doc]


def lemmatize(text: str, language_model: 'spacy.language.Language') -> str:
    """Applies lemmatization on the text.

    Parameters
    ----------
    text : str, required
        text to be lemmatized
    language_model : spacy.language.Language, required
        spacy language model to be used

    Returns
    -------
    str
        Lemmatized text.

    Raises
    ------
    RuntimeError
        if lemmatization fails due to an IOError on spacy.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/lemmatize.py
    """
    if not text:
        return text

    return ''.join(
        (x.lemma_ if x.pos_ in ('VERB', 'AUX') else x.text) + x.whitespace_
        for x in language_model(text)
        if x.text).strip()


# endregion


def stem(tokens: List[str], model: 'nltk.stem.api.StemmerI') -> List[str]:
    """Applies stemming on some text.

    Parameters
    ----------
    tokens : list of str, required
        words to be stemmed
    model : nltk.stem.api.StemmerI, optional
        NLTK stemmer model to be used. Defaults to ``SnowballStemmer("portuguese")``

    Returns
    -------
    str
        Stemmed text.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/stem.py
    """
    return [model.stem(word) for word in tokens]


def contains(text: str, pattern: str) -> bool:
    """Check if the text contains the RegExp pattern.

    Parameters
    ----------
    text : str, required
        text to be searched
    pattern : str, required
        regular expression pattern

    Returns
    -------
    bool
        Search result.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/contains.py
    """
    return re.search(pattern, text) is not None


def replace(text: str, matching: 're.Pattern', replacement: str) -> str:
    """Replace values in a text based on RegExp pattern.

    Parameters
    ----------
    text : str, required
        text to be modified
    matching : re.Pattern, required
        regular expression pattern
    replacement : str, required
        Replace value.

    Returns
    -------
    str
        Updated text.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/replace.py
    """
    if text is None:
        return None

    return matching.sub(replacement, text)


def extract(text: str, matching: str, flags=0) -> str:
    """Replace values in a text based on RegExp pattern.

    Parameters
    ----------
    text : str, required
        text to be searched
    matching : str, required
        regular Expression Pattern
    flags : int, optional
        search flags passed to :code:`re`. Examples are
        :code:`re.DOTALL` and :code:`re.MULTILINE`

    Returns
    -------
    str
        Extracted text.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/extract.py
    """
    if text is None:
        return None

    groups = re.search(matching, text, flags)
    return groups[0] if groups else None


def to_boolean(text: str) -> bool:
    """Parse ``sim`` or ``não`` to boolean value.

    Parameters
    ----------
    text : str, required
        Portuguese boolean name.

    Returns
    -------
    bool
        Boolean equivalent.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/to_boolean.py
    """
    if text is None:
        return None

    return text.lower() == 'sim'


# region anonymization

def patterns_as_masks(patterns: Dict[str, str]):
    return {pattern: '{%s}' % name for name, pattern in patterns.items()}


def replacement_mask(vocabulary: List[str], replacement_token: str = ';') -> np.ndarray:
    """Replacement Mask retrieve a numpy array containing a mask for the vocabulary.

    Parameters
    ----------
    vocabulary : list of str, required
        array-like of words containing the language vocabulary
    replacement_token : str, optional
        the token representing any replaced words. The default value is ``;``

    Returns
    -------
    np.ndarray
        Replaced values map.

    Examples
    --------
    .. jupyter-execute:: /examples/text/utils/replacement_mask.py
    """
    return (np.asarray([1 - int(contains(w, replacement_token)) for w in vocabulary])
            if replacement_token
            else np.ones(len(vocabulary)))

# endregion
