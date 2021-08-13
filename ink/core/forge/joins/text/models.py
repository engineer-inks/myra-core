from typing import Union, List

import pyspark.sql.functions as F
from pyspark.ml import Model
from pyspark.ml.feature import (CountVectorizer, IDF, CountVectorizerModel,
                                NGram, Word2Vec, Word2VecModel, Tokenizer,
                                StopWordsRemover)
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as SkPipeline

from ink.core.forge.joins.core.io.stream import merge
from ink.core.forge.joins.core.models import base

from . import transformers, utils

def tokenizing_model(input_col: str = '_text',
                     n_grams: int = None,
                     stop_words: Union[bool, str, List[str]] = 'portuguese',
                     lemmatization: Union[bool, 'spacy.language.Language'] = False,
                     stemming: Union[bool, 'nltk.stem.api.StemmerI'] = False):
    stages = []
    step = input_col

    if lemmatization:
        if lemmatization is True:
            lemmatization = None

        stages.append(transformers.Lemmatizer(inputCol=step,
                                              outputCol=f'{input_col}_lemma',
                                              languageModel=lemmatization))
        step = f'{input_col}_lemma'

    stages.append(Tokenizer(inputCol=step, outputCol=f'{input_col}_words'))
    step = f'{input_col}_words'

    if stop_words:
        if isinstance(stop_words, str):
            stop_words = StopWordsRemover.loadDefaultStopWords(stop_words)

        stages.append(StopWordsRemover(inputCol=step,
                                       outputCol=f'{input_col}_filtered',
                                       stopWords=stop_words))
        step = f'{input_col}_filtered'

    if stemming:
        if stemming is True:
            stemming = None

        stages.append(transformers.Stemmer(inputCol=step, outputCol=f'{input_col}_stem', model=stemming))
        step = f'{input_col}_stem'

    if n_grams:
        stages.append(NGram(n=n_grams, inputCol=step, outputCol=f'{input_col}_{n_grams}_grams'))
        step = f'{input_col}_{n_grams}_grams'

    return Pipeline(stages=stages)


def text_to_word_importance_vector(input_col: str = '_text',
                                   output_col: str = 'features',
                                   features: int = 2048,
                                   min_doc_freq: int = 0,
                                   n_grams: int = None,
                                   stop_words: List[str] = 'portuguese',
                                   lemmatization: Union[bool, 'spacy.language.Language'] = False,
                                   stemming: Union[bool, 'nltk.stem.api.StemmerI'] = False) -> Pipeline:
    tk = tokenizing_model(input_col,
                          n_grams=n_grams,
                          stop_words=stop_words,
                          lemmatization=lemmatization,
                          stemming=stemming)

    return Pipeline(stages=[
        tk,
        CountVectorizer(
            vocabSize=features,
            inputCol=tk.getStages()[-1].getOutputCol(),
            outputCol=f'{input_col}_word_count',
            minDF=min_doc_freq),
        IDF(inputCol=f'{input_col}_word_count',
            outputCol=output_col)
    ])


def word2vec(input_col: str = 'text',
             output_col: str = 'features',
             features: int = 2048,
             n_grams: int = None,
             stop_words: List[str] = 'portuguese',
             lemmatization: Union[bool, 'spacy.language.Language'] = False,
             stemming: Union[bool, 'nltk.stem.api.StemmerI'] = False):
    tk = tokenizing_model(input_col,
                          n_grams=n_grams,
                          stop_words=stop_words,
                          lemmatization=lemmatization,
                          stemming=stemming)
    return Pipeline(stages=[
        tk,
        Word2Vec(vectorSize=features,
                 inputCol=tk.getStages()[-1].getOutputCol(),
                 outputCol=output_col)
    ])


def synonyms(model: Union[PipelineModel, Word2VecModel],
             word: Union[str, List[str]],
             n: int = 5) -> F.DataFrame:
    if isinstance(model, PipelineModel):
        model = next(s for s in model.stages if isinstance(s, Word2VecModel))

    return merge([
        model.findSynonyms(t, n)
        for t in utils.to_list(word)])


def vocabulary_of(model: Union[PipelineModel, CountVectorizerModel]) -> List[str]:
    if isinstance(model, PipelineModel):
        model = next(s for s in model.stages if isinstance(s, (CountVectorizerModel, Word2VecModel)))

    if isinstance(model, Word2VecModel):
        vocab = [r.word for r in model.getVectors().collect()]
    else:
        if model is None or not hasattr(model, 'vocabulary'):
            raise ValueError('This model does not seem to have a vocabulary. '
                             'Check if it contains a word count vectorization '
                             'component and if it\'s fitted.')
        vocab = model.vocabulary

    return vocab


def explain(model,
            vocabulary: Union[PipelineModel, List[str]],
            terms: int = 16,
            **kwargs):
    '''
    Generate a report that explains the model's behavior.

    :param model:      the model that should be explained.
    :param vocabulary: the vocabulary mapping words (or bi-grams, tri-grams) to the feature-vector-space
                       by the TF-IDF algorithm. If a pipeline is passed, we search for the vocabulary within
                       the trained CountVectorizer model.
    :param terms:      the maximum amount of terms shown during the explanation.
    :param **kwargs:  see below.

    :Keyword Arguments:
        * *svd* -- SVD model used for unsupervised learning with KMeans

    :return:           str, a report of the model
    '''
    
    model = (next(e for n, e in model.steps if isinstance(e, base.SUPPORTED_ESTIMATORS))
             if isinstance(model, SkPipeline)
             else model)
    model = model.best_estimator_ if isinstance(model, GridSearchCV) else model

    if isinstance(vocabulary, PipelineModel):
        vocabulary = vocabulary_of(vocabulary)

    return C.models.explain(model, vocabulary, features_used=terms, **kwargs)


def output_columns(model: Union[PipelineModel, Model]):
    cols = []
    for t in model.stages:
        if isinstance(t, PipelineModel):
            cols += output_columns(t)
        else:
            cols.append(t.getOrDefault('outputCol'))
    return cols
