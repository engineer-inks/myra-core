"""Declare the overrides of the base processors in :py:mod:`dextra.dna.core`,
adding functions for the text context.

"""

import abc
import os
import logging
from typing import List, Union, Callable, Optional, Dict

from ink.core.forge.joins.core import processors
from py4j.protocol import Py4JJavaError
from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram, CountVectorizer, IDF
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.sql import DataFrame, Column, functions as F, types as T

from ..text import (models,
               utils)
from ..text import datasets, functions as F2


class Rawing(processors.Rawing, metaclass=abc.ABCMeta):
    """Processor capable of tearing up transient data after processing it.

    """

    def exclude_sensible_info(self,
                              x: DataFrame,
                              col: Union[str, Column],
                              patterns: Optional[Dict[str, str]] = None) -> DataFrame:
        """Exclude sensible info within a free text column
        by replacing it with its mask.

        Parameters
        ----------
        x: DataFrame
            the input dataframe containing the column ``col``
        col: str or Column
            the column in the frame containing sensible info
        patterns: dict
            dictionary of patterns used for encryption.

        Returns
        -------
        DataFrame
            A copy of the input frame ``x`` with a free text column without
            sensible info.

        Examples
        --------
        .. jupyter-execute::

            import pyspark.sql.functions as F

            class SafeIngestion(T.processors.Rawing):
                def call(self, x):  
                    x = x.withColumn('text', F.lower('text'))
                    x = self.exclude_sensible_info(x, 'text', patterns={
                        'email': r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
                        **T.datasets.all_patterns()
                    })

                    return x

            ingest = SafeIngestion(None, None)

            (ingest(T.datasets.newsgroups20())
            .limit(5)
            .toPandas())

        """
        patterns = patterns or datasets.all_patterns()
        masks = utils.patterns_as_masks(patterns)
        logging.info(f'The following patterns will be encrypted: {list(patterns.keys())}')

        masked = F.col(col)

        for rule, mask in masks.items():
            masked = F2.replace(masked, rule, mask)

        x = x.withColumn(col, masked)

        return x


class Trusting(processors.Trusting, metaclass=abc.ABCMeta):
    ...


class Refining(processors.Refining, metaclass=abc.ABCMeta):
    text_encoder: PipelineModel

    def infer_from_keywords(self,
                            x: DataFrame,
                            text_col: str,
                            tag: str,
                            positive_words: List[str] = None,
                            negative_words: List[str] = None) -> DataFrame:
        """Infer a certain characteristic ``tag`` from the dataframe.

        Parameters
        ----------
        x: DataFrame
           the frame from which the inference is made
        text_col: str
           the text column in ``x``.
        tag: str
           the inferred property
        positive_words: list of str
           words that positively affect the inference of ``tag``
        negative_words: list of str
           words that negatively affect the inference of ``tag``

        Returns
        -------
        DataFrame
            A modified copy of ``x``, adding a new integer
            (binary) column ``tag``.
        """
        x = x.withColumn(tag, F.lit(1))  # 1-True, by default.

        if positive_words:
            # True and contains terms.
            x = x.withColumn(tag, F.col(tag) * F2.contains(F.col(text_col), positive_words))

        if negative_words:
            # (True and contains positive words if there are any) and does not contain negative words.
            x = x.withColumn(tag, F.col(tag) * (1 - F2.contains(F.col(text_col), negative_words)))

        return x

    def encode_text(self,
                    x: DataFrame,
                    input_col: str = 'message',
                    model_path: str = None,
                    encoder_fn: Callable = models.text_to_word_importance_vector,
                    training: DataFrame = None,
                    keep_intermediate: bool = False,
                    train_if_missing: bool = True):
        """Encode base text into a feature vector that can be
           fed to machine learning models.

        Parameters
        ----------
        x: DataFrame
           frame to be encoded
        input_col: str
           name of column containing the text of interest
        model_path: str
           name under which the model will be saved
        encoder_fn: Callable
           function to build a encoding model
        training: DataFrame
           frame used to train the encoder. If none is passsed, ``x`` is used.
        keep_intermediate: bool
           if true, the intermediate representation generated will be returned
           within the frame. This is useful when reusing the encode pipeline
           for other ends, such as for computing n-grams.
        train_if_missing: bool
           if true, the encoder is trained whenever not found

        Returns
        -------
            DataFrame
            A modified copy of ``x``, containing the encoded text in
            the ``features`` column.
        """
        if not model_path:
            model_path = os.path.join(self.config.lakes.models,
                                      self.fullname().lower(),
                                      'text_encoder')

        x = x.withColumn('_text', F.col(input_col))
        training = x if not training else training.withColumn('_text', F.col(input_col))

        try:
            logging.info(f'Loading text encoder from {model_path}.')
            self.text_encoder = PipelineModel.load(model_path)

        except Py4JJavaError as error:
            if not train_if_missing:
                raise models.errors.TrainingNotFound(error)

            logging.info(f'Failed due to the following error: {error}. Retraining.')

            self.text_encoder = encoder_fn(input_col='_text').fit(training)
            self.text_encoder.save(model_path)
            logging.info(f'Training completed. Saved at {model_path}.')

        x = self.text_encoder.transform(x)
        x = x.drop('_text')

        return (x
                if keep_intermediate
                else x.drop(*models.output_columns(self.text_encoder)[:-1]))

    def most_relevant_ngram(self,
                            x: DataFrame,
                            text_column: str,
                            output_column_prefix: str,
                            id_field: str = 'id',
                            where: Column = None,
                            stop_words='portuguese',
                            features: int = 4096,
                            min_df: float = 3.0,
                            texts_to_filter: List[str] = (),
                            replacement: str = ';',
                            keep_intermediate: bool = False):
        """Compute Most Relevant N-Grams.

        As it is now, this method receives a ``DataFrame``, filters it and creates
        features to select the most relevant word, 2-grams and 3-grams.

        :param x: ``DataFrame`` containing pre-processed text.
        :param text_column: name of the column containing the evaluating documents.
        :param output_column_prefix: prefix used to create the output column. For example, when
               `output_column_prefix='high_friction'`, the following columns will be added to the
               returning data frame: `high_friction_word`, `high_friction_bigram`, `high_friction_trigram`.
        :param id_field: name of the column of the unique id to join the result on the original data frame in case filter_expr was not null.
        :param where: expression used to filter the dataframe if you want to extract the most relevant ngrams for a specific context.
            Eg: if you want the most relevant high_friction ngram then filter_expr=(F.col('high_friction') == 1)
        :param stop_words: list of words fed to `StopWordsRemover` transformer
        :param features: CountVectorizer param
        :param min_df: CountVectorizer param
        :param texts_to_filter: list of terms to filter before computing the n-grams
        :param replacement: tag that will replace all terms described by `texts_to_filter`
        :param keep_intermediate: retain intermediate representation

        :return: ``DataFrame`` containing most relevant word, 2-grams and 3-grams columns.
        """
        o = x

        if isinstance(stop_words, str):
            stop_words = StopWordsRemover.loadDefaultStopWords('portuguese')

        if where is not None:
            x = o.filter(where)

        if texts_to_filter:
            x = x.withColumn(text_column, F2.replace(text_column, texts_to_filter, replacement))

        tokenizer = Tokenizer(inputCol=text_column, outputCol=f'{text_column}_aux_tokenized')
        words_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol=f'{text_column}_aux_filtered',
                                         stopWords=stop_words)
        bigram = NGram(n=2, inputCol=words_remover.getOutputCol(), outputCol=f'{text_column}_aux_bigrams')
        trigram = NGram(n=3, inputCol=words_remover.getOutputCol(), outputCol=f'{text_column}_aux_trigrams')
        words_cv = CountVectorizer(inputCol=words_remover.getOutputCol(), outputCol=f'{text_column}_aux_words_cv',
                                   vocabSize=features, minDF=min_df)
        bigram_cv = CountVectorizer(inputCol=bigram.getOutputCol(), outputCol=f'{text_column}_aux_bigrams_cv',
                                    vocabSize=features, minDF=min_df)
        trigram_cv = CountVectorizer(inputCol=trigram.getOutputCol(), outputCol=f'{text_column}_aux_trigrams_cv',
                                     vocabSize=features, minDF=min_df)
        words_idf = IDF(inputCol=words_cv.getOutputCol(), outputCol=f'{output_column_prefix}_words_idf')
        bigram_idf = IDF(inputCol=bigram_cv.getOutputCol(), outputCol=f'{output_column_prefix}_bigrams_idf')
        trigram_idf = IDF(inputCol=trigram_cv.getOutputCol(), outputCol=f'{output_column_prefix}_trigrams_idf')

        pipeline = Pipeline(stages=[tokenizer, words_remover, bigram, trigram,
                                    words_cv, bigram_cv, trigram_cv,
                                    words_idf, bigram_idf, trigram_idf])

        model = pipeline.fit(x)
        x = model.transform(x)

        word_field = f'{output_column_prefix}_word'
        bi_gram_field = f'{output_column_prefix}_bigram'
        tri_gram_field = f'{output_column_prefix}_trigram'

        for field, col, vocabulary in ((word_field, words_idf.getOutputCol(), model.stages[4].vocabulary),
                                       (bi_gram_field, bigram_idf.getOutputCol(), model.stages[5].vocabulary),
                                       (tri_gram_field, trigram_idf.getOutputCol(), model.stages[6].vocabulary)):
            x = x.withColumn(field, F2.argmax(col, vocabulary, utils.replacement_mask(vocabulary, replacement)))

        if not keep_intermediate:
            x = x.drop(bigram.getOutputCol(), trigram.getOutputCol(), tokenizer.getOutputCol(),
                       words_remover.getOutputCol(), bigram_cv.getOutputCol(),
                       trigram_cv.getOutputCol(), words_cv.getOutputCol(), words_idf.getOutputCol(),
                       bigram_idf.getOutputCol(), trigram_idf.getOutputCol())

        if where is not None:
            id_r = f'filtered_{id_field}'
            x = (x.select(id_field, bi_gram_field, tri_gram_field, word_field)
                 .withColumnRenamed(id_field, id_r))

            x = o.join(x, o[id_field] == x[id_r], how="left").drop(id_r)

        return x, model


__all__ = ['Rawing',
           'Trusting',
           'Refining']
