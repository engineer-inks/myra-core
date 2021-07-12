from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType


class NGramFilter(Transformer, HasInputCol, HasOutputCol):
    """N-Grams Filter Transformer.

    See https://github.com/apache/spark/blob/master/python/pyspark/ml/base.py
    Talvez devesse extender UnaryTransformer, embora aparentemente não mude nada em questão de performance
    """

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, stringsToRemove=None, matchFunctions=None):
        super().__init__()

        self.stringsToRemove = Param(self, "stringsToRemove", "Array de strings a serem filtradas")
        self.matchFunctions = Param(self, "matchFunctions",
                                    "Array de funcoes que dao match em ngramas que devem ser filtrados")
        self._setDefault(stringsToRemove=[])
        self._setDefault(matchFunctions=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, stringsToRemove=None, matchFunctions=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setStringsToRemove(self, value):
        return self._set(stringsToRemove=value)

    def setMatchFunctions(self, value):
        return self._set(matchFunctions=value)

    def getStringsToRemove(self):
        return self.getOrDefault(self.stringsToRemove) or []

    def getMatchFunctions(self):
        return self.getOrDefault(self.matchFunctions) or []

    def _build_string_match_function(self, to_remove):
        def contains(big, small):
            for i in range(len(big) - len(small) + 1):
                for j in range(len(small)):
                    if big[i + j] != small[j]:
                        break
                else:
                    return i, i + len(small)
            return False

        # returns (isMatch, startIndex, endIndex
        def has_match_at_index(ngrams, i, ngram_size):
            # if to_remove is smaller than ngram it could be inside the final ngram not starting at first word
            # this is confusing right?
            if i == (len(ngrams) - 1):
                if len(to_remove) <= ngram_size:
                    c = contains(ngrams[i], to_remove)
                    if c:
                        # startIndex > endIndex because endindex for the last ngram is irrelevant
                        return True, i + c[0], i, len(to_remove)

                return False, None, None, None

            match_end = i + (max(0, len(to_remove) - ngram_size))
            ngram_last_index = ngram_size - 1
            for j, word in enumerate(to_remove):
                current_ngram_index = i + max(0, j - ngram_last_index)
                if current_ngram_index >= len(ngrams):
                    return False, None, None, None

                current_ngram = ngrams[current_ngram_index]
                if word != current_ngram[min(j, ngram_last_index)]:
                    return False, None, None, None

            return True, i, match_end, len(to_remove)

        return has_match_at_index

    def _transform(self, dataset):
        stringsToRemove = self.getStringsToRemove()
        stringsToRemove = [s.split() for s in stringsToRemove]
        string_match_functions = [self._build_string_match_function(s) for s in stringsToRemove]

        matchFunctions = self.getMatchFunctions()

        all_match_functions = string_match_functions + matchFunctions

        def transform_udf(ngrams):
            ngrams = [n.split() for n in ngrams]
            ngram_size = len(next(iter(ngrams or []), []))

            indices_to_be_removed = []

            for mf in all_match_functions:
                for i in range(len(ngrams)):
                    match = mf(ngrams, i, ngram_size)
                    if match[0]:
                        start_of_match = match[1]
                        end_of_match = match[2]
                        match_length_in_tokens = match[3]
                        indices_to_be_removed += list(
                            range(start_of_match - (ngram_size - 1),
                                  end_of_match + (min(ngram_size, match_length_in_tokens) - 1) + 1)
                        )

            return [' '.join(n) for i, n in enumerate(ngrams) if i not in indices_to_be_removed]

        op = udf(transform_udf, ArrayType(StringType()))(dataset[self.getInputCol()])
        return dataset.withColumn(self.getOutputCol(), op)
