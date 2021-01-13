from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
from numpy import testing
import re
import unittest
import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        newsgroup = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=('headers', 'footers', 'quotes'))

        self.X = [
            Counter(tok.lower() for tok in re.findall(r"\w+", text))
            for text in newsgroup.data
        ]

    def check_model(self, model):
        expected_vectors = model.fit_transform(self.X)

        serialized_model = skljson.to_dict(model)
        deserialized_model = skljson.from_dict(serialized_model)

        actual_vectors = deserialized_model.fit_transform(self.X)

        if model.sparse:
            testing.assert_array_equal(expected_vectors.indptr, actual_vectors.indptr)
            testing.assert_array_equal(expected_vectors.indices, actual_vectors.indices)
            testing.assert_array_equal(expected_vectors.data, actual_vectors.data)
        else:
            testing.assert_array_equal(expected_vectors, actual_vectors)

    def test_dict_vectorization(self):
        self.check_model(DictVectorizer())
        self.check_model(DictVectorizer(sparse=False))
