from sklearn.datasets import make_classification
from sklearn.feature_extraction import FeatureHasher
from sklearn import svm, discriminant_analysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import unittest
import random
import numpy as np
from numpy import testing
import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=50, n_features=3, n_classes=3, n_informative=3, n_redundant=0, random_state=0, shuffle=False)

        feature_hasher = FeatureHasher(n_features=3)
        features = []
        for i in range(0, 100):
            features.append({'a': random.randint(0, 2), 'b': random.randint(3, 5), 'c': random.randint(6, 8)})
        self.y_sparse = [random.randint(0, 2) for i in range(0, 100)]
        self.X_sparse = feature_hasher.transform(features)

    def check_model(self, model, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X), self.y)
        else:
            model.fit(self.X, self.y)

        # When
        serialized_model = skljson.to_dict(model)
        deserialized_model = skljson.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        testing.assert_array_equal(expected_predictions, actual_predictions)

    def check_sparse_model(self, model, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X_sparse), self.y_sparse)
        else:
            model.fit(self.X_sparse, self.y_sparse)

        # When
        serialized_model = skljson.to_dict(model)
        deserialized_model = skljson.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        testing.assert_array_equal(expected_predictions, actual_predictions)

    def check_model_json(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X), self.y)
        else:
            model.fit(self.X, self.y)

        # When
        serialized_model = skljson.to_json(model, model_name)
        deserialized_model = skljson.from_json(model_name)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        testing.assert_array_equal(expected_predictions, actual_predictions)

    def check_sparse_model_json(self, model, model_name, abs=False):
        # Given
        if abs:
            model.fit(np.absolute(self.X_sparse), self.y_sparse)
        else:
            model.fit(self.X_sparse, self.y_sparse)

        # When
        serialized_model = skljson.to_json(model, model_name)
        deserialized_model = skljson.from_json(model_name)

        # Then
        expected_predictions = model.predict(self.X)
        actual_predictions = deserialized_model.predict(self.X)

        testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_bernoulli_nb(self):
        self.check_model(BernoulliNB())
        self.check_sparse_model(BernoulliNB())

        model_name = 'bernoulli-nb.json'
        self.check_model_json(BernoulliNB(), model_name)
        self.check_sparse_model_json(BernoulliNB(), model_name)

    def test_guassian_nb(self):
        self.check_model(GaussianNB())

        model_name = 'gaussian-nb.json'
        self.check_model_json(GaussianNB(), model_name)

        # No sklearn implementation for sparse matrix

    def test_multinomial_nb(self):
        self.check_model(MultinomialNB(), abs=True)
        self.check_sparse_model(MultinomialNB(), abs=True)

        model_name = 'multinomial-nb.json'
        self.check_model_json(MultinomialNB(), model_name, abs=True)
        self.check_sparse_model_json(MultinomialNB(), model_name, abs=True)

    def test_complement_nb(self):
        self.check_model(ComplementNB(), abs=True)

        model_name = 'complement-nb.json'
        self.check_model_json(ComplementNB(), model_name, abs=True)

        # No sklearn implementation for sparse matrix

    def test_logistic_regression(self):
        self.check_model(LogisticRegression())
        self.check_sparse_model(LogisticRegression())

        model_name = 'lr.json'
        self.check_model_json(LogisticRegression(), model_name)
        self.check_sparse_model_json(LogisticRegression(), model_name)

    def test_lda(self):
        self.check_model(discriminant_analysis.LinearDiscriminantAnalysis())

        model_name = 'lda.json'
        self.check_model_json(discriminant_analysis.LinearDiscriminantAnalysis(), model_name)

        # No sklearn implementation for sparse matrix

    def test_qda(self):
        self.check_model(discriminant_analysis.QuadraticDiscriminantAnalysis())

        model_name = 'qda.json'
        self.check_model_json(discriminant_analysis.QuadraticDiscriminantAnalysis(), model_name)

        # No sklearn implementation for sparse matrix

    def test_svm(self):
        self.check_model(svm.SVC(gamma=0.001, C=100., kernel='linear'))
        self.check_sparse_model(svm.SVC(gamma=0.001, C=100., kernel='linear'))

        model_name = 'svm.json'
        self.check_model_json(svm.SVC(), model_name)
        self.check_sparse_model_json(svm.SVC(), model_name)

    def test_decision_tree(self):
        self.check_model(DecisionTreeClassifier())
        self.check_sparse_model(DecisionTreeClassifier())

        model_name = 'dt.json'
        self.check_model_json(DecisionTreeClassifier(), model_name)
        self.check_sparse_model_json(DecisionTreeClassifier(), model_name)

    def test_gradient_boosting(self):
        self.check_model(GradientBoostingClassifier(n_estimators=25, learning_rate=1.0))
        self.check_sparse_model(GradientBoostingClassifier(n_estimators=25, learning_rate=1.0))

        model_name = 'gb.json'
        self.check_model_json(GradientBoostingClassifier(), model_name)
        self.check_sparse_model_json(GradientBoostingClassifier(), model_name)

    def test_random_forest(self):
        self.check_model(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0))
        self.check_sparse_model(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0))

        model_name = 'rf.json'
        self.check_model_json(RandomForestClassifier(), model_name)
        self.check_sparse_model_json(RandomForestClassifier(), model_name)

    def test_perceptron(self):
        self.check_model(Perceptron())
        self.check_sparse_model(Perceptron())

        model_name = 'perceptron.json'
        self.check_model_json(Perceptron(), model_name)
        self.check_sparse_model_json(Perceptron(), model_name)

    def test_mlp(self):
        self.check_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
        self.check_sparse_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))

        model_name = 'mlp.json'
        self.check_model_json(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), model_name)
        self.check_sparse_model_json(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), model_name)

