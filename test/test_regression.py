import random

import pytest
from numpy import testing
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import sklearn_json as skljson

X, y = make_regression(
    n_samples=50, n_features=3, n_informative=3, random_state=0, shuffle=False
)

feature_hasher = FeatureHasher(n_features=3)
features = []
for i in range(0, 100):
    features.append(
        {
            "a": random.randint(0, 2),
            "b": random.randint(3, 5),
            "c": random.randint(6, 8),
        }
    )
y_sparse = [random.random() for i in range(0, 100)]
X_sparse = feature_hasher.transform(features)


@pytest.mark.usefixtures("tmp_path")
class TestRegression:
    def check_model(self, model):
        # Given
        model.fit(X, y)

        # When
        serialized_model = skljson.to_dict(model)
        deserialized_model = skljson.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(X)
        actual_predictions = deserialized_model.predict(X)

        testing.assert_array_equal(expected_predictions, actual_predictions)

    def check_sparse_model(self, model):
        # Given
        model.fit(X_sparse, y_sparse)

        # When
        serialized_model = skljson.to_dict(model)
        deserialized_model = skljson.from_dict(serialized_model)

        # Then
        expected_predictions = model.predict(X_sparse)
        actual_predictions = deserialized_model.predict(X_sparse)

        testing.assert_array_equal(expected_predictions, actual_predictions)

    def test_linear_regression(self):
        self.check_model(LinearRegression())
        self.check_sparse_model(LinearRegression())

    def test_lasso_regression(self):
        self.check_model(Lasso(alpha=0.1))
        self.check_sparse_model(Lasso(alpha=0.1))

    def test_ridge_regression(self):
        self.check_model(Ridge(alpha=0.5))
        self.check_sparse_model(Ridge(alpha=0.5))

    def test_svr(self):
        self.check_model(SVR(gamma="scale", C=1.0, epsilon=0.2))
        self.check_sparse_model(SVR(gamma="scale", C=1.0, epsilon=0.2))

    def test_decision_tree_regression(self):
        self.check_model(DecisionTreeRegressor())
        self.check_sparse_model(DecisionTreeRegressor())

    def test_gradient_boosting_regression(self):
        self.check_model(GradientBoostingRegressor())
        self.check_sparse_model(GradientBoostingRegressor())

    def test_random_forest_regression(self):
        self.check_model(
            RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
        )
        self.check_sparse_model(
            RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
        )

    def test_mlp_regression(self):
        self.check_model(MLPRegressor())
        self.check_sparse_model(MLPRegressor())
