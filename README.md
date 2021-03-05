# model-json
JSON serialization ans seserialization of scikit-learn, dask-ml and lightGBM model files, including transformers

# Initial source
The solution is the extension of https://github.com/mlrequest/sklearn-json library


# Getting Started

model-json makes exporting/importing the models and transformers files to/from JSON simple

## Install
```
pip install git+https://github.com/FireFlyTy/sklearn-json@th/test
```
## Example Usage

```python
import datrics_json as datjson
from sklearn.ensemble import IsolationForest

model = IsolationForest().fit(X)

datjson.to_json(model, file_name)
deserialized_model = datjson.from_json(file_name)

deserialized_model.predict(X)
```

# Features
The list of supported models is rapidly growing. If you have a request for a model or feature, please reach out to support@mlrequest.com.

sklearn-json requires scikit-learn >= 0.21.3.

## Supported scikit-learn Models

* Classification
    * **`sklearn.linear_model.LogisticRegression`**
    * **`sklearn.linear_model.Perceptron`
    * **`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
    * **`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
    * **`sklearn.svm.SVC`
    * **`sklearn.ensemble.IsolationForest`
    * **`sklearn.clustering.KMeans`
    * **`sklearn.clustering.DBSCAN`
    * *`sklearn.naive_bayes.GaussianNB`
    * *`sklearn.naive_bayes.MultinomialNB`
    * *`sklearn.naive_bayes.ComplementNB`
    * *`sklearn.naive_bayes.BernoulliNB`
    * *`sklearn.tree.DecisionTreeClassifier`
    * *`sklearn.ensemble.RandomForestClassifier`
    * *`sklearn.ensemble.GradientBoostingClassifier`
    * *`sklearn.neural_network.MLPClassifier`

* Regression
    * **`sklearn.linear_model.LinearRegression`
    * **`sklearn.linear_model.Ridge`
    * **`sklearn.linear_model.Lasso`
    * **`sklearn.linear_model.ElasticNet`
    * *`sklearn.svm.SVR`
    * *`sklearn.tree.DecisionTreeRegressor`
    * *`sklearn.ensemble.RandomForestRegressor`
    * *`sklearn.ensemble.GradientBoostingRegressor`
    * *`sklearn.neural_network.MLPRegressor`

## Supported lightGBM Models
   * **`lightgbm.LGBMClassifier - binary - Gradient Boosting Trees`
   * **`lightgbm.LGBMClassifier - multiclass - Gradient Boosting Trees`
   * **`lightgbm.LGBMClassifier - binary - Random Forest`
   * **`lightgbm.LGBMClassifier - multiclass - Random Forest`
   * **`lightgbm.LGBMRegressor - Gradient Boosting Trees`
   * **`lightgbm.LGBMRegressor - Random Forest`

## Supported dask-ml Models
   * **`dask-ml.preprocessing.LabelEncoder`
   * **`dask-ml.preprocessing.OneHotEncoder`
   * **`dask-ml.preprocessing.MinMaxScaler`


# Example
   * [I'm a relative reference to a repository file](../blob/master/LICENSE)
