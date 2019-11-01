# sklearn-json
Export scikit-learn model files to JSON for sharing or deploying predictive models with peace of mind.

# Why sklearn-json?
Other methods for exporting scikit-learn models require Pickle or Joblib (based on Pickle). Serializing model files with Pickle provides a simple attack vector for malicious users-- they give an attacker the ability to execute arbitrary code wherever the file is deserialized. For an example see: https://www.smartfile.com/blog/python-pickle-security-problems-and-solutions/.

sklearn-json is a safe and transparent solution for exporting scikit-learn model files.

### Safe
Export model files to 100% JSON which cannot execute code on deserialization.

### Transparent
Model files are serialized in JSON (i.e., not binary), so you have the ability to see exactly what's inside.

# Getting Started

sklearn-json makes exporting model files to JSON simple.

## Install
```
pip install sklearn-json
```
## Example Usage

```python
import sklearn_json as skljson
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(X, y)

skljson.to_json(model, file_name)
deserialized_model = skljson.from_json(file_name)

deserialized_model.predict(X)
```

# Features
The list of supported models is rapidly growing. If you have a request for a model or feature, please reach out to support@mlrequest.com.

sklearn-json requires scikit-learn >= 0.21.3.

## Supported scikit-learn Models

* Classification
    * `sklearn.linear_model.LogisticRegression`
    * `sklearn.linear_model.Perceptron`
    * `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
    * `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
    * `sklearn.svm.SVC`
    * `sklearn.naive_bayes.GaussianNB`
    * `sklearn.naive_bayes.MultinomialNB`
    * `sklearn.naive_bayes.ComplementNB`
    * `sklearn.naive_bayes.BernoulliNB`
    * `sklearn.tree.DecisionTreeClassifier`
    * `sklearn.ensemble.RandomForestClassifier`
    * `sklearn.ensemble.GradientBoostingClassifier`
    * `sklearn.neural_network.MLPClassifier`

* Regression
    * `sklearn.linear_model.LinearRegression`
    * `sklearn.linear_model.Ridge`
    * `sklearn.linear_model.Lasso`
    * `sklearn.svm.SVR`
    * `sklearn.tree.DecisionTreeRegressor`
    * `sklearn.ensemble.RandomForestRegressor`
    * `sklearn.ensemble.GradientBoostingRegressor`
    * `sklearn.neural_network.MLPRegressor`
