from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.tree import ExtraTreeRegressor
from datrics_json import csr
import numpy as np
import scipy as sp
from datrics_json import classification as clf
from datrics_json import regression as reg



def serialize_kmeans_clustering(model):
    serialized_model = {
        'meta': 'kmeans_clustering',
        'cluster_centers_': model.cluster_centers_.tolist(),
        'labels_': model.labels_.tolist(),
        'inertia_': model.inertia_,
        'n_features_in_': model.n_features_in_,
        'n_iter_': model.n_iter_,
        '_n_threads': model._n_threads,
        '_tol': model._tol,

        'params': model.get_params()
    }

    return serialized_model


def deserialize_kmeans_clustering(model_dict):
    model = KMeans(model_dict['params'])

    model.cluster_centers_ = np.array(model_dict['cluster_centers_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.inertia_ = model_dict['inertia_']
    model.n_features_in_ = model_dict['n_features_in_']
    model.n_iter_ = model_dict['n_iter_']
    model._n_threads = model_dict['_n_threads']
    model._tol = model_dict['_tol']

    return model


def serialize_dbscan_clustering(model):
    serialized_model = {
        'meta': 'dbscan_clustering',
        'components_': model.components_.tolist(),
        'core_sample_indices_': model.core_sample_indices_.tolist(),
        'labels_': model.labels_.tolist(),
        'n_features_in_': model.n_features_in_,
        '_estimator_type': model._estimator_type,

        'params': model.get_params()
    }

    return serialized_model


def deserialize_dbscan_clustering(model_dict):
    model = DBSCAN(**model_dict['params'])
    #model.eps = model_dict['params']['eps']

    model.components_ = np.array(model_dict['components_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.core_sample_indices_ = model_dict['core_sample_indices_']
    model.n_features_in_ = model_dict['n_features_in_']
    model._estimator_type = model_dict['_estimator_type']

    return model


def serialize_iforest(model):
    params = model.get_params()
    n_features_ = model.n_features_
    n_features_in_ = model.n_features_in_
    max_samples_ = model.max_samples_
    _max_features = model._max_features
    max_features = model.max_features
    estimators_features_ = model.estimators_features_
    _seeds = model._seeds
    _n_samples = model._n_samples

    base_estimator_ = model.base_estimator_.get_params()
    base_estimator_.pop('splitter')

    offset_ = model.offset_
    oob_score = model.oob_score

    estimators_ = []
    for est in model.estimators_:

        tree_, dtypes = clf.serialize_tree(est.tree_)

        tree_dtypes = []
        for i in range(0, len(dtypes)):
            tree_dtypes.append(dtypes[i].str)

        et_params = est.get_params()
        et_params.pop('splitter')

        serialized_tree = {
            "params": et_params,
            "splitter": est.splitter,
            "max_features_'": est.max_features_,
            "n_features_": est.n_features_,
            "n_features_in_": est.n_features_in_,
            "n_outputs_": est.n_outputs_,
            "tree_": tree_}

        serialized_tree['tree_']['nodes_dtype'] = tree_dtypes

        estimators_.append(serialized_tree)

    estimators_features_ = list(map(lambda x: x.tolist(), estimators_features_))

    serialized_model = {
        "params": params,
        "base_estimator_": base_estimator_,
        "estimators_features_": estimators_features_,
        "_seeds": _seeds.tolist(),
        "n_features_": n_features_,
        "n_features_in_": n_features_in_,
        "max_samples_": max_samples_,
        "max_features": max_features,
        "_max_features": _max_features,
        "_n_samples": _n_samples,
        "offset_": offset_,
        "oob_score": oob_score,
        "estimators_": estimators_}

    return serialized_model

def deserialize_iforest(model_dict):
    model = IsolationForest(**model_dict['params'])

    for param in list(model_dict.keys())[4:-1]:
        setattr(model, param, model_dict[param])

    model.base_estimator_ = ExtraTreeRegressor(**model_dict['base_estimator_'])

    estimators_features_ = list(map(lambda x: np.array(x), model_dict['estimators_features_']))
    model.estimators_features_ = estimators_features_

    _seeds = np.array(model_dict['_seeds'])
    model._seeds = _seeds

    new_estimators = []
    for est_dict in model_dict['estimators_']:
        est = ExtraTreeRegressor(**est_dict['params'])
        for param in list(est_dict.keys())[1:-1]:
            setattr(est, param, est_dict[param])
        est.tree_ = reg.deserialize_tree(est_dict['tree_'],
                                         est_dict['n_features_'],
                                         est.n_classes_[0],
                                         est_dict['n_outputs_'])
        new_estimators.append(est)
    model.estimators_ = new_estimators

    return model