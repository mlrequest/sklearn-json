from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.tree._tree import Tree
from datrics_json import csr
import numpy as np
import scipy as sp


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

def serialize_isolation_forest(model):
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
