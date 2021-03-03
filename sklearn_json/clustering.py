from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.tree._tree import Tree
from sklearn_json import csr
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
        'params': model.get_params()
    }

    return serialized_model


def deserialize_kmeans_clustering(model_dict):
    model = KMeans(model_dict['params'])

    model.cluster_centers_ = np.array(model_dict['cluster_centers_'])
    model.labels_ = np.array(model_dict['labels_'])
    model.inertia_ = model_dict['inertia_'].astype(np.float)
    model.n_features_in_ = model_dict['n_features_in_'].astype(np.int)
    model.n_iter_ = model_dict['n_iter_'].astype(np.int)

    return model
