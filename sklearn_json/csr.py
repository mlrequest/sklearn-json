import numpy as np
import scipy as sp


def serialize_csr_matrix(csr_matrix):
    serialized_csr_matrix = {
        'meta': 'csr',
        'data': csr_matrix.data.tolist(),
        'indices': csr_matrix.indices.tolist(),
        'indptr': csr_matrix.indptr.tolist(),
        '_shape': csr_matrix._shape,
    }
    return serialized_csr_matrix


def deserialize_csr_matrix(csr_dict, data_type=np.float64, indices_type=np.int32, indptr_type=np.int32):
    csr_matrix = sp.sparse.csr_matrix(tuple(csr_dict['_shape']))
    csr_matrix.data = np.array(csr_dict['data']).astype(data_type)
    csr_matrix.indices = np.array(csr_dict['indices']).astype(indices_type)
    csr_matrix.indptr = np.array(csr_dict['indptr']).astype(indptr_type)

    return csr_matrix
