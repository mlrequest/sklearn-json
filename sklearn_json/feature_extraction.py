from sklearn.feature_extraction import DictVectorizer


def serialize_dict_vectorizer(model):
    serialized_model = {
        'meta': 'dict-vectorizer',
        'dtype': model.dtype,
        'separator': model.separator,
        'sparse': model.sparse,
        'sort': model.sort,
        'feature_names': model.get_feature_names(),
        'vocabulary': model.vocabulary_,
    }

    return serialized_model


def deserialize_dict_vectorizer(model_dict):
    model = DictVectorizer()

    model.dtype = model_dict['dtype']
    model.separator = model_dict['separator']
    model.sparse = model_dict['sparse']
    model.sort = model_dict['sort']
    model.feature_names_ = model_dict['feature_names']
    model.vocabulary_ = model_dict['vocabulary']

    return model
