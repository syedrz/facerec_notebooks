from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin, clone
import numpy as np


class MultiDimensionalModel(BaseEstimator, TransformerMixin):
    def __init__(self, models, dimensions=None):
        if isinstance(models, list):
            self.models = models
        else:
            self.models = [clone(models) for _ in range(dimensions)]

    def fit(self, X, y=None):
        for space, model in zip(np.swapaxes(X, 0, 1), self.models):
            model.fit(space, y)
        return self

    def transform(self, X):
        retval = None
        # TODO: pre-allocate and set slice
        for space, model in zip(np.swapaxes(X, 0, 1), self.models):
            fitted = model.transform(space)
            if retval is None:
                retval = fitted
            else:
                retval = np.concatenate((retval, fitted), axis=1)
        return retval


class SKLRecogniser:
    def __init__(self, model_spec_path, name=None):
        if name is None:
            self.name = model_spec_path.split('/')[-1].split('\\')[-1]
        else:
            self.name = name
        self.model = joblib.load(model_spec_path)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)
