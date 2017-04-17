from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
import numpy as np


class ColorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, transform, classify):
        if type(transform) is list:
            self.transform = transform
        else:
            self.transform = [clone(transform) for i in range(3)]

        self.classify = classify

    def fit(self, X, y):
        X = ColorClassifier.subsample(ColorClassifier.to_yuv(X))

        X_transformed = []

        for i in range(3):
            Xt = X[i]
            Xt = Xt.reshape(Xt.shape[0], Xt.shape[1] * Xt.shape[2])
            X_transformed.append(self.transform[i].fit_transform(Xt, y))

        self.classify.fit(ColorClassifier.flatten(X_transformed), y)
        return self

    def predict(self, X):
        X = ColorClassifier.subsample(ColorClassifier.to_yuv(X))

        X_transformed = []

        for i in range(3):
            Xt = X[i]
            Xt = Xt.reshape(Xt.shape[0], Xt.shape[1] * Xt.shape[2])
            X_transformed.append(self.transform[i].transform(Xt))

        return self.classify.predict(ColorClassifier.flatten(X_transformed))
    

    @staticmethod
    def to_yuv(X):
        rgb2yuv = np.array([[0.299, 0.587, 0.114],
                           [-0.14713, -0.28886, 0.436],
                           [0.615, -0.51499, -0.10001]])
        return np.dot(X, rgb2yuv.T)

    @staticmethod
    def subsample(X):
        Y = X[:, :, :, 0]
        U = X[:, ::2, ::2, 1]
        V = X[:, ::2, ::2, 2]
        return (Y, U, V)

    @staticmethod
    def flatten(X):
        X = np.swapaxes(np.swapaxes(X, 0, 1), 1, 2)
        return X.reshape(X.shape[0], X.shape[1] * X.shape[2])
