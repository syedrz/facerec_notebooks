from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC


class EigenClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.pca = PCA()
        self.svm = LinearSVC()

    def fit(self, X, y):
        self.pca.fit(X, y)
        X_transformed = self.pca.transform(X)

        self.svm.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.pca.transform(X)
        return self.svm.predict(X_transformed)
