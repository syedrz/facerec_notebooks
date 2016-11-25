from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV


class EigenClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.pca = PCA()
        self.svm = LinearSVC()

    def fit(self, X, y):
        X_transformed = self.pca.fit_transform(X, y)

        self.svm.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.pca.transform(X)
        return self.svm.predict(X_transformed)



class EigenClassifierGridSearch(BaseEstimator, ClassifierMixin):
    def __init__(self, pca_params, svc_params, param_grid):
        self.pca_params = pca_params
        self.svc_params = svc_params
        self.param_grid = param_grid

        self.pca = PCA(**pca_params)
        self.svm = GridSearchCV(SVC(**svc_params), param_grid)

    def fit(self, X, y):
        X_transformed = self.pca.fit_transform(X, y)

        self.svm.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.pca.transform(X)
        return self.svm.predict(X_transformed)
