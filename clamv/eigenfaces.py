from sklearn.base import BaseEstimator, ClassifierMixin


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, transform, classify):
        self.transform = transform
        self.classify = classify

    def fit(self, X, y):
        X_transformed = self.transform.fit_transform(X, y)

        self.classify.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.transform.transform(X)
        return self.classify.predict(X_transformed)
