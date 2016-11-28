from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class OpenCVClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, get_recognizer):
        self.get_recognizer = get_recognizer
        self.recognizer = get_recognizer()

    def fit(self, X, y):
        self.recognizer.train(X, y)

        return self

    def predict(self, X):
        predicted = []
        
        for x in X:
            y = self.recognizer.predict(x)

            if type(y) is tuple:
                predicted.append(y[0])
            else:
                predicted.append(y)

        return np.array(predicted)
