from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import cross_val_score
import numpy as np


def get_images():
  d = fetch_lfw_people(min_faces_per_person=70, resize=1)
  X = d.data
  y = d.target

  return X, y

def test(model, X, y, f='accuracy', k=10):

  results = cross_val_score(model, X, y, scoring=f, cv=k)

  return np.mean(results), np.std(results)
