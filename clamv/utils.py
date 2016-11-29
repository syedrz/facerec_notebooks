from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt
import time


def get_images(color=False, min_faces_per_person=70):
    d = fetch_lfw_people(color=color, min_faces_per_person=min_faces_per_person, resize=1)

    if color:
        X = d.images
    else:
        X = d.data

    y = d.target

    return X, y

def test(model, X, y, filename, k=10):
    start = time.time()
    print('start: ', start)

    results = []

    skf = StratifiedKFold(n_splits=k, random_state=42)

    for train, test in skf.split(X, y):
        m = clone(model)
        m.fit(X[train], y[train])
        y_pred = m.predict(X[test])
        results.append((y[test], y_pred))

    end = time.time()
    print('end', end, 'duration', end - start)

    np.save(filename, results)

def show(img, *args, **kwargs):
    plt.imshow(img.astype(np.uint8), *args, **kwargs)
