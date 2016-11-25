from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import time


def get_images(color=False):
    d = fetch_lfw_people(color=color, min_faces_per_person=70, resize=1)

    if color:
        X = d.images
    else:
        X = d.data

    y = d.target

    return X, y

def test(model, X, y, f='accuracy', k=10):
    start = time.time()
    print('start: ', start)

    np.random.seed(42)

    results = cross_val_score(model, X, y, scoring=f, cv=k)

    end = time.time()
    print('end', end, 'duration', end - start)

    return np.mean(results), np.std(results)

def show(img, *args, **kwargs):
    plt.imshow(img.astype(np.uint8), *args, **kwargs)
