from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from scipy.misc import imread, imresize
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import os

DATA_FOLDER = 'results/'

def get_images(color=False, min_faces_per_person=70):
    return fetch_lfw_people(color=color, min_faces_per_person=min_faces_per_person, resize=1)

def get_clamv_images(base_dir='clamv_faces', face_type='face', resize_size=None):
    X = []
    y = []
    
    classes = {}
    
    # Get all the known classes
    subdir, dirs, files = next(os.walk(base_dir))
    
    for i in range(len(dirs)):
        classes[dirs[i]] = i
    
    # Go through each class and get the face type, placing it into X and 
    for subdir, dirs, files in os.walk(base_dir):
        split = subdir.split('/')
        
        # It's not a directory with images
        if len(split) != 3:
            continue
        
        # It's not the correct face type
        if split[2] != face_type:
            continue
        
        for f in files:
            # Do not use images that are not cutouts
            if 'face' not in f:
                continue
                
            face_path = os.path.join(subdir, f)
            
            img = imread(face_path)
            X.append(img)
            
            y.append(classes[split[1]])
    
    
    if resize_size is not None:
        new_X = []
        for x in X:
            new_X.append(imresize(x, size=resize_size))
        
        return np.array(new_X), np.array(y)
    else:
        return np.array(X), np.array(y)
            

def score(filename, metric, *args, **kwargs):
    D = np.load(DATA_FOLDER + filename + '.npy')

    results = []

    for y, y_pred in D:
        results.append(metric(y, y_pred, *args, **kwargs))

    return np.mean(results), np.std(results)

def test(model, X, y, filename, k=10):
    start = time.time()
    print(time.ctime(start))

    results = []

    skf = StratifiedKFold(n_splits=k, random_state=42)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Variables are collinear.')
        warnings.filterwarnings('ignore', message='The priors do not sum to 1. Renormalizing')

        for train, test in skf.split(X, y):
            m = clone(model)
            m.fit(X[train], y[train])
            y_pred = m.predict(X[test])
            
            print("Fold Accuracy:" + str(accuracy_score(y[test], y_pred)))
            results.append((y[test], y_pred))

    print('duration:', time.time() - start)

    np.save(DATA_FOLDER + filename + '.npy', results)

    return score(filename, accuracy_score)

def show(img, *args, **kwargs):
    plt.imshow(img.astype(np.uint8), *args, **kwargs)

def up_to_per_person(X, y, n=10):
    out = set()
    d = dict()

    for k, v in enumerate(y):
        if v in d:
            d[v] += 1
        else:
            d[v] = 1

        if d[v] <= n:
            out.add(k)

    print('classes', len(d))

    out = sorted(out)

    return X[out], y[out]
