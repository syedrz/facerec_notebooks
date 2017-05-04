import numpy as np
import keras
import cv2
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from skimage.transform import resize

# Load the dataset
y = []
X = []
for path in glob('../final_dataset/*/*'):
    _, _, y1, name = path.split('/')
    x1 = cv2.imread(path)
    
    y.append(y1)
    X.append(x1)

clmv_dict = {}
for idx, val in enumerate(np.unique(y)):
    clmv_dict[val] = idx

clmv_y = []
for n in y:
    clmv_y.append(clmv_dict[n])

y = keras.utils.to_categorical(clmv_y, num_classes=len(np.unique(clmv_y)))

# Resize the output from the CNN
resized_X = []
for x in X:
	resized_X.append(resize(x, (100, 100)))

resized_X = np.array(resized_X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(resized_X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(clmv_y)), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(resized_X, y, batch_size=32, epochs=15)
model.save('cnn.h5')
