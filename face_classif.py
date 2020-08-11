from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.applications.vgg16 import VGG16
from skimage import io

import numpy as np
import os

IMG_CHANNELS = 3
IMG_ROWS = 128
IMG_COLS = 128

BATCH_SIZE = 20
NB_EPOCH = 40
NB_CLASSES = 28
VERBOSE = 1
VALIDATION_SPLIT = 0.2

# np.random.seed(1845)

dense_num = 1024

N = 0
shape = 128
rgb = 3

# считывание количества фотографий
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path=os.path.join(base_dir, "face_net_mkr\dataset")

for folder in os.listdir(path):
    for name in os.listdir(path + '/' + folder):
        N += 1

x_train = np.zeros((N, shape, shape, rgb))
y_train = np.zeros((N))

step = 0

# считывание фоток для создания датасета

for i, folder in enumerate(os.listdir(path)):
    for j, name in enumerate(os.listdir(path + '/' + folder)):
        image = io.imread(path + '/' + folder + '/' + name)
        x_train[step] = image
        y_train[step] = int(folder)
        step += 1

num = 2000

X_train = np.zeros((num, shape, shape, rgb))
Y_train = np.zeros((num))

X_test = np.zeros((num, shape, shape, rgb))
Y_test = np.zeros((num))
# y_train=y_train.reshape((y_train.shape[0],1))

# создание датасета с примением перемешивания

shape_v = x_train.shape[0] - 1
for i in range(num):
    x = np.random.randint(0, shape_v)
    X_train[x] = x_train[x]
    Y_train[x] = y_train[x]

num_test = 500
X_test = np.zeros((num_test, shape, shape, rgb))
Y_test = np.zeros((num_test))
for i in range(num_test):
    x = np.random.randint(0, shape_v)
    X_train[x] = x_train[x]
    Y_train[x] = y_train[x]

X_train = X_train.astype('float16')
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)

X_test = X_test.astype('float16')
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

X_train /= 255

# adam 0.11


# model = VGG_16('vgg16_weights.h5')
model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
model = Sequential()

for layer in model_vgg.layers:
    model.add(layer)

model.layers.pop()
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES, activation='softmax'))
model.summary()

model.compile(metrics=['categorical_accuracy'], optimizer='Adam', loss='categorical_crossentropy')

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs=18, validation_split=0.2,
                    verbose=1)

print('Testing...')
score = model.evaluate(X_test, Y_test,
                       batch_size=10, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
open('face_net.json', 'w').write(model_json)
model.save_weights('face_net.h5', overwrite=True)
