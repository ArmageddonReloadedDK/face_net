from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.applications.vgg16 import VGG16
from skimage import io
from sklearn.model_selection import train_test_split

import numpy as np
import os


# считывание количества фотографий
def getData(NB_CLASSES):
    N = 0
    shape = 128
    rgb = 3
    # base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # path = os.path.join(base_dir, "face_net\dataset")
    path = "dataset"
    for folder in os.listdir(path):
        for name in os.listdir(path + '/' + folder):
            N += 1

    X = np.zeros((N, shape, shape, rgb))
    Y = np.zeros((N))

    step = 0

    # считывание фоток для создания датасета

    for i, folder in enumerate(os.listdir(path)):
        for j, name in enumerate(os.listdir(path + '/' + folder)):
            image = io.imread(path + '/' + folder + '/' + name)
            X[step] = image
            Y[step] = int(folder)
            step += 1

    Y = np_utils.to_categorical(Y, NB_CLASSES)

    X /= 255

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y,
                                                    test_size=0.2, random_state=42)

    return Xtrain, Xtest, Ytrain, Ytest


# adam 0.11

# model = VGG_16('vgg16_weights.h5')
def getModel(NB_CLASSES):
    model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Sequential()

    for layer in model_vgg.layers:
        model.add(layer)

    model.layers.pop()
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    #  model.add(Dropout(0.2))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    model.summary()
    return model


def train(model, Xtrain, Ytrain):
    model.compile(metrics=['categorical_accuracy'], optimizer='Adam', loss='categorical_crossentropy')

    history = model.fit(Xtrain, Ytrain, batch_size=16,
                        epochs=18, validation_split=0.2,
                        verbose=1)
    model.save_weights('face_net.h5', overwrite=True)
    return history


def test(model, Xtest, Ytest):
    print('Testing...')
    score = model.evaluate(Xtest, Ytest,
                           batch_size=10, verbose=True)
    print("\nTest score:", score[0])
    print('Test accuracy:', score[1])

    model_json = model.to_json()
    open('face_net.json', 'w').write(model_json)
    model.save_weights('face_net.h5', overwrite=True)
    return score


if __name__ == '__main__':
    NB_CLASSES = 28
    Xtrain, Xtest, Ytrain, Ytest = getData(NB_CLASSES)
    model = getModel(NB_CLASSES)
    history = train(model, Xtrain, Ytrain)
    score = test(model, Xtest, Ytest)
