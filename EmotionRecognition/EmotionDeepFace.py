from os.path import exists
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	from keras.models import Model, Sequential
	from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
elif tf_version == 2:
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from glob import glob
import datetime

# Eu meti este código no if em cima estava a dar erro

if tf_version == 1:
    from keras import callbacks
    from keras.models import load_model
if tf_version == 2:
    from tensorflow.keras import callbacks
    from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator


labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
def loadModel(path):
    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(7, activation="softmax"))

    model.load_weights(path)

    return model

def getClassIndices():
    return labels