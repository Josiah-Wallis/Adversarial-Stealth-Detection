# %% 1-Client Adversarial Stealth Detection (1-CLient ASD)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# %%
X, y = fetch_openml('mnist_784', version = 1, return_X_y = True, as_frame = False)

# %%
plt.imshow(X[1].reshape((28, 28)), cmap = 'gray')
plt.show()

# %%
np.sum(y == '4')

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 20)
X_train = X_train.reshape((46900, 28, 28, 1))
X_test = X_test.reshape((23100, 28, 28, 1))
y_train = to_categorical(y_train.astype('int64'), 10)
y_test = to_categorical(y_test.astype('int64'), 10)


# %%
model = Sequential([
    Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Flatten(),
    Dense(units = 10, activation = 'softmax')
])

# %%
model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# %%
model.fit(x = X_train, y = y_train, validation_split = 0.1, epochs = 20, verbose = 2)

# %%
model.evaluate(X_test, y_test)

# %%
