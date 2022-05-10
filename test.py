# %% 1-Client Adversarial Stealth Detection (1-CLient ASD)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from distribute_data import generate_mnist_client_data
from asd import FedAvg


# %%
bunch = generate_mnist_client_data()

# %%
w, b = FedAvg(pkg['Client Train Data'], pkg['Client Train Labels'], rounds = 50)

# %%
model = generate_model(w, b, [0, 2, 5])

# %%
model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(pkg['Client Train Data'][0], pkg['Client Train Labels'][0], validation_split = 0.2, batch_size = 100, epochs = 5, verbose = 2, use_multiprocessing = True)

# %%
