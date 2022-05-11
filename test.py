# %% 1-Client Adversarial Stealth Detection (1-CLient ASD) - test run via ipython notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from distribute_data import generate_mnist_client_data, create_adversary
from asd import FedAvg, generate_model

# %%
bunch = generate_mnist_client_data()

# %%
adv_labels = create_adversary(bunch['Client Train Labels'], 3, 4, 9)

# %%
np.sum(adv_labels[3] != bunch['Client Train Labels'][3])

# %%
#w, b = FedAvg(bunch['Client Train Data'], adv_labels, rounds = 50)
w, b = FedAvg(bunch['Client Train Data'], adv_labels, rounds = 20)


# %%
model = generate_model(w, b, [0, 2, 5])

# %%
model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# %%
model.evaluate(bunch['Client Test Data'][0], bunch['Client Test Labels'][0], use_multiprocessing = True)

