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
from asd import FedAvg, generate_model
from copy import deepcopy


# %%
bunch = generate_mnist_client_data()

# %%
def create_adversary(client_train_labels, client, true, target):
    adv_labels = deepcopy(client_train_labels)
    num_labels = len(adv_labels[0][0])
    target_label = to_categorical(target, num_labels)

    rev_one_hot = np.argmax(adv_labels[client], axis = 1)
    adv_labels[client][rev_one_hot == true] = target_label
    return adv_labels

# %%
adv_labels = create_adversary(bunch['Client Train Labels'], 3, 4, 9)

# %%
np.sum(adv_labels[3] != bunch['Client Train Labels'][3])

# %%
#w, b = FedAvg(bunch['Client Train Data'], adv_labels, rounds = 50)
w, b = FedAvg(bunch['Client Train Data'], bunch['Client Train Labels'], rounds = 50)

# %%
model = generate_model(w, b, [0, 2, 5])

# %%
model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#model.fit(bunch['Client Train Data'][0], bunch['Client Train Labels'][0], validation_split = 0.2, batch_size = 100, epochs = 5, verbose = 2, use_multiprocessing = True)

# %%
model.evaluate(bunch['Client Test Data'][0], bunch['Client Test Labels'][0], use_multiprocessing = True)
# %%
