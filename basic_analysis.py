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
def test_acc(ws, bs, test_data, test_labels):
    T = len(ws)
    test_accs = []

    for t in range(T):
        model = generate_model(ws[t], bs[t], [0, 2, 5])
        model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        acc = model.evaluate(test_data, test_labels, verbose = 0, use_multiprocessing = True)[1]
        test_accs.append(acc)

    return test_accs

# %%
bunch = generate_mnist_client_data()

# %%
test_data = np.concatenate([x for x in bunch['Client Test Data']])
test_labels = np.concatenate([y for y in bunch['Client Test Labels']])

# %%
adv_labels = create_adversary(bunch['Client Train Labels'], 3, 4, 9)

# %%
#w, b = FedAvg(bunch['Client Train Data'], adv_labels, rounds = 50)
w, b, ws, bs = FedAvg(bunch['Client Train Data'], bunch['Client Train Labels'], rounds = 55)

# %%
r55_standard_test_acc = test_acc(ws, bs, test_data, test_labels)

# %%
w, b, ws, bs = FedAvg(bunch['Client Train Data'], adv_labels, rounds = 55)

# %%
r55_adv_test_acc = test_acc(ws, bs, test_data, test_labels)

# %%
w, b, ws, bs = FedAvg(bunch['Client Train Data'], bunch['Client Train Labels'], learning_rate = 0.01, rounds = 55)

# %%
r55_standard_01_acc = test_acc(ws, bs, test_data, test_labels)

# %%
w, b, ws, bs = FedAvg(bunch['Client Train Data'], adv_labels, learning_rate = 0.01, rounds = 55)

# %%
r55_adv_01_acc = test_acc(ws, bs, test_data, test_labels)

# %%
x = range(56)
plt.figure(figsize = (10, 7))
plt.plot(x, r55_standard_test_acc, color = 'red', label = 'standard - learning_rate = 0.001', marker = '.')
plt.plot(x, r55_adv_test_acc, color = 'green', label = 'adv - learning_rate = 0.001', marker = '.')
plt.plot(x, r55_standard_01_acc, color = 'blue', label = 'standard - learning_rate = 0.01', marker = '.')
plt.plot(x, r55_adv_01_acc, color = 'purple', label = 'adv - learning_rate = 0.01', marker = '.')
plt.axis([0, 57, 0.8, 1])
plt.xlabel('Round')
plt.ylabel('Test Accuracy')
plt.title('MNIST Test Data Non-IID')
plt.legend()

# %%
