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
from fedavg import FedAvg, generate_model
from tensorflow.keras.datasets import fashion_mnist

from asd import *
# %%
x = fashion_mnist.load_data()
# %%
x
# %%
type(x)
# %%
len(x)
# %%
thing = x[0]
# %%
other = x[1]
# %%
thing
# %%
other
# %%
type(thing[0])
# %%
data = pd.read_csv('adult.data', sep = ',', header = None)
# %%
data
# %%
data[14][50]
# %%
