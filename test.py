# %% 1-Client Adversarial Stealth Detection (1-CLient ASD) - test run via ipython notebook
from distribute_data import Datasets
from fedavg import FederatedSystem
import matplotlib.pyplot as plt
from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

from asd import *

# %%
dataHandler = Datasets()
fashion_system = FederatedSystem('fashion')
fashion_adv_system = FederatedSystem('fashion')
mnist_system = FederatedSystem('digits')
mnist_adv_system = FederatedSystem('digits')


fashion_train_labels = fashion_adv_system.client_train_labels
fashion_system.client_train_data = fashion_adv_system.client_train_data
fashion_system.client_train_labels = fashion_train_labels
fashion_adv_labels = dataHandler.create_adversary(fashion_train_labels, client = 5, true = 8, target = 3)

fashion_adv_system.poison(fashion_adv_labels)

# %%
a, b, c = fashion_system.FedAvg(rounds = 10)

# %%
d, e, f = fashion_adv_system.FedAvg(rounds = 10, enable = 1, threshold = 5)

# %%
#z, h = fashion_system.test_acc()
t, v = fashion_adv_system.test_acc()
# %%
x = range(11)
#plt.plot(x, z, color = 'blue', marker = 's')
plt.plot(x, t, color = 'red', marker = 'o')
# %%
c
# %%

f
# %%
