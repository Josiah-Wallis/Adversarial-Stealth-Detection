# %% 1-Client Adversarial Stealth Detection (1-CLient ASD) - test run via ipython notebook
from distribute_data import Datasets
from fedavg import FederatedSystem
import matplotlib.pyplot as plt
from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

from asd import *

# %%
model = FederatedSystem('fashion', tolerance = 3000)
# %%
w, b, tally = model.FedAvg(rounds = 20)

# %%
test_accs, test_losses = model.test_acc()
# %%
x = range(21)
plt.plot(x, test_accs, color = 'blue', marker = 's')
# %%
