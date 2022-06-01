# %% 1-Client Adversarial Stealth Detection (1-CLient ASD) - test run via ipython notebook
from distribute_data import Datasets
from fedavg import FederatedSystem
import matplotlib.pyplot as plt

from asd import *

# %%
model = FederatedSystem('cifar', seed = 30)
# %%
w, b, tally = model.FedAvg(rounds = 15)

# %%
test_accs, test_losses = model.test_acc()
# %%
x = range(16)
plt.plot(x, test_losses, color = 'blue', marker = 's')
# %%
