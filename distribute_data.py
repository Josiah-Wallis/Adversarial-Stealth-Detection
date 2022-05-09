# %% 1-Client Adversarial Stealth Detection (1-CLient ASD)
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# %%
X, y = fetch_openml('mnist_784', version = 1, return_X_y = True, as_frame = False)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 20)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
y_train = to_categorical(y_train.astype('int64'), 10)
y_test = to_categorical(y_test.astype('int64'), 10)

# %% add a tolerance distance between idxs
def split_amongst_clients(X, y, client_num):
    #np.random.seed(5)
    split_idx = np.random.uniform(0, X.shape[0], client_num - 1)
    split_idx = np.sort(split_idx).astype('int64')

    clients_X = []
    clients_y = []
    start = 0
    for end in split_idx:
        data = X[start : end]
        labels = y[start : end]

        clients_X.append(data)
        clients_y.append(labels)

        start = end

    data = X[start:]
    labels = y[start:]

    clients_X.append(data)
    clients_y.append(data)

    return split_idx, clients_X, clients_y

# %%
_, clients_train_X, clients_train_y = split_amongst_clients(X_train, y_train, 10)
_, clients_test_X, clients_test_y = split_amongst_clients(X_test, y_test, 10)

# %%
for i in range(len(clients_train_X)):
    print(clients_train_X[i].shape)
    print(clients_train_y[i].shape)


# %%
