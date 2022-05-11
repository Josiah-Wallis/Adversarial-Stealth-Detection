# %% Splitting up the data among the clients
import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Creates a single adversarial client (targeted)
def create_adversary(client_train_labels, client, true, target):
    adv_labels = deepcopy(client_train_labels)
    num_labels = len(adv_labels[0][0])
    target_label = to_categorical(target, num_labels)

    rev_one_hot = np.argmax(adv_labels[client], axis = 1)
    adv_labels[client][rev_one_hot == true] = target_label
    return adv_labels

# Distributes data into client_num datasets
def split_among_clients(X, y, split_idxs):
    clients_X = []
    clients_y = []
    start = 0
    for end in split_idxs:
        data = X[start : end]
        labels = y[start : end]

        clients_X.append(data)
        clients_y.append(labels)

        start = end

    data = X[start:]
    labels = y[start:]

    clients_X.append(data)
    clients_y.append(labels)

    return clients_X, clients_y

# Z-normalizes data
def normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

# Checks if each client has more than tolerance samples
def check_tolerance(idxs, size, tolerance):
    start = 0
    for idx in idxs:
        if (idx - start) <= tolerance:
            return False
        start = idx

    if (size - start) < tolerance:
        return False
    else:
        return True

# Suggests options if tolerance not met
def validate_distribution(split_idxs, N, tolerance, client_num):
    count = 0
    while True:
        if check_tolerance(split_idxs, N, tolerance):
            return split_idxs
            break
        else:
            count += 1

            if count == 500:
                print('The program is having trouble fitting the specified tolerance.\nPlease try a smaller tolernace. Exiting with error code -1...')
                return -1

            split_idxs = np.random.uniform(0, N, client_num - 1)
            split_idxs = np.sort(split_idxs).astype('int64')

# Generates the federated dataset
def generate_mnist_client_data(client_num = 10, tolerance = 2000, test_size = 0.25):
    X, y = fetch_openml('mnist_784', version = 1, return_X_y = True, as_frame = False)
    N = X.shape[0]

    # Data preparation
    X = normalize(X)
    X = X.reshape(X.shape[0], 28, 28, 1)
    y = to_categorical(y.astype('int64'), 10)

    # Defining data split
    split_idxs = np.random.uniform(0, N, client_num - 1)
    split_idxs = np.sort(split_idxs).astype('int64')
    split_idxs = validate_distribution(split_idxs, N, tolerance, client_num)

    assert type(split_idxs) != int

    client_X, client_y = split_among_clients(X, y, split_idxs)

    # For each client, split data into train and test sets
    client_train_data = []
    client_train_labels = []
    client_test_data = []
    client_test_labels = []
    for i in range(len(client_X)):
        X_train, X_test, y_train, y_test = train_test_split(client_X[i], client_y[i], test_size = test_size)
        client_train_data.append(X_train)
        client_test_data.append(X_test)
        client_train_labels.append(y_train)
        client_test_labels.append(y_test)

    return {'Client Train Data': client_train_data, 'Client Train Labels':  client_train_labels, 'Client Test Data': client_test_data, 'Client Test Labels': client_test_labels}

