# Splitting up the data among the clients
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist, cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

class Datasets:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if self.dataset_name == 'cifar':
            self.dataset = cifar10.load_data()
        elif self.dataset_name == 'fashion':
            self.dataset = fashion_mnist.load_data()

        #self.seed = seed
    
    # Creates a single adversarial client (targeted)
    def create_adversary(self, client_train_labels, client, true, target):
        adv_labels = deepcopy(client_train_labels)
        num_labels = len(adv_labels[0][0])
        target_label = to_categorical(target, num_labels)

        rev_one_hot = np.argmax(adv_labels[client], axis = 1)
        adv_labels[client][rev_one_hot == true] = target_label
        return adv_labels

    # Distributes data into client_num datasets
    def split_among_clients(self, X, y, split_idxs):
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
    def normalize(self, X):
        scaler = StandardScaler().fit(X)
        return scaler.transform(X)

    # Checks if each client has more than tolerance samples
    def check_tolerance(self, idxs, size, tolerance):
        start = 0
        for idx in idxs:
            if (idx - start) <= tolerance:
                return False
            start = idx

        if (size - start) < tolerance:
            return False
        else:
            return True

    # Forces tolerance if tolerance not met
    def validate_distribution(self, split_idxs, N, tolerance, client_num):
        count = 0
        while True:
            if self.check_tolerance(split_idxs, N, tolerance):
                return split_idxs
                break
            else:
                count += 1

                if count == 1000:
                    print('The program is having trouble fitting the specified tolerance.\nPlease try a smaller tolernace. Exiting with error code -1...')
                    return -1

                #np.random.seed(self.seed)
                split_idxs = np.random.uniform(0, N, client_num - 1)
                split_idxs = np.sort(split_idxs).astype('int32')

    # Generates the federated dataset
    def generate_data(self, client_num = 10, tolerance = 2000, test_size = 0.25):
        (x1, y1), (x2, y2) = self.dataset
        orig_shape = list(x1.shape)
        orig_shape[0] += x2.shape[0]
        orig_shape = tuple(orig_shape)
        x1 = x1.reshape((x1.shape[0], np.prod(x1.shape[1:])))
        x2 = x2.reshape((x2.shape[0], np.prod(x2.shape[1:])))
        X = np.append(x1, x2, axis = 0)
        y = np.append(y1, y2)

        N = X.shape[0]

        # Data preparation
        X = self.normalize(X)
        if self.dataset_name == 'cifar':
            X = X.reshape(orig_shape)
        else:
            X = X.reshape(orig_shape + tuple([1]))
        y = to_categorical(y.astype('int32'), 10)

        # Defining data split
        #np.random.seed(self.seed)
        split_idxs = np.random.uniform(0, N, client_num - 1)
        split_idxs = np.sort(split_idxs).astype('int32')
        split_idxs = self.validate_distribution(split_idxs, N, tolerance, client_num)

        assert type(split_idxs) != int

        client_X, client_y = self.split_among_clients(X, y, split_idxs)

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