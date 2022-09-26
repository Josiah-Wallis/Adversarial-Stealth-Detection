# Splitting up the data among the clients
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.image import per_image_standardization
from copy import deepcopy

class Datasets:
    def __init__(self, dataset_name: str, seed: int = 11) -> None:
        """
        Data handler for fashion mnist and mnist test scenarios.
        Distributes data among clients and produces test set.
        """
        
        self.dataset_name = dataset_name
        if self.dataset_name == 'fashion':
            self.dataset = fashion_mnist.load_data()
        elif self.dataset_name == 'digits':
            self.dataset = mnist.load_data()
            
        self.seed = seed

    def create_adversary(self, client_train_labels: np.array, client: int, true: int , target: int) -> np.array:
        """
        Creates a single adversarial client (targeted).
        Selected client will change provided true label to target label.
        """
        
        adv_labels = deepcopy(client_train_labels)
        num_labels = len(adv_labels[0][0])
        target_label = to_categorical(target, num_labels)

        rev_one_hot = np.argmax(adv_labels[client], axis = 1)
        adv_labels[client][rev_one_hot == true] = target_label
        return adv_labels

    def check_tolerance(self, idxs, N, tolerance):
        #Checks if index list idxs satisfies the tolerance.

        start = 0
        for idx in idxs:
            if (idx - start) <= tolerance:
                return False
            start = idx

        if (N - start) < tolerance:
            return False
        else:
            return True

    def validate_distribution(self, split_idxs, N, tolerance, client_num):
        # Wrapper for check_tolerance.

        count = 0

        while True:
            if self.check_tolerance(split_idxs, N, tolerance):
                return split_idxs
            else:
                count += 1

                if count == 10000:
                    print('The program is having trouble fitting the specified tolerance.\nPlease try a smaller tolerance. Exiting with error code -1...')
                    return -1
                
                split_idxs = np.random.uniform(0, N, client_num - 1)
                split_idxs = np.sort(split_idxs).astype('int32')

    def split_among_clients(self, X, y, split_idxs):
        #Distributes X and y amongst clients using the indices from split_idxs.

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

    def generate_data(self, normalize: bool = True, client_num: int = 10, tolerance: int = 1000, test_size: float = 0.25) -> dict:
        """
        Splits dataset X and label vector y into client_num clients with at least tolerance samples. 
        Also creates test set.
        """
        
        tf.keras.utils.set_random_seed(self.seed)
        
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
        X = X.reshape(orig_shape + tuple([1]))
        y = to_categorical(y.astype('int32'), 10)
        
        # Shuffle
        idx = np.arange(N)
        idx = np.random.permutation(idx)
        X = X[idx]
        y = y[idx]
        
        # Separate training/test data
        split = int(N * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        M = X_train.shape[0]

        # Defining data split
        split_idxs = np.random.uniform(0, M, client_num - 1)
        split_idxs = np.sort(split_idxs).astype('int32')
        split_idxs = self.validate_distribution(split_idxs, M, tolerance, client_num)

        assert type(split_idxs) != int

        clients_X, clients_y = self.split_among_clients(X_train, y_train, split_idxs)
        
        if normalize:
            X_test = per_image_standardization(X_test)
            for i, _ in enumerate(clients_X):
                clients_X[i] = per_image_standardization(clients_X[i])

        

        return {'Client Train Data': clients_X, 'Client Train Labels':  clients_y, 'Client Test Data': X_test, 'Client Test Labels': y_test}