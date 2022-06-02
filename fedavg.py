# Federation and Main Algorithms
import tensorflow as tf
import numpy as np

from random import sample
from tensorflow import keras
from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from distribute_data import Datasets

from asd import *

# Class for all federated processes
class FederatedSystem:
    def __init__(self, dataset_name, client_num = 10, tolerance = 2000, test_size = 0.25):
        self.dataset_name = dataset_name
        if self.dataset_name == 'fashion' or self.dataset_name == 'digits':
            if self.dataset_name == 'fashion':
                self.metadata = Datasets('fashion')
            else:
                self.metadata = Datasets('digits')
        self.data = self.metadata.generate_data(client_num, tolerance, test_size)
        self.model = Sequential([
            Conv2D(8, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)),
            MaxPool2D((2, 2), strides = 2),
            Conv2D(16, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
            MaxPool2D(pool_size = (2, 2), strides = 2),
            Flatten(),
            Dense(10, activation = 'softmax')
        ])
        
        self.client_train_data = self.data['Client Train Data']
        self.client_train_labels = self.data['Client Train Labels']
        self.client_test_data = self.data['Client Test Data']
        self.client_test_labels = self.data['Client Test Labels']
        self.trainable_layers = None
        self.final_w = None
        self.final_b = None
        self.w_history = None
        self.b_history = None
        self.final_tally = None

    def poison(self, adv_labels):
        self.client_train_labels = adv_labels

    # Initializes weights for FedAvg according to CNN parameters
    def initialize_weights(self):
        w = []
        b = []

        for l in self.trainable_layers:
            w_shape = self.model.layers[l].weights[0].shape
            b_shape = self.model.layers[l].weights[1].shape
            w.append(np.random.standard_normal(w_shape))
            b.append(np.random.standard_normal(b_shape))

        return w, b

    # Generates CNN with 2 convolutional layers and a dense layer. Initializes weights with passed w, b
    def generate_model(self, w, b, skip = 0):
        model = clone_model(self.model)

        if not skip:
            for i, x in enumerate(self.trainable_layers):
                model.layers[x].set_weights([w[i], b[i]])

        return model

    # Trains CNN locally and returns weight updates
    def ClientUpdate(self, X, y, w, b, E):
        model = self.generate_model(w, b)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.fit(X, y, validation_split = 0.2, epochs = E, verbose = 0, use_multiprocessing = True)

        w = []
        b = []
        for l in self.trainable_layers:
            w.append(model.layers[l].get_weights()[0])
            b.append(model.layers[l].get_weights()[1])

        return w, b

    # Aggregates weights from all client models
    def aggregate(self, w_updates, b_updates, n_k, S_t):
        n = np.sum(n_k[S_t])
        num_trainable_layers = len(w_updates[S_t[0]])
        w = [0 for _ in range(num_trainable_layers)]
        b = [0 for _ in range(num_trainable_layers)]

        for k in S_t:
            w_k = w_updates[k]
            b_k = b_updates[k]
            for l in range(num_trainable_layers):
                w[l] += (n_k[k] / n) * w_k[l]
                b[l] += (n_k[k] / n) * b_k[l]

        return w, b

    # Federated Averaging with 1-Client  ASD
    def FedAvg(self, epochs = 5, frac_clients = 1, rounds = 20, enable = 0, threshold = 10):
        # Find which layers have trainable parameters
        trainable_layers = []
        for i, x in enumerate(self.model.layers):
            if x.weights:
                trainable_layers.append(i)

        self.trainable_layers = trainable_layers

        # Initialize from a standard normal distribution
        w, b = self.initialize_weights()
        K = len(self.client_train_data)
        m = max(int(frac_clients * K), 1)
        client_set = range(K)

        ws = []
        ws.append(w)
        bs = []
        bs.append(b)

        # Record number of samples per client
        n_k = []
        for x in self.client_train_data:
            n_k.append(x.shape[0])
        n_k = np.array(n_k)


        # Start federating process
        tally = [0] * K
        for t in range(rounds):
            print(f'Round {t + 1}')
            w_updates = [None for _ in range(m)]
            b_updates = [None for _ in range(m)]
            S_t = sample(client_set, m)

            # Client updates while trying to avoid the one less client after threshold round
            for k in S_t:
                w_temp, b_temp = self.ClientUpdate(self.client_train_data[k], self.client_train_labels[k], w, b, epochs)
                if ((k == 9) and (t > threshold - 1)) and (enable == 1):
                    w_updates.append(w_temp)
                    b_updates.append(b_temp)
                else:
                    w_updates[k], b_updates[k] = w_temp, b_temp

            # Check threshold, then exclude the most questionable client after threshold round
            if t <= (threshold - 1):
                qClients = asd_cancel(w_updates, b_updates, tally)
            if (enable == 1) and (t == (threshold - 1)):
                bad_client = tally.index(max(tally))
                client_set = [i for i in client_set if i != bad_client]
                m -= 1
            w, b = self.aggregate(w_updates, b_updates, n_k, S_t)
            ws.append(w)
            bs.append(b)

        self.final_w = w
        self.final_b = b
        self.w_history = ws
        self.b_history = bs
        self.final_tally = tally
            
        return w, b, tally

    def test_acc(self):
        T = len(self.w_history)
        test_accs = []
        test_losses = []

        test_data = np.concatenate([x for x in self.client_test_data])
        test_labels = np.concatenate([y for y in self.client_test_labels])

        for t in range(T):
            model = self.generate_model(self.w_history[t], self.b_history[t])
            model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
            loss, acc = model.evaluate(test_data, test_labels, verbose = 0, use_multiprocessing = True)
            test_accs.append(acc)
            test_losses.append(loss)

        return test_accs, test_losses