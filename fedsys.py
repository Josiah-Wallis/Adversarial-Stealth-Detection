import os
import tensorflow as tf
import numpy as np

from random import sample
from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from asd import *
os.environ['PYTHONHASHSEED'] = str(50)

# FL Class
class FederatedSystem:
    def __init__(self, clients_X: list[np.array], clients_y: list[np.array], seed: int = 50) -> None:
        """
        Unpack client data, labels, and distances.
        Initialize containers for storing run history.
        """

        # Unpack data used for training and timing
        self.seed = seed
        self.clients_X = clients_X
        self.clients_y = clients_y

        # Define default local model
        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()
        self.model = Sequential([
            Conv2D(2, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)),
            MaxPool2D((2, 2), strides = 2),
            Conv2D(4, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
            MaxPool2D(pool_size = (2, 2), strides = 2),
            Flatten(),
            Dense(10, activation = 'softmax')
        ])

        # Initialize FL containers
        self.trainable_layers = []
        self.w_history = []
        self.b_history = []

        # Record trainable layers
        for i, x in enumerate(self.model.layers):
            if x.weights:
                self.trainable_layers.append(i)

        # Performance
        self.test_data = None

    def clear_history(self) -> None:
        """
        Refreshes FL system to be trained again.
        """

        self.w_history = []
        self.b_history = []

    def DefaultModel(self) -> None:
        """
        If local models have been altered, reverts back to base models. 
        """

        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()
        self.model = Sequential([
            Conv2D(8, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)),
            MaxPool2D((2, 2), strides = 2),
            Conv2D(16, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
            MaxPool2D(pool_size = (2, 2), strides = 2),
            Flatten(),
            Dense(10, activation = 'softmax')
        ])

        # Record trainable layers
        for i, x in enumerate(self.model.layers):
            if x.weights:
                self.trainable_layers.append(i)

    # Setters + Getters
    def SetModel(self, model: tf.keras.Model) -> None:
        """
        Sets FL local models to model.
        """

        self.model = model

        # Record trainable layers
        for i, x in enumerate(self.model.layers):
            if x.weights:
                self.trainable_layers.append(i)

    def SetTestData(self, test_data: np.array) -> None:
        """
        FL system evaluates on test_data at test time.
        Should be set prior to using self.test_loss.
        """

        self.test_data = test_data
    
    def SetSeed(self, seed: int) -> None:
        """
        Sets seed for numpy, random, and Python.
        """

        self.seed = seed

    # FedAvg
    def initialize_weights(self):
    # Returns initial model parameters of self.model
        w = []
        b = []

        for l in self.trainable_layers:
            w.append(self.model.layers[l].get_weights()[0])
            b.append(self.model.layers[l].get_weights()[1])

        return w, b

    def generate_model(self, w: list[np.array], b: list[np.array], skip: int = 0) -> tf.keras.Model:
        """
        Given model parameters and bias of same shape as self.model,
        initializes copy of self.model with said parameters.
        Must be tweaked for custom or more complex NN layers.
        """
        
        model = clone_model(self.model)
        if not skip:
            for i, x in enumerate(self.trainable_layers):
                model.layers[x].set_weights([w[i], b[i]])

        return model

    def ClientUpdate(self, X, y, lr, w, b, E):
        # Trains a local model
        model = self.generate_model(w, b)
        model.compile(optimizer = Adam(learning_rate = lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = model.fit(X, y, validation_split = 0.2, epochs = E, verbose = 0, shuffle = False, use_multiprocessing = True)

        w = []
        b = []
        for l in self.trainable_layers:
            w.append(model.layers[l].get_weights()[0])
            b.append(model.layers[l].get_weights()[1])

        return w, b, model, history

    def aggregate(self, w_updates, b_updates, n_k, S_t):
        # FedAvg model aggregation step
        n = np.sum(n_k[S_t])
        num_trainable_layers = len(self.trainable_layers)
        w = [0 for _ in range(num_trainable_layers)]
        b = [0 for _ in range(num_trainable_layers)]

        for k in S_t:
            w_k = w_updates[k]
            b_k = b_updates[k]
            for l in range(num_trainable_layers):
                w[l] += (n_k[k] / n) * w_k[l]
                b[l] += (n_k[k] / n) * b_k[l]

        return w, b

    def initialize(self, system, frac_clients = None):
        # Initializes FedAvg model with appropriate parameters

        # Initialize w, b from standard normal
        w, b = self.initialize_weights()
        self.w_history.append(w)
        self.b_history.append(b)

        # Number of Clients
        K = len(self.clients_X)

        # Record number of samples per client
        n_k = []
        for x in self.clients_X:
            n_k.append(x.shape[0])
        n_k = np.array(n_k)

        if system == 'fedavg':
            m = max(int(frac_clients * K), 1)
            client_set = range(K)

            return {'K': K, 'm': m, 'client set': client_set, 'n_k': n_k, 'w': w, 'b': b}

    def FedAvg(self, lr: float = 0.001, epochs: int = 5, frac_clients: float = 1, rounds: int = 20) -> tuple[list[np.array], list[np.array]]:
        """
        Performs Federated Averaging on the given client data.
        Local client models are aggregated rounds times.
        """

        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        initializer = self.initialize('fedavg', frac_clients = frac_clients)

        K = initializer['K']
        m = initializer['m']
        client_set = initializer['client set']
        n_k = initializer['n_k']
        w = initializer['w']
        b = initializer['b']

        # Start federating process
        for t in range(rounds):
            print(f'Round {t + 1}')
            w_updates = [None for _ in range(K)]
            b_updates = [None for _ in range(K)]
            S = sample(client_set, m)

            # Client updates
            for k in S:
                w_updates[k], b_updates[k], _, _ = self.ClientUpdate(self.clients_X[k], self.clients_y[k], lr, w, b, epochs)

            w, b = self.aggregate(w_updates, b_updates, n_k, S)
            self.w_history.append(w)
            self.b_history.append(b)

        return w, b

    def ASD_FedAvg(self, enable = 0, threshold = 10, lr: float = 0.001, epochs: int = 5, frac_clients: float = 1, rounds: int = 20) -> tuple[list[np.array], list[np.array]]:
        """
        Performs Federated Averaging on the given client data.
        Local client models are aggregated rounds times.
        Performs ASD cancellaion if enable == 1, and tallies up to round threshold.
        """

        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        initializer = self.initialize('fedavg', frac_clients = frac_clients)

        K = initializer['K']
        m = initializer['m']
        client_set = initializer['client set']
        n_k = initializer['n_k']
        w = initializer['w']
        b = initializer['b']

        # Start federating process
        tally = [0] * K
        for t in range(rounds):
            print(f'Round {t + 1}')
            w_updates = [None for _ in range(K)]
            b_updates = [None for _ in range(K)]
            S = sample(client_set, m)

            # Client updates
            for k in S:
                w_temp, b_temp, _, _ = self.ClientUpdate(self.clients_X[k], self.clients_y[k], lr, w, b, epochs)
                if ((k == 9) and (t > threshold - 1)) and (enable == 1):
                    w_updates.append(w_temp)
                    b_updates.append(b_temp)
                else:
                    w_updates[k], b_updates[k] = w_temp, b_temp
            
            # Tally clients before threshold round
            if t <= (threshold - 1):
                qClients = asd_cancel(w_updates, b_updates, tally)
                
            # At threshold round, remove client with most tallies
            if (enable == 1) and (t == (threshold - 1)):
                bad_client = tally.index(max(tally))
                client_set = [i for i in client_set if i != bad_client]
                m -= 1

            w, b = self.aggregate(w_updates, b_updates, n_k, S)
            self.w_history.append(w)
            self.b_history.append(b)

        return w, b, tally

    # Performance
    def test_loss(self) -> list[float]:
        """
        For each global model generated during aggregation, evaluates on self.test_data.
        """

        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        T = len(self.w_history)
        mse_losses = []

        test_data = self.test_data['Test Data']
        test_labels = self.test_data['Test Labels']

        for t in range(T):
            model = self.generate_model(self.w_history[t], self.b_history[t])
            model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
            mse_loss = model.evaluate(test_data, test_labels, verbose = 0, use_multiprocessing = True)
            mse_losses.append(mse_loss)

        return mse_losses

    