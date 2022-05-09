# %%Federation and Main Algorithms
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from distribute_data import generate_mnist_client_data

# %%
pkg = generate_mnist_client_data()

# %%
client_train_data = pkg['Client Train Data']
client_train_labels = pkg['Client Train Labels']
client_test_data = pkg['Client Test Data']
client_test_labels = pkg['Client Test Labels']

# %%
model = Sequential()
# was 32 filters
model.add(Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))
model.add(MaxPool2D(pool_size = (2, 2), strides = 2))
# was 64 filters
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPool2D(pool_size = (2, 2), strides = 2))
model.add(Flatten())
model.add(Dense(units = 10, activation = 'softmax'))

# %%
model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# %%
model.fit(x = client_train_data[0], y = client_train_labels[0], validation_split = 0.2, epochs = 20, verbose = 2, use_multiprocessing = True)

# %%
model.summary()

# %%
l = model.layers[0]
x = l.weights[0]
x + x

# %%
model.layers[0].weights[1]


# %%
for layer in model.layers:
    if len(layer.weights):
        print(layer.weights[0].shape)
        print(layer.weights[1].shape, end = '\n\n')

# %%
def initialize_w():
    w = {}
    w['l1 w'] = tf.random.normal([3, 3, 1, 8], 0, 1, tf.float32)
    w['l1 b'] = tf.random.normal([8], 0, 1, tf.float32)
    w['l2 w'] = tf.random.normal([3, 3, 8, 16], 0, 1, tf.float32)
    w['l2 b'] = tf.random.normal([16], 0 , 1, tf.float32)
    w['l3 w'] = tf.random.normal([784, 10], 0, 1, tf.float32)
    w['l3 b'] = tf.random.normal([10], 0, 1, tf.float32)

    return w

def FedAvg(data, batch_size, epochs, learning_rate, frac_clients, rounds):
    w = initialize_w()
    K = len(data['Client Train Data'])

    for t in range(rounds):
        pass

# %%
