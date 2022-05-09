import unittest
from distribute_data import *

# Unit Tests for distribute_data functions
class TestNormalize(unittest.TestCase):
    def test_mean_and_std(self):
        x = np.random.uniform(0, 100, 50).reshape((10, 5))
        
        mu = x.mean(axis = 0)
        sigma = x.std(axis = 0)
        x_normalized = (x - mu) / sigma
        normalize_test = normalize(x)

        check = np.all((x_normalized == normalize_test))

        self.assertTrue(check)

    def test_mean_and_std_seeded(self):
        np.random.seed(5)
        x = np.random.uniform(0, 1000, 500).reshape((100, 5))
        
        mu = x.mean(axis = 0)
        sigma = x.std(axis = 0)
        x_normalized = (x - mu) / sigma
        normalize_test = normalize(x)

        check = np.all((x_normalized == normalize_test))

        self.assertTrue(check) 

class TestCheckTolerance(unittest.TestCase):
    def test_true_tolerance(self):
        np.random.seed(101)
        x = np.sort(np.random.uniform(0, 1000, 10)).astype('int64')
        size = 1000
        tolerance = 15

        self.assertTrue(check_tolerance(x, size, tolerance))

    def test_false_tolerance(self):
        np.random.seed(101)
        x = np.sort(np.random.uniform(0, 1000, 10)).astype('int64')
        size = 1000
        tolerance = 20

        self.assertFalse(check_tolerance(x, size, tolerance))

class TestValidateDist(unittest.TestCase):
    def test_new_tolerance(self):
        np.random.seed(101)
        x = np.sort(np.random.uniform(0, 1000, 10)).astype('int64')
        size = 1000
        tolerance = 20

        print('(Choose 1 -> 15)')
        idxs = validate_distribution(x, size, tolerance, 10)
        print()

        self.assertTrue(type(idxs) != int)

    def test_diff_dist(self):
        np.random.seed(101)
        x = np.sort(np.random.uniform(0, 1000, 10)).astype('int64')
        size = 1000
        tolerance = 20

        print('(Choose 2 until you pass)')
        idxs = validate_distribution(x, size, tolerance, 10)
        print()

        self.assertTrue(type(idxs) != int)

    def test_quit(self):
        np.random.seed(101)
        x = np.sort(np.random.uniform(0, 1000, 10)).astype('int64')
        size = 1000
        tolerance = 20

        print('(Choose 3)')
        idxs = validate_distribution(x, size, tolerance, 10)
        print()

        self.assertTrue(idxs == -1)

class TestSplitAmongClients(unittest.TestCase):
    def test_split_datasets(self):
        X = np.zeros((1000, 3))
        y = np.zeros(1000)
        idxs = np.sort(np.random.uniform(0, 1000, 9)).astype('int64')

        clients_X, clients_y = split_among_clients(X, y, idxs)

        self.assertTrue(len(clients_X) == 10)
        
class TestGenerateData(unittest.TestCase):
    def test_default_settings(self):
        pkg = generate_mnist_client_data()

        client_train_data = pkg['Client Train Data']
        client_train_labels = pkg['Client Train Labels']
        client_test_data = pkg['Client Test Data']
        client_test_labels = pkg['Client Test Labels']

        self.assertTrue(len(client_train_data) == 10)
        self.assertTrue(len(client_train_labels) == 10)
        self.assertTrue(len(client_test_data) == 10)
        self.assertTrue(len(client_test_labels) == 10)

    def test_diff_settings(self):
        pkg = generate_mnist_client_data(15, tolerance = 1, test_size = 0.2)

        client_train_data = pkg['Client Train Data']
        client_train_labels = pkg['Client Train Labels']
        client_test_data = pkg['Client Test Data']
        client_test_labels = pkg['Client Test Labels']

        self.assertTrue(len(client_train_data) == 15)
        self.assertTrue(len(client_train_labels) == 15)
        self.assertTrue(len(client_test_data) == 15)
        self.assertTrue(len(client_test_labels) == 15)

if __name__ == '__main__':
    unittest.main()