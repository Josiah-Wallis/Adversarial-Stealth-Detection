All code presented in these files have been written by me, Josiah Wallis, except for the imported libraries.

Files contain the following contents:
- distribute_data.py: class used for accessing, generating, and splitting data amongst clients
- fedavg.py: class used for accessing Federated Averaging algorithm
- asd.py: My primary algorithm, adversarial sneak detection. Currently only contains that one algorithm
- analysis.ipynb: ipython notebook containing demonstrable results as well as a demonstration of how to run the code

HOW TO USE FILE CONTENTS SEPARATELY:
**** distribute_data.py
1) Initialize Datasets class object with the name of the dataset you'd like to input into the FedAvg algorithm. Choices: "fashion" for fashion_mnist or "digits" for mnist

2) All methods in this class, except generate_data, are only meant for generate_data to use. Although the user is welcome to use them, that is not their purpose, and are commented appropriately in the file

3) Call Datasets.generate_data(client_num, tolerance, test_size) to produce a federated dataset from the dataset specified in 1). 
-- client_num is the number of clients you'd like to distribute the data across
-- tolerance is the minimum number of samples you'd like each client to have
-- test_size is the fraction of samples each client will have in their test set with respect to their total number of samples
The method will return a dictionary of "Client Train Data", "Client Train Labels", "Client Test Data", and "Client Test Labels"

4) For future use, create_adversary will be used to create a single targetted adversary. It takes in client_train_labels, a client to turn into an adversary, an existing label, and the target label you'd like the existing label to be misclassified as. It will return a federated dataset the same shape as the one passed in. The specified client will have poisoned labels

** Practical Methods:
- Datasets(dataset_name = None) -> class object
- generate_data(client_num = 10, tolerance = 2000, test_Size = 0.25) -> {'Client Train Data': client_train_data, 'Client Train Labels':  client_train_labels, 'Client Test Data': client_test_data, 'Client Test Labels': client_test_labels}
- create_adversary(client_train_labels, client, true, target) -> adv_labels


**** fedavg.py
1) Initialize FederatedSystem class object with a dataset name, like Datasets, the number of clients you'd like in your federated system, the tolerance of the clients' datasets, and the test_size. These parameters are intermingled with the Datasets class object, as it is internally used within FederatedSystem to automatically create your federated dataset upon initialization

2) All the functions except for FedAvg, test_acc, and poison are not meant to be used by the user, so their comments detail their functionality in the source code

3) To poison your model with a single adversary, you must first as Datasets.create_adversary as described above. The labels returned by the method should be passed to an initialized FederatedSystem via its poison method. It sets its own training labels to the poisoned labels

4) You can now call FedAvg to run the federated averaging algorithm on the internally federated datasets. You can specify the number of epochs, fraction of clients selected per round frac_clients, the number of rounds it will aggregate, whether or not to enable my algorithm, and the threshold for my algorithm

5) After FedAvg runs the specified number of rounds, it returns the latest model parameters, biases, and the tally list my algorithm generates. More on that later

6) Run FederatedSystem.test_acc to see the federated model's test accuracy and test loss at every round, using the parameters it aggregated that round

** Practical Methods:
- FederatedSystem(dataset_name, client_num = 10, tolerance = 2000, test_size = 0.25) -> class object
- FedAvg(epochs = 5, frac_clients = 1, rounds = 20, enable = 0, threshold = 10) -> w, b, tally
- test_acc() -> test_accs, test_losses


**** asd.py
The only method in this module, asd_cancel, is my designed algorithm used to detect a single adversary in a federated setting. It is automatically implemented within the FedAvg algorithm. Given a round's parameters, biases, a tally vector, and a distance function, asd_cancel does the following:
1) Calculates pairwise distances between every pair of parameters
2) Notes the two furthest away parameters as being "questionable", or in other words, two "questionable" clients
3) Computes the distances between the two "questionable" parameters with every other parameter
4) Sums these computed distances for the two clients respectively
5) Takes the max of these two sums
6) Whichever client produced the parameters resulting in the max distance has 1 added to their tally in the tally vector, a vector containing "questionability" of every client computed within this method
7) It then returns the two most questionable clients that round
The tally vector, as long as enable == 1 in FedAvg, will be used from threshold rounds onward to exclude one client from the aggregation process. Whichever client has the highest tally will be canceled/excluded after threshold rounds, as it is most likely the adversary

The two function headers at the bottom of the file are meant for future extensions of this project, as I have some ideas on how to improve asd_cancel


**** analysis.ipynb
This file essentially contains a demonstration of how to use the algorithms provided in the manner I've mentioned above. It also provides an analysis of the test accuracies before and after my algorithm is enabled. The ipython notebook will walk you through everything  with details behind why I'm doing what I'm doing



**** Steps for practical use 
1) Instantiate a Datasets object with no initialization. This will be purely used to generate adversaries as FederatedSystem implicitly calls Datasets methods to generate the data for you
2) Initialize a FederatedSystem object with the name of the dataset you'd like to run the algorithm on, the number of clients desired, client dataset tolerance, and client test set size/fraction
3) Create an adversary by passing the FederatedSystem attribute client_train_labels to Datasets.create_adversary along with the parameters specified above
4) Pass the returned adversarial labels to your FederatedSystem.poison method
5) You may now run FederatedSystem.FedAvg 
6) After it finishes running, call FederatedSystem.test_acc to view the model's test accuracy and test loss at every round using the internally generated client_test_data and client_test_labels (aka sit back and let the algorithm run!)

FedAvg returns the latest parameters, biases, and tally vector. Through the tally vector, as well as the losses and accuracies after round threshold, you can see the effects of my algorithm. It will have automatically eliminated the adversary, improving accuracy post round threshold. The tally vector will contain a history of which clients it considered questionable. To apply my algorithm in your own federated setting, simply use it after the client update phase. Pass it the computed parameters, biases, and tally vector, then you may use the computed tallies to cancel out a supposed adversarial client.








