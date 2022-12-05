# 1-Client Adversarial Stealth Detection (ASD)
In the field of **federated learning**, some clients aim to lessen the quality of the global aggregated model during training time. One known method of doing so is [**targeted model poisoning**][model poisoning] where an **adversarial client** attempts to learn an **adversarial objective** to misclassify one or more labels labels at prediction time. The **global model** is expected to **converge**, but the overall quality of the **predictions will be worse**. I propose a preliminary **adversarial client detection** scheme that takes advantage of **model parameter dissimilarities** at aggregation time. My paper draft can be found [here][my paper].

# Implementation
### Overview
1-Client ASD detects a single adversarial client by introducing a **tally system**. Every round, **pairwise distances** are taken between the model parameters of each clients' updates and the client that stands out the most will be tallied. After a certain threshold,  the client with the most tallies is **removed** from the client aggregation process in an attempt to stop the model poisoning. This approach assumes that the model produced by the adversary will be noticeably different from the other client updates. \
\
In addition, **principal component analysis** is performed on flattened model parameter matrices of each layer in the local model architectures to lessen the effects of the curse of dimensionality. This approach strives to more easily make the **adversarial updates stand out**. \
\
In my tests, I use the [Fashion MNIST][fashion] and regular [MNIST][mnist] datasets.

### Usage
To get started creating adversaries and capturing tallies, below outlines a typical setup:
```python
from distribute_data import Datasets
from fedsys import FederatedSystem

def main():
    dataHandler = Datasets('digits')                                        # Handler for MNIST
    data = dataHandler.generate_data()                                      # Splits data among clients + generates test data
    X_train = data['Client Train Data']         
    y_train = data['Client Train Labels']
    X_test = data['Client Test Data']
    y_test = data['Client Test Labels']
    
    '''
    Uses client 3 as adversary, 
    changing "1" labels to "7" labels as the target
    '''
    adv_labels = dataHandler.create_adversary(y_train, 3, 1, 7)
    
    fed = FederatedSystem(X_train, adv_labels)                              # Creates FL system with one adversary
    fed.SetTestData({'Test Data': X_test, 'Test Labels': y_test)            # Sets test data for evaluation
    
    w, b, tally = fed.ASD_FedAvg(enable = 1, threshold = 5, rounds = 10)    # Runs FedAvg with tally-counting until round 5
```

# Results
![asd.png](https://www.dropbox.com/s/cjdietuofhgwmfr/asd.png?dl=0&raw=1)
Shown above are validation accuracy comparisons with FedAvg ASD enabled on the Fashion MNIST and regular MNIST datasets. `*_c` represents clean labels while `*_adv` represents adversarial labels. Notice ASD doesn't have a significant impact on the performance of either of these models, with the bump in the `F_adv Acc` curve being due to randomness. 

# Conclusion
The impact of ASD should be further studied as the results may be due to a plethora of circumstances. The curse of dimensionality manifests itself still due to a loss of information dropping from thousand dimensional spaces down to hundreds or tens of dimensions. Also, ASD is sensitive to client updates from clients with a much greater sample size with respect to other clients. From a theoretical standpoint, there is promise.

# Contact
Please send any questions, comments, or inquiries to jwall014@ucr.edu. A paper rough draft with mathematical details an further discussion may be found [here][my paper]. Please note that the paper formatting says it is published although it is **not** a published or submitted paper. It is a draft written for practice and educational/academic purposes only, and any statements of publication on the paper are written in by the LaTeX formatting.

[model poisoning]: <https://arxiv.org/abs/1811.12470>

[fashion]: <https://www.kaggle.com/datasets/zalando-research/fashionmnist>

[mnist]: <https://www.tensorflow.org/datasets/catalog/mnist>

[my paper]: <https://drive.google.com/file/d/1ZpFtym77Qg__nYeT5xQzEQUmxZWFKR2n/view?usp=sharing>
