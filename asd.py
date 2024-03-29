import numpy as np
from sklearn.decomposition import PCA

def conv_pca(w, b):
    # Performs PCA on stack of layer parameters, aka PCA on matrix of flattened layer parameters for each layer.
    
    layers = len(w[0])
    transformed_w = [[] for _ in range(layers)]
    transformed_b = [[] for _ in range(layers)]
    k = len(w)
    w_pca = PCA(n_components = min(w[0][0].shape))
    b_pca = PCA(n_components = min(b[0][0].shape))
    
    # Flattens layer parameters
    for i in range(k):
        for j in range(layers):
            transformed_w[j].append(w[i][j].flatten())
            transformed_b[j].append(b[i][j].flatten())
    
    # Transforms parameters down to their lowest allowable dimension
    for layer in range(layers):
        transformed_w[layer] = w_pca.fit_transform(np.stack([i for i in transformed_w[layer]]))
        transformed_b[layer] = b_pca.fit_transform(np.stack([i for i in transformed_b[layer]]))
        
    # Reformats w and b lists for asd_cancel
    reformed_w = [[] for _ in range(k)]
    reformed_b = [[] for _ in range(k)]
    for i in range(k):
        for layer in range(layers):
            reformed_w[i].append(transformed_w[layer][i])
            reformed_b[i].append(transformed_b[layer][i])
            
    return reformed_w, reformed_b

# Performs the tallying process
def asd_cancel(w_ref, b_ref, tally, dist = lambda x: np.linalg.norm(x)):
    k = len(w_ref)
    num_layers = len(w_ref[0])
    dists = np.zeros((k, k))
    w, b = conv_pca(w_ref, b_ref)

    # Calculates Pairwise Distances
    for i in range(k):
        for j in range(i + 1, k):
            distance = 0
            for weight in range(num_layers):
                w1_temp, w2_temp = w[i][weight], w[j][weight]
                b1_temp, b2_temp = b[i][weight], b[j][weight]
                w1_flat = np.append(w1_temp, b1_temp)
                w2_flat = np.append(w2_temp, b2_temp)
                
                distance += dist(w1_flat - w2_flat)
            dists[i, j] = distance
    
    qClients = np.where(dists == dists.max())
    qClient_1 = qClients[0][0]
    qClient_2 = qClients[1][0]

    client1_dists = np.zeros(k)
    client2_dists = np.zeros(k)
    w_c1, w_c2 = w[qClient_1], w[qClient_2]
    b_c1, b_c2 = b[qClient_1], b[qClient_2]
    
    # Calculates distances from the two questionable clients to all other clients
    for i in range(k):
        c1_distance = 0
        c2_distance = 0
        for weight in range(num_layers):
            w_flat = np.append(w[i][weight].flatten(), b[i][weight].flatten())

            if i != qClient_1:
                w_c1_flat = np.append(w_c1[weight].flatten(), b_c1[weight].flatten())
                c1_distance += dist(w_flat - w_c1_flat)
            if i != qClient_2:
                w_c2_flat = np.append(w_c2[weight].flatten(), b_c2[weight].flatten())
                c2_distance += dist(w_flat - w_c2_flat)

        client1_dists[i] = c1_distance
        client2_dists[i] = c2_distance

    # Checks the total of the distances
    c1_dist_total = client1_dists.sum()
    c2_dist_total = client2_dists.sum()

    # Whichever client has greater total is the most questionable client
    qClient = np.array([c1_dist_total, c2_dist_total]).argsort()[1]

    questionable_clients = []
    if qClient == 0:
        qClient = qClient_1
        questionable_clients.append(qClient)
        questionable_clients.append(qClient_2)
    else:
        qClient = qClient_2
        questionable_clients.append(qClient)
        questionable_clients.append(qClient_1)

    tally[qClient] += 1

    return questionable_clients

             
# For the future
def asd_weights():
    pass

# For the future
def asd_dynamic_weights():
    pass