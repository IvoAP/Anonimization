import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans


def anonimization(data):
    #calculate the mean of each column
    mean = np.array(np.mean(data, axis=0).T)

    # center data
    data_centered = data - mean

    # calculate the covariance matrix
    cov_matrix = np.cov(data_centered, rowvar=False)
   
    # calculate the eignvalues and eignvectors
    evals, evecs = la.eigh(cov_matrix)

    # sort them
    idx = np.argsort(evals)[::-1]

    # Each columns of this matrix is an eingvector
    evecs = evecs[:,idx]
    evals = evals[idx]

    # explained variance
    variance_retained=np.cumsum(evals)/np.sum(evals)

    # calculate the transformed data
    data_transformed=np.dot(evecs.T, data_centered.T).T

    # randomize eignvectors
    new_evecs = evecs.copy().T
    for i in range(len(new_evecs)):
        np.random.shuffle(new_evecs[i])
    new_evecs = np.array(new_evecs).T

    # go back to the original dimension
    data_original_dimension = np.dot(data_transformed, new_evecs.T) 
    data_original_dimension += mean

    return data_original_dimension

def find_clusters(X, k):   
    Kmean = KMeans(n_clusters=k)
    Kmean.fit(X)
    return Kmean.labels_

def anonimization_clustering(data, y, k):
    # generate K data clusters
    clusters = find_clusters(data, k)

    # bucketize the index of each cluster
    indices = dict()
    for i in range(len(clusters)):
        if clusters[i] not in indices.keys():
            indices[ clusters[i] ] = []    
        indices[ clusters[i] ].append(i)

    data_anonymized, y_in_new_order = None, None

    # anonymize each cluster individually
    for k in indices.keys():
        if data_anonymized is None and y_in_new_order is None:
            data_anonymized = anonimization(data[ indices[k] ])
            y_in_new_order = y[ indices[k] ]
            empty_flag = False
        else:
            data_anonymized = np.concatenate((data_anonymized, anonimization(data[ indices[k] ]) ), axis=0)
            y_in_new_order = np.concatenate((y_in_new_order, y[ indices[k] ]), axis=0)

    # data_anonymized = np.concatenate((data_anonymized, np.array([y_in_new_order]).T), axis=1)
    # print(pd.DataFrame(data_anonymized))

    return data_anonymized, y_in_new_order
