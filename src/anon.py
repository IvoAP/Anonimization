import numpy as np
import scipy.linalg as la


def anonymization(data):
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    cov_matrix = np.cov(data_centered, rowvar=False)
    evals, evecs = la.eigh(cov_matrix)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    data_transformed = np.dot(evecs.T, data_centered.T).T
    new_evecs = evecs.copy().T
    for i in range(len(new_evecs)):
        np.random.shuffle(new_evecs[i])
    new_evecs = np.array(new_evecs).T
    data_anonymized = np.dot(data_transformed, new_evecs.T) + mean
    return data_anonymized

