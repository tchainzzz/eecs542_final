import numpy as np

def rbf(x1, x2, gamma=0.5):
    return np.exp(-gamma * np.square(x1 - x2).sum())


def mmd_penalty(Z0, Z1, k_fn=rbf):
    n_Z0 = Z0.size(0)
    n_Z1 = Z1.size(0)
    Z0_ = 0.
    Z0Z1 = 0.
    Z1_ = 0.
    for i in range(n_Z0):
        for j in range(n_Z0):
            if i == j: continue
            Z0_ += k_fn(Z0[i], Z0[j]) / (n_Z0 * (n_Z0 - 1))
    for i in range(n_Z0):
        for j in range(n_Z1):
            Z0Z1 += -2 * k_fn(Z0[i], Z1[j]) / (n_Z0 * n_Z1)
    for i in range(n_Z1):
        for j in range(n_Z1):
            if i == j: continue
            Z1_ += k_fn(Z1[i], Z1[j]) / (n_Z1 * (n_Z1 - 1))
    return Z0_ + Z0Z1 + Z1_

