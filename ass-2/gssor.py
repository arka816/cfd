# Gauss-Seidel with SOR
# SOR: successive over relaxation

import numpy as np
import matplotlib.pyplot as plt

TOLERANCE = 1e-6
INT_MAX = float('inf')

def iterate(A, b, rf):
    m, n = A.shape
    x = np.ones(n)
    last_x = np.zeros(n)
    
    iters = 0
    
    while True:
        if np.max(np.abs(x - last_x)) < TOLERANCE:
            # successive improvement less than tolerance
            break
        
        iters += 1
        
        for i in range(n):
            last_x[i] = x[i]
            x[i] = (1 - rf) * x[i]
            s = b[i]
            s -= np.dot(A[i, :i], x[:i])
            s -= np.dot(A[i, i+1:], x[i+1:])
            s *= (rf / A[i, i])
            x[i] += s
            
    return x, iters            
    

def GSSOR(A, b, rf = None):
    # solves Ax = b
    # with relaxation factor rf
    
    rfs = [1.2, 1.4, 1.6, 1.7, 1.75, 1.80, 1.85, 1.90, 1.95, 1.97, 1.99]
    iter_dict = {}
    
    if rf is not None:
        x, iters = iterate(A, b, rf)
        return x, rf, iters
    
    for rf in rfs:
        x, iters = iterate(A, b, rf)
        iter_dict[rf] = iters

    best_rf = min(iter_dict, key=iter_dict.get)
    
    plt.figure()
    plt.xlabel('weights')
    plt.ylabel('iters')
    plt.plot(rfs, [iter_dict[rf] for rf in rfs], marker='o')
    plt.show()
            
    return x, best_rf, iter_dict[best_rf]
    