from os import environ
environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from numpy.linalg import norm
import math
import time
from commonFunctions import *

print("Generating W")

# Write your settings here
m, n = 10000, 300
matrixType = 0
cond = 1e4
save = False

W = genMatrix(m, n, matrixType)
wt = time.time()

print("Starting factorization")

Q = np.empty((m,n))
R = np.empty((n,n))

A = W.T @ W
R = np.linalg.cholesky(A).T

for i in range(n):
	Q[:,i] = (W[:,i] - np.dot(Q[:,0:i],R[0:i,i]))/R[i,i]

wt = time.time()-wt
saveTestOnFile(save, "results.csv", 0, "cholesky", wt, W, matrixType, Q, R)