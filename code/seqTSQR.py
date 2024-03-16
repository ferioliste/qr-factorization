from os import environ
environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from numpy.linalg import norm
import math
import time
from commonFunctions import *

s = 4

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
R = Q[0:0,:]

Q_list = []
m_step = m//s
toFactor = W[0:m_step,:]

for step in range(s):
	toFactor = np.vstack((R, W[(step*m_step):((step+1)*m_step),:]))
	(Q_step, R) = np.linalg.qr(toFactor, mode='reduced')
	Q_list.append(Q_step)

for step in reversed(range(1,s)):
	Q_list[step-1] = Q_list[step-1] @ Q_list[step][0:n,:]
	Q[(step*m_step):((step+1)*m_step),:] = Q_list[step][n:,:]
Q[0:m_step] = Q_list[0]

wt = time.time()-wt
saveTestOnFile(save, "results.csv", 0, "TSQR", wt, W, matrixType, Q, R)