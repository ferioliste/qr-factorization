from os import environ
environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
import numpy as np
from numpy.linalg import norm
import math
from commonFunctions import *

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
s = comm.Get_size()

#==================== Matrix generation ====================

(m, n) = (None, None)
W = None

if rank == 0:
	print("Generating W")
	
	# Write your settings here
	m, n = 10000, 300
	matrixType = 0
	cond = 1e4
	save = False
	
	W = genMatrix(m, n, matrixType, cond)
	
	wt = MPI.Wtime()

#=================== Matrix distribution ===================

(m,n) = comm.bcast((m, n), root = 0)
s_local = m//s
W_local = np.empty((s_local, n))
comm.Scatterv(W, W_local, root=0)

#==================== QR factorization =====================

if rank == 0:
	print("Starting factorization")

A_local = W_local.T @ W_local
A = np.empty((n,n))
comm.Reduce(A_local, A, op = MPI.SUM, root = 0)

if rank == 0:
	try:
		R = np.linalg.cholesky(A).T
	except:
		print("The matrix is numerically singular")
		eps = 1e-12
		while True:
			try:
				R = np.linalg.cholesky(A + eps*np.identity(n)).T
				break
			except:
				eps = eps*5
else:
	R = None

R = comm.bcast(R, root = 0)
for i in range(n):
	W_local[:,i] = (W_local[:,i] - W_local[:,0:i] @ R[0:i,i])/R[i,i]

Q = np.empty((m,n))
comm.Gatherv(W_local, Q, root = 0)

if rank == 0:
	wt = MPI.Wtime()-wt
	saveTestOnFile(save, "results.csv", s, "cholesky", wt, W, matrixType, Q, R)