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
	
	Q = np.empty((m,n)) 
	R = np.zeros((n,n))
else:
	Q = None
	R = None

Q_local = np.empty((s_local, n))

for i in range(n):
	ri_local = Q_local[:,0:i].T @ W_local[:,i]
	ri = np.empty((i,1))
	comm.Allreduce(ri_local, ri, op = MPI.SUM)
	Q_local[:,i,None] = W_local[:,i,None] - Q_local[:,0:i] @ ri
	qi_norm_local2 = norm(Q_local[:,i])**2
	qi_norm = comm.allreduce(qi_norm_local2, op = MPI.SUM)**0.5
	Q_local[:,i] = Q_local[:,i]/qi_norm
	
	if rank == 0:
		R[0:i,i,None] = ri
		R[i,i] = qi_norm

comm.Gatherv(Q_local, Q, root = 0)

if rank == 0:
	wt = MPI.Wtime()-wt
	saveTestOnFile(save, "results.csv", s, "CGS", wt, W, matrixType, Q, R)
