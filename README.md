# Orthogonalization techniques for a set of vectors

This repository contains three orhogonalization algorithms for sets of vectors using MPI.

The code was developed for the course "HPC for numerical methods and data analysis" at EPFL in the academic year 2023-2024, fall semester.

Check out the final report here: [`ferioli_project1.pdf`](./ferioli_project1.pdf).

## Implementations of the algorithms
* `code/seqCGS.py` and `code/parallelCGS.py` contain our implementations of the Classical Gram-Schmidt algorithm (sequentially and in parallel, respectively)
* `code/seqCholeskyQR.py` and `code/parallelCholeskyQR.py` contain our implementations of Cholesky QR (sequentially and in parallel, respectively)
* `code/seqTSQR.py` and `code/parallelTSQR.py` contain our implementations of the TSQR algorithm (sequentially and in parallel, respectively)

## How to run a test
To setup a test modify direclty the file of the algorithm you want to run. If `save` is set to `True` the results of the test are saved in the csv file `testing/results.csv`. 
To run a test, call
```
python3 seqAlgorithm.py
```
or
```
mpirun -n 4 python3 parallelAlgorithm.py
```
from `/code`, where 'seqAlgorithm.py' and 'parallelAlgorithm.py' are any sequential and parallel algorithm, respectively. The number of processors `4` can be changed to any other power of 2.

## Testing parameters
The following parameters have to be set at the beginning of the algorithm file, in order to run a test:
* `m` and `n` specify the matrix size (rows and columns, respectively)
* `matrix_type` specifies the testing matrix.
  - `0`: Type 1
  - `1`: Type 2
  - `2`: Type 3
* `cond` specifies the conditional number for Type 3 matrices. It is ignored otherwise.

More information about what this settings are can be found in the final report.
