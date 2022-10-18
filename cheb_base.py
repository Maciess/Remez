import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
n=2
base = [Chebyshev.basis(i) for i in range(n+1)]

def make_matrix(array):
	M = np.array([0,0,0,0])
	k=1
	for arg in array:
		row = np.concatenate( (np.array([T(arg) for T in base]), np.array([k]) ) )
		k*=-1
		M = np.vstack((M,row))
	return M[1:,:]


t= Chebyshev([1,2,1])
print(t(-1))
