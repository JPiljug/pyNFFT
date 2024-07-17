# Add the parent directory to the system path to have access to nfft3 until proper pip
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# proper test of nfft3
import nfft3
import numpy as np 

N = np.array([16, 8, 4]) 
M = 100
d = len(N)
Ns = np.prod(N)

X = 0.5 * np.random.rand(M,d) 
fhat = np.random.rand(Ns)

# test inits
plan = nfft3.NFCT(N,M)
plan2 = nfft3.NFCT(N,M,2*N,8,8145,65)
plan.X = X
plan.fhat = fhat

# test plans
plan.trafo() # value is in plan.f

# compare with directly computed
I = [[k, i, j] for  k in range(0,N[0]) for i in range(0,N[1]) for j in range(0,N[2])]

F = np.array([[ np.cos(2 * np.pi * np.dot(X[j,:][0], I[l][0])) *
                np.cos(2 * np.pi * np.dot(X[j,:][1], I[l][1])) *
                np.cos(2 * np.pi * np.dot(X[j,:][2], I[l][2])) for l in range (0,Ns) ] for j in range(0,M)])

f1 = F @ fhat

print(np.linalg.norm(f1-plan.f) / np.linalg.norm(f1))
print(np.linalg.norm(f1-plan.f,np.inf) / np.linalg.norm(fhat, 1))
