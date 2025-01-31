# Add the parent directory to the system path to have access to nfft3 until proper pip
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# proper test of nfft3
import nfft3
import numpy as np 


# import numpy as np
N = np.array([16,])
M = 100
d = len(N)
Ns = np.prod(N)

X = np.random.rand(M,) - 0.5
fhat = np.random.rand(Ns) +  1.0j * np.random.rand(Ns)
f = np.random.rand(M) +  1.0j * np.random.rand(M)

# test init and setting
plan = nfft3.NFFT(N,M)
plan.X = X
plan.f = f   # this gets overwritten
plan.fhat = fhat

# test traffo
plan.trafo() # value is in plan.f

# compare with directly computed
I = [k for  k in range(int(-N[0]/2),int(N[0]/2)) ]

F = np.array([[np.exp(-2 * np.pi * 1j * (X[j] * I[l])) for l in range (0,Ns) ] for j in range(0,M)])

f1 = F @ fhat

print(np.linalg.norm(f1-plan.f) / np.linalg.norm(f1))
print(np.linalg.norm(f1-plan.f,np.inf) / np.linalg.norm(fhat, 1))

