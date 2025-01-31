# pyNFFT
A ctypes based Python Interface for [NFFT C library](https://github.com/NFFT/nfft). The implementation is modelled after the already existing Interface to the nfft in Julia with [NFFT3.jl](https://github.com/NFFT/NFFT3.jl) to allow a smooth transition for existing users.

`pyNFFT` provides the following fast algorithms:
- nonequispaced fast Fourier transform (NFFT) 
- nonequispaced fast cosine transform (NFCT) 
- nonequispaced fast sine transform (NFST)
- fast summation (fastsum)                  // in progress


## Getting started

In Python you can get started by typing

```
import nfft3
import numpy as np

# set up parameters
N = np.array([16,])
M = 10000

# set up data
X = np.random.rand(M,) - 0.5
fhat = np.random.rand(N[0]) +  1.0j * np.random.rand(N[0])

# initialize a NFFT plan and set data
plan = nfft3.NFFT(N,M)
plan.X = X
plan.fhat = fhat 

# perform the NFFT transformation
plan.trafo()

#transformed vector can be accessed as 
plan.f 

```

Documentation is a work in progress.