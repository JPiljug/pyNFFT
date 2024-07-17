"""
    PRE_PHI_HUT

precompute and store values ``\hat{\phi}(k)`` of the Fourier transform of the window function ``\hat{\phi}``.
"""
PRE_PHI_HUT = 1 << 0
"""
    FG_PSI

fast Gaussian gridding.
"""
FG_PSI = 1 << 1
"""
    PRE_LIN_PSI

linear interpolation of the window function from a lookup table.
"""
PRE_LIN_PSI = 1 << 2
"""
    PRE_FG_PSI

fast Gaussian gridding.
"""
PRE_FG_PSI = 1 << 3
"""
    PRE_PSI

precomputation based on tensor product structure of the window function.
"""
PRE_PSI = 1 << 4
"""
    PRE_FULL_PSI

calculate and store all values ``\tilde{\psi}(x_j - \frac{1}{n} \odot l)``.
"""
PRE_FULL_PSI = 1 << 5
"""
    MALLOC_X

allocate memory for node ``x_j``.
"""
MALLOC_X = 1 << 6
"""
    MALLOC_F_HAT

allocate memory for coefficient``\hat{f}_k``.
"""
MALLOC_F_HAT = 1 << 7
"""
    MALLOC_F

allocate memory for approximate function value ``f_j``.
"""
MALLOC_F = 1 << 8
"""
    FFT_OUT_OF_PLACE

FFTW uses disjoint input/output vector.
"""
FFT_OUT_OF_PLACE = 1 << 9
"""
    FFTW_INIT

initialize FFTW plan.
"""
FFTW_INIT = 1 << 10
"""
    NFFT_SORT_NODES

internal sorting of the nodes ``x_j`` that may increase performance.
"""
NFFT_SORT_NODES = 1 << 11
"""
    NFFT_OMP_BLOCKWISE_ADJOINT

blockwise calculation for adjoint NFFT in the case of OpenMP.
"""
NFFT_OMP_BLOCKWISE_ADJOINT = 1 << 12
"""
    NFCT_SORT_NODES

internal sorting of the nodes ``x_j`` that may increase performance.
"""
NFCT_SORT_NODES = 1 << 11
"""
    NFCT_OMP_BLOCKWISE_ADJOINT

blockwise calculation for adjoint NFFT in the case of OpenMP.
"""
NFCT_OMP_BLOCKWISE_ADJOINT = 1 << 12
"""
    NFST_SORT_NODES

internal sorting of the nodes ``x_j`` that may increase performance.
"""
NFST_SORT_NODES = 1 << 11
"""
    NFST_OMP_BLOCKWISE_ADJOINT

blockwise calculation for adjoint NFFT in the case of OpenMP.
"""
NFST_OMP_BLOCKWISE_ADJOINT = 1 << 12
"""
    PRE_ONE_PSI
"""
PRE_ONE_PSI = (PRE_LIN_PSI | PRE_FG_PSI | PRE_PSI | PRE_FULL_PSI)

# FFTW flags
"""
    FFTW_MEASURE

find optimal plan by executing several FFTs and compare times.
"""
FFTW_MEASURE = 0
"""
    FFTW_DESTROY_INPUT

an out-of-place transform is allowed to overwrite the input array with arbitrary data.
"""
FFTW_DESTROY_INPUT = 1 << 0
"""
    FFTW_UNALIGNED

the algorithm may not impose any unusual alignment requirements on the input/output arrays (not necessary in most context).
"""
FFTW_UNALIGNED = 1 << 1
"""
    FFTW_CONSERVE_MEMORY

conserving memory.
"""
FFTW_CONSERVE_MEMORY = 1 << 2
"""
    FFTW_EXHAUSTIVE

behaves like FFTW_PATIENT with an even wider range of tests.
"""
FFTW_EXHAUSTIVE = 1 << 3
"""
    FFTW_PRESERVE_INPUT

input vector is preserved and unchanged.
"""
FFTW_PRESERVE_INPUT = 1 << 4
"""
    FFTW_PATIENT

behaves like FFTW_MEASURE with a wider range of tests.
"""
FFTW_PATIENT = 1 << 5
"""
    FFTW_ESTIMATE

use simple heuristic instead of measurements to pick a plan.
"""
FFTW_ESTIMATE = 1 << 6
"""
    FFTW_WISDOM_ONLY

a plan is only created if wisdom from tests is available.
"""
FFTW_WISDOM_ONLY = 1 << 21

# default flag values
"""
    f1_default_1d
"""
f1_default_1d = ( PRE_PHI_HUT |
                  PRE_PSI |
                  MALLOC_X |
                  MALLOC_F_HAT |
                  MALLOC_F |
                  FFTW_INIT |
                  FFT_OUT_OF_PLACE)
"""
    f1_default
"""
f1_default = ( PRE_PHI_HUT |
               PRE_PSI |
               MALLOC_X |
               MALLOC_F_HAT |
               MALLOC_F |
               FFTW_INIT |
               FFT_OUT_OF_PLACE |
               NFCT_SORT_NODES |
               NFCT_OMP_BLOCKWISE_ADJOINT )

"""
    f2_default
"""
f2_default = FFTW_ESTIMATE | FFTW_DESTROY_INPUT
"""
default window cut off
"""
default_window_cut_off = 8

BASES = {
    "exp": 0,
    "cos": 1,
    "alg": 2
}