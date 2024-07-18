import ctypes as ct
import numpy as np

from .. import flags
from .. import _init_paths

# Define  dummy structure for C nfct_plan
class _NFSTPlan(ct.Structure):
    pass

# load matching C shared object file for NFFT
_libnfct = ct.CDLL(_init_paths.NFST_PATH)  
  
# Define the function prototypes already known prior to runtime
_libnfct.jnfst_init.argtypes = [ct.POINTER(_NFSTPlan), ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32, 
                               ct.POINTER(ct.c_int32), ct.c_int32, ct.c_uint32, ct.c_uint32]
_libnfct.jnfst_alloc.restype = ct.POINTER(_NFSTPlan)
_libnfct.jnfst_finalize.argtypes = [ct.POINTER(_NFSTPlan)]
_libnfct.jnfst_set_f.argtypes = [ct.POINTER(_NFSTPlan), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C')] 
_libnfct.jnfst_set_f.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C') 
_libnfct.jnfst_set_fhat.argtypes = [ct.POINTER(_NFSTPlan), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C')] 
_libnfct.jnfst_set_fhat.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C') 
_libnfct.jnfst_trafo.argtypes = [ct.POINTER(_NFSTPlan)]
_libnfct.jnfst_adjoint.argtypes = [ct.POINTER(_NFSTPlan)]
_libnfct.jnfst_trafo_direct.argtypes = [ct.POINTER(_NFSTPlan)]
_libnfct.jnfst_adjoint_direct.argtypes = [ct.POINTER(_NFSTPlan)]


# Python class for NFCT
class NFST:
    def __init__(self, *args):
        if len(args) == 2:
            self._N, self._M = args
            self._plan = _libnfct.jnfst_alloc()
            self._n = (2 ** (np.ceil( np.log(self.N) / np.log(2)) + 1 )).astype('int32') 
            self._m = flags.default_window_cut_off
            if self._D == 1:
                self._f1 = flags.f1_default_1d | flags.MALLOC_X | flags.MALLOC_F_HAT | flags.MALLOC_F | flags.FFTW_INIT
            else:
                self._f1 = flags.f1_default | flags.MALLOC_X | flags.MALLOC_F_HAT | flags.MALLOC_F | flags.FFTW_INIT
            self._f2 = flags.f2_default  
            _libnfct.jnfst_init(self._plan, self._D, self.__N, self.__M, self.__n, self.__m, self.__f1, self.__f2)
            self._init_done = True
            self._finalized = False
        elif len(args) == 6:
            self._N, self._M, self._n, self._m, self._f1, self._f2 = args
            self._plan = _libnfct.jnfst_alloc()
            _libnfct.jnfst_init(self._plan, self._D, self.__N, self.__M, self.__n, self.__m, self.__f1, self.__f2)
            self._init_done = True
            self._finalized = False
        else:
            raise RuntimeError("Invalid number of argumentes given to NFFT constructor.")

    # use setters and getters to ensure correct data types 
    @property
    def N(self):
        return np.ctypeslib.as_array(self.__N, shape=(self._D,))

    @N.setter 
    def _N(self,value):
        if not np.shape(value) or not np.size(value) or len(np.shape(value)) > 1:
            raise TypeError("N has to be 1 dimensional, iterable and nonempty.")
        if isinstance(value,np.ndarray) and value.dtype == np.int32:
            np_value = value
        else:
            if all(isinstance(i, int) | isinstance(i, np.integer)  for i in value)  and np.max(value) < 2**31:
                np_value = np.array(value, dtype=np.int32)
            else:
                raise TypeError("N has to contain integers smaller than 2^31-1.")
        if np.any(np_value < 0) or np.any(np_value % 2 != 0):
            raise RuntimeError("N must only contain even and positive integers.")
        self._D = len(np_value)
        self.__N = np_value.ctypes.data_as(ct.POINTER(ct.c_int32))

    @property 
    def M(self):
        return self.__M.value

    @M.setter
    def _M(self,value):
        if not isinstance(value, int) or value <= 0 or value >= 2**31:
            raise TypeError("M has to be an integer between 1 and 2^31-1.")
        self.__M = ct.c_int32(value) 

    @property 
    def D(self):
        return self._D

    @property
    def n(self):
        return np.ctypeslib.as_array(self.__n, shape=(self._D,))

    @n.setter
    def _n(self,value):
        if not np.shape(value) or not np.size(value):
            raise TypeError("n has to be iterable and nonempty.")
        if isinstance(value,np.ndarray) and value.dtype == np.int32:
            np_value = value
        else:
            if all(isinstance(i, int) | isinstance(i, np.integer)  for i in value) and np.max(value) < 2**31:
                np_value = np.array(value, dtype=np.int32)
            else:
                raise TypeError("n has to contain integers smaller than 2^31-1.")
        if np.any(np_value < 0) or np.any(np_value % 2 != 0):
            raise RuntimeError("n must only contain even and positive integers.")
        elif np.any(np_value <= self.N):
            raise RuntimeError("n must fulfil n_i > N_i.") 
        self.__n = np_value.ctypes.data_as(ct.POINTER(ct.c_int32))

    @property 
    def m(self):
        return self.__m.value

    @m.setter
    def _m(self,value):
        if not isinstance(value, int) or value <= 0 or value >= 2**31:
            raise TypeError("m has to be an integer between 1 and 2^31-1.")
        self.__m = ct.c_int32(value) 

    @property 
    def f1(self):
        return self.__f1.value

    @m.setter
    def _f1(self,value):
        if not isinstance(value, int) or value <= 0 or value >= 2**32:
            raise TypeError("f1 has to be an integer between 0 and 2^32-1 .")
        self.__f1 = ct.c_uint32(value)

    @property 
    def f2(self):
        return self.__f2.value

    @m.setter
    def _f2(self,value):
        if not isinstance(value, int) or value <= 0 or value >= 2**32:
            raise TypeError("f2 has to be an integer between 0 and 2^32-1.")
        self.__f2 = ct.c_uint32(value)  
    
    @property
    def X(self):
        return self._X

    @X.setter 
    def X(self, value):
        if self._finalized:
            raise RuntimeError("NFFT already finalized.")
        if not (isinstance(value,np.ndarray) and value.dtype == np.float64 and value.flags['C']):
            raise TypeError("X has to be C-continuous, float64 numpy array.")
        if self._D == 1:
            if value.shape != (self.M, ):
                raise RuntimeError("X has to be 1-dimensional array of shape (M,).")
            _libnfct.jnfst_set_x.argtypes = [ct.POINTER(_NFSTPlan), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C')]
            _libnfct.jnfst_set_x.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(self.M,), flags='C')
        else:
            if value.shape != (self.M, self._D):
                raise RuntimeError("X has to be 2-dimensional array of shape (M,D).")
            _libnfct.jnfst_set_x.argtypes = [ct.POINTER(_NFSTPlan), np.ctypeslib.ndpointer(np.float64, ndim=2, flags='C')]
            _libnfct.jnfst_set_x.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=(self.M,self._D), flags='C')
        self._X = _libnfct.jnfst_set_x(self._plan, value)
    
    @property
    def f(self):
        return self._f
    
    @f.setter 
    def f(self, value):
        if not (isinstance(value,np.ndarray) and value.dtype == np.float64 and value.flags['C']):
            raise RuntimeError("f has to be C-continuous, numpy float64 array")
        if value.shape != (self.M, ):
            raise RuntimeError("f has to be 1-dimensional array of shape (M,).")  
        _libnfct.jnfst_set_f.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=(self.M,), flags='C') 
        self._f = _libnfct.jnfst_set_f(self._plan, value)

    @property
    def fhat(self):
        return self._fhat
    
    @fhat.setter 
    def fhat(self, value):
        if not (isinstance(value,np.ndarray) and value.dtype == np.float64 and value.flags['C']):
            raise RuntimeError("fhat has to be C-continuous, numpy float64 array") 
        Ns = np.prod(self.N - 1)
        if value.shape != (Ns, ):
            raise RuntimeError("fhat has to be 1-dimensional array of shape (np.prod(N-1),).") 
        _libnfct.jnfst_set_fhat.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=(Ns,), flags='C') 
        self._fhat = _libnfct.jnfst_set_fhat(self._plan, value)   

    def trafo_direct(self):
        if not self._init_done:
            raise RuntimeError("NFFT plan not initialized")
        elif self._finalized:
            raise RuntimeError("NFFT plan already finalized.")
        elif not hasattr(self, 'X'):
            raise RuntimeError("X has not been set.")
        elif not hasattr(self, 'fhat'):
            raise RuntimeError("fhat has not been set.")
        _libnfct.jnfst_trafo.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=(self.M,), flags='C') 
        self.f = _libnfct.jnfst_trafo_direct(self._plan)

    def trafo(self):
        if not self._init_done:
            raise RuntimeError("NFFT plan not initialized")
        elif self._finalized:
            raise RuntimeError("NFFT plan already finalized.")
        elif not hasattr(self, 'X'):
            raise RuntimeError("X has not been set.")
        elif not hasattr(self, 'fhat'):
            raise RuntimeError("fhat has not been set.")
        _libnfct.jnfst_trafo.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=(self.M,), flags='C') 
        self.f = _libnfct.jnfst_trafo(self._plan)

    def adjoint_direct(self):
        if not self._init_done:
            raise RuntimeError("NFFT plan not initialized")
        elif self._finalized:
            raise RuntimeError("NFFT plan already finalized.")
        elif not hasattr(self, 'X'):
            raise RuntimeError("X has not been set.")
        elif not hasattr(self, 'f'):
            raise RuntimeError("f has not been set.")
        Ns = np.prod(self.N-1)
        _libnfct.jnfst_adjoint_direct.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=(Ns,), flags='C') 
        self._fhat = _libnfct.jnfst_adjoint_direct(self._plan)

    def adjoint(self):
        if not self._init_done:
            raise RuntimeError("NFFT plan not initialized")
        elif self._finalized:
            raise RuntimeError("NFFT plan already finalized.")
        elif not hasattr(self, 'X'):
            raise RuntimeError("X has not been set.")
        elif not hasattr(self, 'f'):
            raise RuntimeError("f has not been set.")
        Ns = np.prod(self.N-1)
        _libnfct.jnfst_adjoint.restype = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=(Ns,), flags='C') 
        self._fhat = _libnfct.jnfst_adjoint(self._plan)

    # finalization method
    def finalize(self):
        if not self._init_done:
            raise RuntimeError("NFFT plan not initialized.")
        if self._finalized:
            raise RuntimeError("NFFT plan already finalized.")
        _libnfct.jnfst_finalize(self._plan)
        self._finalized = True

    # default destructor method 
    def __del__(self):
        self.finalize() 