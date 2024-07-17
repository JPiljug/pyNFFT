import os 
import sys

# Check loading path, compatibility with ide for comfort pasting 
if '__file__' not in globals():
    _dir_path = os.path.dirname(os.path.realpath((__name__)))
else:
    _dir_path = os.path.abspath(os.path.dirname(__file__))

# @TODO: check for avx support -> linux 'grep avx /proc/cpuinfo' possible but in general python tricky
# there is something like cpuinfo , but that is not in general python
# for now restrict to systems with avx support
_avx_flag = "avx"

# Check os and load data
if sys.platform == 'linux':
    _os_flag = '.so'
elif sys.platform == 'win32':
    _os_flag = '.avx.dll'
elif sys.platform == 'darwin':
    _os_flag = '.dylib'
else:
    raise RuntimeError("Operating System not supported.")

NFFT_PATH = _dir_path + "/nfft/libnfftjulia" + _avx_flag + _os_flag
NFCT_PATH = _dir_path + "/nfct/libnfctjulia" + _avx_flag + _os_flag