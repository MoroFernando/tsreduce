from importlib.metadata import version as _version
__version__ = _version("tsreduce")

from .base import BaseReducer
from .reducers.paa import PAA
from .reducers.dft import DFT
from .reducers.dwt import DWT
from .reducers.uniform_downsampling import UniformDownsampling
from .reducers.pca import PCA
from .reducers.svd import SVD
from .reducers.kpca import KPCA
from .reducers.isomap import Isomap
from .reducers.ae import AE
from .reducers.cae import CAE
from .reducers.tae import TAE
from .reducers.tcn import TCN
from .reducers.s2v import S2V
from .reducers.siamese import Siamese
from .reducers.ccnn import CCNN

__all__ = [
    "__version__",
    "BaseReducer",
    "PAA", "DFT", "DWT", "UniformDownsampling",
    "PCA", "SVD", "KPCA", "Isomap",
    "AE", "CAE", "TAE", "TCN", "S2V", "Siamese", "CCNN",
]
