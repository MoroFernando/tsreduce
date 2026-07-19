from importlib.metadata import version as _version
__version__ = _version("tsreduce")

from .base import BaseReducer
from .reducers.paa import PAA
from .reducers.pca import PCA
from .reducers.ae import AE

__all__ = ["__version__", "BaseReducer", "PAA", "PCA", "AE"]
