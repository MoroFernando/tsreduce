# TSreduce

Time series dimensionality reduction — from classical methods like PAA and PCA to deep learning encoders.

**Documentation:** [tsreduce.readthedocs.io](https://tsreduce.readthedocs.io/)

## Installation

```bash
pip install git+https://github.com/MoroFernando/tsreduce.git
```

### Local development

```bash
# conda
conda create -n tsreduce python=3.10 -y
conda activate tsreduce
pip install -e .

# venv
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
pip install -e .
```

## Usage

```python
import numpy as np
from tsreduce import PAA, AE

X_train = np.random.randn(200, 1000)
X_test  = np.random.randn(50,  1000)

# Classical method
paa = PAA(retention_rate=0.2)
paa.fit(X_train)
X_reduced = paa.transform(X_test)   # (50, 200)

# Neural method with progress bar
ae = AE(retention_rate=0.1, epochs=30, verbose=True, random_state=42)
ae.fit(X_train)
X_reduced = ae.transform(X_test)    # (50, 100)
print(ae.fit_time_, ae.transform_time_)
```

All reducers are sklearn-compatible (`fit`, `transform`, `fit_transform`, `get_params`, `set_params`) and accept both 2-D `(n_samples, n_timepoints)` and 3-D `(n_samples, n_channels, n_timepoints)` inputs.

## Available methods

| Class | Description |
|-------|-------------|
| `PAA` | Piecewise Aggregate Approximation — segment-mean pooling, stateless |
| `DFT` | Discrete Fourier Transform — retains lowest-frequency coefficients, stateless |
| `DWT` | Discrete Wavelet Transform — selects coefficients by energy ranking |
| `UniformDownsampling` | Picks uniformly-spaced indices from each series, stateless |
| `PCA` | Principal Component Analysis — per-channel sklearn PCA |
| `SVD` | Singular Value Decomposition — per-channel right singular vectors |
| `KPCA` | Kernel PCA — non-linear per-channel embedding |
| `Isomap` | Geodesic manifold embedding — per-channel |
| `AE` | Dense Autoencoder — fully-connected encoder–decoder |
| `CAE` | Convolutional Autoencoder — 1-D conv encoder–decoder |
| `TAE` | Transformer Autoencoder — Transformer encoder with adaptive pooling |
| `TCN` | Temporal Convolutional Network — causal dilated residual blocks |
| `S2V` | Series2Vec — dual-branch (time + frequency) contrastive encoder |
| `CCNN` | Contrastive CNN — nearest-neighbour queue contrastive learning |

## Extending

Subclass `BaseReducer` and implement `_fit` and `_transform`. Both receive 3-D arrays; 2-D normalisation, shape validation, and timing are handled by the base class. `y` is forwarded from `fit()` and available for supervised methods.

```python
import numpy as np
from tsreduce.base import BaseReducer


class MyReducer(BaseReducer):

    def __init__(self, *, target_len=None, retention_rate=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)

    def _fit(self, X: np.ndarray, y=None) -> None:
        # X is always (n_samples, n_channels, n_timepoints)
        pass  # stateless example

    def _transform(self, X: np.ndarray) -> np.ndarray:
        # Must return (n_samples, n_channels, self.n_timepoints_out_)
        n_samples, n_channels, n_timepoints = X.shape
        w = self.n_timepoints_out_
        step = n_timepoints // w
        return X[:, :, ::step][:, :, :w]
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.
