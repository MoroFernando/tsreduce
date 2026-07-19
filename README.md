# tsreduce

Time series dimensionality reduction — from classical methods like PAA and PCA to deep learning encoders.

## Installation

```bash
pip install tsreduce
```

## Local development

### With conda

```bash
conda create -n tsreduce python=3.10 -y
conda activate tsreduce
pip install -e .
```

### With venv

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -e .
```

### Development tools

```bash
pip install -r requirements-dev.txt
```

## Usage

```python
import numpy as np
from tsreduce import PAA, AE

X_train = np.random.randn(200, 1000)
X_test  = np.random.randn(50,  1000)

paa = PAA(retention_rate=0.2)
paa.fit(X_train)
X_reduced = paa.transform(X_test)   # (50, 200)

ae = AE(retention_rate=0.1, epochs=30, random_state=42)
ae.fit(X_train)
X_reduced = ae.transform(X_test)    # (50, 100)
print(ae.fit_time_)
```

## Available methods

| Name | Class | Description |
|------|-------|-------------|
| PAA  | `tsreduce.PAA` | Piecewise Aggregate Approximation — segment-mean pooling, stateless |
| PCA  | `tsreduce.PCA` | Principal Component Analysis — per-channel sklearn PCA |
| AE   | `tsreduce.AE`  | Dense Autoencoder — learns a compressed latent representation |

All reducers are sklearn-compatible (`fit`, `transform`, `fit_transform`, `get_params`, `set_params`) and accept both 2-D `(n_samples, n_timepoints)` and 3-D `(n_samples, n_channels, n_timepoints)` inputs.

## Adding a new method

Subclass `BaseReducer` and implement `_fit` and `_transform`. Both receive and must return 3-D arrays; 2-D normalisation, shape validation, and timing are handled by the base class.

```python
import numpy as np
from tsreduce.base import BaseReducer


class MyReducer(BaseReducer):

    def __init__(self, *, target_length=None, retention_rate=None):
        super().__init__(target_length=target_length, retention_rate=retention_rate)

    def _fit(self, X: np.ndarray) -> None:
        # X is always (n_samples, n_channels, n_timepoints)
        pass  # stateless example

    def _transform(self, X: np.ndarray) -> np.ndarray:
        # Must return (n_samples, n_channels, self.n_timepoints_out_)
        n_samples, n_channels, n_timepoints = X.shape
        w = self.n_timepoints_out_
        step = n_timepoints // w
        return X[:, :, ::step][:, :, :w]
```
