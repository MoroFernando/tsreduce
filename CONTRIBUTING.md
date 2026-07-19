# Contributing to tsreduce

## Local setup

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

## Adding a method

Follow these steps to add a new dimensionality-reduction method.

**1. Create the file**

Add `tsreduce/reducers/my_method.py`.

**2. Subclass `BaseReducer`**

Implement `_fit` and `_transform`. Both receive a 3-D array
`(n_samples, n_channels, n_timepoints)`; 2-D normalisation, shape validation,
and timing are handled by the base class.

```python
import numpy as np
from tsreduce.base import BaseReducer


class MyMethod(BaseReducer):

    def __init__(self, *, target_length=None, retention_rate=None):
        super().__init__(target_length=target_length, retention_rate=retention_rate)

    def _fit(self, X: np.ndarray) -> None:
        pass  # store whatever state the method needs

    def _transform(self, X: np.ndarray) -> np.ndarray:
        # Must return (n_samples, n_channels, self.n_timepoints_out_)
        ...
```

**3. Register in `tsreduce/reducers/__init__.py`**

```python
from .my_method import MyMethod
```

**4. Re-export from `tsreduce/__init__.py`**

```python
from .reducers import MyMethod
```

## Conventions

- All constructor parameters must be keyword-only (declared after `*`).
- Heavy imports (torch, tensorflow, etc.) belong inside `_fit` and `_transform`, never at module level — keeps import time fast for users who don't need that backend.
- Neural methods must accept `verbose: int = 0` and use `tqdm` to wrap the training loop so users can opt into progress reporting.
- Write comments only for non-obvious decisions — skip anything the code already says clearly.
