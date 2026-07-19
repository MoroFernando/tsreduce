Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/MoroFernando/tsreduce.git

Basic Usage
-----------

All reducers share the same sklearn-compatible interface.
Provide either ``target_len`` or ``retention_rate``:

.. code-block:: python

   import numpy as np
   from tsreduce import PAA

   X = np.random.randn(100, 500)  # 100 series of length 500

   # by absolute length
   paa = PAA(target_len=100)
   X_reduced = paa.fit_transform(X)  # shape (100, 100)

   # by fraction
   paa = PAA(retention_rate=0.2)
   X_reduced = paa.fit_transform(X)  # shape (100, 100)

Multi-channel Data
------------------

3-D arrays ``(n_samples, n_channels, n_timepoints)`` are supported natively:

.. code-block:: python

   X_mc = np.random.randn(50, 6, 500)
   X_reduced = PAA(target_len=100).fit_transform(X_mc)  # (50, 6, 100)

sklearn Pipelines
-----------------

Reducers are fully compatible with ``sklearn.pipeline.Pipeline``:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.svm import SVC
   from tsreduce import PAA

   pipe = Pipeline([
       ("reduce", PAA(target_len=50)),
       ("clf",    SVC()),
   ])
   pipe.fit(X_train, y_train)
