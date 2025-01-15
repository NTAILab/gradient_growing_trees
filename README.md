# Gradient Growing Trees

This repository contains a reference implementation of Gradient Growing Trees and examples.

The Gradient Growing Trees are of the same structure as CART, but the construction algorithm is different:

- value at each node is calculated by its *parent node value* plus *gradient step*;
- split at each point is chosen in order to minimize 2nd order loss approximation, calculated by
using the given gradients and hessian diagonals.

## Installation

The package is called `gradient_growing_trees` and can be installed by running:

```bash
pip install -r requirements.txt
pip install -e .
```

Please, note that tree construction algorithm is implemented in **Cython**,
and therefore it is required for the build process.
So the packages mentioned in `requirements.txt` have to be installed in advance.

## Usage

**The detailed examples are provided in [notebooks/](notebooks/) directory.**

The package provides scikit-learn compatible interface for regression:

```python
from gradient_growing_trees.tree import GradientGrowingTreeRegressor

model = GradientGrowingTreeRegressor(
    lam_2=0.5,              # regularization, the higher the slower training
    lr=1.0,                 # it is preferred to set it to 1.0
    criterion='batch_mse',  # for regression, 'batch_gce' for classification and survival tasks
    splitter='best',
    max_depth=3,
    random_state=1,
)
# assuming the tuple (X_train, y_train) is already prepared
if y_train.ndim == 1:
    y_train = y_train.reshape((-1, 1))
model.fit(X_train, y_train)
# ...
preds_test = model.predict(X_test)
```

The main feature is that **an arbitrary twice differentiable loss function** can be used for tree construction.
For example:

```python
def custom_loss(ys, sample_ids, cur_value, g_out, h_out):
    g_out = np.asarray(g_out)
    h_out = np.asarray(h_out)
    sample_ids = np.asarray(sample_ids)
    # 1nd order derivative of squared error:
    g_out[sample_ids] = -2.0 * (np.asarray(ys)[sample_ids] - np.asarray(cur_value)[np.newaxis])
    # 2nd order derivative of squared error is constant:
    h_out[sample_ids] = 2.0

model = GradientGrowingTreeRegressor(
    # ...
    criterion=custom_loss,
)
model.fit(X_train, y_train)
preds_test = model.predict(X_test)  # output dimensionality is the same as `y_train`
```

There are some known peculiarities, which would be fixed after interface stabilization:

1. Currently only `GradientGrowingTreeRegressor` is implemented which is a *Regressor* in terms of scikit-learn.
However, it [can be directly used for classification tasks](notebooks/01_classification.ipynb), the only limitation is that the output of `predict(X)`
method would be the logits of probability distribution (not the most common label).
2. Target variable (`y`) should be always with two axes (dimensions).
E.g. for 1d regression the label should be of shape `(n_samples, 1)`.
3. The output dimensionality is determined by the target label shape (`y.shape[1]`), while
for some applications it is more convenient to have different output and label dimensions
(e.g. in classification the output is a probability distribution, while labels are categorical).
This limitation can be overcome by using a specific custom loss function and fake `y_train`, or by extending `TreeNN`.

### TreeNN

The Gradient Growing Trees method allows to create an effective algorithm for building
combinations of trees and neural networks, where all components are trained simultaneously.

The reference implementation can be found at `gradient_growing_trees.tree_nn.TreeNN`.
It is a base class from which a model class should be derived.
The example can be found [here](notebooks/04_nn.ipynb).
