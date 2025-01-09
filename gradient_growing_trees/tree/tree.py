import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    _assert_all_finite_element_wise,
    assert_all_finite,
    check_is_fitted,
    check_random_state,
)
from numbers import Integral, Real
from ._tree import DepthFirstTreeBuilder, Tree
from . import _splitter, _criterion
from ._splitter import Splitter
from ._criterion import BatchArbitraryLoss


DENSE_SPLITTERS = {"best": _splitter.BestSplitter, "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {
    "best": _splitter.BestSparseSplitter,
    "random": _splitter.RandomSparseSplitter,
}

CRITERIONS = {
    "mse": _criterion.MSE2,
    "ce": _criterion.CrossEntropy2,
    "gce": _criterion.GeneralizedLogLoss,
    "batch_mse": _criterion.BatchMSE,
    "batch_gce": _criterion.BatchGeneralizedLogLoss,
}


class GradientGrowingTreeRegressor(BaseEstimator, RegressorMixin):  #, MultiOutputMixin):
    def __init__(
        self,
        *,
        lam_2=0.1,
        lr=1.0,
        splitter="best",
        criterion="batch_mse",
        n_update_iterations=1,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None
    ):
        self.lam_2 = lam_2
        self.lr = lr
        self.splitter = splitter
        self.criterion = criterion
        self.n_update_iterations = n_update_iterations
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a survival tree from the training set (X, y).

        If ``splitter='best'``, `X` is allowed to contain missing
        values. In addition to evaluating each potential threshold on
        the non-missing data, the splitter will evaluate the split
        with all the missing values going to the left node or the
        right node. See :ref:`tree_missing_value_support` for details.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self
        """
        self._fit(X, y, sample_weight, check_input)
        return self

    def _check_max_features(self):
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))

        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, (Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        if not 0 < max_features <= self.n_features_in_:
            raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

    def _check_params(self, n_samples):
        # self._validate_params()

        # Check parameters
        max_depth = (2**31) - 1 if self.max_depth is None else self.max_depth

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if isinstance(self.min_samples_leaf, (Integral, np.integer)):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(np.ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(np.ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        self._check_max_features()

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")

        min_weight_leaf = self.min_weight_fraction_leaf * n_samples

        return {
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "min_weight_leaf": min_weight_leaf,
        }

    def _fit(self, X, y, sample_weight=None, check_input=True, missing_values_in_feature_mask=None):
        random_state = check_random_state(self.random_state)
        assert y.ndim == 2
        self.n_outputs_ = y.shape[1]
        n_samples = y.shape[0]
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = np.ones(self.n_outputs_, dtype=np.int64) * 1
        params = self._check_params(n_samples)

        # Build tree
        if not isinstance(self.criterion, _criterion.Criterion):
            if isinstance(self.criterion, str):
                criterion_cls = CRITERIONS[self.criterion]
                if 'batch' in self.criterion:
                    criterion = criterion_cls(self.n_outputs_, n_samples, self.lam_2, self.lr, self.n_update_iterations)
                else:
                    criterion = criterion_cls(self.n_outputs_, n_samples, self.lam_2, self.lr)
            elif callable(self.criterion):
                criterion = _criterion.BatchArbitraryLoss(
                    self.n_outputs_, n_samples, self.lam_2, self.lr, self.n_update_iterations
                )
                criterion.set_loss(self.criterion)
            else:
                raise ValueError(f'Wrong criterion type: {type(self.criterion)!r}')
        else:
            criterion = self.criterion

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                params["min_samples_leaf"],
                params["min_weight_leaf"],
                random_state,
                None,  # monotonic_cst
            )

        self.tree_ = Tree(self.n_features_in_, self.n_classes_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        assert params['max_leaf_nodes'] < 0, 'Only DepthFirstTreeBuilder is implemented'
        builder = DepthFirstTreeBuilder(
            splitter,
            params["min_samples_split"],
            params["min_samples_leaf"],
            params["min_weight_leaf"],
            params["max_depth"],
            0.0,  # min_impurity_decrease
        )

        builder.build(self.tree_, X, y, sample_weight, missing_values_in_feature_mask)
        return self

    def apply(self, X):
        return self.tree_.apply(X.astype(np.float32))

    def predict(self, X):
        return self.tree_.predict(X.astype(np.float32))
