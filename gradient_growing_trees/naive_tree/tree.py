import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Optional
from .losses import *


def calc_values_multioutput(grad, hess, ind_in, lam_2: float):
    n_outputs = grad.shape[1]

    EPS = 1.e-9
    n_total = ind_in.shape[0]
    n_in = np.sum(ind_in)

    v_in = np.zeros((1, n_outputs), dtype=np.float64)
    v_out = np.zeros((1, n_outputs), dtype=np.float64)

    if n_in > 0:
        G_in = ind_in.astype(np.float64) @ grad
        H_in = ind_in.astype(np.float64) @ hess
        v_in += (-(G_in) / np.maximum(lam_2 * n_total + H_in, EPS))

    ind_out = ~ind_in
    n_out = len(ind_in) - n_in
    if n_out > 0:
        G_out = ind_out.astype(np.float64) @ grad
        H_out = ind_out.astype(np.float64) @ hess
        v_out += (-(G_out) / np.maximum(lam_2 * n_total + H_out, EPS))
    return v_in, v_out


def calc_loss_multioutput(x, grad, hess, feature, threshold, lam_2: float,
              use_reg_in_loss: bool):
    ind_in = x[:, feature] <= threshold
    ind_out = ~ind_in
    v_in, v_out = calc_values_multioutput(grad, hess, ind_in, lam_2)
    n_total = ind_in.shape[0]
    loss = ind_in.astype(np.float64) @ (v_in * (grad + 0.5 * v_in * hess)) + \
           ind_out.astype(np.float64) @ (v_out * (grad + 0.5 * v_out * hess))
    if use_reg_in_loss:
        loss += 0.5 * lam_2 * n_total * (v_in * v_in + v_out * v_out)
    return loss.sum(), v_in, v_out


def randomized_split_multioutput(x, grad, hess, n_guess: int = 1, lam_2: float = 0.0,
                     use_reg_in_loss: bool = False):
    n_features = x.shape[1]
    best_loss = np.inf
    best_v_in = 0.0
    best_v_out = 0.0
    best_feature = -1
    best_threshold = 0.0
    if n_guess > 0:
        for i in range(n_features):
            for j in range(n_guess):
                random_threshold = np.random.uniform(x[:, i].min(), x[:, i].max())
                loss, v_in, v_out = calc_loss_multioutput(
                    x, grad, hess, i, random_threshold, lam_2, use_reg_in_loss
                )
                if loss < best_loss:
                    best_loss = loss
                    best_v_in = v_in
                    best_v_out = v_out
                    best_feature = i
                    best_threshold = random_threshold
    else:
        for i in range(n_features):
            unique = np.unique(x[:, i])
            for t in unique:
                loss, v_in, v_out = calc_loss_multioutput(
                    x, grad, hess, i, t, lam_2, use_reg_in_loss
                )
                if loss < best_loss:
                    best_loss = loss
                    best_v_in = v_in
                    best_v_out = v_out
                    best_feature = i
                    best_threshold = t
    return best_feature, best_threshold, best_v_in, best_v_out


class BaseFnMultiOutput:
    def __init__(self, n_guess, lam_2: float, use_reg_in_loss: bool):
        self.n_guess = n_guess
        self.lam_2 = lam_2
        self.use_reg_in_loss = use_reg_in_loss

    def fit(self, X, grad, hess):
        assert grad.ndim == 2
        assert hess.ndim == 2
        self.feature_, self.threshold_, self.v_in_, self.v_out_ = randomized_split_multioutput(
            X,
            grad,
            hess,
            n_guess=self.n_guess,
            lam_2=self.lam_2,
            use_reg_in_loss=self.use_reg_in_loss,
        )
        return self

    def get_indicators(self, X):
        ind_in = X[:, self.feature_] <= self.threshold_
        ind_out = ~ind_in
        return ind_in[:, np.newaxis], ind_out[:, np.newaxis]

    def predict(self, X):
        ind_in, ind_out = self.get_indicators(X)
        return self.v_in_ * ind_in + self.v_out_ * ind_out


class GAMGRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, max_depth: int = 3, n_guess: int = 1):
        self.max_depth = max_depth
        self.n_guess = n_guess

    def fit(self, X, y):
        self.estimators_ = []
        preds = 0.0
        for i in range(self.max_depth):
            fn = BaseFnMultiOutput(n_guess=self.n_guess, lam_2=0.0, use_reg_in_loss=False)
            grads = 2 * (preds - y)
            hess = np.full_like(grads, 2.0)
            fn.fit(X, grads, hess)
            self.estimators_.append(fn)
            if i != self.max_depth - 1:
                preds += fn.predict(X)
        return self

    def predict(self, X):
        preds = 0.0
        for fn in self.estimators_:
            preds += fn.predict(X)
        return preds


class FullGTreeRegressorMultiOutput(RegressorMixin, BaseEstimator):
    """Non oblivious Gradient-Tree.
    """
    def __init__(self, max_depth: int = 3, n_guess: int = 1, learning_rate: float = 1.0,
                 loss_grad_hess: Optional[BaseLossGradHess] = None,
                 lam_2: float = 0.0,
                 use_reg_in_loss: bool = False):
        self.max_depth = max_depth
        self.n_guess = n_guess
        self.learning_rate = learning_rate
        if loss_grad_hess is None:
            loss_grad_hess = MultiOutputMSELossGradHess()
        self.loss_grad_hess = loss_grad_hess
        self.lam_2 = lam_2
        self.use_reg_in_loss = use_reg_in_loss

    def fit(self, X, y):
        assert y.ndim == 2
        self.n_outputs_ = y.shape[1]

        self.estimators_ = []
        preds = np.zeros_like(y)
        current_nodes = [np.arange(len(X))]
        for i in range(self.max_depth):
            grads, hess = self.loss_grad_hess(y, preds)
            next_nodes = []
            for node in current_nodes:
                point_ids = node
                fn = BaseFnMultiOutput(
                    n_guess=self.n_guess,
                    lam_2=self.lam_2,
                    use_reg_in_loss=self.use_reg_in_loss,
                )
                if len(point_ids) != 0:
                    fn.fit(X[point_ids], grads[point_ids], hess[point_ids])
                    fn.v_in_ *= self.learning_rate
                    fn.v_out_ *= self.learning_rate
                    self.estimators_.append(fn)
                    if i != self.max_depth - 1:
                        preds[point_ids] += fn.predict(X[point_ids])
                        ind_in, ind_out = fn.get_indicators(X[point_ids])
                        next_nodes.append(point_ids[ind_in.ravel()])
                        next_nodes.append(point_ids[ind_out.ravel()])
                else:
                    fn.feature_, fn.threshold_, fn.v_in_, fn.v_out_ = 0, 0.0, 0.0, 0.0
                    self.estimators_.append(fn)
                    if i != self.max_depth - 1:
                        next_nodes.append(point_ids)
                        next_nodes.append(point_ids)
            current_nodes = next_nodes
        return self

    def predict(self, X):
        preds = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        current_nodes = [np.arange(len(X))]
        fn_iter = iter(self.estimators_)
        for i in range(self.max_depth):
            next_nodes = []
            for node in current_nodes:
                point_ids = node
                fn = next(fn_iter)
                if i != self.max_depth - 1:
                    preds[point_ids] += fn.predict(X[point_ids])
                    ind_in, ind_out = fn.get_indicators(X[point_ids])
                    next_nodes.append(point_ids[ind_in.ravel()])
                    next_nodes.append(point_ids[ind_out.ravel()])
            current_nodes = next_nodes
        return preds
