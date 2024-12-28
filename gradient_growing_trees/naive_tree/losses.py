import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from scipy.special import softmax


class BaseLossGradHess(ABC):
    @abstractmethod
    def __call__(self, target, preds) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Pair (gradients, hessian diagonal).

        """
        ...


class MultiOutputMSELossGradHess(BaseLossGradHess):
    def __call__(self, target, preds) -> Tuple[np.ndarray, np.ndarray]:
        grads = 2 * (preds - target)
        hess = np.full_like(grads, 2.0)
        return grads, hess


class MultiOutputMAELossGradHess(BaseLossGradHess):
    def __call__(self, target, preds) -> Tuple[np.ndarray, np.ndarray]:
        grads = np.sign(preds - target)
        hess = np.full_like(grads, 1.0)
        return grads, hess


class CELossGradHess(BaseLossGradHess):
    def __init__(self, l2_lam: float = 0.0, l1_lam: float = 0.0):
        self.l2_lam = l2_lam
        self.l1_lam = l1_lam

    def __call__(self, target, preds) -> Tuple[np.ndarray, np.ndarray]:
        # preds are logits

        # softmax
        s = softmax(preds, axis=1)  # shape: (N, C)
        # gradient
        C = preds.shape[1]  # number of classes

        delta = np.eye(C).reshape((1, C, C))
        delta_j_k_minus_s_k = delta - s[:, np.newaxis, :]  # shape: (N, C, C)

        grad = -np.einsum('ij,ijk->ik', target, delta_j_k_minus_s_k)  # shape: (N, C)
        # hessian
        hess = s * (1.0 - s)
        return grad, hess


class SurvivalLogLossGradHess(BaseLossGradHess):
    eps: float = 1.e-18

    def __call__(self, target, preds) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            target: Array consisting of ones and zeros.
                    Probabilities of entries labeled as 1, will be summed up in the loss.
                    I.e. if `target` is one-hot encoded label, the result will be the same as in
                    classical Cross-Entropy.
            preds: Array of predicted logits, shape (n_samples, n_outputs).

        """
        # softmax
        s = softmax(preds, axis=1)  # shape: (N, C)
        # gradient
        C = preds.shape[1]  # number of classes

        y = target
        y_dot_s = np.einsum('nj,nj->n', y, s).reshape((-1, 1))
        grad = s * (1 - y / np.maximum(y_dot_s, self.eps))
        hess = s * (1 - s - y * (y_dot_s - y * s) / np.maximum(y_dot_s * y_dot_s, self.eps))
        return grad, hess
