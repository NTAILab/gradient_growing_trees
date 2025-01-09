import numpy as np
import torch
from gradient_growing_trees.tree import BatchArbitraryLoss
from sklearn.base import BaseEstimator, RegressorMixin, clone
from abc import ABCMeta, abstractmethod


class TreeNN(RegressorMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self, base_estimator=None, n_estimators: int = 10,
                 lam_2: float=0.1,
                 lr: float = 1.0,
                 tree_loss_on_sample_ids: bool = True,
                 n_update_iterations: int = 1,
                 embedding_size: int = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.lam_2 = lam_2
        self.lr = lr
        # training params
        self.tree_loss_on_sample_ids = tree_loss_on_sample_ids
        self.n_update_iterations = n_update_iterations
        self.embedding_size = embedding_size
        self.__post_init__()

    def __post_init__(self):
        pass

    def _make_tree(self, criterion):
        tree = clone(self.base_estimator)
        tree.criterion = criterion  # inject criterion
        tree.lam_2 = self.lam_2
        tree.lr = self.lr
        tree.n_update_iterations = self.n_update_iterations
        return tree

    def _pretrain_nn(self, X_nn_torch, y_torch):
        return

    def _pre_update_nn(self, X_nn_torch, y_torch, sample_ids_torch):
        pass

    @abstractmethod
    def _predict_nn(self, cur_X_nn_torch, cur_trees_predictions_torch):
        """Predict with torch-based NN using NN-specific input and trees predictions.

        Args:
            cur_X_nn_torch: Training NN input.
            cur_trees_predictions_torch: Tree(s) predictions for each sample.

        Returns:
            Predictions tensor (arbitrary shape, type).

        Example implementation:

            >>> return user_nn_predict_fn(cur_X_nn_torch, cur_trees_predictions_torch)

        """
        pass

    @abstractmethod
    def _calc_sample_grads(self, cur_X_nn_torch, cur_y_torch,
                           cur_trees_predictions_torch,
                           cur_sample_predictions):
        """Calculate grads w.r.t. `cur_sample_predictions`.

        Args:
            cur_X_nn_torch: Training NN input.
            cur_y_torch: Training labels.
            cur_trees_predictions_torch: Tree(s) predictions for each sample, depends on `cur_sample_predictions`.
            cur_sample_predictions: Predictions w.r.t. which gradients should be calculated.

        Returns:
            Loss gradients w.r.t. `cur_sample_predictions`.

        Example implementation:

            >>> loss = user_loss_fn(cur_y_torch, self._predict_nn(cur_X_nn_torch, cur_trees_predictions_torch))
            >>> grads, = torch.autograd.grad(loss, cur_sample_predictions)
            >>> return grads

        """
        pass

    def _post_update_nn(self, X_nn_torch, y_torch, sample_ids_torch, cumulative_predictions):
        pass

    def _postiter_nn(self, X_nn_torch, y_torch, cumulative_predictions,
                     eval_X_nn=None,
                     eval_y=None,
                     eval_cumulative_predictions=None):
        pass

    def _posttrain_nn(self, X_nn_torch, y_torch, cumulative_predictions,
                      eval_X_nn=None,
                      eval_y=None,
                      eval_cumulative_predictions=None):
        return

    def fit(self, X, y, X_nn=None, eval_XyXnn=None):
        if self.embedding_size is None:
            y_for_tree = y
        else:
            y_for_tree = np.zeros((X.shape[0], self.embedding_size), dtype=np.float64)
        embedding_size = y_for_tree.shape[1]

        if X_nn is not None:
            X_nn_torch = torch.tensor(X_nn, dtype=torch.float64)
        else:
            X_nn_torch = None
        y_torch = torch.tensor(y, dtype=torch.float64)
        cumulative_predictions = torch.zeros(
            y_for_tree.shape,
            dtype=torch.float64
        )  # cumulative tree predictions
        # evaluation data
        if eval_XyXnn is not None:
            eval_X, eval_y, *eval_rest = eval_XyXnn
            eval_X_nn = eval_rest[0] if len(eval_rest) > 0 else None
            eval_X_nn_torch = (
                torch.tensor(eval_X_nn, dtype=torch.float64)
                if eval_X_nn is not None
                else None
            )
            eval_y_torch = torch.tensor(eval_y, dtype=torch.float64)
            cumulative_predictions_eval = torch.zeros(
                (eval_y_torch.shape[0], embedding_size),
                dtype=torch.float64
            )
        else:
            eval_X_nn_torch = None
            eval_y_torch = None
            cumulative_predictions_eval = None

        self._pretrain_nn(X_nn_torch, y_torch)

        self.estimators_ = []
        for tree_index in range(self.n_estimators):
            all_sample_predictions = torch.zeros_like(cumulative_predictions)

            def _loss(_ys, sample_ids, cur_value, g_out, h_out):
                g_out = np.asarray(g_out)
                h_out = np.asarray(h_out)
                sample_ids = np.asarray(sample_ids)
                cur_value = np.asarray(cur_value)
                # update sample predictions in order to do NN training step
                all_sample_predictions[sample_ids] = torch.tensor(cur_value)

                sample_ids_torch = torch.tensor(sample_ids)

                self._pre_update_nn(X_nn_torch, y_torch, sample_ids_torch)

                if self.tree_loss_on_sample_ids:
                    cur_sample_predictions = all_sample_predictions[sample_ids].requires_grad_(True)
                    cur_X_nn_torch = X_nn_torch[sample_ids_torch] if X_nn_torch is not None else None
                    cur_y_torch = y_torch[sample_ids_torch]
                    cur_trees_predictions_torch = (
                        cumulative_predictions[sample_ids_torch]
                        +
                        cur_sample_predictions
                    )
                    grads = self._calc_sample_grads(
                        cur_X_nn_torch,
                        cur_y_torch,
                        cur_trees_predictions_torch,
                        cur_sample_predictions,
                    )
                else:  # not tree_loss_on_sample_ids
                    mask = torch.ones(all_sample_predictions.shape[0], dtype=bool)
                    mask[sample_ids_torch] = False
                    if X_nn_torch is None:
                        cur_X_nn_torch = None
                    else:
                        cur_X_nn_torch = X_nn_torch
                    cur_y_torch = y_torch
                    curp = all_sample_predictions.clone().requires_grad_(True)
                    # NOTE: here we compute gradients for all samples, not only `sample_ids`
                    # It is a sacrifice for training sample order preservation.
                    # Possibly it can be overcome by using concatenation of tensors with and without gradients and sorting.
                    grads = self._calc_sample_grads(
                        X_nn_torch,
                        y_torch,
                        cumulative_predictions + curp,
                        curp,
                    )[sample_ids]

                g_out[sample_ids] = grads.to('cpu').numpy()
                h_out[sample_ids] = 1.0

                self._post_update_nn(
                    X_nn_torch,
                    y_torch,
                    sample_ids_torch,
                    cumulative_predictions,
                )

            criterion = BatchArbitraryLoss(
                y.shape[1] if self.embedding_size is None else self.embedding_size,
                y.shape[0],
                self.lam_2,
                self.lr,
                self.n_update_iterations
            )
            criterion.set_loss(_loss)
            tree = self._make_tree(criterion)
            tree.fit(X, y_for_tree)
            self.estimators_.append(tree)
            cumulative_predictions += torch.tensor(tree.predict(X).squeeze(2))
            if cumulative_predictions_eval is not None:
                cumulative_predictions_eval += torch.tensor(tree.predict(eval_X).squeeze(2))
            self._postiter_nn(
                X_nn_torch,
                y_torch,
                cumulative_predictions,
                eval_X_nn=eval_X_nn_torch,
                eval_y=eval_y_torch,
                eval_cumulative_predictions=cumulative_predictions_eval,
            )

        self._posttrain_nn(
            X_nn_torch,
            y_torch,
            cumulative_predictions,
            eval_X_nn=eval_X_nn_torch,
            eval_y=eval_y_torch,
            eval_cumulative_predictions=cumulative_predictions_eval,
        )
        return self

    def _predict_trees(self, X):
        predictions = 0.0
        for tree in self.estimators_:
            predictions += tree.predict(X).squeeze(2)
        return predictions

    def predict(self, X, X_nn=None):
        X_nn_torch = torch.tensor(X_nn) if X_nn is not None else None
        with torch.inference_mode():
            return self._predict_nn(X_nn_torch, torch.tensor(self._predict_trees(X)))
