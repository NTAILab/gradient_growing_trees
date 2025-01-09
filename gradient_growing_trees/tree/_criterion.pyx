# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport INFINITY, NAN, fabs, sqrt, exp
from libc.stdlib cimport free, malloc
from libc.string cimport memset
from libc.string cimport memcpy
# from libc.stdio cimport printf
from libc.math cimport fabs, INFINITY

import numpy as np

cimport numpy as cnp

cnp.import_array()

from sklearn.utils._typedefs cimport float64_t, intp_t, uint64_t, uint8_t
from ._parent_info cimport ParentInfo


# sklearn Criterion
from scipy.special.cython_special cimport xlogy

from sklearn.tree._utils cimport log
from sklearn.tree._utils cimport WeightedMedianCalculator

# EPSILON is used in the Poisson criterion
cdef float64_t EPSILON = 10 * np.finfo('double').eps

cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
        ParentInfo* parent,
    ) except -1 nogil:
        """Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=float64_t
            y is a buffer that can store values for n_outputs target variables
            stored as a Cython memoryview.
        sample_weight : ndarray, dtype=float64_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : float64_t
            The total weight of the samples being considered
        sample_indices : ndarray, dtype=intp_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : intp_t
            The first sample to be used on this node
        end : intp_t
            The last sample used on this node

        """
        pass

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]

        Parameters
        ----------
        n_missing: intp_t
            Number of missing values for specific feature.
        """
        pass

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """
        pass

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        This updates the collected statistics by moving sample_indices[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : intp_t
            New starting index position of the sample_indices in the right child
        """
        pass

    cdef float64_t node_impurity(self) noexcept nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of sample_indices[start:end]. This is the
        primary function of the criterion class. The smaller the impurity the
        better.
        """
        pass

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of sample_indices[start:pos] + the impurity
        of sample_indices[pos:end].

        Parameters
        ----------
        impurity_left : float64_t pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : float64_t pointer
            The memory address where the impurity of the right child should be
            stored
        """
        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of sample_indices[start:end] and save the value into dest.

        Parameters
        ----------
        dest : float64_t pointer
            The memory address where the node value should be stored.
        """
        pass

    cdef void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        pass

    cdef float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints

        This method is implemented in ClassificationCriterion and RegressionCriterion.
        """
        pass

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef float64_t impurity_left
        cdef float64_t impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                        float64_t impurity_left,
                                        float64_t impurity_right) noexcept nogil:
        """Compute the improvement in impurity.

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity_parent : float64_t
            The initial impurity of the parent node before the split

        impurity_left : float64_t
            The impurity of the left child

        impurity_right : float64_t
            The impurity of the right child

        Return
        ------
        float64_t : improvement in impurity after the split occurs
        """
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right /
                                    self.weighted_n_node_samples * impurity_right)
                                 - (self.weighted_n_left /
                                    self.weighted_n_node_samples * impurity_left)))

    cdef bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        pass

    cdef inline bint _check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
        float64_t value_left,
        float64_t value_right,
    ) noexcept nogil:
        cdef:
            bint check_lower_bound = (
                (value_left >= lower_bound) &
                (value_right >= lower_bound)
            )
            bint check_upper_bound = (
                (value_left <= upper_bound) &
                (value_right <= upper_bound)
            )
            bint check_monotonic_cst = (
                (value_left - value_right) * monotonic_cst <= 0
            )
        return check_lower_bound & check_upper_bound & check_monotonic_cst

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""

# end of sklearn Criterion


# cdef inline void _move_sums_regression(
cdef inline void _move_sums_regression(
    Criterion criterion,
    float64_t[::1] sum_1,
    float64_t[::1] sum_2,
    float64_t* weighted_n_1,
    float64_t* weighted_n_2,
    bint put_missing_in_1,
    float64_t* sum_missing,
    float64_t* sum_total,
) noexcept nogil:
    """Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values to go sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    cdef:
        intp_t i
        intp_t n_bytes = criterion.n_outputs * sizeof(float64_t)
        bint has_missing = criterion.n_missing != 0

    if has_missing and put_missing_in_1:
        # memcpy(&sum_1[0], &criterion.sum_missing[0], n_bytes)
        memcpy(&sum_1[0], &sum_missing[0], n_bytes)
        for i in range(criterion.n_outputs):
            sum_2[i] = sum_total[i] - sum_missing[i]
        weighted_n_1[0] = criterion.weighted_n_missing
        weighted_n_2[0] = criterion.weighted_n_node_samples - criterion.weighted_n_missing
    else:
        memset(&sum_1[0], 0, n_bytes)
        # Assigning sum_2 = sum_total for all outputs.
        memcpy(&sum_2[0], &sum_total[0], n_bytes)
        weighted_n_1[0] = 0.0
        weighted_n_2[0] = criterion.weighted_n_node_samples


cdef class GradientCriterion(Criterion):
    r"""Abstract gradient criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`.
    """

    cdef float64_t sq_sum_total
    cdef float64_t lam_2
    cdef float64_t lr

    # Gradient
    cdef float64_t[::1] g_sum_total    # The sum of w*y.
    cdef float64_t[::1] g_sum_left     # Same as above, but for the left side of the split
    cdef float64_t[::1] g_sum_right    # Same as above, but for the right side of the split
    cdef float64_t[::1] g_sum_missing  # Same as above, but for missing values in X
    cdef float64_t[::1] g_sum_tmp      # For children impurity
    cdef float64_t[::1] g_cur          # Gradient for the current point

    # Hessian
    cdef float64_t[::1] h_sum_total    # The sum of w*y.
    cdef float64_t[::1] h_sum_left     # Same as above, but for the left side of the split
    cdef float64_t[::1] h_sum_right    # Same as above, but for the right side of the split
    cdef float64_t[::1] h_sum_missing  # Same as above, but for missing values in X
    cdef float64_t[::1] h_sum_tmp    # For children impurity
    cdef float64_t[::1] h_cur          # Hessian diagonal for the current point

    # Node value
    cdef float64_t[::1] current_node_value  # Current node value, calculated at `init`

    # cdef float64_t[::1] sum_

    def __cinit__(self, intp_t n_outputs, intp_t n_samples, float64_t lam_2, float64_t lr):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets to be predicted

        n_samples : intp_t
            The total number of samples to fit on
        """
        # Default values
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        self.sq_sum_total = 0.0

        self.lam_2 = lam_2
        self.lr = lr

        self.g_sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.g_sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.g_sum_right = np.zeros(n_outputs, dtype=np.float64)
        self.g_sum_tmp = np.zeros(n_outputs, dtype=np.float64)
        self.g_cur = np.zeros(n_outputs, dtype=np.float64)

        self.h_sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.h_sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.h_sum_right = np.zeros(n_outputs, dtype=np.float64)
        self.h_sum_tmp = np.zeros(n_outputs, dtype=np.float64)
        self.h_cur = np.zeros(n_outputs, dtype=np.float64)

        self.current_node_value = np.zeros(n_outputs, dtype=np.float64)

        self.init_sum_missing()  # TODO: try to remove this

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
        ParentInfo* parent
    ) except -1 nogil:
        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.
        # !modified
        # self.parent_node_value = parent_node_value
        self.parent = parent
        # end of modified

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t g_ik
        cdef float64_t w_g_ik
        cdef float64_t h_ik
        cdef float64_t w_h_ik
        cdef float64_t w = 1.0
        memset(&self.g_sum_total[0], 0, self.n_outputs * sizeof(float64_t))
        memset(&self.h_sum_total[0], 0, self.n_outputs * sizeof(float64_t))


        # Calculating current node value:
        # 1. Set current value = parent value
        # 2. Calculate gradients and hessians
        # 3. Update current node value
        cdef uint64_t parent_node_id = self.parent.node_id
        cdef float64_t* parent_tree_value = self.parent.tree_value
        cdef uint64_t parent_tree_value_stride = self.parent.tree_value_stride
        cdef float64_t* parent_value_ptr = parent_tree_value + parent_node_id * parent_tree_value_stride
        cdef uint8_t is_root = (parent_node_id >= (<uint64_t> -3))
        if not is_root:
            memcpy(&self.current_node_value[0], parent_value_ptr, self.n_outputs * sizeof(float64_t))
        else:
            memset(&self.current_node_value[0], 0, self.n_outputs * sizeof(float64_t))

        # Calculate gradients at parent value
        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            self.gradients_hessians(i)  # calculate gradients and hessians for the i-th point

            for k in range(self.n_outputs):
                g_ik = self.g_cur[k]
                h_ik = self.h_cur[k]
                w_g_ik = w * g_ik
                w_h_ik = w * h_ik
                self.g_sum_total[k] += w_g_ik
                self.h_sum_total[k] += w_h_ik

            self.weighted_n_node_samples += w

        # Update current node value
        self.update_current_node_value()

        # Calculate gradients at current value
        memset(&self.g_sum_total[0], 0, self.n_outputs * sizeof(float64_t))
        memset(&self.h_sum_total[0], 0, self.n_outputs * sizeof(float64_t))
        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            self.gradients_hessians(i)  # calculate gradients and hessians for the i-th point

            for k in range(self.n_outputs):
                g_ik = self.g_cur[k]
                h_ik = self.h_cur[k]
                w_g_ik = w * g_ik
                w_h_ik = w * h_ik
                self.g_sum_total[k] += w_g_ik
                self.h_sum_total[k] += w_h_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        self.g_sum_missing = np.zeros(self.n_outputs, dtype=np.float64)
        self.h_sum_missing = np.zeros(self.n_outputs, dtype=np.float64)

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        cdef intp_t i, p, k
        cdef float64_t g_ik
        cdef float64_t w_g_ik
        cdef float64_t h_ik
        cdef float64_t w_h_ik
        cdef float64_t w = 1.0

        self.n_missing = n_missing
        if n_missing == 0:
            return

        memset(&self.g_sum_missing[0], 0, self.n_outputs * sizeof(float64_t))
        memset(&self.h_sum_missing[0], 0, self.n_outputs * sizeof(float64_t))

        self.weighted_n_missing = 0.0

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            self.gradients_hessians(i)  # calculate gradients and hessians for the i-th point

            for k in range(self.n_outputs):
                g_ik = self.g_cur[k]
                h_ik = self.h_cur[k]
                w_g_ik = w * g_ik
                w_h_ik = w * h_ik
                self.g_sum_missing[k] += w_g_ik
                self.h_sum_missing[k] += w_h_ik

            self.weighted_n_missing += w

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start
        _move_sums_regression(
            self,
            self.g_sum_left,
            self.g_sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left,
            &self.g_sum_missing[0],
            &self.g_sum_total[0],
        )
        _move_sums_regression(
            self,
            self.h_sum_left,
            self.h_sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left,
            &self.h_sum_missing[0],
            &self.h_sum_total[0],
        )
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end."""
        self.pos = self.end
        _move_sums_regression(
            self,
            self.g_sum_right,
            self.g_sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left,
            &self.g_sum_missing[0],
            &self.g_sum_total[0],
        )
        _move_sums_regression(
            self,
            self.h_sum_right,
            self.h_sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left,
            &self.h_sum_missing[0],
            &self.h_sum_total[0],
        )
        return 0

    cdef void gradients_hessians(self, intp_t sample_id) noexcept nogil:
        pass

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        cdef intp_t pos = self.pos

        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        cdef intp_t end_non_missing = self.end - self.n_missing
        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t w = 1.0
        cdef float64_t g_ik = 0.0
        cdef float64_t h_ik = 0.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                self.gradients_hessians(i)  # calculate gradients and hessians for the i-th point

                for k in range(self.n_outputs):
                    # NOTE: here loss function is approximated at `z = parent_preds + v(current_node)`, not at `z = parent_preds`!
                    g_ik = self.g_cur[k]
                    h_ik = self.h_cur[k]
                    self.g_sum_left[k] += w * g_ik
                    self.h_sum_left[k] += w * h_ik

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                self.gradients_hessians(i)  # calculate gradients and hessians for the i-th point

                for k in range(self.n_outputs):
                    g_ik = self.g_cur[k]
                    h_ik = self.h_cur[k]
                    self.g_sum_left[k] -= w * g_ik
                    self.h_sum_left[k] -= w * h_ik

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            self.g_sum_right[k] = self.g_sum_total[k] - self.g_sum_left[k]
            self.h_sum_right[k] = self.h_sum_total[k] - self.h_sum_left[k]

        self.pos = new_pos
        return 0

    cdef void update_current_node_value(self) noexcept nogil:
        """Compute the node value of sample_indices[start:end] into dest."""
        cdef intp_t k

        cdef float64_t step = 0.0  # for k-th component
        cdef float64_t lam_2 = self.lam_2
        cdef intp_t n_samples_in_parent_node = self.parent.n_samples_in_node  # number of samples in the parent node
        cdef float64_t lr = self.lr
        cdef float64_t EPS = 1.e-38
        cdef float64_t[::1] dest = self.current_node_value

        for k in range(self.n_outputs):
            step = -(self.g_sum_total[k] / max(EPS, self.h_sum_total[k] + lam_2 * n_samples_in_parent_node))
            dest[k] += lr * step

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] into dest."""
        memcpy(dest, &self.current_node_value[0], self.n_outputs * sizeof(float64_t))

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""
        if dest[0] < lower_bound:
            dest[0] = lower_bound
        elif dest[0] > upper_bound:
            dest[0] = upper_bound

    cdef float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.

        Monotonicity constraints are only supported for single-output trees we can safely assume
        n_outputs == 1.
        """
        # TODO: implement monotonicity constraints
        return 0.0
        # return (
        #     (self.sum_left[0] / (2 * self.weighted_n_left)) +
        #     (self.sum_right[0] / (2 * self.weighted_n_right))
        # )

    cdef bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current regression split"""
        # TODO: implement monotonicity constraints
        return 1
        # cdef:
        #     float64_t value_left = self.sum_left[0] / self.weighted_n_left
        #     float64_t value_right = self.sum_right[0] / self.weighted_n_right

        # return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        # TODO: implement "impturity" calculation
        return 0.0

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        The MSE proxy is derived from

            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2

        Neglecting constant terms, this gives:

            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
        """
        cdef intp_t k
        cdef float64_t proxy_impurity_left = 0.0
        cdef float64_t proxy_impurity_right = 0.0

        cdef float64_t step_left = 0.0  # for k-th component, u_k
        cdef float64_t step_right = 0.0  # for k-th component, v_k

        cdef float64_t lam_2 = self.lam_2
        # cdef intp_t n_samples = self.n_samples
        cdef intp_t n_samples_in_node = self.end - self.start
        cdef float64_t impurity = 0.0
        cdef float64_t EPS = 1.e-38

        # approximated loss function (up to second term)
        for k in range(self.n_outputs):
            step_left = -(self.g_sum_left[k] / max(EPS, self.h_sum_left[k] + lam_2 * n_samples_in_node))
            step_right = -(self.g_sum_right[k] / max(EPS, self.h_sum_right[k] + lam_2 * n_samples_in_node))
            proxy_impurity_left += step_left * (self.g_sum_left[k] + step_left * self.h_sum_left[k] * 0.5)
            proxy_impurity_right += step_right * (self.g_sum_right[k] + step_right * self.h_sum_right[k] * 0.5)

        # return (proxy_impurity_left / self.weighted_n_left +
        #         proxy_impurity_right / self.weighted_n_right)
        impurity = (proxy_impurity_left + proxy_impurity_right) / n_samples_in_node
        return -impurity

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        """
        # TODO: implement "impurity" calculation
        impurity_left[0] = 0.0
        impurity_right[0] = 0.0


cdef class MSE2(GradientCriterion):
    cdef void gradients_hessians(self, intp_t sample_id) noexcept nogil:
        cdef float64_t[::1] g = self.g_cur
        cdef float64_t[::1] h = self.h_cur
        cdef intp_t n_outputs = self.n_outputs
        cdef const float64_t[::1] cur_node_values = self.current_node_value

        for out_id in range(n_outputs):
            g[out_id] = 2.0 * (cur_node_values[out_id] - self.y[sample_id, out_id])
            h[out_id] = 2.0


cdef void softmax(float64_t* dst, const float64_t* values, intp_t size) noexcept nogil:
    cdef float64_t denom = 0.0
    cdef float64_t max_value = -INFINITY
    for k in range(size):
        max_value = max(max_value, values[k])
    for k in range(size):
        dst[k] = exp(values[k] - max_value)
        denom += dst[k]
    # denom = max(denom, 1.e-38)
    if denom < 1.e-38:
        denom = 1.0 / size
        for k in range(size):
            dst[k] = denom
    else:
        for k in range(size):
            dst[k] /= denom

cdef class CrossEntropy2(GradientCriterion):
    cdef void gradients_hessians(self, intp_t sample_id) noexcept nogil:
        cdef float64_t[::1] g = self.g_cur
        cdef float64_t[::1] h = self.h_cur
        cdef intp_t n_outputs = self.n_outputs
        cdef const float64_t[::1] cur_node_values = self.current_node_value
        cdef float64_t* s = &self.g_sum_tmp[0]  # using an empty buffer for softmax result

        softmax(s, &cur_node_values[0], n_outputs)

        for out_id in range(n_outputs):
            g[out_id] = s[out_id] - self.y[sample_id, out_id]
            h[out_id] = s[out_id] * (1.0 - s[out_id])


cdef float64_t dot(const float64_t[::1] dst, const float64_t* values, intp_t size) noexcept nogil:
    cdef float64_t result = 0.0
    for k in range(size):
        result += dst[k] * values[k]
    return result

cdef class GeneralizedLogLoss(GradientCriterion):
    cdef void gradients_hessians(self, intp_t sample_id) noexcept nogil:
        cdef float64_t EPS = 1.e-38
        cdef float64_t[::1] g = self.g_cur
        cdef float64_t[::1] h = self.h_cur
        cdef intp_t n_outputs = self.n_outputs
        cdef const float64_t[::1] cur_node_values = self.current_node_value
        cdef float64_t* s = &self.g_sum_tmp[0]  # using an empty buffer for softmax result
        cdef float64_t y_dot_s = 0.0
        cdef float64_t cur_y = 0.0

        softmax(s, &cur_node_values[0], n_outputs)
        y_dot_s = dot(self.y[sample_id], s, self.n_outputs)

        for out_id in range(n_outputs):
            cur_y = self.y[sample_id, out_id]
            g[out_id] = s[out_id] * (1.0 - cur_y / max(y_dot_s, EPS))
            h[out_id] = s[out_id] * (1.0 - s[out_id] - cur_y * (y_dot_s - cur_y * s[out_id]) / max(y_dot_s * y_dot_s, EPS))


cdef class ArbitraryLoss(GradientCriterion):
    cdef public object loss_fn

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    cdef void gradients_hessians(self, intp_t sample_id) noexcept nogil:
        cdef float64_t[::1] g = self.g_cur
        cdef float64_t[::1] h = self.h_cur
        cdef intp_t n_outputs = self.n_outputs
        cdef const float64_t[::1] cur_node_values = self.current_node_value

        with gil:
            self.loss_fn(sample_id, self.y, cur_node_values, g, h)


# Batch criteria


cdef class BatchGradientCriterion(Criterion):
    r"""Abstract gradient criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`.
    """

    cdef float64_t lam_2
    cdef float64_t lr

    # Gradient
    cdef float64_t[::1] g_sum_total    # The sum of w*y.
    cdef float64_t[::1] g_sum_left     # Same as above, but for the left side of the split
    cdef float64_t[::1] g_sum_right    # Same as above, but for the right side of the split
    cdef float64_t[::1] g_sum_missing  # Same as above, but for missing values in X
    cdef float64_t[::1] g_sum_tmp      # For children impurity
    cdef float64_t[:, :] g_cur       # Gradient for the node points

    # Hessian
    cdef float64_t[::1] h_sum_total    # The sum of w*y.
    cdef float64_t[::1] h_sum_left     # Same as above, but for the left side of the split
    cdef float64_t[::1] h_sum_right    # Same as above, but for the right side of the split
    cdef float64_t[::1] h_sum_missing  # Same as above, but for missing values in X
    cdef float64_t[::1] h_sum_tmp      # For children impurity
    cdef float64_t[:, :] h_cur       # Hessian diagonal for the node points

    # Node value
    cdef public float64_t[::1] current_node_value  # Current node value, calculated at `init`

    # Update steps
    cdef intp_t n_update_iterations

    def __cinit__(self, intp_t n_outputs, intp_t n_samples, float64_t lam_2, float64_t lr,
                  intp_t n_update_iterations):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets to be predicted

        n_samples : intp_t
            The total number of samples to fit on
        """
        # Default values
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_update_iterations = n_update_iterations
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        self.lam_2 = lam_2
        self.lr = lr

        self.g_sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.g_sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.g_sum_right = np.zeros(n_outputs, dtype=np.float64)
        self.g_sum_tmp = np.zeros(n_outputs, dtype=np.float64)
        self.g_cur = np.zeros((n_samples, n_outputs), dtype=np.float64)

        self.h_sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.h_sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.h_sum_right = np.zeros(n_outputs, dtype=np.float64)
        self.h_sum_tmp = np.zeros(n_outputs, dtype=np.float64)
        self.h_cur = np.zeros((n_samples, n_outputs), dtype=np.float64)

        self.current_node_value = np.zeros(n_outputs, dtype=np.float64)

        self.init_sum_missing()  # TODO: try to remove this

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
        ParentInfo* parent
    ) except -1 nogil:
        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.
        # !modified
        # self.parent_node_value = parent_node_value
        self.parent = parent
        # end of modified

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t g_ik
        cdef float64_t w_g_ik
        cdef float64_t h_ik
        cdef float64_t w_h_ik
        cdef float64_t w = 1.0
        memset(&self.g_sum_total[0], 0, self.n_outputs * sizeof(float64_t))
        memset(&self.h_sum_total[0], 0, self.n_outputs * sizeof(float64_t))

        # Calculating current node value:
        # 1. Set current value = parent value
        # 2. Calculate gradients and hessians
        # 3. Update current node value
        cdef uint64_t parent_node_id = self.parent.node_id
        cdef float64_t* parent_tree_value = self.parent.tree_value
        cdef uint64_t parent_tree_value_stride = self.parent.tree_value_stride
        cdef float64_t* parent_value_ptr = parent_tree_value + parent_node_id * parent_tree_value_stride
        cdef uint8_t is_root = (parent_node_id >= (<uint64_t> -3))
        cdef uint8_t need_recalculate_grads = 0
        if not is_root:
            memcpy(&self.current_node_value[0], parent_value_ptr, self.n_outputs * sizeof(float64_t))
        else:
            # memset(&self.current_node_value[0], 0, self.n_outputs * sizeof(float64_t))
            pass  # current_node_value can be prefilled after criterion construction

        need_recalculate_grads = (end - start >= len(sample_indices))
        for cur_iter in range(self.n_update_iterations):
            if need_recalculate_grads:
                self.batch_gradients_hessians(start, end)  # calculate gradients and hessians for the node points

            # Calculate gradients at parent value
            for p in range(start, end):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    g_ik = self.g_cur[i, k]
                    h_ik = self.h_cur[i, k]
                    w_g_ik = w * g_ik
                    w_h_ik = w * h_ik
                    self.g_sum_total[k] += w_g_ik
                    self.h_sum_total[k] += w_h_ik

                self.weighted_n_node_samples += w

            # Update current node value
            self.update_current_node_value()
            need_recalculate_grads = True

        # Calculate gradients at current value
        memset(&self.g_sum_total[0], 0, self.n_outputs * sizeof(float64_t))
        memset(&self.h_sum_total[0], 0, self.n_outputs * sizeof(float64_t))

        self.batch_gradients_hessians(start, end)  # calculate actual gradients and hessians for the node points

        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                g_ik = self.g_cur[i, k]
                h_ik = self.h_cur[i, k]
                w_g_ik = w * g_ik
                w_h_ik = w * h_ik
                self.g_sum_total[k] += w_g_ik
                self.h_sum_total[k] += w_h_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        self.g_sum_missing = np.zeros(self.n_outputs, dtype=np.float64)
        self.h_sum_missing = np.zeros(self.n_outputs, dtype=np.float64)

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        cdef intp_t i, p, k
        cdef float64_t g_ik
        cdef float64_t w_g_ik
        cdef float64_t h_ik
        cdef float64_t w_h_ik
        cdef float64_t w = 1.0

        self.n_missing = n_missing
        if n_missing == 0:
            return

        memset(&self.g_sum_missing[0], 0, self.n_outputs * sizeof(float64_t))
        memset(&self.h_sum_missing[0], 0, self.n_outputs * sizeof(float64_t))

        self.weighted_n_missing = 0.0

        # if self.end - n_missing < self.end:
        #     # TODO: maybe the line below is redundant?
        #     self.batch_gradients_hessians(self.end - n_missing, self.end)  # calculate gradients and hessians for the node points

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            # self.gradients_hessians(i)  # calculate gradients and hessians for the i-th point

            for k in range(self.n_outputs):
                g_ik = self.g_cur[i, k]
                h_ik = self.h_cur[i, k]
                w_g_ik = w * g_ik
                w_h_ik = w * h_ik
                self.g_sum_missing[k] += w_g_ik
                self.h_sum_missing[k] += w_h_ik

            self.weighted_n_missing += w

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start
        _move_sums_regression(
            self,
            self.g_sum_left,
            self.g_sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left,
            &self.g_sum_missing[0],
            &self.g_sum_total[0],
        )
        _move_sums_regression(
            self,
            self.h_sum_left,
            self.h_sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left,
            &self.h_sum_missing[0],
            &self.h_sum_total[0],
        )
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end."""
        self.pos = self.end
        _move_sums_regression(
            self,
            self.g_sum_right,
            self.g_sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left,
            &self.g_sum_missing[0],
            &self.g_sum_total[0],
        )
        _move_sums_regression(
            self,
            self.h_sum_right,
            self.h_sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left,
            &self.h_sum_missing[0],
            &self.h_sum_total[0],
        )
        # self.batch_gradients_hessians(self.start, self.end)  # calculate actual gradients and hessians for the node points
        return 0

    cdef void batch_gradients_hessians(self, intp_t start, intp_t end) noexcept nogil:
        pass

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        cdef intp_t pos = self.pos

        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        cdef intp_t end_non_missing = self.end - self.n_missing
        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t w = 1.0
        cdef float64_t g_ik = 0.0
        cdef float64_t h_ik = 0.0

        # Update statistics up to new_pos
        if (new_pos - pos) <= (end_non_missing - new_pos):
            # self.batch_gradients_hessians(pos, new_pos)  # gradients are already calculated
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                # self.gradients_hessians(i)  # calculate gradients and hessians for the i-th point

                for k in range(self.n_outputs):
                    # NOTE: here loss function is approximated at `z = parent_preds + v(current_node)`, not at `z = parent_preds`!
                    g_ik = self.g_cur[i, k]
                    h_ik = self.h_cur[i, k]
                    self.g_sum_left[k] += w * g_ik
                    self.h_sum_left[k] += w * h_ik

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            # self.batch_gradients_hessians(new_pos, end_non_missing)  # gradients are already calculated
            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                # self.gradients_hessians(i)  # calculate gradients and hessians for the i-th point

                for k in range(self.n_outputs):
                    g_ik = self.g_cur[i, k]
                    h_ik = self.h_cur[i, k]
                    self.g_sum_left[k] -= w * g_ik
                    self.h_sum_left[k] -= w * h_ik

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            self.g_sum_right[k] = self.g_sum_total[k] - self.g_sum_left[k]
            self.h_sum_right[k] = self.h_sum_total[k] - self.h_sum_left[k]

        self.pos = new_pos
        return 0

    cdef void update_current_node_value(self) noexcept nogil:
        """Compute the node value of sample_indices[start:end] into dest."""
        cdef intp_t k

        cdef float64_t step = 0.0  # for k-th component
        cdef float64_t lam_2 = self.lam_2
        cdef intp_t n_samples_in_parent_node = self.parent.n_samples_in_node  # number of samples in the parent node
        cdef float64_t lr = self.lr
        cdef float64_t EPS = 1.e-38
        cdef float64_t[::1] dest = self.current_node_value

        for k in range(self.n_outputs):
            step = -(self.g_sum_total[k] / max(EPS, self.h_sum_total[k] + lam_2 * n_samples_in_parent_node))
            dest[k] += lr * step

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] into dest."""
        memcpy(dest, &self.current_node_value[0], self.n_outputs * sizeof(float64_t))

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""
        if dest[0] < lower_bound:
            dest[0] = lower_bound
        elif dest[0] > upper_bound:
            dest[0] = upper_bound

    cdef float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.

        Monotonicity constraints are only supported for single-output trees we can safely assume
        n_outputs == 1.
        """
        # TODO: implement monotonicity constraints
        return 0.0

    cdef bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current regression split"""
        # TODO: implement monotonicity constraints
        return 1

    cdef float64_t node_impurity(self) noexcept nogil:
        return 0.0

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        cdef intp_t k
        cdef float64_t proxy_impurity_left = 0.0
        cdef float64_t proxy_impurity_right = 0.0

        cdef float64_t step_left = 0.0  # for k-th component, u_k
        cdef float64_t step_right = 0.0  # for k-th component, v_k

        cdef float64_t lam_2 = self.lam_2
        # cdef intp_t n_samples = self.n_samples
        cdef intp_t n_samples_in_node = self.end - self.start
        cdef float64_t impurity = 0.0
        cdef float64_t EPS = 1.e-38

        # approximated loss function (up to second term)
        for k in range(self.n_outputs):
            step_left = -(self.g_sum_left[k] / max(EPS, self.h_sum_left[k] + lam_2 * n_samples_in_node))
            step_right = -(self.g_sum_right[k] / max(EPS, self.h_sum_right[k] + lam_2 * n_samples_in_node))
            proxy_impurity_left += step_left * (self.g_sum_left[k] + step_left * self.h_sum_left[k] * 0.5)
            proxy_impurity_right += step_right * (self.g_sum_right[k] + step_right * self.h_sum_right[k] * 0.5)

        impurity = (proxy_impurity_left + proxy_impurity_right) / n_samples_in_node
        return -impurity

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        impurity_left[0] = 0.0
        impurity_right[0] = 0.0


cdef class BatchMSE(BatchGradientCriterion):
    cdef void batch_gradients_hessians(self, intp_t start, intp_t end) noexcept nogil:
        cdef float64_t[:, :] g = self.g_cur
        cdef float64_t[:, :] h = self.h_cur
        cdef intp_t n_outputs = self.n_outputs
        cdef const float64_t[::1] cur_node_values = self.current_node_value
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef intp_t i = 0

        for p in range(start, end):
            i = sample_indices[p]
            for out_id in range(n_outputs):
                g[i, out_id] = 2.0 * (cur_node_values[out_id] - self.y[i, out_id])
                h[i, out_id] = 2.0


cdef class BatchGeneralizedLogLoss(BatchGradientCriterion):
    cdef void batch_gradients_hessians(self, intp_t start, intp_t end) noexcept nogil:
        cdef float64_t EPS = 1.e-38
        cdef float64_t[:, :] g = self.g_cur
        cdef float64_t[:, :] h = self.h_cur
        cdef intp_t n_outputs = self.n_outputs
        cdef const float64_t[::1] cur_node_values = self.current_node_value
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef float64_t* s = &self.g_sum_tmp[0]  # using an empty buffer for softmax result
        cdef float64_t y_dot_s = 0.0
        cdef float64_t cur_y = 0.0
        cdef intp_t i = 0

        softmax(s, &cur_node_values[0], n_outputs)

        for p in range(start, end):
            i = sample_indices[p]
            y_dot_s = dot(self.y[i], s, self.n_outputs)
            for out_id in range(n_outputs):
                cur_y = self.y[i, out_id]
                g[i, out_id] = s[out_id] * (1.0 - cur_y / max(y_dot_s, EPS))
                h[i, out_id] = s[out_id] * (1.0 - s[out_id] - cur_y * (y_dot_s - cur_y * s[out_id]) / max(y_dot_s * y_dot_s, EPS))


cdef class BatchArbitraryLoss(BatchGradientCriterion):
    cdef public object loss_fn

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    cdef void batch_gradients_hessians(self, intp_t start, intp_t end) noexcept nogil:
        cdef float64_t[:, :] g = self.g_cur
        cdef float64_t[:, :] h = self.h_cur
        cdef intp_t n_outputs = self.n_outputs
        cdef const float64_t[::1] cur_node_values = self.current_node_value
        cdef const intp_t[:] sample_indices = self.sample_indices

        with gil:
            self.loss_fn(self.y, sample_indices[start:end], cur_node_values, g, h)
