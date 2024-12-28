from sklearn.utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint8_t, uint32_t


cdef struct ParentInfo:
    # Structure to store information about the parent of a node
    # This is passed to the splitter, to provide information about the previous split

    float64_t lower_bound           # the lower bound of the parent's impurity
    float64_t upper_bound           # the upper bound of the parent's impurity
    float64_t impurity              # the impurity of the parent
    intp_t n_constant_features      # the number of constant features found in parent
    # !modified
    float64_t* tree_value
    intp_t tree_value_stride
    intp_t node_id
    intp_t n_samples_in_node
    # !end of modified
