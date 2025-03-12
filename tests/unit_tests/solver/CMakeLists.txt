# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_compute_k_col_inds.cpp
    test_compute_k_row_ptrs.cpp
    test_compute_number_of_non_zeros_constraints.cpp
    test_fill_unshifted_row_ptrs.cpp
    test_populate_sparse_row_ptrs_col_inds_constraints.cpp
    test_populate_sparse_row_ptrs_col_inds_transpose.cpp
)
