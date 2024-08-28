# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_compute_number_of_non_zeros.cpp
    test_copy_into_sparse_matrix.cpp
    test_populate_sparse_indices.cpp
    test_populate_sparse_row_ptrs.cpp
)
