target_sources(openturbine_library PRIVATE)

install(FILES
        calculate_error_sum_squares.hpp
        compute_b_num_non_zero.hpp
        compute_k_col_inds.hpp
        compute_k_num_non_zero.hpp
        compute_k_row_ptrs.hpp
        compute_number_of_non_zeros_constraints.hpp
        compute_num_system_dofs.hpp
        condition_system.hpp
        contribute_beams_to_sparse_matrix.hpp
        contribute_beams_to_vector.hpp
        contribute_constraints_system_residual_to_vector.hpp
        contribute_forces_to_vector.hpp
        contribute_lambda_to_vector.hpp
        contribute_masses_to_sparse_matrix.hpp
        contribute_masses_to_vector.hpp
        contribute_springs_to_sparse_matrix.hpp
        contribute_springs_to_vector.hpp
        copy_constraints_residual_to_vector.hpp
        copy_constraints_to_sparse_matrix.hpp
        copy_constraints_transpose_to_sparse_matrix.hpp
        create_constraints_matrix_full.hpp
        create_full_matrix.hpp
        create_matrix_spadd.hpp
        create_sparse_dense_solver.hpp
        create_system_matrix_full.hpp
        create_transpose_matrix_full.hpp
        fill_unshifted_row_ptrs.hpp
        populate_sparse_row_ptrs_col_inds_constraints.hpp
        populate_sparse_row_ptrs_col_inds_transpose.hpp
        solver.hpp
        DESTINATION include/OpenTurbine/solver/
)
