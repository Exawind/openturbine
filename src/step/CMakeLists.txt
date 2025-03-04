target_sources(openturbine_library PRIVATE)

install(FILES
        assemble_constraints_matrix.hpp
        assemble_constraints_residual.hpp
        assemble_inertia_matrix_beams.hpp
        assemble_residual_vector_beams.hpp
        assemble_stiffness_matrix_beams.hpp
        assemble_system_matrix.hpp
        assemble_system_residual.hpp
        assemble_tangent_operator.hpp
        calculate_convergence_error.hpp
        predict_next_state.hpp
        reset_constraints.hpp
        solve_system.hpp
        step.hpp
        step_parameters.hpp
        update_constraint_prediction.hpp
        update_constraint_variables.hpp
        update_state_prediction.hpp
        update_system_variables_beams.hpp
        update_system_variables.hpp
        update_system_variables_masses.hpp
        update_system_variables_springs.hpp
        update_tangent_operator.hpp
        DESTINATION include/OpenTurbine/step/
)
