# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_integrate_stiffness_matrix.cpp
    test_integrate_inertia_matrix.cpp
    test_integrate_residual_vector.cpp
    test_update_node_state.cpp
    test_calculate_RR0.cpp
    test_rotate_section_matrix.cpp
    test_calculate_strain.cpp
    test_calculate_mass_matrix_components.cpp
    test_calculate_temporary_variables.cpp
    test_calculate_force_FC.cpp
    test_calculate_force_FD.cpp
    test_calculate_inertial_forces.cpp
    test_calculate_gravity_force.cpp
    test_calculate_Ouu.cpp
    test_calculate_Puu.cpp
    test_calculate_Quu.cpp
    test_calculate_gyroscopic_matrix.cpp
    test_calculate_inertia_stiffness_matrix.cpp
)

add_subdirectory(springs)