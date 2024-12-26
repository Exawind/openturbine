# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_assemble_residual_vector_springs.cpp
    test_assemble_stiffness_matrix_springs.cpp
)
