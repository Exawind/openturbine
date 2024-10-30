# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_matrix_operations.cpp
    test_quaternion_operations.cpp
    test_vector_operations.cpp
)
